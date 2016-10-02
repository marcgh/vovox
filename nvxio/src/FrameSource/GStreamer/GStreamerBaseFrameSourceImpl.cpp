/*
# Copyright (c) 2014-2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef USE_GSTREAMER

#include <memory>

#include "NVXIO/Application.hpp"
#include "NVXIO/Utility.hpp"
#include "FrameSource/GStreamer/GStreamerBaseFrameSourceImpl.hpp"

#include <cuda_runtime_api.h>

#include <gst/pbutils/missing-plugins.h>
#include <gst/app/gstappsink.h>

namespace nvxio
{

void convertFrame(vx_context vxContext, vx_image frame,
                  const FrameSource::Parameters & configuration,
                  vx_imagepatch_addressing_t & decodedImageAddr,
                  void * decodedPtr, bool is_cuda, void *& devMem,
                  size_t & devMemPitch, vx_image & scaledImage);

GStreamerBaseFrameSourceImpl::GStreamerBaseFrameSourceImpl(vx_context context, FrameSource::SourceType type, const std::string & name):
    FrameSource(type, name),
    pipeline(NULL), bus(NULL),
    end(true),
    vxContext(context),
    sink(NULL),
    devMem(NULL),
    devMemPitch(0),
    scaledImage(NULL)
{
}

void GStreamerBaseFrameSourceImpl::newGstreamerPad(GstElement * /*elem*/, GstPad *pad, gpointer data)
{
    GstElement *color = (GstElement *) data;

    std::unique_ptr<GstPad, GStreamerObjectDeleter> sinkpad(gst_element_get_static_pad (color, "sink"));
    if (!sinkpad)
    {
        NVXIO_PRINT("Gstreamer: no pad named sink");
        return;
    }

    gst_pad_link(pad, sinkpad.get());
}

bool GStreamerBaseFrameSourceImpl::open()
{
    if (pipeline)
    {
        close();
    }

    if (!InitializeGstPipeLine())
    {
        NVXIO_PRINT("Cannot initialize Gstreamer pipeline");
        return false;
    }

    NVXIO_ASSERT(!end);

    return true;
}

FrameSource::FrameStatus GStreamerBaseFrameSourceImpl::extractFrameParams(GstCaps* bufferCaps,
    gint & width, gint & height, gint & fps, gint & depth)
{
    // fail out if no caps
    assert(gst_caps_get_size(bufferCaps) == 1);
    GstStructure * structure = gst_caps_get_structure(bufferCaps, 0);

    // fail out if width or height are 0
    if (!gst_structure_get_int(structure, "width", &width))
    {
        NVXIO_PRINT("Failed to retrieve width");
        return FrameSource::CLOSED;
    }
    if (!gst_structure_get_int(structure, "height", &height))
    {
        NVXIO_PRINT("Failed to retrieve height");
        return FrameSource::CLOSED;
    }

    gint num = 0, denom = 1;
    if (!gst_structure_get_fraction(structure, "framerate", &num, &denom))
    {
        NVXIO_PRINT("Cannot query video fps");
        return FrameSource::CLOSED;
    }
    else
        fps = static_cast<float>(num) / denom;

    depth = 3;
#if GST_VERSION_MAJOR > 0
    depth = 0;
    const gchar* name = gst_structure_get_name(structure);
    const gchar* format = gst_structure_get_string(structure, "format");

    if (!name || !format)
    {
        return FrameSource::CLOSED;
    }

    // we support 3 types of data:
    //     video/x-raw, format=RGBA  -> 8bit, 4 channels
    //     video/x-raw, format=RGB   -> 8bit, 3 channels
    //     video/x-raw, format=GRAY8 -> 8bit, 1 channel
    if (strcasecmp(name, "video/x-raw") == 0)
    {
        if (strcasecmp(format, "RGBA") == 0)
            depth = 4;
        else if (strcasecmp(format, "RGB") == 0)
            depth = 3;
        else if(strcasecmp(format, "GRAY8") == 0)
            depth = 1;
    }
#endif

    return FrameSource::OK;
}

FrameSource::FrameStatus GStreamerBaseFrameSourceImpl::fetch(vx_image image, vx_uint32 /*timeout*/)
{
    if (end)
    {
        close();
        return FrameSource::CLOSED;
    }

    handleGStreamerMessages();

    if (gst_app_sink_is_eos(GST_APP_SINK(sink)))
    {
        close();
        return FrameSource::CLOSED;
    }

#if GST_VERSION_MAJOR == 0
    std::unique_ptr<GstBuffer, GStreamerObjectDeleter> bufferHolder(
        gst_app_sink_pull_buffer(GST_APP_SINK(sink)));
    GstBuffer* buffer = bufferHolder.get();
#else
    std::unique_ptr<GstSample, GStreamerObjectDeleter> sample;

    if (sampleFirstFrame)
    {
        sample = std::move(sampleFirstFrame);
        NVXIO_ASSERT(sampleFirstFrame == nullptr);
    }
    else
        sample.reset(gst_app_sink_pull_sample(GST_APP_SINK(sink)));

    if (!sample)
    {
        close();
        return FrameSource::CLOSED;
    }

    GstBuffer* buffer = gst_sample_get_buffer(sample.get());
#endif

#if GST_VERSION_MAJOR == 0
    std::unique_ptr<GstCaps, GStreamerObjectDeleter> bufferCapsHolder(gst_buffer_get_caps(buffer));
    GstCaps* bufferCaps = bufferCapsHolder.get();
#else
    GstCaps* bufferCaps = gst_sample_get_caps(sample.get());
#endif

    gint width, height, fps, depth;
    if (extractFrameParams(bufferCaps, width, height, fps, depth) == FrameSource::CLOSED || depth == 0)
    {
        close();
        return FrameSource::CLOSED;
    }

    vx_imagepatch_addressing_t decodedImageAddr;
    decodedImageAddr.dim_x = width;
    decodedImageAddr.dim_y = height;
    decodedImageAddr.stride_x = depth;
    // GStreamer uses as stride width rounded up to the nearest multiple of 4
    decodedImageAddr.stride_y = ((width*depth+3)/4)*4;
    decodedImageAddr.scale_x = decodedImageAddr.scale_y = VX_SCALE_UNITY;

#if GST_VERSION_MAJOR == 0
    void * decodedPtr = GST_BUFFER_DATA(buffer);
#else
    GstMapInfo info;

    gboolean success = gst_buffer_map(buffer, &info, (GstMapFlags)GST_MAP_READ);
    if (!success)
    {
        NVXIO_PRINT("GStreamer: unable to map buffer");
        close();
        return FrameSource::CLOSED;
    }

    void * decodedPtr = info.data;
#endif

    convertFrame(vxContext,
                 image,
                 configuration,
                 decodedImageAddr,
                 decodedPtr,
                 false,
                 devMem,
                 devMemPitch,
                 scaledImage);

#if GST_VERSION_MAJOR != 0
    gst_buffer_unmap(buffer, &info);
#endif

    return FrameSource::OK;
}

FrameSource::Parameters GStreamerBaseFrameSourceImpl::getConfiguration()
{
    return configuration;
}

bool GStreamerBaseFrameSourceImpl::setConfiguration(const FrameSource::Parameters& params)
{
    bool result = true;

    if (end)
    {
        configuration.frameHeight = params.frameHeight;
        configuration.frameWidth = params.frameWidth;
    }
    else
    {
        if ((params.frameWidth != (vx_uint32)-1) && (params.frameWidth != configuration.frameWidth))
            result = false;
        if ((params.frameHeight != (vx_uint32)-1) && (params.frameHeight != configuration.frameHeight))
            result = false;
    }

    if ((params.fps != (vx_uint32)-1) && (params.fps != configuration.fps))
        result = false;

    configuration.format = params.format;

    return result;
}

void GStreamerBaseFrameSourceImpl::close()
{
    handleGStreamerMessages();
    FinalizeGstPipeLine();

    if (devMem != NULL)
    {
        cudaFree(devMem);
        devMem = NULL;
    }

    if (scaledImage)
    {
        vxReleaseImage(&scaledImage);
        scaledImage = NULL;
    }
}

void GStreamerBaseFrameSourceImpl::handleGStreamerMessages()
{
    std::unique_ptr<GstMessage, GStreamerObjectDeleter> msg;
    GError *err = NULL;
    gchar *debug = NULL;
    GstStreamStatusType tp;
    GstElement * elem = NULL;

    if (!bus)
        return;

    while (gst_bus_have_pending(bus))
    {
        msg.reset(gst_bus_pop(bus));

        if (gst_is_missing_plugin_message(msg.get()))
        {
            NVXIO_PRINT("GStreamer: your gstreamer installation is missing a required plugin!");
            end = true;
        }
        else
        {
            switch (GST_MESSAGE_TYPE(msg.get()))
            {
                case GST_MESSAGE_STATE_CHANGED:
                    GstState oldstate, newstate, pendstate;
                    gst_message_parse_state_changed(msg.get(), &oldstate, &newstate, &pendstate);
                    break;
                case GST_MESSAGE_ERROR:
                {
                    gst_message_parse_error(msg.get(), &err, &debug);
                    std::unique_ptr<char[], GlibDeleter> name(gst_element_get_name(GST_MESSAGE_SRC(msg.get())));

                    NVXIO_PRINT("GStreamer Plugin: Embedded video playback halted; module %s reported: %s",
                           name.get(), err->message);

                    g_error_free(err);
                    g_free(debug);
                    end = true;
                    break;
                }
                case GST_MESSAGE_EOS:
                    end = true;
                    break;
                case GST_MESSAGE_STREAM_STATUS:
                    gst_message_parse_stream_status(msg.get(), &tp, &elem);
                    break;
                default:
                    break;
            }
        }
    }
}

void GStreamerBaseFrameSourceImpl::FinalizeGstPipeLine()
{
    if (pipeline)
    {
        handleGStreamerMessages();

        gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_NULL);
        handleGStreamerMessages();

        gst_object_unref(GST_OBJECT(bus));
        bus = NULL;

        gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        pipeline = NULL;
    }
}

GStreamerBaseFrameSourceImpl::~GStreamerBaseFrameSourceImpl()
{
    close();
}

}

#endif // USE_GSTREAMER
