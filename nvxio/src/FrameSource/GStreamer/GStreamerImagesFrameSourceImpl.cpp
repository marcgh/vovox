/*
# Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
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

#include "FrameSource/GStreamer/GStreamerImagesFrameSourceImpl.hpp"

#if GST_VERSION_MAJOR == 0
#define DECODEBIN_ELEM "decodebin2"
#else
#define DECODEBIN_ELEM "decodebin"
#endif

#include <gst/app/gstappsink.h>

#include <map>

namespace nvxio
{

GStreamerImagesFrameSourceImpl::GStreamerImagesFrameSourceImpl(vx_context context, FrameSource::SourceType type,
                                                               const std::string & fileName_) :
    GStreamerBaseFrameSourceImpl(context, type, "GstreamerImagesFrameSource"),
    fileName(fileName_)
{
    const std::map<std::string, guint> features_list =
    {
        { "nvjpegenc", GST_RANK_NONE },
        { "nvjpegdec", GST_RANK_NONE }
    };

    std::unique_ptr<GstElementFactory, GStreamerObjectDeleter> factory;
    for (auto p : features_list)
    {
        factory.reset(gst_element_factory_find(p.first.c_str()));
        if (factory)
        {
            gst_plugin_feature_set_rank(GST_PLUGIN_FEATURE(factory.get()), p.second);
        }
    }
}

GstAutoplugSelectResult GStreamerImagesFrameSourceImpl::autoPlugSelect(GstElement *, GstPad *,
                              GstCaps * caps, GstElementFactory *, gpointer)
{
    std::unique_ptr<char[], GlibDeleter> capsStr(gst_caps_to_string(caps));
    if (strstr(capsStr.get(), "image"))
    {
        return GST_AUTOPLUG_SELECT_TRY;
    }
    else
    {
        return GST_AUTOPLUG_SELECT_EXPOSE;
    }
}

bool GStreamerImagesFrameSourceImpl::InitializeGstPipeLine()
{
    GstStateChangeReturn status;
    end = true;

    pipeline = GST_PIPELINE(gst_pipeline_new(NULL));
    if (pipeline == NULL)
    {
        NVXIO_PRINT("Cannot create Gstreamer pipeline");
        return false;
    }

    bus = gst_pipeline_get_bus(GST_PIPELINE (pipeline));

    // create filesrc
    bool isImageSequence = sourceType == FrameSource::IMAGE_SEQUENCE_SOURCE;

    const char * elementFactoryName =
            isImageSequence ? "multifilesrc" : "filesrc";

    GstElement * filesrc = gst_element_factory_make(elementFactoryName, NULL);
    if (filesrc == NULL)
    {
        NVXIO_PRINT("Cannot create filesrc");
        FinalizeGstPipeLine();

        return false;
    }

    g_object_set(G_OBJECT(filesrc), "location", fileName.c_str(), NULL);

    if (isImageSequence)
        g_object_set(G_OBJECT(filesrc), "start-index", 1, NULL);

#if GST_VERSION_MAJOR == 0
    if (isImageSequence)
    {
        std::unique_ptr<GstCaps, GStreamerObjectDeleter> caps(
                    gst_caps_new_simple("image/png", "framerate", GST_TYPE_FRACTION, 30, 1, NULL));
        if (caps)
        {
            g_object_set(G_OBJECT(filesrc), "caps", caps.get(), NULL);
        }
    }
#endif

    gst_bin_add(GST_BIN(pipeline), filesrc);

    // create decodebin[2] element
    GstElement * decodebin = gst_element_factory_make(DECODEBIN_ELEM, NULL);
    if (decodebin == NULL)
    {
        NVXIO_PRINT("Cannot create " DECODEBIN_ELEM " element");
        FinalizeGstPipeLine();

        return false;
    }

    gst_bin_add(GST_BIN(pipeline), decodebin);

    // create color convert element
    GstElement * color = gst_element_factory_make(COLOR_ELEM, NULL);
    if (color == NULL)
    {
        NVXIO_PRINT("Cannot create %s element", COLOR_ELEM);
        FinalizeGstPipeLine();

        return false;
    }

    gst_bin_add(GST_BIN(pipeline), color);

    // create appsink element
    sink = gst_element_factory_make("appsink", NULL);
    if (sink == NULL)
    {
        NVXIO_PRINT("Cannot create appsink element");
        FinalizeGstPipeLine();

        return false;
    }

    gst_bin_add(GST_BIN(pipeline), sink);

    g_signal_connect(decodebin, "autoplug-select", G_CALLBACK(GStreamerImagesFrameSourceImpl::autoPlugSelect), NULL);
    g_signal_connect(decodebin, "pad-added", G_CALLBACK(GStreamerBaseFrameSourceImpl::newGstreamerPad), color);

    // link elements
    if (!gst_element_link(filesrc, decodebin))
    {
        NVXIO_PRINT("GStreamer: cannot link filesrc -> decodebin");
        FinalizeGstPipeLine();

        return false;
    }
    if (!gst_element_link(color, sink))
    {
        NVXIO_PRINT("GStreamer: cannot link color -> appsink");
        FinalizeGstPipeLine();

        return false;
    }

    gst_app_sink_set_max_buffers (GST_APP_SINK(sink), 4);
    gst_app_sink_set_drop (GST_APP_SINK(sink), false);
    gst_app_sink_set_emit_signals (GST_APP_SINK(sink), 0);

#if GST_VERSION_MAJOR == 0
    std::unique_ptr<GstCaps, GStreamerObjectDeleter> caps_appsink(
                gst_caps_new_simple("video/x-raw-rgb",
                                    "bpp",        G_TYPE_INT, 24,
                                    "red_mask",   G_TYPE_INT, 0xFF0000,
                                    "green_mask", G_TYPE_INT, 0x00FF00,
                                    "blue_mask",  G_TYPE_INT, 0x0000FF,
                                    NULL));
#else
    // support 1 and 4 channel 8 bit data
    std::unique_ptr<GstCaps, GStreamerObjectDeleter> caps_appsink(
                gst_caps_from_string("video/x-raw, format=(string){RGBA, GRAY8};"));
#endif
    gst_app_sink_set_caps(GST_APP_SINK(sink), caps_appsink.get());

    // Force pipeline to play video as fast as possible, ignoring system clock
    gst_pipeline_use_clock(pipeline, NULL);

    status = gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING);
    handleGStreamerMessages();

    if (status == GST_STATE_CHANGE_ASYNC)
    {
        // wait for status update
        status = gst_element_get_state(GST_ELEMENT(pipeline), NULL, NULL, GST_CLOCK_TIME_NONE);
    }
    if (status == GST_STATE_CHANGE_FAILURE)
    {
        NVXIO_PRINT("GStreamer: unable to start playback");
        FinalizeGstPipeLine();

        return false;
    }

    if (!updateConfiguration(color, configuration))
    {
        FinalizeGstPipeLine();
        return false;
    }

    end = false;

    return true;
}

}

#endif // USE_GSTREAMER
