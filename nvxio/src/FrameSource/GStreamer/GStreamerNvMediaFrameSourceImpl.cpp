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

#ifdef USE_GSTREAMER_NVMEDIA

#include "NVXIO/Application.hpp"
#include "FrameSource/GStreamer/GStreamerNvMediaFrameSourceImpl.hpp"

#include <VX/vx.h>
#include <NVX/nvx.h>
#include <cuda_runtime.h>

namespace nvxio
{

GStreamerNvMediaFrameSourceImpl::GStreamerNvMediaFrameSourceImpl(vx_context vxcontext, const std::string & path):
    GStreamerEGLStreamSinkFrameSourceImpl(vxcontext, FrameSource::VIDEO_SOURCE, "GstreamerNvMediaFrameSource", true),
    fileName(path)
{
}

void GStreamerNvMediaFrameSourceImpl::setNvMediaPluginRunk()
{
    const std::map<std::string, guint> features_list =
    {
        { "avdec_mpeg4",          GST_RANK_SECONDARY },
        { "avdec_h264",           GST_RANK_SECONDARY },
        { "nvmediamp3auddec",     GST_RANK_PRIMARY },
        { "nvmediaaacauddec",     GST_RANK_PRIMARY },
        { "nvmediawmaauddec",     GST_RANK_PRIMARY },
        { "nvmediaaacaudenc",     GST_RANK_PRIMARY },
        { "nvmediampeg2viddec",   GST_RANK_PRIMARY },
        { "nvmediampeg4viddec",   GST_RANK_PRIMARY },
        { "nvmediavc1viddec",     GST_RANK_PRIMARY },
        { "nvmediamjpegviddec",   GST_RANK_PRIMARY },
        { "nvmediah264viddec",    GST_RANK_PRIMARY },
        { "nvmediah264videnc",    GST_RANK_PRIMARY },
        { "nvmediacapturesrc",    GST_RANK_PRIMARY },
        { "nvmediaoverlaysink",   GST_RANK_PRIMARY },
        { "nvmediaeglstreamsink", GST_RANK_PRIMARY },
        { "nvmediavp8viddec",     GST_RANK_PRIMARY }
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

void GStreamerNvMediaFrameSourceImpl::newGstreamerPad(GstElement * /*elem*/, GstPad *pad, gpointer data)
{
    GstElement *mixer = (GstElement *) data;

    std::unique_ptr<GstPad, GStreamerObjectDeleter> sinkpad(gst_element_get_static_pad(mixer, "sink"));
    if (!sinkpad)
    {
        NVXIO_PRINT("Gstreamer: no pad named \"sink\"");
        return;
    }

    gst_pad_link(pad, sinkpad.get());
}

GStreamerNvMediaFrameSourceImpl::GstAutoplugSelectResult GStreamerNvMediaFrameSourceImpl::autoPlugSelect(GstElement *, GstPad *,
                              GstCaps * caps, GstElementFactory *, gpointer)
{
    std::unique_ptr<char[], GlibDeleter> msg_str(gst_caps_to_string(caps));
    GStreamerNvMediaFrameSourceImpl::GstAutoplugSelectResult result = GST_AUTOPLUG_SELECT_EXPOSE;

    if (strstr(msg_str.get(), "video"))
    {
        result = GST_AUTOPLUG_SELECT_TRY;
    }

    return result;
}

bool GStreamerNvMediaFrameSourceImpl::InitializeGstPipeLine()
{
    end = true;
    setNvMediaPluginRunk();

    GstStateChangeReturn status;

    std::string uri;
    const char * fileNameS = fileName.c_str();

    if(!gst_uri_is_valid(fileNameS))
    {
        char* real = realpath(fileNameS, NULL);
        if (!real)
        {
            NVXIO_PRINT("Can't resolve path \"%s\": %s", fileNameS, strerror(errno));
            return false;
        }

        std::unique_ptr<char[], GlibDeleter> pUri(g_filename_to_uri(real, NULL, NULL));
        free(real);
        uri = pUri.get();
    }
    else
    {
        uri = fileName;
    }

    pipeline = GST_PIPELINE(gst_pipeline_new(NULL));
    if (pipeline == NULL)
    {
        NVXIO_PRINT("Cannot create Gstreamer pipeline");
        return false;
    }

    bus = gst_pipeline_get_bus(GST_PIPELINE (pipeline));

    // create uriDecodeBin
    GstElement * uriDecodeBin = gst_element_factory_make("uridecodebin", NULL);
    handleGStreamerMessages();
    if (uriDecodeBin == NULL)
    {
        NVXIO_PRINT("Cannot create uridecodebin");
        return false;
    }

    g_object_set(G_OBJECT(uriDecodeBin), "uri", uri.c_str(), NULL);

    std::unique_ptr<GstCaps, GStreamerObjectDeleter> caps(gst_caps_from_string("video/x-nvmedia"));
    if (!caps)
    {
        NVXIO_PRINT("Cannot create nvmedia caps");
        return false;
    }

    g_object_set(G_OBJECT(uriDecodeBin), "caps", caps.get(), NULL);

    gst_bin_add(GST_BIN(pipeline), uriDecodeBin);

    // create nvmediasurfmixer
    GstElement * surfaceMixer = gst_element_factory_make("nvmediasurfmixer", NULL);
    if (surfaceMixer == NULL)
    {
        NVXIO_PRINT("Cannot create surface mixer");
        FinalizeGstPipeLine();
        return false;
    }

    gst_bin_add(GST_BIN(pipeline), surfaceMixer);

    g_signal_connect(uriDecodeBin, "pad-added", G_CALLBACK(GStreamerNvMediaFrameSourceImpl::newGstreamerPad), surfaceMixer);
    g_signal_connect(uriDecodeBin, "autoplug-select", G_CALLBACK(GStreamerNvMediaFrameSourceImpl::autoPlugSelect), NULL);

    // create nvmediaeglstreamsink
    GstElement * eglSink = gst_element_factory_make("nvmediaeglstreamsink", NULL);
    if (eglSink == NULL)
    {
        NVXIO_PRINT("Cannot create EGL sink");
        FinalizeGstPipeLine();
        return false;
    }

    gst_bin_add(GST_BIN(pipeline), eglSink);

    g_object_set(G_OBJECT(eglSink), "display", context.display, NULL);
    g_object_set(G_OBJECT(eglSink), "stream", context.stream, NULL);
    g_object_set(G_OBJECT(eglSink), "fifo", fifoMode, NULL);
    g_object_set(G_OBJECT(eglSink), "max-lateness", G_GINT64_CONSTANT(-1), NULL);
    g_object_set(G_OBJECT(eglSink), "throttle-time", G_GUINT64_CONSTANT(0), NULL);
    g_object_set(G_OBJECT(eglSink), "render-delay", G_GUINT64_CONSTANT(0), NULL);
    g_object_set(G_OBJECT(eglSink), "qos", FALSE, NULL);
    g_object_set(G_OBJECT(eglSink), "sync", FALSE, NULL);
    g_object_set(G_OBJECT(eglSink), "async", TRUE, NULL);

    // link elements
    if (!gst_element_link(surfaceMixer, eglSink))
    {
        NVXIO_PRINT("Cannot link SurfaceMixer and EGL sink");
        FinalizeGstPipeLine();
        return false;
    }

    // Force pipeline to play video as fast as possible, ignoring system clock
    gst_pipeline_use_clock(pipeline, NULL);

    status = gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING);
    handleGStreamerMessages();

    if (status == GST_STATE_CHANGE_ASYNC)
    {
        // wait for status update
        status = gst_element_get_state(GST_ELEMENT(pipeline), NULL, NULL, GST_CLOCK_TIME_NONE);
    }

//    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");

    if (status == GST_STATE_CHANGE_FAILURE)
    {
        NVXIO_PRINT("GStreamer: unable to start playback");
        FinalizeGstPipeLine();
        return false;
    }

    if (!updateConfiguration(surfaceMixer, configuration))
    {
        FinalizeGstPipeLine();
        return false;
    }

    end = false;

    return true;
}

}

#endif // USE_GSTREAMER_NVMEDIA
