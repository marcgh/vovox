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

#if defined USE_GUI && defined USE_GSTREAMER

#include "Render/GStreamer/GStreamerImagesRenderImpl.hpp"
#include "Private/GStreamerUtils.hpp"

nvxio::GStreamerImagesRenderImpl::GStreamerImagesRenderImpl(vx_context context) :
    GStreamerBaseRenderImpl(context, nvxio::Render::IMAGE_RENDER, "GStreamerImagesOpenGlRender")
{
}

bool nvxio::GStreamerImagesRenderImpl::InitializeGStreamerPipeline()
{
    // multifilesink does not report erros in case of URI is a directory,
    // let's make some check manually
    {
        // if uri is directory
        if (g_file_test(windowTitle_.c_str(), G_FILE_TEST_IS_DIR))
            return false;
    }

    std::ostringstream stream;

    pipeline = GST_PIPELINE(gst_pipeline_new(NULL));
    if (pipeline == NULL)
    {
        NVXIO_PRINT("Cannot create Gstreamer pipeline");
        return false;
    }

    bus = gst_pipeline_get_bus(GST_PIPELINE (pipeline));

    // create appsrc
    GstElement * appsrcelem = gst_element_factory_make("appsrc", NULL);
    if (appsrcelem == NULL)
    {
        NVXIO_PRINT("Cannot create appsrc");
        FinalizeGStreamerPipeline();

        return false;
    }

    g_object_set(G_OBJECT(appsrcelem), "is-live", 0, NULL);
    g_object_set(G_OBJECT(appsrcelem), "num-buffers", -1, NULL);
    g_object_set(G_OBJECT(appsrcelem), "emit-signals", 0, NULL);
    g_object_set(G_OBJECT(appsrcelem), "block", 1, NULL);
    g_object_set(G_OBJECT(appsrcelem), "size", static_cast<guint64>(wndHeight_ * wndWidth_ * 4), NULL);
    g_object_set(G_OBJECT(appsrcelem), "format", GST_FORMAT_TIME, NULL);
    g_object_set(G_OBJECT(appsrcelem), "stream-type", GST_APP_STREAM_TYPE_STREAM, NULL);

    appsrc = GST_APP_SRC_CAST(appsrcelem);
#if GST_VERSION_MAJOR == 0
    std::unique_ptr<GstCaps, GStreamerObjectDeleter> caps(
                gst_caps_new_simple("video/x-raw-rgb",
                                    "bpp", G_TYPE_INT, 32,
                                    "endianness", G_TYPE_INT, 4321,
                                    "red_mask", G_TYPE_INT, -16777216,
                                    "green_mask", G_TYPE_INT, 16711680,
                                    "blue_mask", G_TYPE_INT, 65280,
                                    "alpha_mask", G_TYPE_INT, 255,
                                    "width", G_TYPE_INT, wndWidth_,
                                    "height", G_TYPE_INT, wndHeight_,
                                    "framerate", GST_TYPE_FRACTION, GSTREAMER_DEFAULT_FPS, 1,
                                    NULL));
    if (!caps)
    {
        NVXIO_PRINT("Failed to create caps");
        FinalizeGStreamerPipeline();

        return false;
    }

#else
    // support 4 channel 8 bit data
    stream << "video/x-raw"
           << ", width=" << wndWidth_
           << ", height=" << wndHeight_
           << ", format=(string){RGBA}"
           << ", framerate=" << GSTREAMER_DEFAULT_FPS << "/1;";
    std::unique_ptr<GstCaps, GStreamerObjectDeleter> caps(
                gst_caps_from_string(stream.str().c_str()));

    if (!caps)
    {
        NVXIO_PRINT("Failed to create caps");
        FinalizeGStreamerPipeline();

        return false;
    }

    gst_caps_ref(caps.get());
    caps.reset(gst_caps_fixate(caps.get()));
#endif

    gst_app_src_set_caps(appsrc, caps.get());

    gst_bin_add(GST_BIN(pipeline), appsrcelem);

    // create color convert element
    GstElement * color = gst_element_factory_make(COLOR_ELEM, NULL);
    if (color == NULL)
    {
        NVXIO_PRINT("Cannot create " COLOR_ELEM " element");
        FinalizeGStreamerPipeline();

        return false;
    }
    gst_bin_add(GST_BIN(pipeline), color);

    // create videoflip element
    GstElement * videoflip = gst_element_factory_make("videoflip", NULL);
    if (videoflip == NULL)
    {
        NVXIO_PRINT("Cannot create videoflip element");
        FinalizeGStreamerPipeline();

        return false;
    }

    g_object_set(G_OBJECT(videoflip), "method", 5, NULL);

    gst_bin_add(GST_BIN(pipeline), videoflip);

    // create color2 convert element
    GstElement * color2 = gst_element_factory_make(COLOR_ELEM, NULL);
    if (color2 == NULL)
    {
        NVXIO_PRINT("Cannot create " COLOR_ELEM " element");
        FinalizeGStreamerPipeline();

        return false;
    }
    gst_bin_add(GST_BIN(pipeline), color2);

    // create pngenc element
    GstElement * pngenc = gst_element_factory_make("pngenc", NULL);
    if (pngenc == NULL)
    {
        NVXIO_PRINT("Cannot create pngenc element");
        FinalizeGStreamerPipeline();

        return false;
    }

    g_object_set(G_OBJECT(pngenc), "snapshot", 0, NULL);
    gst_bin_add(GST_BIN(pipeline), pngenc);

    // create multifilesink element
    GstElement * multifilesink = gst_element_factory_make("multifilesink", NULL);
    if (multifilesink == NULL)
    {
        NVXIO_PRINT("Cannot create multifilesink element");
        FinalizeGStreamerPipeline();

        return false;
    }

    g_object_set(G_OBJECT(multifilesink), "location", windowTitle_.c_str(), NULL);
    g_object_set(G_OBJECT(multifilesink), "max-lateness", G_GINT64_CONSTANT(-1), NULL);
    g_object_set(G_OBJECT(multifilesink), "async", 0, NULL);
    g_object_set(G_OBJECT(multifilesink), "render-delay", G_GUINT64_CONSTANT(0), NULL);
    g_object_set(G_OBJECT(multifilesink), "throttle-time", G_GUINT64_CONSTANT(0), NULL);
    g_object_set(G_OBJECT(multifilesink), "index", 1, NULL);
    g_object_set(G_OBJECT(multifilesink), "max-files", 9999, NULL);
    g_object_set(G_OBJECT(multifilesink), "post-messages", 1, NULL);
    g_object_set(G_OBJECT(multifilesink), "next-file", 0, NULL);

    gst_bin_add(GST_BIN(pipeline), multifilesink);

#if GST_VERSION_MAJOR == 0
    std::unique_ptr<GstCaps, GStreamerObjectDeleter> caps_color(
                gst_caps_new_simple("video/x-raw-rgb",
                                    "bpp", G_TYPE_INT, 32,
                                    "depth", G_TYPE_INT, 32,
                                    "endianness", G_TYPE_INT, 4321,
                                    "red_mask", G_TYPE_INT, -16777216,
                                    "green_mask", G_TYPE_INT, 16711680,
                                    "blue_mask", G_TYPE_INT, 65280,
                                    "alpha_mask", G_TYPE_INT, 255,
                                    "width", G_TYPE_INT, wndWidth_,
                                    "height", G_TYPE_INT, wndHeight_,
                                    "framerate", GST_TYPE_FRACTION, GSTREAMER_DEFAULT_FPS, 1,
                                    NULL));
    if (!caps_color)
    {
        NVXIO_PRINT("Failed to create caps");
        FinalizeGStreamerPipeline();
        return false;
    }

    if (!gst_element_link_filtered(appsrcelem, color, caps_color.get()))
    {
        NVXIO_PRINT("GStreamer: cannot link " COLOR_ELEM
                    " -> videoflip -> pngenc -> multifilesink");
        FinalizeGStreamerPipeline();

        return false;
    }

#else
    if (!gst_element_link(appsrcelem, color))
    {
        NVXIO_PRINT("GStreamer: cannot link appsrc -> " COLOR_ELEM);
        FinalizeGStreamerPipeline();
        return false;
    }
#endif

    if (!gst_element_link_many(color, videoflip,
                               pngenc, multifilesink, NULL))
    {
        NVXIO_PRINT("GStreamer: cannot link " COLOR_ELEM
                    " -> videoflip -> pngenc -> multifilesink");
        FinalizeGStreamerPipeline();

        return false;
    }

    // Force pipeline to play video as fast as possible, ignoring system clock
    gst_pipeline_use_clock(pipeline, NULL);

    num_frames = 0;

    GstStateChangeReturn status = gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING);
    if (status == GST_STATE_CHANGE_FAILURE)
    {
        NVXIO_PRINT("GStreamer: unable to start playback");
        FinalizeGStreamerPipeline();

        return false;
    }

    return true;
}

#endif // USE_GUI && USE_GSTREAMER
