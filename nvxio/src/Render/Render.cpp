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

#include <memory>

#include "NVXIO/Render.hpp"
#include "NVXIO/Application.hpp"
#include "Render/EventLogger.hpp"
#include "Render/EventPlayer.hpp"

#ifdef USE_GUI
# include "Render/GlfwUIRenderImpl.hpp"
# ifdef USE_GSTREAMER
#  include "Render/GStreamer/GStreamerVideoRenderImpl.hpp"
#  include "Render/GStreamer/GStreamerImagesRenderImpl.hpp"
# endif
# ifdef USE_OPENCV
#  include "Render/OpenCV/OpenGLOpenCVRenderImpl.hpp"
# endif
#endif

#include "Render/StubRenderImpl.hpp"

namespace nvxio
{

static std::string patchWindowTitle(const std::string & windowTitle)
{
    std::string retVal = windowTitle;

    std::replace(retVal.begin(), retVal.end(), '/', '|');
    std::replace(retVal.begin(), retVal.end(), '\\', '|');

    return retVal;
}

static std::unique_ptr<Render> createSmartRender(std::unique_ptr<Render> specializedRender)
{
    Application &app = Application::get();
    std::unique_ptr<Render> render = std::move(specializedRender);

    if (!app.getScenarioName().empty())
    {
        std::unique_ptr<EventPlayer> player(new EventPlayer);
        if (player->init(app.getScenarioName(), app.getScenarioLoopCount()))
        {
            // To read data for the first frame
            player->flush();
            player->setEfficientRender(std::move(render));
            render = std::move(player);
        }
        else
        {
            NVXIO_THROW_EXCEPTION("Warning: cannot open scenario \"" << app.getScenarioName() << "\"");
        }
    }

    if (!app.getEventLogName().empty())
    {
        std::unique_ptr<EventLogger> logger(new EventLogger(app.getEventLogDumpFramesFlag()));
        if (logger->init(app.getEventLogName()))
        {
            logger->setEfficientRender(std::move(render));
            render = std::move(logger);
        }
        else
        {
            fprintf(stderr, "Warning: cannot open log file \"%s\"\n", app.getEventLogName().c_str());
        }
    }

    return render;
}

std::unique_ptr<Render> createVideoRender(vx_context context, const std::string& path, vx_uint32 width, vx_uint32 height, vx_uint32 format)
{
#if defined USE_GUI && defined USE_GSTREAMER
    std::unique_ptr<GStreamerVideoRenderImpl> gst_render(new GStreamerVideoRenderImpl(context));

    if (!gst_render->open(path, width, height, format))
        return nullptr;

    return createSmartRender(std::move(gst_render));
#elif defined USE_GUI && defined USE_OPENCV
    std::unique_ptr<OpenGLOpenCVRenderImpl> ocv_render(new OpenGLOpenCVRenderImpl(context,
        Render::VIDEO_RENDER, "OpenGLOpenCVVideoRenderImpl"));

    if (!ocv_render->open(path, width, height, format))
        return nullptr;

    return createSmartRender(std::move(ocv_render));
#else
    (void)context;
    (void)path;
    (void)width;
    (void)height;
    (void)format;

    return nullptr;
#endif
}

std::unique_ptr<Render> createImageRender(vx_context context, const std::string& path, vx_uint32 width, vx_uint32 height, vx_uint32 format)
{
#if defined USE_GUI && defined USE_GSTREAMER
    std::unique_ptr<GStreamerImagesRenderImpl> gst_render(new GStreamerImagesRenderImpl(context));

    if (!gst_render->open(path, width, height, format))
        return nullptr;

    return createSmartRender(std::move(gst_render));
#elif defined USE_GUI && defined USE_OPENCV
    std::unique_ptr<OpenGLOpenCVRenderImpl> ocv_render(new OpenGLOpenCVRenderImpl(context,
        Render::IMAGE_RENDER, "OpenGLOpenCVImagesRenderImpl"));

    if (!ocv_render->open(path, width, height, format))
        return nullptr;

    return createSmartRender(std::move(ocv_render));
#else
    (void)context;
    (void)path;
    (void)width;
    (void)height;
    (void)format;

    return nullptr;
#endif
}

std::unique_ptr<Render> createWindowRender(vx_context context, const std::string& title, vx_uint32 width, vx_uint32 height,
                                           vx_uint32 format, bool doScale, bool fullscreen)
{
#ifdef USE_GUI
    std::unique_ptr<GlfwUIImpl> render(new GlfwUIImpl(context, nvxio::Render::WINDOW_RENDER, "GlfwOpenGlRender"));

    if (!render->open(title, width, height, format, doScale, fullscreen))
        return nullptr;

    return createSmartRender(std::move(render));
#else
    (void)context;
    (void)title;
    (void)width;
    (void)height;
    (void)format;
    (void)doScale;
    (void)fullscreen;

    return nullptr;
#endif
}

std::unique_ptr<Render> createDefaultRender(vx_context context, const std::string& title, vx_uint32 width, vx_uint32 height,
                                            vx_uint32 format, bool doScale, bool fullscreen)
{
    std::string prefferedRenderName = Application::get().getPreferredRenderName();

    if (prefferedRenderName == "default")
    {
        std::unique_ptr<Render> render = createWindowRender(context, title, width, height, format, doScale, fullscreen);

        if (!render)
            render = createVideoRender(context, title + ".avi", width, height, format);

        return render;
    }
    else if (prefferedRenderName == "window")
    {
        return createWindowRender(context, title, width, height, format, doScale, fullscreen);
    }
    else if (prefferedRenderName == "video")
    {
        return createVideoRender(context, patchWindowTitle(title + ".avi"), width, height, format);
    }
    else if (prefferedRenderName == "image")
    {
        return createImageRender(context, patchWindowTitle(title + "_%05d.png"), width, height, format);
    }
    else if (prefferedRenderName == "stub")
    {
        std::unique_ptr<StubRenderImpl> render(new StubRenderImpl());
        NVXIO_ASSERT(render->open(title, width, height, format));
        return createSmartRender(std::move(render));
    }

    return nullptr;
}

}
