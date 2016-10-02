/*
# Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
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

#ifdef USE_GUI

#ifdef _WIN32
# define NOMINMAX
# include <Windows.h>
#endif

#include <string>

#include "Render/GlfwUIRenderImpl.hpp"
#include "Private/LogUtils.hpp"
#include "NVXIO/Application.hpp"

nvxio::GlfwUIImpl::GlfwUIImpl(vx_context context, TargetType type, const std::string & name) :
    OpenGLRenderImpl(type, name),
    context_(context),
    window_(NULL), prevWindow_(NULL),
    keyboardCallback_(NULL),
    keyboardCallbackContext_(NULL),
    mouseCallback_(NULL),
    mouseCallbackContext_(NULL),
    scaleRatioWindow(1.0),
    xBorder_(0), yBorder_(0)
{
}

nvxio::GlfwUIImpl::~GlfwUIImpl()
{
    close();
}

namespace {

std::string getDisplayName()
{
    char * displayName = getenv("NVXIO_DISPLAY");
    return std::string(displayName ? displayName : glfwGetMonitorName(glfwGetPrimaryMonitor()));
}

}

bool nvxio::GlfwUIImpl::open(const std::string& title, vx_uint32 width, vx_uint32 height, vx_uint32 format,
                             bool doScale, bool fullScreen)
{
    NVXIO_ASSERT(format == VX_DF_IMAGE_RGBX);

    windowTitle_ = title;
    doScale_ = doScale;

    if (!nvxio::Application::get().initGui())
    {
        NVXIO_PRINT("Error: Failed to init GUI");
        return false;
    }

    vx_uint32 wndWidth = 0, wndHeight = 0;
    const GLFWvidmode * mode = NULL;
    bool renderToFile = (targetType == Render::VIDEO_RENDER) ||
            (targetType == Render::IMAGE_RENDER);

    GLFWmonitor * monitor = NULL;

    if (fullScreen)
    {
        NVXIO_PRINT("Full Screen mode is used. Both specified width and height are ignored");
    }

    if (!renderToFile)
    {
        int count = 0;
        GLFWmonitor ** monitors = glfwGetMonitors(&count);

        if (count == 0)
        {
            NVXIO_PRINT("Glfw: no monitors found");
            return false;
        }

        int maxPixels = 0;
        std::string specifiedDisplayName = getDisplayName();

        for (int i = 0; i < count; ++i)
        {
            const GLFWvidmode* currentMode = glfwGetVideoMode(monitors[i]);
            int currentPixels = currentMode->width * currentMode->height;

            if (maxPixels < currentPixels)
            {
                mode = currentMode;
                maxPixels = currentPixels;
            }

            if (fullScreen)
            {
                std::string monitorName = glfwGetMonitorName(monitors[i]);

                if (monitorName == specifiedDisplayName)
                {
                    monitor = monitors[i];
                    mode = currentMode;
                    break;
                }
            }
        }

#ifdef _WIN32
        int clientWidth = GetSystemMetrics(SM_CXFULLSCREEN),
            clientHeight = GetSystemMetrics(SM_CYFULLSCREEN);
#else
        int clientWidth = mode->width, clientHeight = mode->height;
#endif

        // use full client area if we are in full-screen mode
        if (fullScreen)
        {
            width = clientWidth;
            height = clientHeight;
        }

        if (width <= (vx_uint32)clientWidth && height <= (vx_uint32)clientHeight)
        {
            wndWidth = width;
            wndHeight = height;
        }
        else
        {
            // calculate scale to keep aspect ratio
            vx_float64 widthRatio = static_cast<vx_float64>(clientWidth) / width;
            vx_float64 heightRatio = static_cast<vx_float64>(clientHeight) / height;
            scaleRatioWindow = std::min(widthRatio, heightRatio);

            // apply max contraints
            wndWidth = static_cast<vx_uint32>(width * scaleRatioWindow);
            wndHeight = static_cast<vx_uint32>(height * scaleRatioWindow);
        }
    }
    else // set window params for video/image renders
    {
        wndWidth = width;
        wndHeight = height;
    }

    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

#ifdef USE_GLES
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif
    if (renderToFile)
        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

    window_ = glfwCreateWindow(wndWidth, wndHeight,
                               windowTitle_.c_str(),
                               monitor, NULL);
    if (!window_)
    {
        NVXIO_PRINT("Error: Failed to create GLFW window");
        return false;
    }

    if (!renderToFile)
    {
        // as it's said in documentation, actual window and context
        // parameters may differ from specified ones. So, we need to query
        // actual params and use them later.
        glfwGetFramebufferSize(window_, (int *)&wndWidth, (int *)&wndHeight);

#ifdef _WIN32
        // update sizes
        vx_float64 widthRatio = static_cast<vx_float64>(wndWidth) / width;
        vx_float64 heightRatio = static_cast<vx_float64>(wndHeight) / height;
        scaleRatioWindow = std::min(heightRatio, widthRatio);

        wndHeight = static_cast<vx_uint32>(scaleRatioWindow * height);
        vx_float64 aspectRatio = static_cast<vx_float64>(width) / height;
        wndWidth = static_cast<vx_uint32>(aspectRatio * scaleRatioWindow * height);

        // update window size
        glfwSetWindowSize(window_, wndWidth, wndHeight);
#endif

        // GLFW says that we don't have to set window position
        // for full screen mode.
        if (!fullScreen)
        {
            NVXIO_ASSERT(mode != nullptr);

            int xpos = (mode->width - wndWidth) >> 1;
            int ypos = (mode->height - wndHeight) >> 1;
            glfwSetWindowPos(window_, xpos, ypos);
        }
    }

    glfwSetWindowUserPointer(window_, this);
    glfwSetInputMode(window_, GLFW_STICKY_KEYS, GL_TRUE);

    // create OpenGL context holder
    createOpenGLContextHolder();

    // Must be done after glfw is initialized!
    return initGL(context_, wndWidth, wndHeight);
}

void nvxio::GlfwUIImpl::putImage(vx_image image)
{
    NVXIO_ASSERT(image != nullptr);

    vx_uint32 imgWidth = 0, imgHeight = 0;
    NVXIO_SAFE_CALL( vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &imgWidth, sizeof(imgWidth)) );
    NVXIO_SAFE_CALL( vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &imgHeight, sizeof(imgHeight)) );

    // calculate borders
    GLfloat scaleUniformX_ = static_cast<GLfloat>(wndWidth_) / imgWidth;
    GLfloat scaleUniformY_ = static_cast<GLfloat>(wndHeight_) / imgHeight;
    GLfloat scale = std::min(scaleUniformX_, scaleUniformY_);

    GLint viewportWidth = static_cast<GLint>(imgWidth * scale);
    GLint viewportHeight = static_cast<GLint>(imgHeight * scale);

    xBorder_ = (wndWidth_ - viewportWidth) >> 1;
    yBorder_ = (wndHeight_ - viewportHeight) >> 1;

    // render image
    nvxio::OpenGLRenderImpl::putImage(image);
}

void nvxio::GlfwUIImpl::getCursorPos(vx_float64 & x, vx_float64 & y) const
{
    glfwGetCursorPos(window_, &x, &y);

    x = x / (scaleRatioWindow) - xBorder_;
    y = y / (scaleRatioWindow) - yBorder_;
    x /= scaleRatioImage_;
    y /= scaleRatioImage_;
}

void nvxio::GlfwUIImpl::cursor_pos(GLFWwindow* window, double x, double y)
{
    GlfwUIImpl* impl = static_cast<GlfwUIImpl*>(glfwGetWindowUserPointer(window));

    x = x / (impl->scaleRatioWindow) - impl->xBorder_;
    y = y / (impl->scaleRatioWindow) - impl->yBorder_;
    x /= impl->scaleRatioImage_;
    y /= impl->scaleRatioImage_;

    if (impl->mouseCallback_)
    {
        (impl->mouseCallback_)(impl->mouseCallbackContext_, Render::MouseMove,
                               static_cast<vx_uint32>(x),
                               static_cast<vx_uint32>(y));
    }
}

// callback for keys
void nvxio::GlfwUIImpl::key_fun(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    GlfwUIImpl* impl = static_cast<GlfwUIImpl*>(glfwGetWindowUserPointer(window));

    if (impl->keyboardCallback_ && action == GLFW_PRESS)
    {
        double x = 0, y = 0;
        impl->getCursorPos(x, y);

        if (key == GLFW_KEY_ESCAPE)
            key = 27;

        (impl->keyboardCallback_)(impl->keyboardCallbackContext_, tolower(key),
                                  static_cast<vx_uint32>(x),
                                  static_cast<vx_uint32>(y));
    }
}

void nvxio::GlfwUIImpl::setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void* context)
{
    keyboardCallback_ = callback;
    keyboardCallbackContext_ = context;

    glfwSetKeyCallback(window_, key_fun);
}

void nvxio::GlfwUIImpl::setOnMouseEventCallback(OnMouseEventCallback callback, void* context)
{
    mouseCallback_ = callback;
    mouseCallbackContext_ = context;

    glfwSetMouseButtonCallback(window_, mouse_button);
    glfwSetCursorPosCallback(window_, cursor_pos);
}

// callback for mouse
void nvxio::GlfwUIImpl::mouse_button(GLFWwindow* window, int button, int action, int /*mods*/)
{
    GlfwUIImpl* impl = static_cast<GlfwUIImpl*>(glfwGetWindowUserPointer(window));

    if (impl->mouseCallback_)
    {
        Render::MouseButtonEvent event = Render::MouseMove;

        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            if (action == GLFW_RELEASE)
                event = Render::LeftButtonUp;
            else
                event = Render::LeftButtonDown;
        }
        if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            if (action == GLFW_RELEASE)
                event = Render::RightButtonUp;
            else
                event = Render::RightButtonDown;
        }
        if (button == GLFW_MOUSE_BUTTON_MIDDLE)
        {
            if (action == GLFW_RELEASE)
                event = Render::MiddleButtonUp;
            else
                event = Render::MiddleButtonDown;
        }

        double x = 0, y = 0;
        impl->getCursorPos(x, y);

        (impl->mouseCallback_)(impl->mouseCallbackContext_, event,
                               static_cast<vx_uint32>(x),
                               static_cast<vx_uint32>(y));
    }
}

bool nvxio::GlfwUIImpl::flush()
{
    if (!window_)
        NVXIO_THROW_EXCEPTION("The render is closed, you must open it before");

    if (glfwWindowShouldClose(window_))
    {
        close();
        return false;
    }

    // GLFW says that we don't need current OpenGL context, but
    // it's wrong for EGL (OpenGL ES).
    // See EGL 1.4 spec. 3.9.3. Posting Semantics;
    // See EGL 1.5 spec. 3.10.3. Posting Semantics;
    {
        OpenGLContextSafeSetter setter(holder_);
        glfwSwapBuffers(window_);
    }

    glfwPollEvents();
    clearGlBuffer();

    return true;
}

void nvxio::GlfwUIImpl::close()
{
    if (window_)
    {
        // finalize OpenGL resources of base class
        finalGL();

        glfwDestroyWindow(window_);
        window_ = NULL;
    }
}

namespace {

class GLFWContextHolderImpl :
        public nvxio::OpenGLContextHolder
{
public:
    explicit GLFWContextHolderImpl(GLFWwindow * currentWindow_) :
        prevWindow(NULL), currentWindow(currentWindow_)
    {
        if (!currentWindow)
            NVXIO_THROW_EXCEPTION("The render is closed, you must open it before");
    }

    virtual void set()
    {
        // save current context
        prevWindow = glfwGetCurrentContext();
        // attach our OpenGL context
        glfwMakeContextCurrent(currentWindow);
    }

    virtual void unset()
    {
        // attach previous context
        glfwMakeContextCurrent(prevWindow);
    }

private:
    GLFWwindow * prevWindow, * currentWindow;
};

}

void nvxio::GlfwUIImpl::createOpenGLContextHolder()
{
    holder_ = std::make_shared<GLFWContextHolderImpl>(window_);
}

#endif // USE_GUI
