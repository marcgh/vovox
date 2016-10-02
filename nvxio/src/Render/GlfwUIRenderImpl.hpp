/*
# Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
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

#ifndef GLFW_RENDER_IMPL_HPP
#define GLFW_RENDER_IMPL_HPP

#ifdef USE_GUI

#include "Private/LogUtils.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "Render/CUDA-OpenGL/OpenGLRenderImpl.hpp"

namespace nvxio {

class GlfwUIImpl : public nvxio::OpenGLRenderImpl
{
public:
    GlfwUIImpl(vx_context context, TargetType type, const std::string & name);
    virtual ~GlfwUIImpl();

    virtual bool open(const std::string& title, vx_uint32 width, vx_uint32 height, vx_uint32 format,
                      bool doScale, bool fullScreen);

    virtual void setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void* context);
    virtual void setOnMouseEventCallback(OnMouseEventCallback callback, void* context);

    // overload putImage to understand needed offsets
    virtual void putImage(vx_image image);

    virtual bool flush();
    virtual void close();

protected:

    virtual void createOpenGLContextHolder();

    vx_context context_;

    std::string windowTitle_;
    GLFWwindow* window_;

private:

    GLFWwindow* prevWindow_;

    OnKeyboardEventCallback keyboardCallback_;
    void* keyboardCallbackContext_;

    OnMouseEventCallback mouseCallback_;
    void* mouseCallbackContext_;

    vx_float64 scaleRatioWindow;
    vx_uint32 xBorder_, yBorder_;

    static void key_fun(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouse_button(GLFWwindow* window, int button, int action, int mods);
    static void cursor_pos(GLFWwindow* window, double x, double y);

    void getCursorPos(vx_float64 & x, vx_float64 & y) const;
};

}

#endif // USE_GUI

#endif // GLFW_RENDER_IMPL_HPP
