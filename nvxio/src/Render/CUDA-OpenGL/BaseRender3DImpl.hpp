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

#ifndef BASE_RENDER3D_IMPL_HPP
#define BASE_RENDER3D_IMPL_HPP

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <NVX/nvx.h>
#ifdef USE_GUI
#include <NVXIO/OpenGL.hpp>
#endif

#include <NVXIO/Render3D.hpp>
#include "Render/CUDA-OpenGL//OpenGLBasicRenders.hpp"

namespace nvxio
{

class BaseRender3DImpl : public nvxio::Render3D
{
public:
    BaseRender3DImpl(vx_context context);

    virtual void putPlanes(vx_array planes, vx_matrix model, const PlaneStyle& style);

    virtual void putPointCloud(vx_array points, vx_matrix model, const PointCloudStyle& style);

    virtual void putImage(vx_image image);

    virtual void putText(const std::string& text, const nvxio::Render::TextBoxStyle& style);

    virtual bool flush();

    virtual bool open(int xPos, int yPos, vx_uint32 windowWidth, vx_uint32 windowHeight, const std::string& windowTitle);
    virtual void close();

    virtual void setViewMatrix(vx_matrix view);
    virtual void getViewMatrix(vx_matrix view) const;

    virtual void setProjectionMatrix(vx_matrix projection);
    virtual void getProjectionMatrix(vx_matrix projection) const;

    virtual void setDefaultFOV(float fov);//in degrees

    virtual void setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void* context);

    virtual void enableDefaultKeyboardEventCallback();
    virtual void disableDefaultKeyboardEventCallback();
    virtual bool useDefaultKeyboardEventCallback();

    virtual void setOnMouseEventCallback(OnMouseEventCallback callback, void* context);
    virtual vx_uint32 getWidth() const
    {
        return windowWidth_;
    }

    virtual vx_uint32 getHeight() const
    {
        return windowHeight_;
    }

    virtual ~BaseRender3DImpl();

protected:
    vx_context context_;

    std::shared_ptr<nvxio::GLFunctions> gl_;

    vx_matrix model_;
    vx_matrix view_;
    vx_matrix projection_;

    GLFWwindow* window_;

    void initMVP();

    void setModelMatrix(vx_matrix model);

    bool initWindow(int xpos, int ypos, vx_uint32 width, vx_uint32 height, const std::string& wintitle);

    OnKeyboardEventCallback keyboardCallback_;
    void * keyboardCallbackContext_;

    OnMouseEventCallback mouseCallback_;
    void * mouseCallbackContext_;

    static void mouse_button(GLFWwindow* window, int button, int action, int mods);
    static void cursor_pos(GLFWwindow* window, double x, double y);

    static void keyboardCallbackDefault(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    bool useDefaultCallback_;

    vx_uint32 windowWidth_;
    vx_uint32 windowHeight_;

    float defaultFOV_;
    const float Z_NEAR_;
    const float Z_FAR_;
    float fov_;

    struct OrbitCameraParams
    {
        const float R_min;
        const float R_max;
        float xr; // rotation angle around x-axis
        float yr; // rotation angle around y-axis
        float R;  // radius of camera orbit
        OrbitCameraParams(float R_min_, float R_max_);
        void applyConstraints();
        void setDefault();
    };

    OrbitCameraParams orbitCameraParams_;

    void updateView();

    ImageRender imageRender_;
    TextRender textRender_;
    PointCloudRender pointCloudRender_;
    FencePlaneRender fencePlaneRender_;

    vx_float32 scaleRatio_;
};

}

#endif // BASE_RENDER3D_IMPL_HPP
