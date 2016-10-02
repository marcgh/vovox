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

#ifndef STUBRENDERIMPL_HPP
#define STUBRENDERIMPL_HPP

#include "NVXIO/Render.hpp"

namespace nvxio
{

class StubRenderImpl : public Render
{
public:
    StubRenderImpl():
        Render(nvxio::Render::UNKNOWN_RENDER, "Stub")
    {}
    virtual bool open(const std::string& title, vx_uint32 width, vx_uint32 height, vx_uint32 format)
    {
        (void)format;
        windowHeight = height;
        windowWidth = width;
        windowTitle = title;

        return true;
    }
    virtual void setOnKeyboardEventCallback(OnKeyboardEventCallback, void*){}
    virtual void setOnMouseEventCallback(OnMouseEventCallback, void*){}
    virtual void putImage(vx_image){}
    virtual void putTextViewport(const std::string&, const TextBoxStyle&){}
    virtual void putFeatures(vx_array, const FeatureStyle&){}
    virtual void putFeatures(vx_array, vx_array){}
    virtual void putLines(vx_array, const LineStyle&){}
    virtual void putConvexPolygon(vx_array, const LineStyle&){}
    virtual void putMotionField(vx_image, const MotionFieldStyle&){}
    virtual void putObjectLocation(const vx_rectangle_t&, const DetectedObjectStyle&){}
    virtual void putCircles(vx_array, const CircleStyle&){}
    virtual void putArrows(vx_array, vx_array, const LineStyle &) { }

    virtual bool flush(){return true;}
    virtual void close(){}
    virtual vx_uint32 getViewportWidth() const
    {
        return windowWidth;
    }
    virtual vx_uint32 getViewportHeight() const
    {
        return windowHeight;
    }
    virtual ~StubRenderImpl()
    {}
protected:
    vx_uint32 windowWidth;
    vx_uint32 windowHeight;
    std::string windowTitle;
};

}
#endif // STUBRENDERIMPL_HPP
