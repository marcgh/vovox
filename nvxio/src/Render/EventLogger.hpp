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

#ifndef EVENTLOGGER_HPP
#define EVENTLOGGER_HPP

#include <stdio.h>
#include <memory>
#include <string>
#include "NVXIO/Render.hpp"

namespace nvxio
{

class EventLogger: public nvxio::Render
{
public:
    explicit EventLogger(bool _writeSrc);
    void setEfficientRender(std::unique_ptr<Render> render);
    bool init(const std::string& path);
    void final();
    ~EventLogger();

    virtual void setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void* context);
    virtual void setOnMouseEventCallback(OnMouseEventCallback callback, void* context);

    void putImage(vx_image image);
    void putTextViewport(const std::string& text, const Render::TextBoxStyle& style);
    void putFeatures(vx_array location, const Render::FeatureStyle& style);
    void putFeatures(vx_array location, vx_array styles);
    void putLines(vx_array lines, const Render::LineStyle& style);
    void putConvexPolygon(vx_array verticies, const LineStyle& style);
    void putMotionField(vx_image field, const Render::MotionFieldStyle& style);
    void putObjectLocation(const vx_rectangle_t& location, const Render::DetectedObjectStyle& style);
    void putCircles(vx_array circles, const CircleStyle& style);
    void putArrows(vx_array old_points, vx_array new_points, const LineStyle& line_style);

    bool flush();
    void close();

    virtual vx_uint32 getViewportWidth() const
    {
        if (efficientRender)
        {
            return efficientRender->getViewportWidth();
        }
        else
        {
            return 0;
        }
    }
    virtual vx_uint32 getViewportHeight() const
    {
        if (efficientRender)
        {
            return efficientRender->getViewportHeight();
        }
        else
        {
            return 0;
        }
    }

protected:
    static void keyboard(void* context, vx_char key, vx_uint32 x, vx_uint32 y);
    static void mouse(void* context, Render::MouseButtonEvent event, vx_uint32 x, vx_uint32 y);

    std::unique_ptr<Render> efficientRender;
    bool writeSrc;
    FILE* handle;
    std::string srcImageFilePattern;
    int frameCounter;
    OnKeyboardEventCallback keyBoardCallback;
    void* keyboardCallbackContext;
    OnMouseEventCallback mouseCallback;
    void* mouseCallbackContext;
};

}

#endif // EVENTLOGGER_HPP
