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

#ifndef GSTREAMEREGLSTREAMSINKFRAMESOURCEIMPL_HPP
#define GSTREAMEREGLSTREAMSINKFRAMESOURCEIMPL_HPP

#if defined USE_GSTREAMER_NVMEDIA || defined USE_NVGSTCAMERA

#include <VX/vx.h>

#include "NVXIO/FrameSource.hpp"
#include "NVX/nvx_timer.hpp"

#include "FrameSource/EGLAPIAccessors.hpp"
#include "FrameSource/GStreamer/GStreamerCommon.hpp"

#include <cudaEGL.h>

namespace nvxio
{

class GStreamerEGLStreamSinkFrameSourceImpl :
    public FrameSource
{
public:
    GStreamerEGLStreamSinkFrameSourceImpl(vx_context context, SourceType sourceType,
                                          const char * const name, bool fifomode);

    virtual bool open();
    virtual void close();
    virtual FrameSource::FrameStatus fetch(vx_image image, vx_uint32 timeout = 5 /*milliseconds*/);

    virtual FrameSource::Parameters getConfiguration();
    virtual bool setConfiguration(const FrameSource::Parameters& params);
    virtual ~GStreamerEGLStreamSinkFrameSourceImpl();

protected:
    void handleGStreamerMessages();

    virtual bool InitializeGstPipeLine() = 0;
    void CloseGstPipeLineAsyncThread();
    void FinalizeGstPipeLine();

    GstPipeline*  pipeline;
    GstBus*       bus;
    volatile bool end;

    // EGL context and stream
    struct EglContext
    {
        EGLDisplay display;
        EGLStreamKHR stream;
    };

    bool InitializeEGLDisplay();
    bool InitializeEGLStream();
    void FinalizeEglStream();

    EglContext   context;
    int          fifoLength;
    bool         fifoMode;
    int          latency;

    // CUDA consumer
    bool InitializeEglCudaConsumer();
    void FinalizeEglCudaConsumer();

    CUeglStreamConnection cudaConnection;

    // Common FrameSource parameters
    vx_context vxContext;
    FrameSource::Parameters configuration;
    vx_image scaledImage;
};

}

#endif // defined USE_GSTREAMER_NVMEDIA || defined USE_NVGSTCAMERA

#endif // GSTREAMEREGLSTREAMSINKFRAMESOURCEIMPL_HPP
