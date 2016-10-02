/*
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVMEDIA_CSI10640CAMERAFRAMESOURCEIMPL_HPP
#define NVMEDIA_CSI10640CAMERAFRAMESOURCEIMPL_HPP

#ifdef USE_CSI_OV10640

#include <VX/vx.h>

#include "NVXIO/FrameSource.hpp"
#include "NVX/nvx_timer.hpp"

#include "FrameSource/NvMedia/OV10640/interop.hpp"

namespace nvxio
{

class NvMediaCSI10640CameraFrameSourceImpl : public nvxio::FrameSource
{
public:
    NvMediaCSI10640CameraFrameSourceImpl(vx_context vxContext, const std::string & configName, int number);
    virtual bool open();
    virtual FrameSource::FrameStatus fetch(vx_image image, vx_uint32 timeout = 5 /*milliseconds*/);
    virtual FrameSource::Parameters getConfiguration();
    virtual bool setConfiguration(const FrameSource::Parameters& params);
    virtual void close();
    virtual ~NvMediaCSI10640CameraFrameSourceImpl();

protected:
    const char * defaultCameraConfig() const;
    std::string parseCameraConfig(const std::string cameraConfigFile, CaptureConfigParams& captureConfigCollection);

    // camera params
    CaptureConfigParams captureConfigCollection;
    IPPCtx * ctx;
    InteropContext * interopCtx;

    int cameraNumber;
    std::string configPath;

    FrameSource::Parameters configuration;
    vx_context vxContext;

    vx_image nv12Frame;
    vx_image convertedImage;
};

}

#endif // USE_CSI_OV10640

#endif // NVMEDIA_CSI10640CAMERAFRAMESOURCEIMPL_HPP
