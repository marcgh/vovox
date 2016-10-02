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

#ifdef USE_OPENCV
#include <system_error>

#include "FrameSource/OpenCV/OpenCVFrameSourceImpl.hpp"

#include "NVX/nvx.h"
#include <cuda_runtime.h>
#include "NVXIO/Application.hpp"
#include "NVX/nvx_opencv_interop.hpp"
#include <map>

namespace nvxio
{

void convertFrame(vx_context vxContext, vx_image frame,
                  const FrameSource::Parameters & configuration,
                  vx_imagepatch_addressing_t & decodedImageAddr,
                  void * decodedPtr, bool is_cuda, void *& devMem,
                  size_t & devMemPitch, vx_image & scaledImage);

OpenCVFrameSourceImpl::OpenCVFrameSourceImpl(vx_context context, std::unique_ptr<OpenCVBaseFrameSource> source):
    FrameSource(source->getSourceType(), source->getSourceName()),
    alive_(false),
    source_(std::move(source)),
    queue_(4),
    context_(context),
    devMem(NULL),
    devMemPitch(0),
    scaledImage(NULL)
{
}

bool OpenCVFrameSourceImpl::open()
{
    if (source_ == NULL)
        return false;

    if (alive_)
        close();

    try
    {
        alive_ = source_->open();
    }
    catch (const cv::Exception &)
    {
        alive_ = false;
        source_->close();
    }

    if (alive_)
    {
        try
        {
            thread = std::thread(&OpenCVFrameSourceImpl::threadFunc, this);
            return true;
        }
        catch (std::system_error &)
        {
            alive_ = false;
            source_->close();
        }
    }

    return alive_;
}

FrameSource::Parameters OpenCVFrameSourceImpl::getConfiguration()
{
    return source_->getConfiguration();
}

bool OpenCVFrameSourceImpl::setConfiguration(const FrameSource::Parameters &params)
{
    return source_->setConfiguration(params);
}

FrameSource::FrameStatus OpenCVFrameSourceImpl::fetch(vx_image image, vx_uint32 timeout)
{
    cv::Mat frame;

    if (queue_.pop(frame, timeout))
    {
        vx_imagepatch_addressing_t image_addr;
        image_addr.dim_x = frame.cols;
        image_addr.dim_y = frame.rows;
        image_addr.stride_x = static_cast<vx_int32>(frame.channels());
        image_addr.stride_y = static_cast<vx_int32>(frame.step);
        image_addr.scale_x = image_addr.scale_y = VX_SCALE_UNITY;

        convertFrame(context_,
                     image,
                     getConfiguration(),
                     image_addr,
                     frame.data,
                     false,
                     devMem,
                     devMemPitch,
                     scaledImage);

        return OK;
    }
    else
    {
        if (alive_)
        {
            return TIMEOUT;
        }
        else
        {
            close();
            return CLOSED;
        }
    }
}

void OpenCVFrameSourceImpl::close()
{
    alive_ = false;
    if (thread.joinable())
        thread.join();

    queue_.clear();
    source_->close();

    if (devMem != NULL)
    {
        cudaFree(devMem);
        devMem = NULL;
    }

    if (scaledImage)
    {
        vxReleaseImage(&scaledImage);
        scaledImage = NULL;
    }
}

OpenCVFrameSourceImpl::~OpenCVFrameSourceImpl()
{
    close();
}

void OpenCVFrameSourceImpl::threadFunc()
{
    const unsigned int timeout = 30; /*milliseconds*/

    while (alive_ && source_->grab())
    {
        cv::Mat tmp = source_->fetch();
        while (alive_ && !queue_.push(tmp, timeout)) { }
    }

    alive_ = false;
}

}

#endif
