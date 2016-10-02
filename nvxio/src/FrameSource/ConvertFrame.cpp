/*
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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

#include "NVXIO/Utility.hpp"
#include "NVXIO/FrameSource.hpp"

#include <cuda_runtime_api.h>

namespace nvxio
{

void convertFrame(vx_context vxContext,
                  vx_image frame,
                  const FrameSource::Parameters & configuration,
                  vx_imagepatch_addressing_t & decodedImageAddr,
                  void * decodedPtr,
                  bool is_cuda,
                  void *& devMem,
                  size_t & devMemPitch,
                  vx_image & scaledImage
                  )
{
    vx_df_image_e vx_type_map[5] = { VX_DF_IMAGE_VIRT, VX_DF_IMAGE_U8,
                                     VX_DF_IMAGE_VIRT, VX_DF_IMAGE_RGB, VX_DF_IMAGE_RGBX };
    vx_df_image_e decodedFormat = vx_type_map[decodedImageAddr.stride_x];

    // fetch image width and height
    vx_uint32 frameWidth, frameHeight;
    vx_df_image_e frameFormat;
    NVXIO_SAFE_CALL( vxQueryImage(frame, VX_IMAGE_ATTRIBUTE_WIDTH, (void *)&frameWidth, sizeof(frameWidth)) );
    NVXIO_SAFE_CALL( vxQueryImage(frame, VX_IMAGE_ATTRIBUTE_HEIGHT, (void *)&frameHeight, sizeof(frameHeight)) );
    NVXIO_SAFE_CALL( vxQueryImage(frame, VX_IMAGE_ATTRIBUTE_FORMAT, (void *)&frameFormat, sizeof(frameFormat)) );
    bool needScale = frameWidth != decodedImageAddr.dim_x ||
                     frameHeight != decodedImageAddr.dim_y;
    bool needConvert = frameFormat != decodedFormat;

    // config and actual image sized must be the same!
    if ((frameWidth != configuration.frameWidth) ||
            (frameHeight != configuration.frameHeight))
    {
        NVXIO_THROW_EXCEPTION("Actual image [ " << frameWidth << " x " << frameHeight <<
                              " ] is not equal to configuration one [ " << configuration.frameWidth
                              << " x " << configuration.frameHeight << " ]");
    }

    // allocate CUDA memory to copy decoded image to
    if (!is_cuda)
    {
        if (!devMem)
        {
            // we assume that decoded image will have no more than 4 channels per pixel
            NVXIO_ASSERT( cudaSuccess == cudaMallocPitch(&devMem, &devMemPitch, decodedImageAddr.dim_x * 4,
                                                         decodedImageAddr.dim_y) );
        }
    }

    // check if decoded image format has changed
    if (scaledImage)
    {
        vx_df_image_e scaledFormat;
        NVXIO_SAFE_CALL( vxQueryImage(scaledImage, VX_IMAGE_ATTRIBUTE_FORMAT, (void *)&scaledFormat, sizeof(scaledFormat)) );

        if (scaledFormat != decodedFormat)
        {
            NVXIO_SAFE_CALL( vxReleaseImage(&scaledImage) );
            scaledImage = NULL;
        }
    }

    if (needScale && !scaledImage)
    {
        scaledImage = vxCreateImage(vxContext, frameWidth, frameHeight, decodedFormat);
        NVXIO_CHECK_REFERENCE( scaledImage );
    }

    vx_image decodedImage = NULL;

    // 1. create vx_image wrapper
    if (is_cuda)
    {
        // a. create vx_image wrapper from CUDA pointer
        decodedImage = vxCreateImageFromHandle(vxContext, decodedFormat, &decodedImageAddr,
                                               &decodedPtr, NVX_IMPORT_TYPE_CUDA);
    }
    else
    {
        cudaStream_t stream = NULL;

        // a. upload decoded image to CUDA buffer
        NVXIO_ASSERT( cudaSuccess == cudaMemcpy2DAsync(devMem, devMemPitch,
                                                       decodedPtr, decodedImageAddr.stride_y,
                                                       decodedImageAddr.dim_x * decodedImageAddr.stride_x,
                                                       decodedImageAddr.dim_y, cudaMemcpyHostToDevice, stream) );

        NVXIO_ASSERT( cudaStreamSynchronize(stream) == cudaSuccess );

        // b. create vx_image wrapper for decoded buffer
        decodedImageAddr.stride_y = static_cast<vx_int32>(devMemPitch);
        decodedImage = vxCreateImageFromHandle(vxContext, decodedFormat, &decodedImageAddr,
                                               &devMem, NVX_IMPORT_TYPE_CUDA);
    }
    NVXIO_CHECK_REFERENCE( decodedImage );

    // 2. scale if necessary
    if (needScale)
    {
        // a. scale image
        NVXIO_SAFE_CALL( vxuScaleImage(vxContext, decodedImage, scaledImage, VX_INTERPOLATION_TYPE_BILINEAR) );
    }
    else
    {
        scaledImage = decodedImage;
    }

    // 3. convert / copy to dst image
    if (needConvert)
    {
        NVXIO_SAFE_CALL( vxuColorConvert(vxContext, scaledImage, frame) );
    }
    else
    {
        NVXIO_SAFE_CALL( nvxuCopyImage(vxContext, scaledImage, frame) );
    }

    if (!needScale)
        scaledImage = NULL;

    NVXIO_SAFE_CALL( vxReleaseImage(&decodedImage) );
}

}
