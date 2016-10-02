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

#ifdef USE_CSI_OV10640

#include "Private/LogUtils.hpp"

#include "FrameSource/EGLAPIAccessors.hpp"

#include "NVXIO/FrameSource.hpp"
#include "NVXIO/Application.hpp"
#include "NVXIO/ConfigParser.hpp"

#include <cuda_runtime.h>
#include <cstring>

#include "FrameSource/NvMedia/NvMediaCSI10640CameraFrameSourceImpl.hpp"


using namespace nvxio::egl_api;

namespace nvxio
{

NvMediaCSI10640CameraFrameSourceImpl::NvMediaCSI10640CameraFrameSourceImpl(vx_context vxContext_, const std::string & configName, int number) :
    FrameSource(FrameSource::CAMERA_SOURCE, "NvMediaCSI10640CameraFrameSource")
{
    convertedImage = NULL;
    nv12Frame = NULL;

    ctx = NULL;
    interopCtx = NULL;

    cameraNumber = number;
    configPath = configName;

    vxContext = vxContext_;
}

std::string NvMediaCSI10640CameraFrameSourceImpl::parseCameraConfig(const std::string cameraConfigFile,
    CaptureConfigParams& captureConfigCollection)
{
    std::unique_ptr<nvxio::ConfigParser> cameraConfigParser(nvxio::createConfigParser());

    captureConfigCollection.i2cDevice = -1;

    cameraConfigParser->addParameter("capture-name", nvxio::OptionHandler::string(&captureConfigCollection.name));
    cameraConfigParser->addParameter("capture-description", nvxio::OptionHandler::string(&captureConfigCollection.description));
    cameraConfigParser->addParameter("board", nvxio::OptionHandler::string(&captureConfigCollection.board));
    cameraConfigParser->addParameter("input_device", nvxio::OptionHandler::string(&captureConfigCollection.inputDevice));
    cameraConfigParser->addParameter("input_format", nvxio::OptionHandler::string(&captureConfigCollection.inputFormat));
    cameraConfigParser->addParameter("surface_format", nvxio::OptionHandler::string(&captureConfigCollection.surfaceFormat));
    cameraConfigParser->addParameter("resolution", nvxio::OptionHandler::string(&captureConfigCollection.resolution));
    cameraConfigParser->addParameter("csi_lanes", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.csiLanes));
    cameraConfigParser->addParameter("interface", nvxio::OptionHandler::string(&captureConfigCollection.interface));
    cameraConfigParser->addParameter("embedded_lines_top", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.embeddedDataLinesTop));
    cameraConfigParser->addParameter("embedded_lines_bottom", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.embeddedDataLinesBottom));
    cameraConfigParser->addParameter("i2c_device", nvxio::OptionHandler::integer(&captureConfigCollection.i2cDevice));
    cameraConfigParser->addParameter("max9286_address", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.desAddr));

    memset(captureConfigCollection.sensorAddr, 0, sizeof(captureConfigCollection.sensorAddr));
    memset(captureConfigCollection.serAddr, 0, sizeof(captureConfigCollection.serAddr));

    cameraConfigParser->addParameter("max9271_address", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.brdcstSerAddr));
    cameraConfigParser->addParameter("max9271_address_0", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.serAddr[0]));
    cameraConfigParser->addParameter("max9271_address_1", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.serAddr[1]));
    cameraConfigParser->addParameter("max9271_address_2", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.serAddr[2]));
    cameraConfigParser->addParameter("max9271_address_3", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.serAddr[3]));

    cameraConfigParser->addParameter("sensor_address", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.brdcstSensorAddr));
    cameraConfigParser->addParameter("sensor_address_0", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.sensorAddr[0]));
    cameraConfigParser->addParameter("sensor_address_1", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.sensorAddr[1]));
    cameraConfigParser->addParameter("sensor_address_2", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.sensorAddr[2]));
    cameraConfigParser->addParameter("sensor_address_3", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.sensorAddr[3]));

    return cameraConfigParser->parse(cameraConfigFile);
}

bool NvMediaCSI10640CameraFrameSourceImpl::open()
{
    close();

    std::map<std::string, CaptureConfigParams>::iterator conf = cameraConfigCollection.find(configPath);
    if (conf != cameraConfigCollection.end())
    {
        captureConfigCollection = conf->second;
        NVXIO_PRINT("Prebuilt camera config from config preset is used ...");
    }
    else
    {
        Application &app = nvxio::Application::get();
        std::string cameraConfigFile = app.findSampleFilePath("nvxio/cameras/" + configPath + ".ini");

        std::string message = parseCameraConfig(cameraConfigFile, captureConfigCollection);
        if (!message.empty())
        {
            NVXIO_PRINT("Error: %s", message.c_str());
            return false;
        }
    }

    if (cudaFree(NULL) != cudaSuccess)
    {
        NVXIO_PRINT("Error: Failed to initialize CUDA context");
        return false;
    }

    // allocate objects
    ctx = new IPPCtx;
    ctx->imagesNum = cameraNumber;
    ctx->ippManager = NULL;
    ctx->extImgDevice = NULL;
    ctx->device = NULL;
    std::memset(ctx->ipp, 0, sizeof(NvMediaIPPPipeline *) * NVMEDIA_MAX_PIPELINES_PER_MANAGER);

    interopCtx = new InteropContext;
    std::memset(interopCtx, 0, sizeof(InteropContext));
    interopCtx->producerExited = NVMEDIA_TRUE;

    if (IsFailed(IPPInit(ctx, captureConfigCollection)))
    {
        NVXIO_PRINT("Error: Failed to Initialize IPPInit");
        close();
        return false;
    }

    if (IsFailed(InteropInit(interopCtx, ctx)))
    {
        NVXIO_PRINT("Error: Failed to Initialize InteropInit");
        close();
        return false;
    }

    if(IsFailed(InteropProc(interopCtx)))
    {
        NVXIO_PRINT("Error: Failed to start InteropProc");
        close();
        return false;
    }

    if(IsFailed(IPPStart(ctx)))
    {
        NVXIO_PRINT("Error: Failed to start IPPStart");
        close();
        return false;
    }

    // fill frame source configuration
    if (configuration.frameWidth == (vx_uint32)-1)
        configuration.frameWidth = ctx->inputWidth;
    if (configuration.frameHeight == (vx_uint32)-1)
        configuration.frameHeight = ctx->inputHeight;

    configuration.fps = 30;
    configuration.format = VX_DF_IMAGE_NV12;

    nv12Frame = vxCreateImage(vxContext, ctx->inputWidth, ctx->inputHeight, VX_DF_IMAGE_NV12);
    NVXIO_CHECK_REFERENCE(nv12Frame);

    return true;
}

FrameSource::FrameStatus NvMediaCSI10640CameraFrameSourceImpl::fetch(vx_image image, vx_uint32 /* timeout milliseconds*/)
{
    if (!ctx || ctx->quit)
    {
        close();
        return FrameSource::FrameStatus::CLOSED;
    }

    if (cudaSuccess != cudaFree(NULL))
    {
        NVXIO_PRINT("Failed to initialize CUDA context");

        return FrameSource::CLOSED;
    }

    for (vx_uint32 i = 0; i < ctx->imagesNum; ++i)
    {
        CUgraphicsResource cudaResource = NULL;

        // Check for new frames in EglStream
        EGLint streamState = 0;

        for ( ; ; )
        {
            if (!eglQueryStreamKHR(ctx->eglDisplay, ctx->eglStream[i], EGL_STREAM_STATE_KHR, &streamState))
            {
                NVXIO_PRINT("Cuda consumer, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed");
                close();

                return FrameSource::FrameStatus::CLOSED;
            }

            if (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR)
            {
                NVXIO_PRINT("CUDA Consumer: - EGL_STREAM_STATE_DISCONNECTED_KHR received");
                close();

                return FrameSource::FrameStatus::CLOSED;
            }

            if (streamState == EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR)
                break;

            usleep(1000);
        }

        // Acquire frame in CUDA resource from EGLStream
        NVXIO_ASSERT( cuEGLStreamConsumerAcquireFrame(ctx->cudaConnection + i, &cudaResource, NULL, 33000) == CUDA_SUCCESS );

        // If frame is acquired succesfully get the mapped CuEglFrame from CUDA resource
        CUeglFrame cudaEgl;
        NVXIO_ASSERT( cuGraphicsResourceGetMappedEglFrame(&cudaEgl, cudaResource, 0, 0) == CUDA_SUCCESS );

        NVXIO_ASSERT(cudaEgl.frameType == CU_EGL_FRAME_TYPE_ARRAY);
        NVXIO_ASSERT(cudaEgl.cuFormat == CU_AD_FORMAT_UNSIGNED_INT8);
        NVXIO_ASSERT(cudaEgl.eglColorFormat == CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR);
        NVXIO_ASSERT(cudaEgl.planeCount == 2);

        {
            vx_rectangle_t rect = {};

            rect.start_x = cudaEgl.width * i;
            rect.end_x = cudaEgl.width * (i + 1);
            rect.start_y = 0;
            rect.end_y = cudaEgl.height;

            // copy the first plane y

            vx_imagepatch_addressing_t addr;
            void *ptr = NULL;
            NVXIO_SAFE_CALL( vxAccessImagePatch(nv12Frame, &rect, 0, &addr, &ptr, NVX_WRITE_ONLY_CUDA) );

            cudaStream_t stream = NULL;
            NVXIO_ASSERT( cudaMemcpy2DFromArrayAsync(ptr, addr.stride_y,
                                                     (const struct cudaArray *) cudaEgl.frame.pArray[0],
                                                     0, 0,
                                                     cudaEgl.width * sizeof(vx_uint8), addr.dim_y,
                                                     cudaMemcpyDeviceToDevice, stream) == cudaSuccess );

            NVXIO_SAFE_CALL( vxCommitImagePatch(nv12Frame, &rect, 0, &addr, ptr) );

            // copy the second plane u/v

            vx_rectangle_t uv_rect = { rect.start_x >> 1, rect.start_y >> 1,
                                       rect.end_x   >> 1, rect.end_y   >> 1 };
            uv_rect = rect;

            ptr = NULL; // important!!
            NVXIO_SAFE_CALL( vxAccessImagePatch(nv12Frame, &uv_rect, 1, &addr, &ptr, NVX_WRITE_ONLY_CUDA) );

            NVXIO_ASSERT( (cudaMemcpy2DFromArrayAsync(ptr, addr.stride_y,
                                                      (const struct cudaArray *)cudaEgl.frame.pArray[1],
                                                      0, 0,
                                                      (cudaEgl.width >> 1) * sizeof(vx_uint16), addr.dim_y >> 1,
                                                      cudaMemcpyDeviceToDevice, stream) == cudaSuccess) );

            NVXIO_SAFE_CALL( vxCommitImagePatch(nv12Frame, &uv_rect, 1, &addr, ptr) );

            NVXIO_ASSERT( cudaStreamSynchronize(stream) == cudaSuccess );
        }

        NVXIO_ASSERT( cuEGLStreamConsumerReleaseFrame(ctx->cudaConnection + i, cudaResource, NULL) == CUDA_SUCCESS );
    }

    // copy or convert to output image
    {
        // get size and format
        vx_df_image frameFormat = 0;
        vx_uint32 frameWidth = 0, frameHeight = 0;
        vx_uint32 nv12_width = 0, nv12_height = 0;

        NVXIO_SAFE_CALL( vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &frameFormat, sizeof(frameFormat)) );
        NVXIO_SAFE_CALL( vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &frameWidth, sizeof(frameWidth)) );
        NVXIO_SAFE_CALL( vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &frameHeight, sizeof(frameHeight)) );

        NVXIO_SAFE_CALL( vxQueryImage(nv12Frame, VX_IMAGE_ATTRIBUTE_WIDTH, &nv12_width, sizeof(nv12_width)) );
        NVXIO_SAFE_CALL( vxQueryImage(nv12Frame, VX_IMAGE_ATTRIBUTE_HEIGHT, &nv12_height, sizeof(nv12_height)) );

        bool needScale = (frameWidth != nv12_width) || (frameHeight != nv12_height);
        bool needConvert = frameFormat != VX_DF_IMAGE_NV12;

        // config and actual image sized must be the same!
        if ((frameWidth != configuration.frameWidth) ||
                (frameHeight != configuration.frameHeight))
        {
            NVXIO_THROW_EXCEPTION("Actual image [ " << frameWidth << " x " << frameHeight <<
                                  " ] is not equal to configuration one [ " << configuration.frameWidth
                                  << " x " << configuration.frameHeight << " ]");
        }

        // check if result image format has changed
        if (convertedImage)
        {
            vx_df_image_e convertedFormat;
            NVXIO_SAFE_CALL( vxQueryImage(convertedImage, VX_IMAGE_ATTRIBUTE_FORMAT, (void *)&convertedFormat, sizeof(convertedFormat)) );

            if (convertedFormat != frameFormat)
            {
                NVXIO_SAFE_CALL( vxReleaseImage(&convertedImage) );
                convertedImage = NULL;
            }
        }

        // create converted image
        if (needConvert && !convertedImage)
        {
            convertedImage = vxCreateImage(vxContext, nv12_width, nv12_height, frameFormat);
            NVXIO_CHECK_REFERENCE( convertedImage );
        }

        // 1. make convertion if needed
        if (needConvert)
        {
            NVXIO_SAFE_CALL( vxuColorConvert(vxContext, nv12Frame, convertedImage) );
        }
        else
        {
            NVXIO_ASSERT(convertedImage == NULL);

            // just assign
            // and then assign to NULL later
            convertedImage = nv12Frame;
        }

        // 2. make scale if needed
        if (needScale)
        {
            NVXIO_SAFE_CALL( vxuScaleImage(vxContext, convertedImage, image, VX_INTERPOLATION_TYPE_BILINEAR) );
        }
        else
        {
            NVXIO_SAFE_CALL( nvxuCopyImage(vxContext, convertedImage, image) );
        }

        // assign to NULL
        if (!needConvert)
            convertedImage = NULL;
    }

    return FrameSource::FrameStatus::OK;
}

FrameSource::Parameters NvMediaCSI10640CameraFrameSourceImpl::getConfiguration()
{
    return configuration;
}

bool NvMediaCSI10640CameraFrameSourceImpl::setConfiguration(const FrameSource::Parameters& params)
{
    bool result = true;

    if (ctx && !ctx->quit)
    {
        if ((params.frameWidth != (vx_uint32)-1) && (params.frameWidth != configuration.frameWidth))
            result = false;
        if ((params.frameHeight != (vx_uint32)-1) && (params.frameHeight != configuration.frameHeight))
            result = false;
    }
    else
    {
        configuration.frameHeight = params.frameHeight;
        configuration.frameWidth = params.frameWidth;
    }

    if ((params.fps != (vx_uint32)-1) && (params.fps != configuration.fps))
        result = false;

    configuration.format = params.format;

    return result;
}

void NvMediaCSI10640CameraFrameSourceImpl::close()
{
    if (convertedImage)
    {
        vxReleaseImage(&convertedImage);
        convertedImage = NULL;
    }

    if (nv12Frame)
    {
        vxReleaseImage(&nv12Frame);
        nv12Frame = NULL;
    }

    if (ctx)
    {
        ctx->quit = NVMEDIA_TRUE;

        if (IsFailed(IPPStop(ctx)))
        {
            NVXIO_PRINT("Error: Failed to stop IPPStop");
        }

        if (interopCtx)
        {
            InteropFini(interopCtx);

            delete interopCtx;
            interopCtx = NULL;
        }

        IPPFini(ctx);

        delete ctx;
        ctx = NULL;
    }
}

NvMediaCSI10640CameraFrameSourceImpl::~NvMediaCSI10640CameraFrameSourceImpl()
{
    close();
}

}

#endif // USE_CSI_OV10640
