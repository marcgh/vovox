/*
# Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
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

#ifdef USE_CSI_OV10635

#include "Private/LogUtils.hpp"

#include "FrameSource/EGLAPIAccessors.hpp"

#include "NVXIO/FrameSource.hpp"
#include "NVXIO/Application.hpp"
#include "NVXIO/ConfigParser.hpp"

#include <cuda_runtime.h>

#include "FrameSource/NvMedia/NvMediaCSI10635CameraFrameSourceImpl.hpp"
#include "FrameSource/NvMedia/NvMediaCameraConfigParams.hpp"

using namespace nvxio::egl_api;

namespace nvxio
{

NvMediaCSI10635CameraFrameSourceImpl::NvMediaCSI10635CameraFrameSourceImpl(vx_context vxContext_, const std::string & configName, int number) :
    FrameSource(FrameSource::CAMERA_SOURCE, "NvMediaCSI10635CameraFrameSource")
{
    scaledImage = NULL;
    context = NULL;
    cameraNumber = number;
    configPath = configName;

    vxContext = vxContext_;
}

std::string NvMediaCSI10635CameraFrameSourceImpl::parseCameraConfig(const std::string cameraConfigFile,
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

bool NvMediaCSI10635CameraFrameSourceImpl::open()
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

    if (ov10635::ImgCapture_Init(&context, captureConfigCollection, cameraNumber) != NVMEDIA_STATUS_OK)
    {
        NVXIO_PRINT("Error: Failed to Initialize ImgCapture");
        return false;
    }

    // fill frame source configuration
    if (configuration.frameWidth == (vx_uint32)-1)
        configuration.frameWidth = context->outputWidth;
    if (configuration.frameHeight == (vx_uint32)-1)
        configuration.frameHeight = context->outputHeight;

    configuration.fps = 30;
    configuration.format = VX_DF_IMAGE_RGBX;

    return true;
}

void convertFrame(vx_context vxContext, vx_image frame,
                  const FrameSource::Parameters & configuration,
                  vx_imagepatch_addressing_t & decodedImageAddr,
                  void * decodedPtr, bool is_cuda, void *& devMem,
                  size_t & devMemPitch, vx_image & scaledImage);

FrameSource::FrameStatus NvMediaCSI10635CameraFrameSourceImpl::fetch(vx_image image, vx_uint32 timeout /*milliseconds*/)
{
    if (context->quit)
    {
        close();
        return FrameSource::FrameStatus::CLOSED;
    }

    if (cudaSuccess != cudaFree(NULL))
    {
        NVXIO_PRINT("Failed to initialize CUDA context");
        return FrameSource::CLOSED;
    }

    CUresult cuStatus;
    CUgraphicsResource cudaResource;

    EGLint streamState = 0;
    if (!eglQueryStreamKHR(context->eglDisplay, context->eglStream, EGL_STREAM_STATE_KHR, &streamState))
    {
        NVXIO_PRINT("Cuda consumer, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed");
        return FrameSource::FrameStatus::CLOSED;
    }

    if (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR)
    {
        NVXIO_PRINT("CUDA Consumer: - EGL_STREAM_STATE_DISCONNECTED_KHR received");
        return FrameSource::FrameStatus::CLOSED;
    }

    if (streamState != EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR)
    {
        return FrameSource::TIMEOUT;
    }

    cuStatus = cuEGLStreamConsumerAcquireFrame(&context->cudaConnection, &cudaResource, NULL, timeout);
    if (cuStatus != CUDA_SUCCESS)
    {
        NVXIO_PRINT("Cuda Acquire failed cuStatus=%d", cuStatus);

        return FrameSource::FrameStatus::TIMEOUT;
    }

    CUeglFrame eglFrame;
    cuStatus = cuGraphicsResourceGetMappedEglFrame(&eglFrame, cudaResource, 0, 0);
    if (cuStatus != CUDA_SUCCESS)
    {
        const char* error;
        cuGetErrorString(cuStatus, &error);
        NVXIO_PRINT("Cuda get resource failed with error: \"%s\"", error);
        return FrameSource::FrameStatus::CLOSED;
    }

    vx_imagepatch_addressing_t srcImageAddr;
    srcImageAddr.dim_x = eglFrame.width;
    srcImageAddr.dim_y = eglFrame.height;
    srcImageAddr.stride_x = 4;
    srcImageAddr.stride_y = eglFrame.pitch;
    srcImageAddr.scale_x = srcImageAddr.scale_y = VX_SCALE_UNITY;
    srcImageAddr.step_x = srcImageAddr.step_y = 1;

    void * devMem = NULL;
    size_t devMemPitch = 0;

    convertFrame(vxContext, image,
                 configuration, srcImageAddr,
                 eglFrame.frame.pPitch[0], true,
                 devMem, devMemPitch, scaledImage);

    NVXIO_ASSERT(devMem == nullptr && devMemPitch == 0);

    cuStatus = cuEGLStreamConsumerReleaseFrame(&context->cudaConnection, cudaResource, NULL);
    if (cuStatus != CUDA_SUCCESS)
    {
        NVXIO_PRINT("Failed to release EGL frame");
        close();
        return FrameSource::FrameStatus::CLOSED;
    }

    return FrameSource::FrameStatus::OK;
}

FrameSource::Parameters NvMediaCSI10635CameraFrameSourceImpl::getConfiguration()
{
    return configuration;
}

bool NvMediaCSI10635CameraFrameSourceImpl::setConfiguration(const FrameSource::Parameters& params)
{
    bool result = true;

    if (context && !context->quit)
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

void NvMediaCSI10635CameraFrameSourceImpl::close()
{
    if (scaledImage)
    {
        vxReleaseImage(&scaledImage);
        scaledImage = NULL;
    }

    if (context)
    {
        ov10635::ImgCapture_Finish(context);
        context = NULL;
    }
}

NvMediaCSI10635CameraFrameSourceImpl::~NvMediaCSI10635CameraFrameSourceImpl()
{
    close();
}

}

#endif // USE_CSI_OV10635
