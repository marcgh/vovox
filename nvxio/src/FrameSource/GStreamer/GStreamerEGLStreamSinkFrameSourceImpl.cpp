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

#if defined USE_GSTREAMER_NVMEDIA || defined USE_NVGSTCAMERA

#include "NVXIO/FrameSource.hpp"
#include "NVXIO/Application.hpp"
#include "FrameSource/GStreamer/GStreamerEGLStreamSinkFrameSourceImpl.hpp"

#include <VX/vx.h>
#include <NVX/nvx.h>
#include <cuda_runtime.h>

#include <gst/pbutils/missing-plugins.h>

#include <memory>
#include <thread>
#include <string>

using namespace nvxio::egl_api;

namespace nvxio
{

void convertFrame(vx_context vxContext, vx_image frame,
                  const FrameSource::Parameters & configuration,
                  vx_imagepatch_addressing_t & decodedImageAddr,
                  void * decodedPtr, bool is_cuda, void *& devMem,
                  size_t & devMemPitch, vx_image & scaledImage);

GStreamerEGLStreamSinkFrameSourceImpl::GStreamerEGLStreamSinkFrameSourceImpl(vx_context vxcontext, SourceType sourceType,
                                                                             const char * const name, bool fifomode) :
    FrameSource(sourceType, name),
    pipeline(NULL),
    bus(NULL),
    end(true),
    fifoLength(4),
    fifoMode(fifomode),
    latency(0),
    cudaConnection(NULL),
    vxContext(vxcontext),
    scaledImage(NULL)
{
    context.stream = EGL_NO_STREAM_KHR;
    context.display = EGL_NO_DISPLAY;
}

bool GStreamerEGLStreamSinkFrameSourceImpl::open()
{
    if (pipeline)
    {
        close();
    }

    NVXIO_PRINT("Initializing EGL display");
    if (!InitializeEGLDisplay())
    {
        NVXIO_PRINT("Cannot initialize EGL display");
        return false;
    }

    NVXIO_PRINT("Initializing EGL stream");
    if (!InitializeEGLStream())
    {
        NVXIO_PRINT("Cannot initialize EGL Stream");
        return false;
    }

    NVXIO_PRINT("Initializing EGL consumer");
    if (!InitializeEglCudaConsumer())
    {
        NVXIO_PRINT("Cannot initialize CUDA consumer");
        return false;
    }

    NVXIO_PRINT("Creating GStreamer pipeline");
    if (!InitializeGstPipeLine())
    {
        NVXIO_PRINT("Cannot initialize Gstreamer pipeline");
        return false;
    }

    return true;
}

bool GStreamerEGLStreamSinkFrameSourceImpl::InitializeEGLDisplay()
{
    // Obtain the EGL display
    context.display = nvxio::EGLDisplayAccessor::getInstance();
    if (context.display == EGL_NO_DISPLAY)
    {
        NVXIO_PRINT("EGL failed to obtain display.");
        return false;
    }

    return true;
}

bool GStreamerEGLStreamSinkFrameSourceImpl::InitializeEglCudaConsumer()
{
    if (cudaSuccess != cudaFree(NULL))
    {
        NVXIO_PRINT("Failed to initialize CUDA context");
        return false;
    }

    NVXIO_PRINT("Connect CUDA consumer");
    CUresult curesult = cuEGLStreamConsumerConnect(&cudaConnection, context.stream);
    if (CUDA_SUCCESS != curesult)
    {
        NVXIO_PRINT("Connect CUDA consumer ERROR %d", curesult);
        return false;
    }

    return true;
}

bool GStreamerEGLStreamSinkFrameSourceImpl::InitializeEGLStream()
{
    const EGLint streamAttrMailboxMode[] = { EGL_NONE };
    const EGLint streamAttrFIFOMode[] = { EGL_STREAM_FIFO_LENGTH_KHR, fifoLength, EGL_NONE };

    if(!setupEGLExtensions())
        return false;

    context.stream = eglCreateStreamKHR(context.display, fifoMode ? streamAttrFIFOMode : streamAttrMailboxMode);
    if (context.stream == EGL_NO_STREAM_KHR)
    {
        NVXIO_PRINT("Couldn't create stream.");
        return false;
    }

    if (!eglStreamAttribKHR(context.display, context.stream, EGL_CONSUMER_LATENCY_USEC_KHR, latency))
    {
        NVXIO_PRINT("Consumer: streamAttribKHR EGL_CONSUMER_LATENCY_USEC_KHR failed");
    }
    if (!eglStreamAttribKHR(context.display, context.stream, EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, 0))
    {
        NVXIO_PRINT("Consumer: streamAttribKHR EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR failed");
    }

    // Get stream attributes
    if (!eglQueryStreamKHR(context.display, context.stream, EGL_STREAM_FIFO_LENGTH_KHR, &fifoLength))
    {
        NVXIO_PRINT("Consumer: eglQueryStreamKHR EGL_STREAM_FIFO_LENGTH_KHR failed");
    }
    if (!eglQueryStreamKHR(context.display, context.stream, EGL_CONSUMER_LATENCY_USEC_KHR, &latency))
    {
        NVXIO_PRINT("Consumer: eglQueryStreamKHR EGL_CONSUMER_LATENCY_USEC_KHR failed");
    }

    if (fifoMode != (fifoLength > 0))
    {
        NVXIO_PRINT("EGL Stream consumer - Unable to set FIFO mode");
        fifoMode = false;
    }
    if (fifoMode)
    {
        NVXIO_PRINT("EGL Stream consumer - Mode: FIFO Length: %d", fifoLength);
    }
    else
    {
        NVXIO_PRINT("EGL Stream consumer - Mode: Mailbox");
    }

    return true;
}

FrameSource::FrameStatus GStreamerEGLStreamSinkFrameSourceImpl::fetch(vx_image image, vx_uint32 timeout)
{
    handleGStreamerMessages();

    if (end)
    {
        close();
        return FrameSource::CLOSED;
    }

    if (cudaSuccess != cudaFree(NULL))
    {
        NVXIO_PRINT("Failed to initialize CUDA context");
        return FrameSource::CLOSED;
    }

    CUgraphicsResource cudaResource;
    CUeglFrame eglFrame;
    EGLint streamState = 0;

    if (!eglQueryStreamKHR(context.display, context.stream, EGL_STREAM_STATE_KHR, &streamState))
    {
        NVXIO_PRINT("Cuda consumer, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed");
        close();
        return FrameSource::CLOSED;
    }

    if (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR)
    {
        NVXIO_PRINT("CUDA Consumer: - EGL_STREAM_STATE_DISCONNECTED_KHR received");
        close();
        return FrameSource::CLOSED;
    }

    if (streamState != EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR)
    {
        return FrameSource::TIMEOUT;
    }

    CUresult cuStatus = cuEGLStreamConsumerAcquireFrame(&cudaConnection, &cudaResource, NULL, timeout*1000);
    if (cuStatus != CUDA_SUCCESS)
    {
        NVXIO_PRINT("Cuda Acquire failed cuStatus=%d", cuStatus);
        close();
        return FrameSource::CLOSED;
    }

    cuStatus = cuGraphicsResourceGetMappedEglFrame(&eglFrame, cudaResource, 0, 0);
    if (cuStatus != CUDA_SUCCESS)
    {
        NVXIO_PRINT("Cuda get resource failed with %d", cuStatus);
        cuEGLStreamConsumerReleaseFrame(&cudaConnection, cudaResource, NULL);
        close();
        return FrameSource::CLOSED;
    }

    vx_imagepatch_addressing_t decodedImageAddr;
    decodedImageAddr.dim_x = eglFrame.width;
    decodedImageAddr.dim_y = eglFrame.height;
    decodedImageAddr.stride_x = 4;
    decodedImageAddr.stride_y = eglFrame.pitch;
    decodedImageAddr.scale_x = decodedImageAddr.scale_y = VX_SCALE_UNITY;
    decodedImageAddr.step_x = decodedImageAddr.step_y = 1;

    void * devMem = NULL;
    size_t devMemPitch = 0;
    convertFrame(vxContext,
                 image,
                 configuration,
                 decodedImageAddr,
                 eglFrame.frame.pPitch[0],
                 true,
                 devMem,
                 devMemPitch,
                 scaledImage);

    NVXIO_ASSERT(devMem == nullptr && devMemPitch == 0);

    cuStatus = cuEGLStreamConsumerReleaseFrame(&cudaConnection, cudaResource, NULL);

    return FrameSource::OK;
}

FrameSource::Parameters GStreamerEGLStreamSinkFrameSourceImpl::getConfiguration()
{
    return configuration;
}

bool GStreamerEGLStreamSinkFrameSourceImpl::setConfiguration(const FrameSource::Parameters& params)
{
    bool result = true;

    if (end)
    {
        configuration.frameHeight = params.frameHeight;
        configuration.frameWidth = params.frameWidth;
    }
    else
    {
        if ((params.frameWidth != (vx_uint32)-1) && (params.frameWidth != configuration.frameWidth))
            result = false;
        if ((params.frameHeight != (vx_uint32)-1) && (params.frameHeight != configuration.frameHeight))
            result = false;
    }

    if ((params.fps != (vx_uint32)-1) && (params.fps != configuration.fps))
        result = false;

    configuration.format = params.format;

    return result;
}

void GStreamerEGLStreamSinkFrameSourceImpl::handleGStreamerMessages()
{
    std::unique_ptr<GstMessage, GStreamerObjectDeleter> msg;
    GError *err = NULL;
    gchar *debug = NULL;
    GstStreamStatusType tp;
    GstElement * elem = NULL;

    if (!bus)
        return;

    while (gst_bus_have_pending(bus))
    {
        msg.reset(gst_bus_pop(bus));

        if (!msg)
            continue;

        if (gst_is_missing_plugin_message(msg.get()))
        {
            NVXIO_PRINT("GStreamer: your gstreamer installation is missing a required plugin!");
            end = true;
        }
        else
        {
            switch (GST_MESSAGE_TYPE(msg.get()))
            {
                case GST_MESSAGE_STATE_CHANGED:
                    gst_message_parse_state_changed(msg.get(), NULL, NULL, NULL);
                    break;
                case GST_MESSAGE_ERROR:
                {
                    gst_message_parse_error(msg.get(), &err, &debug);
                    std::unique_ptr<char[], GlibDeleter> name(gst_element_get_name(GST_MESSAGE_SRC(msg.get())));
                    NVXIO_PRINT("GStreamer Plugin: Embedded video playback halted; module %s reported: %s",
                           name.get(), err->message);
                    g_error_free(err);
                    g_free(debug);
                    end = true;
                    break;
                }
                case GST_MESSAGE_EOS:
                    end = true;
                    break;
                case GST_MESSAGE_STREAM_STATUS:
                    gst_message_parse_stream_status(msg.get(), &tp, &elem);
                    break;
                default:
                    break;
            }
        }
    }
}

void GStreamerEGLStreamSinkFrameSourceImpl::FinalizeEglStream()
{
    if (context.stream != EGL_NO_STREAM_KHR)
    {
        eglDestroyStreamKHR(context.display, context.stream);
        context.stream = EGL_NO_STREAM_KHR;
    }
}

void GStreamerEGLStreamSinkFrameSourceImpl::FinalizeEglCudaConsumer()
{
    if (cudaConnection != NULL)
    {
        if (cudaSuccess != cudaFree(NULL))
        {
            NVXIO_PRINT("Failed to initialize CUDA context");
            return;
        }

        cuEGLStreamConsumerDisconnect(&cudaConnection);
        cudaConnection = NULL;
    }
}

void GStreamerEGLStreamSinkFrameSourceImpl::CloseGstPipeLineAsyncThread()
{
    gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_NULL);
    end = true;
}

void GStreamerEGLStreamSinkFrameSourceImpl::FinalizeGstPipeLine()
{
    if (pipeline != NULL)
    {
        std::thread t(&GStreamerEGLStreamSinkFrameSourceImpl::CloseGstPipeLineAsyncThread, this);

        if (fifoMode)
        {
            if (cudaSuccess != cudaFree(NULL))
            {
                NVXIO_PRINT("Failed to initialize CUDA context");
                return;
            }

            CUgraphicsResource cudaResource;
            EGLint streamState = 0;
            while (!end)
            {
                if (!eglQueryStreamKHR(context.display, context.stream, EGL_STREAM_STATE_KHR, &streamState))
                {
                    handleGStreamerMessages();
                    break;
                }

                if (streamState == EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR)
                {
                    cuEGLStreamConsumerAcquireFrame(&cudaConnection, &cudaResource, NULL, 1000);
                    cuEGLStreamConsumerReleaseFrame(&cudaConnection, cudaResource, NULL);
                }
                else
                {
                    handleGStreamerMessages();
                    continue;
                }
                handleGStreamerMessages();
            }

        }

        t.join();

        gst_object_unref(GST_OBJECT(bus));
        bus = NULL;

        gst_object_unref(GST_OBJECT(pipeline));
        pipeline = NULL;
    }
}

void GStreamerEGLStreamSinkFrameSourceImpl::close()
{
    handleGStreamerMessages();
    FinalizeGstPipeLine();
    FinalizeEglCudaConsumer();
    FinalizeEglStream();

    if (scaledImage)
    {
        vxReleaseImage(&scaledImage);
        scaledImage = NULL;
    }
}

GStreamerEGLStreamSinkFrameSourceImpl::~GStreamerEGLStreamSinkFrameSourceImpl()
{
    close();
}

}

#endif // defined USE_GSTREAMER_NVMEDIA || defined USE_NVGSTCAMERA
