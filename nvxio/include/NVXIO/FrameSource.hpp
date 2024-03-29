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

#ifndef NVXIO_FRAMESOURCE_HPP
#define NVXIO_FRAMESOURCE_HPP

#include <memory>
#include <string>

#include <VX/vx.h>

/**
 * \file
 * \brief The `FrameSource` interface and utility functions.
 */

namespace nvxio
{
/**
 * \defgroup group_nvxio_frame_source FrameSource
 * \ingroup nvx_nvxio_api
 *
 * This class is intended for reading images from different sources. The source can be:
 * + Single image (PNG, JPEG, JPG, BMP, TIFF).
 * + Sequence of images (PNG, JPEG, JPG, BMP, TIFF).
 * + Video file.
 * + Video for Linux-compatible cameras.
 * + NVIDIA GStreamer Camera on NVIDIA<sup>&reg;</sup> Jetson&tm; Embedded platforms running L4T R24.
 *
 * \note
 * + GStreamer-based pipeline is used for video decoding on Linux platforms. The
 *   support level of video formats depends on the set of installed GStreamer plugins.
 * + GStreamer-based pipeline with NVIDIA hardware-accelerated codecs is used on
 *   NVIDIA Vibrante Linux platform only (V3L, V4L).
 * + Pure NVIDIA hardware-accelerated decoding of H.264 elementary video streams
 *   is used on NVIDIA Vibrante Linux platforms (V3L, V4L).
 * + OpenCV (FFmpeg back-end)-based pipeline is used for video decoding on Windows.
 * + On Vibrante Linux, an active X session is required for `%FrameSource`, because
     it uses EGL as an interop API.
 * + Image decoding, image sequence decoding, and Video4Linux-compatible camera
 *   support require either OpenCV or GStreamer.
 * + `%FrameSource` is thread safe on all platforms except Vibrante Linux.
 *
 * ### URI-based scheme to specify data sources
 * The URI-based scheme is used to describe data sources:
 *
 *      <scheme>:///<host>[?<arg1=value1>&<arg2=value2>[&...]]
 *
 * The scheme can be:
 *  - `file` - Specifies files on the local machine. The host is a path to the file or file
 *     sequence.
 *     \note The scheme can be omitted and you can specify a path to the file or
 *     file sequence directly.
 *  - `device` - Specifies camera devices. The possible `<host>` values:
 *    - `nvmedia` - Specifies Omni Vision cameras. It contains two arguments:
 *      - `config` - The name of the configuration to use (see the table below).
 *      - `number` - The number of cameras to access. Can be 1, 2, or 4.
 *    - `v4l2` - Specifies V4L2-compatible cameras. It has a single argument:
 *      - `index` - The camera device ID to use.
 *    - `nvcamera` - Specifies NVIDIA built-in cameras on Jetson TX1 boards.
 *
 * #### The list of configurations for an access to Omni Vision cameras
 *
 * |  Board name    | Camera Interface | Camera Type | Configuration name          |
 * |:--------------:|:----------------:|:-----------:|:---------------------------:|
 * | Jetson TK1 Pro |        AB        |   OV10635   | dvp-ov10635-yuv422-ab       |
 * | Jetson TK1 Pro |        CD        |   OV10635   | dvp-ov10635-yuv422-cd       |
 * | Jetson TK1 Pro |        EF        |   OV10635   | dvp-ov10635-yuv422-ef       |
 * | DRIVE PX B00   |        AB        |   OV10635   | dvp-ov10635-yuv422-ab-e2379 |
 * | DRIVE PX B00   |        CD        |   OV10635   | dvp-ov10635-yuv422-cd-e2379 |
 * | DRIVE PX B00   |        EF        |   OV10635   | dvp-ov10635-yuv422-ef-e2379 |
 * | DRIVE PX B00   |        AB        |   OV10640   | dvp-ov10640-raw12-ab-e2379  |
 * | DRIVE CX       |        AB        |   OV10635   | dvp-ov10635-yuv422-ab-p2382 |
 * | DRIVE CX       |        AB        |   OV10640   | dvp-ov10640-raw12-ab-p2382  |
 * | E2580          |        EF        |   OV10635   | dvp-ov10635-yuv422-ef-e2580 |
 *
 */

/**
 * \ingroup group_nvxio_frame_source
 * \brief %FrameSource interface.
 *
 * Common interface for reading frames from all sources.
 *
 * \note To create the object of this class call the \ref createDefaultFrameSource method.
 *
 * \see nvx_nvxio_api
 */
class FrameSource
{
public:
    /**
     * \brief `%FrameSource` parameters.
     */
    struct Parameters
    {
        vx_uint32 frameWidth; /**< \brief Holds the width of the frame. */
        vx_uint32 frameHeight; /**< \brief Holds the height of the frame. */
        vx_df_image format; /**< \brief Holds the format of the frame. */
        vx_uint32 fps; /**< \brief Holds the FPS (for video only). */

        Parameters():
            frameWidth(-1),
            frameHeight(-1),
            format(VX_DF_IMAGE_RGBX),
            fps(-1)
        {}
    };

    /**
     * \brief Defines the type of source.
     */
    enum SourceType
    {
        UNKNOWN_SOURCE, /**< \brief Indicates an unknown source. */
        SINGLE_IMAGE_SOURCE, /**< \brief Indicates a single image. */
        IMAGE_SEQUENCE_SOURCE, /**< \brief Indicates a sequence of images. */
        VIDEO_SOURCE, /**< \brief Indicates a video file. */
        CAMERA_SOURCE /**< \brief Indicates a camera. */
    };

    /**
     * \brief Defines the status of read operations.
     */
    enum FrameStatus
    {
        OK, /**< \brief Indicates the frame has been read successfully. */
        TIMEOUT, /**< \brief Indicates a timeout has been exceeded. */
        CLOSED /**< \brief Indicates the frame source has been closed. */
    };

    /**
     * \brief Opens the FrameSource.
     */
    virtual bool open() = 0;

    /**
     * \brief Fetches frames from the source.
     *
     * \param [out] image                The read image.
     * \param [in]  timeout              Specifies the maximum wait time for the next frame in milliseconds (ms).
     * \return A `vx_status` enumerator.
     */
    virtual FrameSource::FrameStatus fetch(vx_image image, vx_uint32 timeout = 5 /*milliseconds*/) = 0;

    /**
     * \brief Gets the configuration of the `%FrameSource`.
     *
     * \return @ref FrameSource::Parameters describing the current configuration of the `%FrameSource`.
     */
    virtual FrameSource::Parameters getConfiguration() = 0;

    /**
     * \brief Sets the configuration of the `%FrameSource`.
     *
     * \param [in] params A reference to the new configuration of the `%FrameSource`.
     *
     * \return Status of the operation (true - success).
     */
    virtual bool setConfiguration(const FrameSource::Parameters& params) = 0;

    /**
     * \brief Closes the FrameSource.
     */
    virtual void close() = 0;

    /**
     * \brief Destructor.
     */
    virtual ~FrameSource()
    {}

    /**
     * \brief Returns the source type of the `%FrameSource`.
     *
     * \return @ref FrameSource::SourceType.
     */
    FrameSource::SourceType getSourceType() const
    {
        return sourceType;
    }

    /**
     * \brief Returns the source name of the FrameSource.
     *
     * \return Source name.
     */
    std::string getSourceName() const
    {
        return sourceName;
    }

protected:
    FrameSource(FrameSource::SourceType type = FrameSource::UNKNOWN_SOURCE, const std::string & name = "Undefined"):
        sourceType(type),
        sourceName(name)
    {}

    const FrameSource::SourceType  sourceType;
    const std::string sourceName;
};

/**
 * \ingroup group_nvxio_frame_source
 * \brief FrameSource interface factory that provides appropriate implementation by source URI.
 *
 * \note
 * + It supports the following image formats: PNG, JPEG, JPG, BMP, TIFF.
 * + To read images from a sequence in which, for example, `0000.jpg` corresponds to the first image, `0001.jpg` - the second and so forth,
 *   use the path like this:
 *   `path_to_folder/%04d.jpg`
 * + Use `"device:///v4l2?index=0"` path to capture images from the first camera. It supports only Video for Linux compatible cameras.
 * + Use `"device:///nvcamera"` path to capture frames from NVIDIA GStreamer camera.
 * + Use `"device:///nvmedia?config=dvp-ov10635-yuv422-ab-e2379&number=4"` path to capture frames from Omni Vision cameras via NvMedia.
 * + Image decoding, image sequence decoding, and Video4Linux-compatible camera support require either OpenCV or GStreamer.
 * + Support level of video formats depends on the set of installed GStreamer plugins.
 *
 * \param [in] context The context.
 * \param [in] uri     A reference to the path to image source.
 * \see nvx_nvxio_api
 */
std::unique_ptr<FrameSource> createDefaultFrameSource(vx_context context, const std::string& uri);

/**
 * \ingroup group_nvxio_frame_source
 * \brief Loads image from file into OpenVX Image object.
 *
 * The method is a wrapper around FrameSource to simplify single images loading.
 *
 * \param [in] context      The OpenVX context.
 * \param [in] fileName     The path to the image file.
 * \param [in] format       The desired output format.
 * \return `vx_image` object. Calling code is responsible for its releasing.
 *
 * \see nvx_nvxio_api
 */
vx_image loadImageFromFile(vx_context context, const std::string& fileName, vx_df_image format = VX_DF_IMAGE_RGB);

}

#endif // NVXIO_FRAMESOURCE_HPP
