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

#ifdef USE_GSTREAMER

#include "FrameSource/GStreamer/GStreamerCommon.hpp"

namespace nvxio
{

bool updateConfiguration(GstElement * element, FrameSource::Parameters & configuration)
{
    std::unique_ptr<GstPad, GStreamerObjectDeleter> pad(gst_element_get_static_pad(element, "src"));

#if GST_VERSION_MAJOR == 0
    std::unique_ptr<GstCaps, GStreamerObjectDeleter> bufferCaps(gst_pad_get_caps(pad.get()));
#else
    std::unique_ptr<GstCaps, GStreamerObjectDeleter> bufferCaps(gst_pad_get_current_caps(pad.get()));
#endif

    if (!bufferCaps)
    {
        NVXIO_PRINT("Width, height, fps can not be queried");
        return false;
    }

    const GstStructure *structure = gst_caps_get_structure(bufferCaps.get(), 0);

    int width, height;
    if (!gst_structure_get_int(structure, "width", &width))
        NVXIO_PRINT("Cannot query video width");

    if (!gst_structure_get_int(structure, "height", &height))
        NVXIO_PRINT("Cannot query video height");

    if (configuration.frameWidth == (vx_uint32)-1)
        configuration.frameWidth = width;
    if (configuration.frameHeight == (vx_uint32)-1)
        configuration.frameHeight = height;

    gint num = 0, denom = 1;
    if (!gst_structure_get_fraction(structure, "framerate", &num, &denom))
        NVXIO_PRINT("Cannot query video fps");

    configuration.fps = static_cast<float>(num) / denom;

    return true;
}

}

#endif // USE_GSTREAMER
