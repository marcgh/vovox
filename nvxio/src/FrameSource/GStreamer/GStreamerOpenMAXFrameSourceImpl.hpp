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

#ifndef GSTREAMEROPENMAXFRAMESOURCEIMPL_HPP
#define GSTREAMEROPENMAXFRAMESOURCEIMPL_HPP

#if defined USE_GSTREAMER_OMX && defined USE_GLES // For L4T R23 and R24 only

#include "GStreamerEGLStreamSinkFrameSourceImpl.hpp"
#include "GStreamerBaseFrameSourceImpl.hpp"

namespace nvxio
{

class GStreamerOpenMAXFrameSourceImpl :
        public GStreamerEGLStreamSinkFrameSourceImpl
{
public:
    GStreamerOpenMAXFrameSourceImpl(vx_context context, const std::string & path);

protected:

    static GstAutoplugSelectResult autoPlugSelect(GstElement *bin, GstPad *pad,
                                                  GstCaps *caps, GstElementFactory *factory,
                                                  gpointer user_data);

    virtual bool InitializeGstPipeLine();

    const std::string fileName;
};

}

#endif // defined USE_GSTREAMER_OMX && defined USE_GLES

#endif // GSTREAMEROPENMAXFRAMESOURCEIMPL_HPP
