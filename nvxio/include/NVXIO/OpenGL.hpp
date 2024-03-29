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

#ifndef NVXIO_OPENGL_HPP
#define NVXIO_OPENGL_HPP

#ifdef USE_GLES
# include <GLES3/gl31.h>
#else
# ifdef _WIN32
// glcorearb.h includes Windows.h
#  define NOMINMAX
# endif
# include <GL/glcorearb.h>
#endif

#ifdef USE_GLES
# define NVXIO_DECLARE_GL_FUNC(name, name_upcase) decltype(gl ## name) *name
#else
# define NVXIO_DECLARE_GL_FUNC(name, name_upcase) PFNGL ## name_upcase ## PROC name
#endif

namespace nvxio {

struct GLFunctions
{
    NVXIO_DECLARE_GL_FUNC(ActiveTexture, ACTIVETEXTURE);
    NVXIO_DECLARE_GL_FUNC(AttachShader, ATTACHSHADER);
    NVXIO_DECLARE_GL_FUNC(BindBuffer, BINDBUFFER);
    NVXIO_DECLARE_GL_FUNC(BindTexture, BINDTEXTURE);
    NVXIO_DECLARE_GL_FUNC(BindVertexArray, BINDVERTEXARRAY);
    NVXIO_DECLARE_GL_FUNC(BlendFunc, BLENDFUNC);
    NVXIO_DECLARE_GL_FUNC(BufferData, BUFFERDATA);
    NVXIO_DECLARE_GL_FUNC(ClearColor, CLEARCOLOR);
    NVXIO_DECLARE_GL_FUNC(Clear, CLEAR);
    NVXIO_DECLARE_GL_FUNC(CompileShader, COMPILESHADER);
    NVXIO_DECLARE_GL_FUNC(CreateProgram, CREATEPROGRAM);
    NVXIO_DECLARE_GL_FUNC(CreateShader, CREATESHADER);
    NVXIO_DECLARE_GL_FUNC(DeleteBuffers, DELETEBUFFERS);
    NVXIO_DECLARE_GL_FUNC(DeleteProgram, DELETEPROGRAM);
    NVXIO_DECLARE_GL_FUNC(DeleteShader, DELETESHADER);
    NVXIO_DECLARE_GL_FUNC(DeleteTextures, DELETETEXTURES);
    NVXIO_DECLARE_GL_FUNC(DeleteVertexArrays, DELETEVERTEXARRAYS);
    NVXIO_DECLARE_GL_FUNC(DepthFunc, DEPTHFUNC);
    NVXIO_DECLARE_GL_FUNC(Disable, DISABLE);
    NVXIO_DECLARE_GL_FUNC(DisableVertexAttribArray, DISABLEVERTEXATTRIBARRAY);
    NVXIO_DECLARE_GL_FUNC(DrawArrays, DRAWARRAYS);
    NVXIO_DECLARE_GL_FUNC(DrawElements, DRAWELEMENTS);
    NVXIO_DECLARE_GL_FUNC(Enable, ENABLE);
    NVXIO_DECLARE_GL_FUNC(EnableVertexAttribArray, ENABLEVERTEXATTRIBARRAY);
    NVXIO_DECLARE_GL_FUNC(GenBuffers, GENBUFFERS);
    NVXIO_DECLARE_GL_FUNC(GenTextures, GENTEXTURES);
    NVXIO_DECLARE_GL_FUNC(GenVertexArrays, GENVERTEXARRAYS);
    NVXIO_DECLARE_GL_FUNC(GetAttribLocation, GETATTRIBLOCATION);
    NVXIO_DECLARE_GL_FUNC(GetError, GETERROR);
    NVXIO_DECLARE_GL_FUNC(GetProgramInfoLog, GETPROGRAMINFOLOG);
    NVXIO_DECLARE_GL_FUNC(GetProgramiv, GETPROGRAMIV);
    NVXIO_DECLARE_GL_FUNC(GetShaderInfoLog, GETSHADERINFOLOG);
    NVXIO_DECLARE_GL_FUNC(GetShaderiv, GETSHADERIV);
    NVXIO_DECLARE_GL_FUNC(IsBuffer, ISBUFFER);
    NVXIO_DECLARE_GL_FUNC(IsTexture, ISTEXTURE);
    NVXIO_DECLARE_GL_FUNC(IsVertexArray, ISVERTEXARRAY);
    NVXIO_DECLARE_GL_FUNC(LinkProgram, LINKPROGRAM);
    NVXIO_DECLARE_GL_FUNC(MapBufferRange, MAPBUFFERRANGE);
    NVXIO_DECLARE_GL_FUNC(ShaderSource, SHADERSOURCE);
    NVXIO_DECLARE_GL_FUNC(TexParameterf, TEXPARAMETERF);
    NVXIO_DECLARE_GL_FUNC(TexParameteri, TEXPARAMETERI);
    NVXIO_DECLARE_GL_FUNC(TexSubImage2D, TEXSUBIMAGE2D);
    NVXIO_DECLARE_GL_FUNC(TexImage2D, TEXIMAGE2D);
    NVXIO_DECLARE_GL_FUNC(Uniform1f, UNIFORM1F);
    NVXIO_DECLARE_GL_FUNC(Uniform1i, UNIFORM1I);
    NVXIO_DECLARE_GL_FUNC(UniformMatrix4fv, UNIFORMMATRIX4FV);
    NVXIO_DECLARE_GL_FUNC(UnmapBuffer, UNMAPBUFFER);
    NVXIO_DECLARE_GL_FUNC(UseProgram, USEPROGRAM);
    NVXIO_DECLARE_GL_FUNC(ValidateProgram, VALIDATEPROGRAM);
    NVXIO_DECLARE_GL_FUNC(VertexAttribPointer, VERTEXATTRIBPOINTER);
    NVXIO_DECLARE_GL_FUNC(ReadPixels, READPIXELS);
    NVXIO_DECLARE_GL_FUNC(PixelStorei, PIXELSTOREI);
    NVXIO_DECLARE_GL_FUNC(IsShader, ISSHADER);
    NVXIO_DECLARE_GL_FUNC(IsProgram, ISPROGRAM);
    NVXIO_DECLARE_GL_FUNC(GetFloatv, GETFLOATV);
    NVXIO_DECLARE_GL_FUNC(LineWidth, LINEWIDTH);
    NVXIO_DECLARE_GL_FUNC(Uniform4f, UNIFORM4F);
    NVXIO_DECLARE_GL_FUNC(BufferSubData, BUFFERSUBDATA);
#ifndef USE_GLES
    NVXIO_DECLARE_GL_FUNC(ClearTexImage, CLEARTEXIMAGE);
#endif
    NVXIO_DECLARE_GL_FUNC(DrawArraysInstanced, DRAWARRAYSINSTANCED);
    NVXIO_DECLARE_GL_FUNC(VertexAttribDivisor, VERTEXATTRIBDIVISOR);
    NVXIO_DECLARE_GL_FUNC(GetBooleanv, GETBOOLEANV);
    NVXIO_DECLARE_GL_FUNC(DeleteFramebuffers, DELETEFRAMEBUFFERS);
    NVXIO_DECLARE_GL_FUNC(IsFramebuffer, ISFRAMEBUFFER);
    NVXIO_DECLARE_GL_FUNC(GenFramebuffers, GENFRAMEBUFFERS);
    NVXIO_DECLARE_GL_FUNC(BindFramebuffer, BINDFRAMEBUFFER);
    NVXIO_DECLARE_GL_FUNC(FramebufferTexture2D, FRAMEBUFFERTEXTURE2D);
    NVXIO_DECLARE_GL_FUNC(CheckFramebufferStatus, CHECKFRAMEBUFFERSTATUS);
    NVXIO_DECLARE_GL_FUNC(GetIntegerv, GETINTEGERV);
    NVXIO_DECLARE_GL_FUNC(Uniform2f, UNIFORM2F);
    NVXIO_DECLARE_GL_FUNC(DispatchCompute, DISPATCHCOMPUTE);
    NVXIO_DECLARE_GL_FUNC(BindBufferBase, BINDBUFFERBASE);
    NVXIO_DECLARE_GL_FUNC(BindImageTexture, BINDIMAGETEXTURE);
    NVXIO_DECLARE_GL_FUNC(MemoryBarrier, MEMORYBARRIER);
    NVXIO_DECLARE_GL_FUNC(Uniform1ui, UNIFORM1UI);
    NVXIO_DECLARE_GL_FUNC(TexStorage2D, TEXSTORAGE2D);
    NVXIO_DECLARE_GL_FUNC(GenProgramPipelines, GENPROGRAMPIPELINES);
    NVXIO_DECLARE_GL_FUNC(DeleteProgramPipelines, DELETEPROGRAMPIPELINES);
    NVXIO_DECLARE_GL_FUNC(BindProgramPipeline, BINDPROGRAMPIPELINE);
    NVXIO_DECLARE_GL_FUNC(UseProgramStages, USEPROGRAMSTAGES);
    NVXIO_DECLARE_GL_FUNC(CreateShaderProgramv, CREATESHADERPROGRAMV);
    NVXIO_DECLARE_GL_FUNC(ProgramUniform1f, PROGRAMUNIFORM1F);
    NVXIO_DECLARE_GL_FUNC(ProgramUniform2f, PROGRAMUNIFORM2F);
    NVXIO_DECLARE_GL_FUNC(ProgramUniform4f, PROGRAMUNIFORM4F);
    NVXIO_DECLARE_GL_FUNC(GetTexLevelParameteriv, GETTEXLEVELPARAMETERIV);
    NVXIO_DECLARE_GL_FUNC(Viewport, VIEWPORT);
    NVXIO_DECLARE_GL_FUNC(Hint, HINT);
};

// Must be called with an active OpenGL context.
void loadGLFunctions(GLFunctions *f);

}

#endif
