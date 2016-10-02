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

#ifdef USE_GUI

#include "OpenGLBasicRenders.hpp"

#include <algorithm>
#include <cstring>
#include <cfloat>

#ifdef _WIN32
// cuda_gl_interop.h includes GL/gl.h, which requires Windows.h to work.
#define NOMINMAX
#include <windows.h>
#endif

#include <cuda_gl_interop.h>

#include "RenderUtils.hpp"
#include "OpenGLShaders.hpp"

#ifndef NDEBUG
void __checkGlError(std::shared_ptr<nvxio::GLFunctions> gl_, const char* file, int line)
{
    GLenum err = gl_->GetError();
    if (err != GL_NO_ERROR)
    {
        const char *errStr;
        switch (err)
        {
            case GL_INVALID_ENUM: errStr = "GL_INVALID_ENUM"; break;
            case GL_INVALID_VALUE: errStr = "GL_INVALID_VALUE"; break;
            case GL_INVALID_OPERATION: errStr = "GL_INVALID_OPERATION"; break;
            case GL_OUT_OF_MEMORY: errStr = "GL_OUT_OF_MEMORY"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: errStr = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
            default: errStr = "UKNOWN OPENGL ERROR CODE"; break;
        };
        NVXIO_PRINT("[%s:%d] OpenGL Error: 0x%x aka %s", file, line, err, errStr);
    }
}
#endif

namespace
{
    struct Vector2f
    {
        float x;
        float y;

        Vector2f()
        {
            x = y = 0.0f;
        }

        Vector2f(float _x, float _y)
        {
            x = _x;
            y = _y;
        }
    };

    struct Vertex
    {
        Vector2f pos;
        Vector2f tex;

        Vertex(Vector2f p, Vector2f t)
        {
            pos = p;
            tex = t;
        }
    };

    class LinesRenderingRules
    {
    public:
        LinesRenderingRules(std::shared_ptr<nvxio::GLFunctions> gl, GLfloat lineWidth) :
            gl_(gl), isLineSmooth_(GL_FALSE), lineSmoothHint_(GL_DONT_CARE), oldLineWidth_(0.0f)
        {
            // save state

#ifndef USE_GLES
            gl_->GetBooleanv(GL_LINE_SMOOTH, &isLineSmooth_);
            NVXIO_CHECK_GL_ERROR();
            gl_->GetIntegerv(GL_LINE_SMOOTH_HINT, &lineSmoothHint_);
            NVXIO_CHECK_GL_ERROR();
#endif // USE_GLES

            gl_->GetFloatv(GL_LINE_WIDTH, &oldLineWidth_);
            NVXIO_CHECK_GL_ERROR();

            // check

            checkWidth(lineWidth);

            // set new state

#ifndef USE_GLES
            gl_->Enable(GL_LINE_SMOOTH);
            NVXIO_CHECK_GL_ERROR();
            gl_->Hint(GL_LINE_SMOOTH_HINT, GL_NICEST);
            NVXIO_CHECK_GL_ERROR();
#endif // USE_GLES

            gl_->LineWidth(lineWidth);
            NVXIO_CHECK_GL_ERROR();
        }

        ~LinesRenderingRules()
        {
#ifndef USE_GLES
            if (!isLineSmooth_)
            {
                gl_->Disable(GL_LINE_SMOOTH);
                NVXIO_CHECK_GL_ERROR();
            }
            gl_->Hint(GL_LINE_SMOOTH_HINT, lineSmoothHint_);
            NVXIO_CHECK_GL_ERROR();
#endif // USE_GLES

            gl_->LineWidth(oldLineWidth_);
            NVXIO_CHECK_GL_ERROR();
        }

    private:

        void checkWidth(GLfloat lineWidth) const
        {
            // get minumum abd maximum supported line widths
            GLfloat widths[2];
            gl_->GetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, widths);
            NVXIO_CHECK_GL_ERROR();

            if ((lineWidth < widths[0]) ||
                    (lineWidth > widths[1]))
            {
                NVXIO_ASSERT(lineWidth > 0);

                NVXIO_PRINT("The specified line width '%f' is clipped to fit into the [%f, %f] interval.\n"
                            "It's performed automatically by OpenGL.", lineWidth, widths[0], widths[1]);
            }
        }

        std::shared_ptr<nvxio::GLFunctions> gl_;

        GLboolean isLineSmooth_;
        GLint lineSmoothHint_;
        GLfloat oldLineWidth_;
    };

    static bool attachShader(std::shared_ptr<nvxio::GLFunctions> gl_, GLuint shaderProgram,
                             const char* shaderText, GLenum shaderType, const char * const options)
    {
        GLuint shaderObj = gl_->CreateShader(shaderType);
        NVXIO_CHECK_GL_ERROR();

        const char * shaderTypeStr =
                shaderType == GL_VERTEX_SHADER ? "VERTEX" :
                shaderType == GL_FRAGMENT_SHADER ? "FRAGMENT" : "COMPUTE";

        if (gl_->IsShader(shaderObj) == GL_FALSE)
        {
            NVXIO_PRINT("Error while creating %s shader", shaderTypeStr);
            return false;
        }

        const GLchar* sources[] =
        {
        #ifdef USE_GLES
            "#version 310 es\n",
            options ? options : "\n",
            "precision mediump float;\n",
        #else
            "#version 430\n",
            options ? options : "\n",
        #endif
            shaderText,
        };

        gl_->ShaderSource(shaderObj, static_cast<GLsizei>(nvxio::dimOf(sources)), sources, NULL);
        NVXIO_CHECK_GL_ERROR();

        gl_->CompileShader(shaderObj);
        NVXIO_CHECK_GL_ERROR();

        GLint status = GL_FALSE;
        GLchar infoLog[1024] = { 0 };

        gl_->GetShaderiv(shaderObj, GL_COMPILE_STATUS, &status);
        NVXIO_CHECK_GL_ERROR();
        if (status == GL_FALSE)
        {
            gl_->GetShaderInfoLog(shaderObj, sizeof(infoLog), NULL, infoLog);
            NVXIO_CHECK_GL_ERROR();
            NVXIO_PRINT("Error compiling %s shader type: %s",
                    shaderTypeStr, infoLog);

            gl_->DeleteShader(shaderObj);
            NVXIO_CHECK_GL_ERROR();

            return false;
        }

        gl_->AttachShader(shaderProgram, shaderObj);
        NVXIO_CHECK_GL_ERROR();

        gl_->DeleteShader(shaderObj);
        NVXIO_CHECK_GL_ERROR();

        return true;
    }

    static bool checkProgramLog(std::shared_ptr<nvxio::GLFunctions> gl_, GLuint shaderProgram)
    {
        GLint status = GL_FALSE;
        GLchar infoLog[1024] = { 0 };

        gl_->GetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
        if (status == GL_FALSE)
        {
            gl_->GetProgramInfoLog(shaderProgram, sizeof(infoLog), NULL, infoLog);
            NVXIO_CHECK_GL_ERROR();
            NVXIO_PRINT("Error linking shader program: %s", infoLog);
            return false;
        }

        gl_->ValidateProgram(shaderProgram);
        NVXIO_CHECK_GL_ERROR();

        gl_->GetProgramiv(shaderProgram, GL_VALIDATE_STATUS, &status);
        if (status == GL_FALSE)
        {
            gl_->GetProgramInfoLog(shaderProgram, sizeof(infoLog), NULL, infoLog);
            NVXIO_CHECK_GL_ERROR();
            NVXIO_PRINT("Invalid shader program: %s", infoLog);
            return false;
        }

        return true;
    }

    static bool compileProgram(std::shared_ptr<nvxio::GLFunctions> gl_, GLuint shaderProgram,
                               const char * const vertexShader,
                               const char * const fragmentShader,
                               const char * const computeShader = NULL,
                               const char * const options = NULL)
    {
        if (!gl_->IsProgram(shaderProgram))
            return false;

        if (vertexShader != NULL)
        {
            if (!attachShader(gl_, shaderProgram, vertexShader, GL_VERTEX_SHADER, options))
                return false;
        }

        if (fragmentShader != NULL)
        {
            if (!attachShader(gl_, shaderProgram, fragmentShader, GL_FRAGMENT_SHADER, options))
                return false;
        }

        if (computeShader != NULL)
        {
            if (!attachShader(gl_, shaderProgram, computeShader, GL_COMPUTE_SHADER, options))
                return false;
        }

        gl_->LinkProgram(shaderProgram);
        NVXIO_CHECK_GL_ERROR();

        return checkProgramLog(gl_, shaderProgram);
    }

    static GLuint createSeparableProgram(std::shared_ptr<nvxio::GLFunctions> gl_,
                                         GLenum shaderType, const char * const shaderText,
                                         const char * const options = NULL)
    {
        const GLchar * sources[] =
        {
        #ifdef USE_GLES
            "#version 310 es\n",
            options ? options : "\n",
            "precision mediump float;\n",
        #else
            "#version 430\n",
            options ? options : "\n",
        #endif
            shaderText,
        };

        GLuint shaderProgram = gl_->CreateShaderProgramv(shaderType, static_cast<GLsizei>(nvxio::dimOf(sources)),
                                                         sources);

        if (!checkProgramLog(gl_, shaderProgram))
        {
            gl_->DeleteProgram(shaderProgram);
            NVXIO_CHECK_GL_ERROR();

            shaderProgram = 0;
        }

        return shaderProgram;
    }
}

//============================================================
// ImageRender
//============================================================

nvxio::ImageRender::ImageRender() :
    gl_(NULL), wndWidth_(0), wndHeight_(0),
    vao_(0), vbo_(0), index_(-1), scaleUniformX_(1.0f), scaleUniformY_(1.0f)
{
    std::memset(tex_, 0, sizeof(tex_[0]) * nvxio::dimOf(tex_));
    std::memset(pipeline_, 0, sizeof(pipeline_[0]) * nvxio::dimOf(pipeline_));
    std::memset(res_, 0, sizeof(res_[0]) * nvxio::dimOf(res_));
    std::memset(fragmentProgram_, 0, sizeof(fragmentProgram_[0]) * nvxio::dimOf(fragmentProgram_));
}

bool nvxio::ImageRender::init(std::shared_ptr<GLFunctions> _gl, vx_uint32 wndWidth, vx_uint32 wndHeight)
{
    gl_ = _gl;

    wndWidth_ = wndWidth;
    wndHeight_ = wndHeight;

    // Generate OpenGL objects

    const char * const defines[] =
    {
        "#extension GL_EXT_shader_io_blocks : enable\n"
        "#define IMAGE_U8\n",

        "#extension GL_EXT_shader_io_blocks : enable\n"
        "#define IMAGE_RGB\n",

        "#extension GL_EXT_shader_io_blocks : enable\n"
        "#define IMAGE_RGBX\n"
    };

    gl_->GenTextures(3, tex_);
    NVXIO_CHECK_GL_ERROR();

    gl_->GenProgramPipelines(3, pipeline_);
    NVXIO_CHECK_GL_ERROR();

    GLuint vertexProgram = ::createSeparableProgram(gl_, GL_VERTEX_SHADER,
                                                    image_render_shader_vs_code,
                                                    "#extension GL_EXT_shader_io_blocks : enable\n");

    for (size_t i = 0; i < nvxio::dimOf(defines); ++i)
    {
        fragmentProgram_[i] = ::createSeparableProgram(gl_, GL_FRAGMENT_SHADER,
                                                       image_render_shader_fs_code,
                                                       defines[i]);

        gl_->UseProgramStages(pipeline_[i], GL_VERTEX_SHADER_BIT, vertexProgram);
        NVXIO_CHECK_GL_ERROR();
        gl_->UseProgramStages(pipeline_[i], GL_FRAGMENT_SHADER_BIT, fragmentProgram_[i]);
        NVXIO_CHECK_GL_ERROR();
    }

    gl_->DeleteProgram(vertexProgram);
    NVXIO_CHECK_GL_ERROR();

    // Vertex buffer

    gl_->GenVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();
    gl_->GenBuffers(1, &vbo_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();

    const Vertex vertices[] =
    {
        Vertex(Vector2f(-1.0f, -1.0f), Vector2f(0.0f, 1.0f)),
        Vertex(Vector2f(-1.0f,  1.0f), Vector2f(0.0f, 0.0f)),
        Vertex(Vector2f( 1.0f, -1.0f), Vector2f(1.0f, 1.0f)),
        Vertex(Vector2f( 1.0f,  1.0f), Vector2f(1.0f, 0.0f)),
    };

    gl_->BindBuffer(GL_ARRAY_BUFFER, vbo_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    NVXIO_CHECK_GL_ERROR();

    gl_->VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->EnableVertexAttribArray(0);
    NVXIO_CHECK_GL_ERROR();
    gl_->VertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const GLvoid*)sizeof(Vector2f));
    NVXIO_CHECK_GL_ERROR();
    gl_->EnableVertexAttribArray(1);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();

    return true;
}

void nvxio::ImageRender::release()
{
    if (!gl_)
        return;

    gl_->DeleteTextures(3, tex_);
    NVXIO_CHECK_GL_ERROR();

    gl_->DeleteProgramPipelines(3, pipeline_);
    NVXIO_CHECK_GL_ERROR();

    for (size_t i = 0; i < nvxio::dimOf(res_); ++i)
    {
        if (res_[i])
        {
            cudaGraphicsUnregisterResource(res_[i]);
            res_[i] = NULL;
        }

        gl_->DeleteProgram(fragmentProgram_[i]);
        NVXIO_CHECK_GL_ERROR();
        fragmentProgram_[i] = 0;
    }

    gl_->DeleteBuffers(1, &vbo_);
    NVXIO_CHECK_GL_ERROR();
    vbo_ = 0;

    gl_->DeleteVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();
    vao_ = 0;
}

void nvxio::ImageRender::render(vx_image image, vx_uint32 imageWidth, vx_uint32 imageHeight)
{
    updateTexture(image, imageWidth, imageHeight);
    renderTexture();
}

void nvxio::ImageRender::updateTexture(vx_image image, vx_uint32 imageWidth, vx_uint32 imageHeight)
{
    cudaStream_t stream = NULL;

    vx_df_image format = 0;
    NVXIO_SAFE_CALL( vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
    NVXIO_ASSERT( format == VX_DF_IMAGE_U8 || format == VX_DF_IMAGE_RGB || format == VX_DF_IMAGE_RGBX );

    GLuint channels = format == VX_DF_IMAGE_U8 ? 1 :
                      format == VX_DF_IMAGE_RGB ? 3 : 4;
    index_ = format == VX_DF_IMAGE_U8 ? 0 :
             format == VX_DF_IMAGE_RGB ? 1 : 2;

    vx_rectangle_t rect = {};
    NVXIO_SAFE_CALL( vxGetValidRegionImage(image, &rect) );

    vx_imagepatch_addressing_t addr;
    void * ptr = NULL;
    NVXIO_SAFE_CALL( vxAccessImagePatch(image, &rect, 0, &addr, &ptr, NVX_READ_ONLY_CUDA) );

    // get actual texture size

    GLuint actualTexWidth_ = 0u, actualTexHeight_ = 0u;

    gl_->BindTexture(GL_TEXTURE_2D, tex_[index_]);
    NVXIO_CHECK_GL_ERROR();
    gl_->GetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, (GLint *)&actualTexWidth_);
    NVXIO_CHECK_GL_ERROR();
    gl_->GetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, (GLint *)&actualTexHeight_);
    NVXIO_CHECK_GL_ERROR();

    if (format == VX_DF_IMAGE_RGB)
        actualTexWidth_ /= 3;

    if ((addr.dim_x != actualTexWidth_) || (addr.dim_y != actualTexHeight_))
    {
        const GLenum internalFormats[] = { GL_R8, GL_R8, GL_RGBA8 };

        scaleUniformX_ = static_cast<GLfloat>(imageWidth) / addr.dim_x;
        scaleUniformY_ = static_cast<GLfloat>(imageHeight) / addr.dim_y;

        // Delete old stuff

        if (res_[index_])
        {
            cudaGraphicsUnregisterResource(res_[index_]);
            res_[index_] = NULL;
        }

        if (tex_[index_])
        {
            gl_->DeleteTextures(1, tex_ + index_);
            NVXIO_CHECK_GL_ERROR();
        }

        // Create new one

        gl_->GenTextures(1, tex_ + index_);
        NVXIO_CHECK_GL_ERROR();

        gl_->ActiveTexture(GL_TEXTURE0);
        NVXIO_CHECK_GL_ERROR();
        gl_->BindTexture(GL_TEXTURE_2D, tex_[index_]);
        NVXIO_CHECK_GL_ERROR();
        gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        NVXIO_CHECK_GL_ERROR();
        gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        NVXIO_CHECK_GL_ERROR();
        gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        NVXIO_CHECK_GL_ERROR();
        gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        NVXIO_CHECK_GL_ERROR();
        gl_->TexStorage2D(GL_TEXTURE_2D, 1, internalFormats[index_],
                          (index_ == 1 ? 3 : 1) * addr.dim_x, addr.dim_y);
        NVXIO_CHECK_GL_ERROR();
        gl_->BindTexture(GL_TEXTURE_2D, 0);
        NVXIO_CHECK_GL_ERROR();

        // CUDA Graphics Resource

        cudaError_t err = cudaGraphicsGLRegisterImage(res_ + index_, tex_[index_], GL_TEXTURE_2D,
                                                      cudaGraphicsRegisterFlagsSurfaceLoadStore);
        if (err != cudaSuccess)
        {
            NVXIO_PRINT("ImageRender error: %s", cudaGetErrorString(err));
            return;
        }

        // Update view port

        vx_float64 scale = std::min(scaleUniformX_, scaleUniformY_);

        GLint viewportWidth = static_cast<GLint>(addr.dim_x * scale);
        GLint viewportHeight = static_cast<GLint>(addr.dim_y * scale);

        NVXIO_ASSERT(wndWidth_ >= (GLuint)viewportWidth);
        NVXIO_ASSERT(wndHeight_ >= (GLuint)viewportHeight);

        GLint xBorder = static_cast<GLint>(wndWidth_ - viewportWidth) >> 1;
        GLint yBorder = static_cast<GLint>(wndHeight_ - viewportHeight) >> 1;

        gl_->Viewport(xBorder, yBorder,
                      viewportWidth, viewportHeight);
    }

    // Copy CUDA image to mapped resource

    NVXIO_CUDA_SAFE_CALL( cudaGraphicsMapResources(1, res_ + index_, stream) );

    cudaArray_t cudaArr = NULL;
    NVXIO_CUDA_SAFE_CALL( cudaGraphicsSubResourceGetMappedArray(&cudaArr, res_[index_], 0, 0) );

    NVXIO_CUDA_SAFE_CALL( cudaMemcpy2DToArrayAsync(cudaArr, 0, 0, ptr, addr.stride_y,
                                                   addr.dim_x * sizeof(GLubyte) * channels, addr.dim_y,
                                                   cudaMemcpyDeviceToDevice, stream) );

    NVXIO_CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, res_ + index_, stream) );

    NVXIO_CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );

    NVXIO_SAFE_CALL( vxCommitImagePatch(image, NULL, 0, &addr, ptr) );
}

void nvxio::ImageRender::renderTexture()
{
    gl_->BindProgramPipeline(pipeline_[index_]);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();

    gl_->ActiveTexture(GL_TEXTURE0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindTexture(GL_TEXTURE_2D, tex_[index_]);
    NVXIO_CHECK_GL_ERROR();

    GLfloat scale = std::min(scaleUniformX_, scaleUniformY_);
    if (index_ == 1)
    {
        gl_->ProgramUniform2f(fragmentProgram_[index_], 0,
                              1 / scale, 1 / scale);
        NVXIO_CHECK_GL_ERROR();

        GLint viewport[4];
        gl_->GetIntegerv(GL_VIEWPORT, viewport);
        NVXIO_CHECK_GL_ERROR();

        gl_->ProgramUniform2f(fragmentProgram_[index_], 1,
                              static_cast<GLfloat>(viewport[0]),
                              static_cast<GLfloat>(viewport[1]));
    }
    else
    {

        gl_->ProgramUniform2f(fragmentProgram_[index_], 0,
                              scaleUniformX_ / scale,
                              scaleUniformY_ / scale);
    }
    NVXIO_CHECK_GL_ERROR();

    gl_->DrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    NVXIO_CHECK_GL_ERROR();

    // reset state
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindTexture(GL_TEXTURE_2D, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindProgramPipeline(0);
    NVXIO_CHECK_GL_ERROR();
}

//============================================================
// RectangleRender
//============================================================

nvxio::RectangleRender::RectangleRender() :
    gl_(NULL), vbo_(0), vao_(0), program_(0)
{
}

bool nvxio::RectangleRender::init(std::shared_ptr<GLFunctions> _gl)
{
    gl_ = _gl;

    // Vertex arrays

    gl_->GenVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();

    // Vertex buffer

    gl_->GenBuffers(1, &vbo_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ARRAY_BUFFER, vbo_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BufferData(GL_ARRAY_BUFFER, sizeof(Vector2f) * 4, NULL, GL_DYNAMIC_DRAW);
    NVXIO_CHECK_GL_ERROR();
    gl_->VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), NULL);
    NVXIO_CHECK_GL_ERROR();
    gl_->EnableVertexAttribArray(0);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();

    // Shaders

    program_ = gl_->CreateProgram();
    NVXIO_CHECK_GL_ERROR();

    if (gl_->IsProgram(program_) == GL_FALSE)
    {
        NVXIO_PRINT("RectangleRender: error creating shader program");
        return false;
    }

    if (!compileProgram(gl_, program_,
                        rectangle_render_shader_vs_code,
                        rectangle_render_shader_fs_code))
        return false;

    return true;
}

void nvxio::RectangleRender::release()
{
    if (!gl_)
        return;

    gl_->DeleteBuffers(1, &vbo_);
    NVXIO_CHECK_GL_ERROR();
    vbo_ = 0;

    gl_->DeleteVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();
    vao_ = 0;

    gl_->DeleteProgram(program_);
    NVXIO_CHECK_GL_ERROR();
    program_ = 0;
}

void nvxio::RectangleRender::render(const vx_rectangle_t& location, const nvxio::Render::DetectedObjectStyle& style,
                                                            vx_uint32 width, vx_uint32 height, vx_float32 scale)
{
    updateArray(location, width, height, scale);
    renderArray(style);
}

void nvxio::RectangleRender::updateArray(const vx_rectangle_t& location, vx_uint32 width, vx_uint32 height, vx_float32 scale)
{
    GLfloat widthScale = 2.0f * scale / (width - 1);
    GLfloat heightScale = 2.0f * scale / (height - 1);

    // normalized coordinates
    GLfloat start_x = location.start_x * widthScale - 1;
    GLfloat end_x = location.end_x * widthScale - 1;
    GLfloat start_y = 1 - location.start_y * heightScale;
    GLfloat end_y = 1 - location.end_y * heightScale;

    // location to array
    Vector2f vectors[4] =
    {
        Vector2f(start_x, start_y),
        Vector2f(end_x, start_y),
        Vector2f(start_x, end_y),
        Vector2f(end_x,  end_y),
    };

    gl_->BindBuffer(GL_ARRAY_BUFFER, vbo_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vectors), vectors);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
}

void nvxio::RectangleRender::renderArray(const nvxio::Render::DetectedObjectStyle& style)
{
    gl_->UseProgram(program_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();

    gl_->Uniform4f(0,
                style.color[0] / 255.0f,
                style.color[1] / 255.0f,
                style.color[2] / 255.0f,
                style.color[3] / 710.0f);
    NVXIO_CHECK_GL_ERROR();

    gl_->DrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    NVXIO_CHECK_GL_ERROR();

    // reset state
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();
    gl_->UseProgram(0);
    NVXIO_CHECK_GL_ERROR();
}

//============================================================
// FeaturesRender
//============================================================

nvxio::FeaturesRender::FeaturesRender() :
    gl_(NULL), bufCapacity_(1000), vbo_(0), vboStyles_(0), vao_(0), res_(NULL),
    pipeline_(0),
    vertexShaderPoints_(0), vertexShaderKeyPoints_(0),
    vertexShaderPointsPerFeature_(0), vertexShaderKeyPointsPerFeature_(0),
    fragmentShader_(0), fragmentShaderPerFeature_(0),
    currentFeatureType_(0), perFeatureStyle_(false)
{
}

bool nvxio::FeaturesRender::init(std::shared_ptr<GLFunctions> _gl)
{
    gl_ = _gl;

    // Vertex arrays

    gl_->GenVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();

    // Vertex buffer

    gl_->GenBuffers(1, &vbo_);
    NVXIO_CHECK_GL_ERROR();

    gl_->GenBuffers(1, &vboStyles_);
    NVXIO_CHECK_GL_ERROR();

    // Create VBO for styles

    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ARRAY_BUFFER, vboStyles_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BufferData(GL_ARRAY_BUFFER, bufCapacity_ * sizeof(Render::FeatureStyle), NULL, GL_DYNAMIC_DRAW);
    NVXIO_CHECK_GL_ERROR();
    gl_->VertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, static_cast<GLsizei>(sizeof(Render::FeatureStyle)),
                             (const GLvoid *)offsetof(Render::FeatureStyle, color));
    gl_->VertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, static_cast<GLsizei>(sizeof(Render::FeatureStyle)),
                             (const GLvoid *)offsetof(Render::FeatureStyle, radius));
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();

    // CUDA Graphics Resource

    NVXIO_CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&resStyles_, vboStyles_, cudaGraphicsMapFlagsWriteDiscard) );

    // Shaders

    gl_->GenProgramPipelines(1, &pipeline_);
    NVXIO_CHECK_GL_ERROR();

    vertexShaderPoints_ = ::createSeparableProgram(gl_,
                                                   GL_VERTEX_SHADER,
                                                   feature_render_shader_vs_code,
                                                   "#extension GL_EXT_shader_io_blocks : enable\n"
                                                   "#define WITH_POINTS\n");

    vertexShaderPointsPerFeature_ = ::createSeparableProgram(gl_,
                                                             GL_VERTEX_SHADER,
                                                             feature_render_shader_vs_code,
                                                             "#extension GL_EXT_shader_io_blocks : enable\n"
                                                             "#define WITH_POINTS\n"
                                                             "#define PER_FEATURE_STYLE\n");

    vertexShaderKeyPoints_ = ::createSeparableProgram(gl_,
                                                      GL_VERTEX_SHADER,
                                                      feature_render_shader_vs_code,
                                                      "#extension GL_EXT_shader_io_blocks : enable\n"
                                                      "#define WITH_KEYPOINTS\n");

    vertexShaderKeyPointsPerFeature_ = ::createSeparableProgram(gl_,
                                                                GL_VERTEX_SHADER,
                                                                feature_render_shader_vs_code,
                                                                "#extension GL_EXT_shader_io_blocks : enable\n"
                                                                "#define WITH_KEYPOINTS\n"
                                                                "#define PER_FEATURE_STYLE\n");

    fragmentShader_ = ::createSeparableProgram(gl_,
                                               GL_FRAGMENT_SHADER,
                                               feature_render_shader_fs_code);

    fragmentShaderPerFeature_ = ::createSeparableProgram(gl_,
                                                         GL_FRAGMENT_SHADER,
                                                         feature_render_shader_fs_code,
                                                         "#define PER_FEATURE_STYLE\n");

    return true;
}

void nvxio::FeaturesRender::release()
{
    if (!gl_)
        return;

    gl_->DeleteBuffers(1, &vbo_);
    NVXIO_CHECK_GL_ERROR();
    vbo_ = 0;

    gl_->DeleteBuffers(1, &vboStyles_);
    NVXIO_CHECK_GL_ERROR();
    vboStyles_ = 0;

    gl_->DeleteVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();
    vao_ = 0;

    // delete CUDA resources

    if (res_)
    {
        cudaGraphicsUnregisterResource(res_);
        res_ = 0;
    }

    if (resStyles_)
    {
        cudaGraphicsUnregisterResource(resStyles_);
        resStyles_ = 0;
    }

    // delete program pipeline

    gl_->DeleteProgramPipelines(1, &pipeline_);
    NVXIO_CHECK_GL_ERROR();

    // delete both fragment and vertex shaders

    GLuint * shaders[] = { &vertexShaderKeyPointsPerFeature_,
                           &vertexShaderPoints_,
                           &vertexShaderKeyPoints_,
                           &vertexShaderPointsPerFeature_,
                           &vertexShaderKeyPointsPerFeature_,
                           &fragmentShader_,
                           &fragmentShaderPerFeature_
                         };

    for (vx_size i = 0; i < nvxio::dimOf(shaders); ++i)
    {
        gl_->DeleteProgram(*(shaders[i]));
        NVXIO_CHECK_GL_ERROR();
        *(shaders[i]) = 0;
    }
}

void nvxio::FeaturesRender::render(vx_array location, const nvxio::Render::FeatureStyle& style, vx_uint32 width, vx_uint32 height, vx_float32 scale)
{
    vx_size num_items = 0;
    NVXIO_SAFE_CALL( vxQueryArray(location, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_items, sizeof(num_items)) );

    // change current rendering mode
    perFeatureStyle_ = false;

    for (vx_size start_x = 0; start_x < num_items; start_x += bufCapacity_)
    {
        vx_size end_x = std::min(start_x + bufCapacity_, num_items);
        updateArray(start_x, end_x, location, nullptr);
        renderArray(end_x - start_x, width, height, scale, style);
    }
}

void nvxio::FeaturesRender::render(vx_array location, vx_array styles, vx_uint32 width, vx_uint32 height, vx_float32 scale)
{
    vx_size num_items = 0, num_style_items = 0;
    NVXIO_SAFE_CALL( vxQueryArray(location, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_items, sizeof(num_items)) );
    NVXIO_SAFE_CALL( vxQueryArray(styles, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_style_items, sizeof(num_style_items)) );

    NVXIO_ASSERT(num_items == num_style_items);

    // change current rendering mode
    perFeatureStyle_ = true;

    for (vx_size start_x = 0; start_x < num_items; start_x += bufCapacity_)
    {
        vx_size end_x = std::min(start_x + bufCapacity_, num_items);
        updateArray(start_x, end_x, location, styles);
        renderArray(end_x - start_x, width, height, scale, Render::FeatureStyle());
    }
}


void nvxio::FeaturesRender::updateArray(vx_size start_x, vx_size end_x, vx_array location, vx_array styles)
{
    cudaStream_t stream = NULL;

    vx_enum item_type = 0;
    NVXIO_SAFE_CALL( vxQueryArray(location, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &item_type, sizeof(item_type)) );
    NVXIO_ASSERT( (item_type == VX_TYPE_KEYPOINT) || (item_type == NVX_TYPE_POINT2F) || (item_type == NVX_TYPE_KEYPOINTF) );

    size_t elemSize = item_type == VX_TYPE_KEYPOINT ? sizeof(vx_keypoint_t) :
                      item_type == NVX_TYPE_POINT2F ? sizeof(nvx_point2f_t) :
                                                      sizeof(nvx_keypointf_t);

    vx_size stride = 0;
    void* src_ptr = NULL;
    NVXIO_SAFE_CALL( vxAccessArrayRange(location, start_x, end_x, &stride, &src_ptr, NVX_READ_ONLY_CUDA) );

    // check if feature type has changed

    if (currentFeatureType_ != item_type)
    {
        gl_->BindVertexArray(vao_);
        NVXIO_CHECK_GL_ERROR();
        gl_->BindBuffer(GL_ARRAY_BUFFER, vbo_);
        NVXIO_CHECK_GL_ERROR();
        gl_->BufferData(GL_ARRAY_BUFFER, bufCapacity_ * elemSize, NULL, GL_DYNAMIC_DRAW);
        NVXIO_CHECK_GL_ERROR();

        // Specify points

        if (item_type == NVX_TYPE_POINT2F)
        {
            gl_->VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, static_cast<GLsizei>(elemSize),
                                     (const GLvoid *)offsetof(nvx_point2f_t, x));
            NVXIO_CHECK_GL_ERROR();
        }
        else if (item_type == VX_TYPE_KEYPOINT)
        {
            gl_->VertexAttribPointer(0, 2, GL_INT, GL_FALSE, static_cast<GLsizei>(elemSize),
                                     (const GLvoid *)offsetof(vx_keypoint_t, x));
            NVXIO_CHECK_GL_ERROR();
        }
        else // NVX_TYPE_KEYPOINTF
        {
            gl_->VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, static_cast<GLsizei>(elemSize),
                                     (const GLvoid *)offsetof(nvx_keypointf_t, x));
            NVXIO_CHECK_GL_ERROR();
        }

        if (!perFeatureStyle_)
        {
            gl_->VertexAttribDivisor(0, 1);
            NVXIO_CHECK_GL_ERROR();
        }

        gl_->EnableVertexAttribArray(0);
        NVXIO_CHECK_GL_ERROR();

        // Specify tracking data

        if (item_type != NVX_TYPE_POINT2F)
        {
            if (item_type == VX_TYPE_KEYPOINT)
            {
                gl_->VertexAttribPointer(1, 1, GL_INT, GL_FALSE, static_cast<GLsizei>(elemSize),
                                         (const GLvoid *)offsetof(vx_keypoint_t, tracking_status));
                NVXIO_CHECK_GL_ERROR();
            }
            else // NVX_TYPE_KEYPOINTF
            {
                gl_->VertexAttribPointer(1, 1, GL_INT, GL_FALSE, static_cast<GLsizei>(elemSize),
                                         (const GLvoid *)offsetof(nvx_keypointf_t, tracking_status));
                NVXIO_CHECK_GL_ERROR();
            }

            if (!perFeatureStyle_)
            {
                gl_->VertexAttribDivisor(1, 1);
                NVXIO_CHECK_GL_ERROR();
            }

            gl_->EnableVertexAttribArray(1);
            NVXIO_CHECK_GL_ERROR();
        }
        else
        {
            gl_->DisableVertexAttribArray(1);
            NVXIO_CHECK_GL_ERROR();
        }

        gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
        NVXIO_CHECK_GL_ERROR();
        gl_->BindVertexArray(0);
        NVXIO_CHECK_GL_ERROR();

        // CUDA Graphics Resource

        if (res_ != NULL)
        {
            NVXIO_CUDA_SAFE_CALL( cudaGraphicsUnregisterResource(res_) );
            res_ = NULL;
        }

        NVXIO_CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&res_, vbo_, cudaGraphicsMapFlagsWriteDiscard) );

        currentFeatureType_ = item_type;
    }

    NVXIO_CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &res_, stream) );

    void* dst_ptr = NULL;
    size_t size = 0;
    NVXIO_CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer(&dst_ptr, &size, res_) );

    // copy elements

    NVXIO_CUDA_SAFE_CALL( cudaMemcpyAsync(dst_ptr, src_ptr, (end_x - start_x) * elemSize,
                                          cudaMemcpyDeviceToDevice, stream) );

    NVXIO_CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &res_, stream) );

    NVXIO_CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );

    NVXIO_SAFE_CALL( vxCommitArrayRange(location, start_x, end_x, src_ptr) );

    // update styles buffer if any

    if (perFeatureStyle_)
    {
        NVXIO_ASSERT(styles);

        vx_size styleElemSize = 0;
        NVXIO_SAFE_CALL( vxQueryArray(styles, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &styleElemSize, sizeof(styleElemSize)) );
        NVXIO_ASSERT( styleElemSize == sizeof(Render::FeatureStyle) );

        vx_size styles_stride = 0;
        void * styles_ptr = NULL;
        NVXIO_SAFE_CALL( vxAccessArrayRange(styles, start_x, end_x, &styles_stride, &styles_ptr, NVX_READ_ONLY_CUDA) );

        NVXIO_CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &resStyles_, stream) );

        // copy elements

        void * dst_styles_ptr = NULL;
        size_t styles_size = 0;
        NVXIO_CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer(&dst_styles_ptr, &styles_size, resStyles_) );

        NVXIO_CUDA_SAFE_CALL( cudaMemcpyAsync(dst_styles_ptr, styles_ptr, (end_x - start_x) * sizeof(nvxio::Render::FeatureStyle),
                                              cudaMemcpyDeviceToDevice, stream) );

        NVXIO_CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &resStyles_, stream) );

        NVXIO_CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );

        NVXIO_SAFE_CALL( vxCommitArrayRange(styles, start_x, end_x, styles_ptr) );

        // update VAO

        gl_->BindVertexArray(vao_);
        NVXIO_CHECK_GL_ERROR();
        gl_->EnableVertexAttribArray(2);
        NVXIO_CHECK_GL_ERROR();
        gl_->EnableVertexAttribArray(3);
        NVXIO_CHECK_GL_ERROR();
        gl_->BindVertexArray(0);
        NVXIO_CHECK_GL_ERROR();
    }
    else
    {
        // update VAO

        gl_->BindVertexArray(vao_);
        NVXIO_CHECK_GL_ERROR();
        gl_->DisableVertexAttribArray(2);
        NVXIO_CHECK_GL_ERROR();
        gl_->DisableVertexAttribArray(3);
        NVXIO_CHECK_GL_ERROR();
        gl_->BindVertexArray(0);
        NVXIO_CHECK_GL_ERROR();
    }
}

void nvxio::FeaturesRender::renderArray(vx_size num_items, vx_uint32 width, vx_uint32 height, vx_float32 scale,
                                        const nvxio::Render::FeatureStyle& style)
{
#ifdef GL_ALIASED_POINT_SIZE_RANGE
    GLfloat pointSizes[2];

    gl_->GetFloatv(GL_ALIASED_POINT_SIZE_RANGE, pointSizes);
    NVXIO_CHECK_GL_ERROR();

    if ((pointSizes[0] > style.radius) ||
            (style.radius > pointSizes[1]))
    {
        NVXIO_PRINT("The specified feature size '%f' is clipped to fit into the [%f, %f] interval.\n"
                    "It's performed automatically by OpenGL.", style.radius, pointSizes[0], pointSizes[1]);
    }
#endif

#ifndef USE_GLES
    GLboolean programPointSize = GL_FALSE;
    gl_->GetBooleanv(GL_VERTEX_PROGRAM_POINT_SIZE, &programPointSize);
    NVXIO_CHECK_GL_ERROR();

    if (programPointSize == GL_FALSE)
    {
        gl_->Enable(GL_VERTEX_PROGRAM_POINT_SIZE);
        NVXIO_CHECK_GL_ERROR();
    }
#endif

    // compose program pipeline
    GLuint vertexShader = 0, fragmentShader = 0;

    if (currentFeatureType_ == NVX_TYPE_POINT2F)
        vertexShader = perFeatureStyle_ ? vertexShaderPointsPerFeature_ : vertexShaderPoints_;
    else
        vertexShader = perFeatureStyle_ ? vertexShaderKeyPointsPerFeature_ : vertexShaderKeyPoints_;

    fragmentShader = perFeatureStyle_ ? fragmentShaderPerFeature_ : fragmentShader_;

    gl_->UseProgramStages(pipeline_, GL_VERTEX_SHADER_BIT, vertexShader);
    NVXIO_CHECK_GL_ERROR();
    gl_->UseProgramStages(pipeline_, GL_FRAGMENT_SHADER_BIT, fragmentShader);
    NVXIO_CHECK_GL_ERROR();

    // bind program

    gl_->BindProgramPipeline(pipeline_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();

    GLfloat scaleRatioX = 2.0f * scale / (width - 1);
    GLfloat scaleRatioY = 2.0f * scale / (height - 1);

    gl_->ProgramUniform2f(vertexShader, 0,
                          scaleRatioX,
                          scaleRatioY);
    NVXIO_CHECK_GL_ERROR();

    if (!perFeatureStyle_)
    {
        gl_->ProgramUniform1f(vertexShader,
                              1, style.radius);
        NVXIO_CHECK_GL_ERROR();

        gl_->ProgramUniform4f(fragmentShader, 0,
                    style.color[0] / 255.0f,
                    style.color[1] / 255.0f,
                    style.color[2] / 255.0f,
                    style.color[3] / 255.0f);
        NVXIO_CHECK_GL_ERROR();


        gl_->DrawArraysInstanced(GL_POINTS, 0, 1, static_cast<GLsizei>(num_items));
        NVXIO_CHECK_GL_ERROR();
    }
    else
    {
        gl_->DrawArrays(GL_POINTS, 0, static_cast<GLsizei>(num_items));
        NVXIO_CHECK_GL_ERROR();
    }

    // reset state
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindProgramPipeline(0);
    NVXIO_CHECK_GL_ERROR();

#ifndef USE_GLES
    if (programPointSize == GL_FALSE)
    {
        gl_->Disable(GL_VERTEX_PROGRAM_POINT_SIZE);
        NVXIO_CHECK_GL_ERROR();
    }
#endif
}

//============================================================
// LinesRender
//============================================================

nvxio::LinesRender::LinesRender() :
    gl_(NULL), res_(NULL), bufCapacity_(500), vbo_(0), vao_(0), program_(0)
{
}

bool nvxio::LinesRender::init(std::shared_ptr<GLFunctions> _gl)
{
    gl_ = _gl;

    // Vertex buffer

    gl_->GenVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();

    // Vertex buffer

    gl_->GenBuffers(1, &vbo_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ARRAY_BUFFER, vbo_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BufferData(GL_ARRAY_BUFFER, bufCapacity_ * 4 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    NVXIO_CHECK_GL_ERROR();
    gl_->VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL);
    NVXIO_CHECK_GL_ERROR();
    gl_->EnableVertexAttribArray(0);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();

    // CUDA Graphics Resource

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&res_, vbo_, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
    {
        NVXIO_PRINT("LinesRender error: %s", cudaGetErrorString(err));
        return false;
    }

    // Shaders

    program_ = gl_->CreateProgram();
    NVXIO_CHECK_GL_ERROR();

    if (gl_->IsProgram(program_) == GL_FALSE)
    {
        NVXIO_PRINT("LinesRender: error creating shader program");
        return false;
    }

    if (!compileProgram(gl_, program_,
                        line_render_shader_vs_code,
                        line_render_shader_fs_code))
        return false;

    return true;
}

void nvxio::LinesRender::release()
{
    if (!gl_)
        return;

    gl_->DeleteBuffers(1, &vbo_);
    NVXIO_CHECK_GL_ERROR();
    vbo_ = 0;

    gl_->DeleteVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();
    vao_ = 0;

    if (res_)
    {
        cudaGraphicsUnregisterResource(res_);
        res_ = 0;
    }

    gl_->DeleteProgram(program_);
    NVXIO_CHECK_GL_ERROR();
    program_ = 0;
}

void nvxio::LinesRender::render(vx_array lines, const nvxio::Render::LineStyle& style, vx_uint32 width, vx_uint32 height, vx_float32 scale)
{
    vx_size num_items = 0;
    NVXIO_SAFE_CALL( vxQueryArray(lines, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_items, sizeof(num_items)) );

    for (vx_size start_x = 0; start_x < num_items; start_x += bufCapacity_)
    {
        vx_size end_x = std::min(start_x + bufCapacity_, num_items);
        updateArray(start_x, end_x, lines);
        renderArray(end_x - start_x, style, width, height, scale);
    }
}

void nvxio::LinesRender::updateArray(vx_size start_x, vx_size end_x, vx_array lines)
{
    cudaStream_t stream = NULL;

    vx_enum item_type = 0;
    NVXIO_SAFE_CALL( vxQueryArray(lines, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &item_type, sizeof(item_type)) );
    NVXIO_ASSERT( item_type == NVX_TYPE_POINT4F );

    vx_size stride = 0;
    void* src_ptr = NULL;
    NVXIO_SAFE_CALL( vxAccessArrayRange(lines, start_x, end_x, &stride, &src_ptr, NVX_READ_ONLY_CUDA) );

    NVXIO_CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &res_, stream) );

    void* dst_ptr = NULL;
    size_t size = 0;
    NVXIO_CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer(&dst_ptr, &size, res_) );

    NVXIO_CUDA_SAFE_CALL( cudaMemcpyAsync(dst_ptr, src_ptr, (end_x - start_x) * sizeof(nvx_point4f_t),
                                          cudaMemcpyDeviceToDevice, stream) );

    NVXIO_CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &res_, stream) );

    NVXIO_CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );

    NVXIO_SAFE_CALL( vxCommitArrayRange(lines, start_x, end_x, src_ptr) );
}

void nvxio::LinesRender::renderArray(vx_size num_items, const nvxio::Render::LineStyle& style,
                                     vx_uint32 width, vx_uint32 height, vx_float32 scale)
{
    LinesRenderingRules rules(gl_, style.thickness);
    (void)rules;

    gl_->UseProgram(program_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();

    GLfloat scaleRatioX = 2.0f * scale / (width - 1);
    GLfloat scaleRatioY = 2.0f * scale / (height - 1);

    gl_->Uniform2f(0,
                   scaleRatioX,
                   scaleRatioY);
    NVXIO_CHECK_GL_ERROR();

    gl_->Uniform4f(1,
                style.color[0] / 255.0f,
                style.color[1] / 255.0f,
                style.color[2] / 255.0f,
                style.color[3] / 255.0f);
    NVXIO_CHECK_GL_ERROR();

    gl_->DrawArrays(GL_LINES, 0, static_cast<GLsizei>(2 * num_items));
    NVXIO_CHECK_GL_ERROR();

    // reset state
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();
    gl_->UseProgram(0);
    NVXIO_CHECK_GL_ERROR();
}

//============================================================
// ArrowsRender
//============================================================

nvxio::ArrowsRender::ArrowsRender() :
    gl_(NULL), resOld_(NULL), resNew_(NULL), bufCapacity_(1000), vbo_(0), ssboOld_(0),
    ssboNew_(0), vao_(0), program_(0), computeShaderProgramPoints_(0),
    computeShaderProgramVxKeyPoints_(0), computeShaderProgramNvxKeyPoints_(0),
    featureType_(0)
{
}

bool nvxio::ArrowsRender::init(std::shared_ptr<GLFunctions> _gl)
{
    gl_ = _gl;

    // Vertex array object

    gl_->GenVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();

    // Vertex buffer

    gl_->GenBuffers(1, &vbo_);
    NVXIO_CHECK_GL_ERROR();

    gl_->GenBuffers(1, &ssboOld_);
    NVXIO_CHECK_GL_ERROR();

    gl_->GenBuffers(1, &ssboNew_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ARRAY_BUFFER, vbo_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BufferData(GL_ARRAY_BUFFER, bufCapacity_ * 6 * 2 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    NVXIO_CHECK_GL_ERROR();
    gl_->VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), NULL);
    NVXIO_CHECK_GL_ERROR();
    gl_->EnableVertexAttribArray(0);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();

    size_t maxElemSize = std::max(sizeof(nvx_point2f_t), std::max(sizeof(nvx_keypointf_t), sizeof(vx_keypoint_t)));

    gl_->BindBuffer(GL_SHADER_STORAGE_BUFFER, ssboOld_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BufferData(GL_SHADER_STORAGE_BUFFER, bufCapacity_ * maxElemSize, NULL, GL_DYNAMIC_DRAW);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_SHADER_STORAGE_BUFFER, ssboNew_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BufferData(GL_SHADER_STORAGE_BUFFER, bufCapacity_ * maxElemSize, NULL, GL_DYNAMIC_DRAW);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();

    // CUDA Graphics Resource

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&resOld_, ssboOld_, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
    {
        NVXIO_PRINT("ArrowsRender error: %s", cudaGetErrorString(err));
        return false;
    }

    err = cudaGraphicsGLRegisterBuffer(&resNew_, ssboNew_, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
    {
        NVXIO_PRINT("ArrowsRender error: %s", cudaGetErrorString(err));
        return false;
    }

    // Shaders

    program_ = gl_->CreateProgram();
    NVXIO_CHECK_GL_ERROR();

    if (gl_->IsProgram(program_) == GL_FALSE)
    {
        NVXIO_PRINT("ArrowsRender: error creating shader program");
        return false;
    }

    if (!compileProgram(gl_, program_,
                        arrow_render_shader_vs_code,
                        arrow_render_shader_fs_code))
        return false;

    // Create compute shaders programs

    GLuint * computePrograms[] =
    {
        &computeShaderProgramPoints_,
        &computeShaderProgramVxKeyPoints_,
        &computeShaderProgramNvxKeyPoints_
    };

    const char * const options[] =
    {
        "#define WITH_POINTS\n",
        "#define WITH_VXKEYPOINTS\n",
        "#define WITH_NVXKEYPOINTS\n",
    };

    for (size_t i = 0; i < nvxio::dimOf(computePrograms); ++i)
    {
        GLuint computeShaderProgram = gl_->CreateProgram();
        NVXIO_CHECK_GL_ERROR();

        if (gl_->IsProgram(computeShaderProgram) == GL_FALSE)
        {
            NVXIO_PRINT("ArrowsRender: error creating compute shader program");
            return false;
        }

        if (!compileProgram(gl_, computeShaderProgram,
                            NULL, NULL, arrow_compute_shader_cs_code,
                            options[i]))
            return false;

        *computePrograms[i] = computeShaderProgram;
    }

    return true;
}

void nvxio::ArrowsRender::release()
{
    if (!gl_)
        return;

    GLuint * buffers[] = { &vbo_, &ssboOld_, &ssboNew_ };

    for (size_t i = 0; i < nvxio::dimOf(buffers); ++i)
    {
        gl_->DeleteBuffers(1, buffers[i]);
        NVXIO_CHECK_GL_ERROR();
        *buffers[i] = 0;
    }

    gl_->DeleteVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();
    vao_ = 0;

    if (resOld_)
    {
        cudaGraphicsUnregisterResource(resOld_);
        resOld_ = 0;
    }

    if (resNew_)
    {
        cudaGraphicsUnregisterResource(resNew_);
        resNew_ = 0;
    }

    GLuint * programs[] = { &program_,
                            &computeShaderProgramPoints_,
                            &computeShaderProgramVxKeyPoints_,
                            &computeShaderProgramNvxKeyPoints_ };

    for (size_t i = 0; i < nvxio::dimOf(programs); ++i)
    {
        gl_->DeleteProgram(program_);
        NVXIO_CHECK_GL_ERROR();
        *programs[i] = 0;
    }
}

void nvxio::ArrowsRender::render(vx_array old_points, vx_array new_points, const nvxio::Render::LineStyle& line_style,
                                 vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio)
{
    vx_size old_size = 0, new_size = 0;

    NVXIO_SAFE_CALL( vxQueryArray(old_points, VX_ARRAY_ATTRIBUTE_NUMITEMS, &old_size, sizeof(old_size)) );
    NVXIO_SAFE_CALL( vxQueryArray(new_points, VX_ARRAY_ATTRIBUTE_NUMITEMS, &new_size, sizeof(new_size)) );

    vx_size num_items = std::min(old_size, new_size);

    for (vx_size start_x = 0; start_x < num_items; start_x += bufCapacity_)
    {
        vx_size end_x = std::min(start_x + bufCapacity_, num_items);
        updateLinesArray(start_x, end_x, old_points, new_points, width, height, scaleRatio);
        renderArray(end_x - start_x, line_style);
    }
}

void nvxio::ArrowsRender::renderArray(vx_size num_items, const nvxio::Render::LineStyle& style)
{
    LinesRenderingRules rules(gl_, static_cast<GLfloat>(style.thickness));
    (void)rules;

    gl_->UseProgram(program_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();

    gl_->Uniform4f(0,
                style.color[0] / 255.0f,
                style.color[1] / 255.0f,
                style.color[2] / 255.0f,
                style.color[3] / 255.0f);
    NVXIO_CHECK_GL_ERROR();

    gl_->DrawArrays(GL_LINES, 0, static_cast<GLsizei>(num_items * 6));
    NVXIO_CHECK_GL_ERROR();

    // reset state
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();
    gl_->UseProgram(0);
    NVXIO_CHECK_GL_ERROR();
}

void nvxio::ArrowsRender::updateLinesArray(vx_size start_x, vx_size end_x, vx_array old_points, vx_array new_points,
                                           vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio)
{
    cudaStream_t stream = NULL;

    NVXIO_SAFE_CALL( vxQueryArray(old_points, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &featureType_, sizeof(featureType_)) );
    NVXIO_ASSERT( (featureType_ == NVX_TYPE_POINT2F) || (featureType_ == NVX_TYPE_KEYPOINTF) || (featureType_ == VX_TYPE_KEYPOINT) );

    size_t elemSize = featureType_ == NVX_TYPE_POINT2F ? sizeof(nvx_point2f_t) :
                      featureType_ == NVX_TYPE_KEYPOINTF ? sizeof(nvx_keypointf_t) :
                                                           sizeof(vx_keypoint_t);

    vx_enum item_type_other = 0;
    NVXIO_SAFE_CALL( vxQueryArray(new_points, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &item_type_other, sizeof(item_type_other)) );
    NVXIO_ASSERT( item_type_other == featureType_);

    vx_size old_stride = 0, new_stride = 0;
    void* old_src_ptr = NULL, * new_src_ptr = NULL;
    NVXIO_SAFE_CALL( vxAccessArrayRange(old_points, start_x, end_x, &old_stride, &old_src_ptr, NVX_READ_ONLY_CUDA) );
    NVXIO_SAFE_CALL( vxAccessArrayRange(new_points, start_x, end_x, &new_stride, &new_src_ptr, NVX_READ_ONLY_CUDA) );

    NVXIO_CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &resOld_, stream) );
    NVXIO_CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &resNew_, stream) );

    void* oldPtr = NULL, * newPtr = NULL;
    size_t sizeOld = 0, sizeNew = 0;

    NVXIO_CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer(&oldPtr, &sizeOld, resOld_) );
    NVXIO_CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer(&newPtr, &sizeNew, resNew_) );

    NVXIO_CUDA_SAFE_CALL( cudaMemcpyAsync(oldPtr, old_src_ptr,
                                          elemSize * (end_x - start_x),
                                          cudaMemcpyDeviceToDevice, stream) );

    NVXIO_CUDA_SAFE_CALL( cudaMemcpyAsync(newPtr, new_src_ptr,
                                          elemSize * (end_x - start_x),
                                          cudaMemcpyDeviceToDevice, stream) );

    NVXIO_CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &resOld_, stream) );
    NVXIO_CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &resNew_, stream) );

    NVXIO_CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );

    NVXIO_SAFE_CALL( vxCommitArrayRange(old_points, start_x, end_x, old_src_ptr) );
    NVXIO_SAFE_CALL( vxCommitArrayRange(new_points, start_x, end_x, new_src_ptr) );

    // OpenGL part

    gl_->UseProgram(featureType_ == NVX_TYPE_POINT2F ? computeShaderProgramPoints_ :
                    featureType_ == NVX_TYPE_KEYPOINTF ? computeShaderProgramNvxKeyPoints_ :
                                                         computeShaderProgramVxKeyPoints_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboOld_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboNew_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vbo_);
    NVXIO_CHECK_GL_ERROR();

    GLfloat scaleRatioX = 2.0f * scaleRatio / (width - 1);
    GLfloat scaleRatioY = 2.0f * scaleRatio / (height - 1);

    gl_->Uniform2f(0, scaleRatioX, scaleRatioY);
    NVXIO_CHECK_GL_ERROR();

    gl_->Uniform1i(1, static_cast<GLint>(end_x - start_x));
    NVXIO_CHECK_GL_ERROR();

    gl_->DispatchCompute(static_cast<GLuint>(end_x - start_x + 255) / 256, 1, 1);
    NVXIO_CHECK_GL_ERROR();

    gl_->MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    NVXIO_CHECK_GL_ERROR();

    gl_->UseProgram(0);
    NVXIO_CHECK_GL_ERROR();
}

//============================================================
// MotionFieldRender
//============================================================

nvxio::MotionFieldRender::MotionFieldRender() :
    gl_(NULL), capacity_(0), numPoints_(0), ssbo_(0), vao_(0), res_(NULL), program_(0),
    computeShaderProgram_(0), ssboTex_(0)
{
}

bool nvxio::MotionFieldRender::init(std::shared_ptr<GLFunctions> _gl, vx_uint32 width, vx_uint32 height)
{
    gl_ = _gl;

    // Vertex arrays

    gl_->GenVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();

    // Vertex buffer

    // 3 lines per arrow = 6 points per arrow
    capacity_ = (width / 16) * (height / 16) * 6;

    gl_->GenBuffers(1, &ssbo_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ARRAY_BUFFER, ssbo_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BufferData(GL_ARRAY_BUFFER, capacity_ * 2 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    NVXIO_CHECK_GL_ERROR();
    gl_->VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), NULL);
    NVXIO_CHECK_GL_ERROR();
    gl_->EnableVertexAttribArray(0);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();

    // Textures

    gl_->GenTextures(1, &ssboTex_);
    NVXIO_CHECK_GL_ERROR();

    gl_->ActiveTexture(GL_TEXTURE0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindTexture(GL_TEXTURE_2D, ssboTex_);
    NVXIO_CHECK_GL_ERROR();
    gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    NVXIO_CHECK_GL_ERROR();
    gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    NVXIO_CHECK_GL_ERROR();
    gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    NVXIO_CHECK_GL_ERROR();
    gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    NVXIO_CHECK_GL_ERROR();
    gl_->TexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, width * 2, height);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindTexture(GL_TEXTURE_2D, 0);
    NVXIO_CHECK_GL_ERROR();

    // CUDA Graphics Resource

    cudaError_t err = cudaGraphicsGLRegisterImage(&res_, ssboTex_, GL_TEXTURE_2D,
                                                  cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
    {
        NVXIO_PRINT("MotionFieldRender error: %s", cudaGetErrorString(err));
        return false;
    }

    // Shaders

    program_ = gl_->CreateProgram();
    NVXIO_CHECK_GL_ERROR();

    if (gl_->IsProgram(program_) == GL_FALSE)
    {
        NVXIO_PRINT("MotionFieldRender: error creating shader program");
        return false;
    }

    if (!compileProgram(gl_, program_,
                        arrow_render_shader_vs_code,
                        arrow_render_shader_fs_code))
        return false;

    // Create compute shader program

    computeShaderProgram_ = gl_->CreateProgram();
    NVXIO_CHECK_GL_ERROR();

    if (gl_->IsProgram(computeShaderProgram_) == GL_FALSE)
    {
        NVXIO_PRINT("MotionFieldRender: error creating shader program");
        return false;
    }

    if (!compileProgram(gl_, computeShaderProgram_,
                        NULL, NULL,
                        motion_field_compute_shader_cs_code))
    {
        NVXIO_PRINT("MotionFieldRender: error creating compute shader program");
        return false;
    }

    return true;
}

void nvxio::MotionFieldRender::release()
{
    if (!gl_)
        return;

    gl_->DeleteBuffers(1, &ssbo_);
    NVXIO_CHECK_GL_ERROR();
    ssbo_ = 0;

    gl_->DeleteTextures(1, &ssboTex_);
    NVXIO_CHECK_GL_ERROR();
    ssboTex_ = 0;

    gl_->DeleteVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();
    vao_ = 0;

    if (res_)
    {
        cudaGraphicsUnregisterResource(res_);
        res_ = 0;
    }

    gl_->DeleteProgram(program_);
    NVXIO_CHECK_GL_ERROR();
    program_ = 0;

    gl_->DeleteProgram(computeShaderProgram_);
    NVXIO_CHECK_GL_ERROR();
    computeShaderProgram_ = 0;
}

void nvxio::MotionFieldRender::render(vx_image field, const nvxio::Render::MotionFieldStyle& style,
                                      vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio)
{
    updateArray(field, width, height, scaleRatio);
    renderArray(style);
}

void nvxio::MotionFieldRender::updateArray(vx_image field, vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio)
{
    cudaStream_t stream = NULL;

    vx_uint32 field_width = 0, field_height = 0;
    vx_df_image field_format = VX_DF_IMAGE_VIRT;
    NVXIO_SAFE_CALL( vxQueryImage(field, VX_IMAGE_ATTRIBUTE_WIDTH, &field_width, sizeof(field_width)) );
    NVXIO_SAFE_CALL( vxQueryImage(field, VX_IMAGE_ATTRIBUTE_HEIGHT, &field_height, sizeof(field_height)) );
    NVXIO_SAFE_CALL( vxQueryImage(field, VX_IMAGE_ATTRIBUTE_FORMAT, &field_format, sizeof(field_format)) );

    NVXIO_ASSERT( field_format == NVX_DF_IMAGE_2F32 );

    numPoints_ = (width / 16) * (height / 16) * 6;

    vx_uint32 mf_scale = std::min(lrintf((width / scaleRatio) / field_width),
                                  lrintf((height / scaleRatio) / field_height));

    vx_rectangle_t rect = {};
    NVXIO_SAFE_CALL( vxGetValidRegionImage(field, &rect) );

    vx_imagepatch_addressing_t addr;
    void *ptr = NULL;
    NVXIO_SAFE_CALL( vxAccessImagePatch(field, &rect, 0, &addr, &ptr, NVX_READ_ONLY_CUDA) );

    NVXIO_CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &res_, stream) );

    cudaArray_t cudaArr = NULL;
    NVXIO_CUDA_SAFE_CALL( cudaGraphicsSubResourceGetMappedArray(&cudaArr, res_, 0, 0) );

    // Copy CUDA memory to mapped OpenGL resource

    NVXIO_CUDA_SAFE_CALL( cudaMemcpy2DToArrayAsync(cudaArr, 0, 0, ptr, addr.stride_y,
                                                   addr.dim_x * sizeof(GLfloat) * 2, addr.dim_y,
                                                   cudaMemcpyDeviceToDevice, stream) );

    NVXIO_CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &res_, stream) );

    NVXIO_CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );

    NVXIO_SAFE_CALL( vxCommitImagePatch(field, NULL, 0, &addr, ptr) );

    // Run compute shader

    gl_->UseProgram(computeShaderProgram_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindImageTexture(0, ssboTex_, 0, GL_FALSE, 0,
                          GL_READ_ONLY, GL_R32F);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_);
    NVXIO_CHECK_GL_ERROR();

    GLfloat scaleRatioX = 2.0f / (width - 1);
    GLfloat scaleRatioY = 2.0f / (height - 1);

    gl_->Uniform2f(0, scaleRatioX, scaleRatioY);
    NVXIO_CHECK_GL_ERROR();

    gl_->Uniform1f(1, scaleRatio);
    NVXIO_CHECK_GL_ERROR();

    gl_->Uniform1ui(2, mf_scale);
    NVXIO_CHECK_GL_ERROR();

    gl_->DispatchCompute(width / 16, height / 16, 1);
    NVXIO_CHECK_GL_ERROR();

    gl_->MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    NVXIO_CHECK_GL_ERROR();

    gl_->UseProgram(0);
    NVXIO_CHECK_GL_ERROR();
}

void nvxio::MotionFieldRender::renderArray(const nvxio::Render::MotionFieldStyle& style)
{
    gl_->UseProgram(program_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();

    gl_->Uniform4f(0,
                style.color[0] / 255.0f,
                style.color[1] / 255.0f,
                style.color[2] / 255.0f,
                style.color[3] / 255.0f);
    NVXIO_CHECK_GL_ERROR();

    gl_->DrawArrays(GL_LINES, 0, static_cast<GLsizei>(numPoints_));
    NVXIO_CHECK_GL_ERROR();

    // reset state
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();
    gl_->UseProgram(0);
    NVXIO_CHECK_GL_ERROR();
}

//============================================================
// CirclesRender
//============================================================

nvxio::CirclesRender::CirclesRender() :
    gl_(NULL), bufCapacity_(3000), vbo_(0), vao_(0), program_(0)
{
}

bool nvxio::CirclesRender::init(std::shared_ptr<GLFunctions> _gl)
{
    gl_ = _gl;

    // Vertex array

    gl_->GenVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();

    // Vertex buffer

    gl_->GenBuffers(1, &vbo_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ARRAY_BUFFER, vbo_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BufferData(GL_ARRAY_BUFFER, bufCapacity_ * 4 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    NVXIO_CHECK_GL_ERROR();
    gl_->VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), NULL);
    NVXIO_CHECK_GL_ERROR();
    gl_->EnableVertexAttribArray(0);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();

    // Shaders

    program_ = gl_->CreateProgram();
    NVXIO_CHECK_GL_ERROR();

    if (gl_->IsProgram(program_) == GL_FALSE)
    {
        NVXIO_PRINT("CirclesRender: error creating shader program");
        return false;
    }

    if (!compileProgram(gl_, program_,
                        line_render_shader_vs_code,
                        line_render_shader_fs_code))
        return false;

    return true;
}

void nvxio::CirclesRender::release()
{
    if (!gl_)
        return;

    gl_->DeleteBuffers(1, &vbo_);
    NVXIO_CHECK_GL_ERROR();
    vbo_ = 0;

    gl_->DeleteVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();
    vao_ = 0;

    gl_->DeleteProgram(program_);
    NVXIO_CHECK_GL_ERROR();
    program_= 0;
}

void nvxio::CirclesRender::render(vx_array circles, const nvxio::Render::CircleStyle& style, vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio)
{
    updateArray(circles);
    renderArray(style, width, height, scaleRatio);
}

void nvxio::CirclesRender::updateArray(vx_array circles)
{
    vx_size num_items = 0;
    NVXIO_SAFE_CALL( vxQueryArray(circles, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_items, sizeof(num_items)) );

    vx_enum item_type = 0;
    NVXIO_SAFE_CALL( vxQueryArray(circles, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &item_type, sizeof(item_type)) );
    NVXIO_ASSERT( item_type == NVX_TYPE_POINT3F );

    points_.clear();

    if (num_items > 0)
    {
        vx_size stride = 0;
        void* ptr = NULL;
        NVXIO_SAFE_CALL( vxAccessArrayRange(circles, 0, num_items, &stride, &ptr, VX_READ_ONLY) );

        for (vx_size i = 0; i < num_items; ++i)
        {
            nvx_point3f_t c = vxArrayItem(nvx_point3f_t, ptr, i, stride);

            int num_segments = getNumCircleSegments(c.z);

            genCircleLines(points_, c.x, c.y, c.z, num_segments);
        }

        NVXIO_SAFE_CALL( vxCommitArrayRange(circles, 0, num_items, ptr) );
    }
}

void nvxio::CirclesRender::renderArray(const nvxio::Render::CircleStyle& style,
                                       vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio)
{
    LinesRenderingRules rules(gl_, static_cast<GLfloat>(style.thickness));
    (void)rules;

    gl_->UseProgram(program_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();

    GLfloat scaleRatioX = 2.0f * scaleRatio / (width - 1);
    GLfloat scaleRatioY = 2.0f * scaleRatio / (height - 1);

    gl_->Uniform2f(0, scaleRatioX, scaleRatioY);
    NVXIO_CHECK_GL_ERROR();

    gl_->Uniform4f(1,
                style.color[0] / 255.0f,
                style.color[1] / 255.0f,
                style.color[2] / 255.0f,
                style.color[3] / 255.0f);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ARRAY_BUFFER, vbo_);
    NVXIO_CHECK_GL_ERROR();

    for (size_t start_x = 0; start_x < points_.size(); start_x += bufCapacity_)
    {
        vx_size end_x = std::min(start_x + bufCapacity_, points_.size());

        gl_->BufferSubData(GL_ARRAY_BUFFER, 0, (end_x - start_x) * 4 * sizeof(GLfloat), &points_[start_x]);
        NVXIO_CHECK_GL_ERROR();

        gl_->DrawArrays(GL_LINES, 0, static_cast<GLsizei>(2 * (end_x - start_x)));
        NVXIO_CHECK_GL_ERROR();
    }

    // reset state
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->UseProgram(0);
    NVXIO_CHECK_GL_ERROR();
}

//============================================================
// TextRender
//============================================================

const int fontSize = 20;

nvxio::TextRender::TextRender() :
    gl_(NULL), ft_(NULL), face_(NULL), programBg_(0), program_(0),
    tex_(0), bufCapacity_(3000), vbo_(0), vboEA_(0), vao_(0), bgVbo_(0), bgVao_(0)
{
}

nvxio::TextRender::~TextRender()
{
    FT_Done_Face(face_);
    FT_Done_FreeType(ft_);
}

bool nvxio::TextRender::init(std::shared_ptr<GLFunctions> _gl)
{
    gl_ = _gl;

    // Freetype library

    if (FT_Init_FreeType(&ft_) != FT_Err_Ok)
    {
        NVXIO_PRINT("TextRender: could not init freetype library");
        return false;
    }

#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
    std::string fontFile = "calibri.ttf";

    CHAR winDir[MAX_PATH];
    GetWindowsDirectoryA(winDir, MAX_PATH);

    std::stringstream stream;
    stream << winDir << "\\Fonts\\" << fontFile;

    std::string fontPath = stream.str();
#elif defined(__ANDROID__)
    std::string fontPath = "/system/fonts/DroidSans.ttf";
#else
    std::string fontPath = "/usr/share/fonts/truetype/freefont/FreeSans.ttf";
#endif

    if (FT_New_Face(ft_, fontPath.c_str(), 0, &face_) != FT_Err_Ok)
    {
        NVXIO_PRINT("TextRender: could not open FreeSans font");
        return false;
    }

    if (FT_Set_Pixel_Sizes(face_, 0, fontSize) != FT_Err_Ok)
    {
        NVXIO_PRINT("TextRender: failed to set font size");
        return false;
    }

    // Shaders for text

    program_ = gl_->CreateProgram();
    NVXIO_CHECK_GL_ERROR();

    if (gl_->IsProgram(program_) == GL_FALSE)
    {
        NVXIO_PRINT("TextRender: error creating shader program");
        return false;
    }

    if (!compileProgram(gl_, program_,
                        text_render_shader_vs_code,
                        text_render_shader_fs_code))
        return false;

    // Shaders for background

    programBg_ = gl_->CreateProgram();
    NVXIO_CHECK_GL_ERROR();

    if (!gl_->IsProgram(programBg_))
    {
        NVXIO_PRINT("TextRender: error creating shader program");
        return false;
    }

    if (!compileProgram(gl_, programBg_,
                        textbg_render_shader_vs_code,
                        textbg_render_shader_fs_code))
        return false;

    // Vertex arrays

    gl_->GenVertexArrays(1, &bgVao_);
    NVXIO_CHECK_GL_ERROR();

    gl_->GenVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();

    // Texture atlas

    FT_GlyphSlot g = face_->glyph;

    int w = 0;
    int h = 0;

    for (int i = 32; i < 128; i++)
    {
        if (FT_Load_Char(face_, i, FT_LOAD_RENDER) != FT_Err_Ok)
            continue;

        w += g->bitmap.width;
        h = std::max<int>(h, g->bitmap.rows);
    }

    atlasWidth_ = w;
    atlasHeight_ = h;

    gl_->GenTextures(1, &tex_);
    NVXIO_CHECK_GL_ERROR();
    gl_->ActiveTexture(GL_TEXTURE0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindTexture(GL_TEXTURE_2D, tex_);
    NVXIO_CHECK_GL_ERROR();
    gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    NVXIO_CHECK_GL_ERROR();
    gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    NVXIO_CHECK_GL_ERROR();
    gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    NVXIO_CHECK_GL_ERROR();
    gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    NVXIO_CHECK_GL_ERROR();
    gl_->PixelStorei(GL_UNPACK_ALIGNMENT, 1);
    NVXIO_CHECK_GL_ERROR();
    gl_->TexStorage2D(GL_TEXTURE_2D, 1, GL_R8, w, h);
    NVXIO_CHECK_GL_ERROR();

    int x = 0;
    for (int i = 32; i < 128; i++)
    {
        if (FT_Load_Char(face_, i, FT_LOAD_RENDER) != FT_Err_Ok)
        {
            NVXIO_PRINT("TextRender: failed to load charachter %c", (char)i);
            continue;
        }

        gl_->TexSubImage2D(GL_TEXTURE_2D, 0, x, 0, g->bitmap.width, g->bitmap.rows, GL_RED, GL_UNSIGNED_BYTE, g->bitmap.buffer);
        NVXIO_CHECK_GL_ERROR();

        c[i].ax = static_cast<float>(g->advance.x >> 6);
        c[i].ay = static_cast<float>(g->advance.y >> 6);

        c[i].bw = static_cast<float>(g->bitmap.width);
        c[i].bh = static_cast<float>(g->bitmap.rows);

        c[i].bl = static_cast<float>(g->bitmap_left);
        c[i].bt = static_cast<float>(g->bitmap_top);

        c[i].tx = (float)x / w;

        x += g->bitmap.width;
    }

    gl_->BindTexture(GL_TEXTURE_2D, 0);
    NVXIO_CHECK_GL_ERROR();

    // Vertex buffer

    gl_->GenBuffers(1, &vbo_);
    NVXIO_CHECK_GL_ERROR();
    gl_->GenBuffers(1, &vboEA_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ARRAY_BUFFER, vbo_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboEA_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BufferData(GL_ARRAY_BUFFER, bufCapacity_ * 4 * 4 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    NVXIO_CHECK_GL_ERROR();
    gl_->BufferData(GL_ELEMENT_ARRAY_BUFFER, bufCapacity_ * 6 * sizeof(GLushort), NULL, GL_DYNAMIC_DRAW);
    NVXIO_CHECK_GL_ERROR();
    gl_->VertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), NULL);
    NVXIO_CHECK_GL_ERROR();
    gl_->EnableVertexAttribArray(0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();

    gl_->GenBuffers(1, &bgVbo_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindVertexArray(bgVao_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ARRAY_BUFFER, bgVbo_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    NVXIO_CHECK_GL_ERROR();
    gl_->VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), NULL);
    NVXIO_CHECK_GL_ERROR();
    gl_->EnableVertexAttribArray(0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();

    return true;
}

void nvxio::TextRender::release()
{
    if (!gl_)
        return;

    gl_->DeleteTextures(1, &tex_);
    NVXIO_CHECK_GL_ERROR();
    tex_ = 0;

    gl_->DeleteBuffers(1, &vbo_);
    NVXIO_CHECK_GL_ERROR();
    vbo_ = 0;

    gl_->DeleteBuffers(1, &vboEA_);
    NVXIO_CHECK_GL_ERROR();
    vboEA_ = 0;

    gl_->DeleteBuffers(1, &bgVbo_);
    NVXIO_CHECK_GL_ERROR();
    bgVbo_ = 0;

    gl_->DeleteVertexArrays(1, &vao_);
    NVXIO_CHECK_GL_ERROR();
    vao_ = 0;

    gl_->DeleteVertexArrays(1, &bgVao_);
    NVXIO_CHECK_GL_ERROR();
    bgVao_ = 0;

    gl_->DeleteProgram(programBg_);
    NVXIO_CHECK_GL_ERROR();
    programBg_ = 0;

    gl_->DeleteProgram(program_);
    NVXIO_CHECK_GL_ERROR();
    program_ = 0;
}

static void addPoint(std::vector<nvx_point4f_t>& points,
                     float x, float y, float tx, float ty,
                     vx_float32 scaleX, vx_float32 scaleY,
                     float& min_x, float& max_x,
                     float& min_y, float& max_y)
{
    nvx_point4f_t pt = {x, y, tx, ty};

    pt.x = 2 * pt.x * scaleX - 1;
    pt.y = 1 - 2 * pt.y * scaleY;

    min_x = std::min(min_x, pt.x);
    max_x = std::max(max_x, pt.x);

    min_y = std::min(min_y, pt.y);
    max_y = std::max(max_y, pt.y);

    points.push_back(pt);
}

void nvxio::TextRender::render(const std::string& text, const nvxio::Render::TextBoxStyle& style,
                               vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio)
{
    NVXIO_ASSERT(!text.empty());

    points_.clear();
    points_.reserve(text.size() * 4);

    elements_.clear();
    elements_.reserve(text.size() * 6);

    vx_float32 scaleX = 1.0f / (width - 1), scaleY = 1.0f / (height - 1);
    nvx_point2f_t origin = { static_cast<GLfloat>(style.origin.x) * scaleRatio,
                             static_cast<GLfloat>(style.origin.y) * scaleRatio };
    float x = origin.x, y = origin.y;

    float min_x = std::numeric_limits<float>::max(), min_y = min_x;
    float max_x = -min_x, max_y = -min_y;

    GLushort ei = 0;
    for (size_t i = 0; i < text.size(); ++i, ++ei)
    {
        int p = text[i];

        // move to the next line
        if (p == '\n')
        {
            x = origin.x;
            y += fontSize;
            --ei;
            continue;
        }

        if (p < 32 || p >= 128)
            continue;

        if (points_.size() + 6 > bufCapacity_)
            break;

        float x2 = x + c[p].bl;
        float y2 = y + fontSize - c[p].bt;
        float w = c[p].bw;
        float h = c[p].bh;

        // Advance the cursor to the start of the next character
        x += c[p].ax;
        y += c[p].ay;

        addPoint(points_, x2    , y2    , c[p].tx                        , 0                     , scaleX, scaleY, min_x, max_x, min_y, max_y);
        addPoint(points_, x2 + w, y2    , c[p].tx + c[p].bw / atlasWidth_, 0                     , scaleX, scaleY, min_x, max_x, min_y, max_y);
        addPoint(points_, x2    , y2 + h, c[p].tx                        , c[p].bh / atlasHeight_, scaleX, scaleY, min_x, max_x, min_y, max_y);
        addPoint(points_, x2 + w, y2 + h, c[p].tx + c[p].bw / atlasWidth_, c[p].bh / atlasHeight_, scaleX, scaleY, min_x, max_x, min_y, max_y);

        elements_.push_back(ei * 4 + 0);
        elements_.push_back(ei * 4 + 1);
        elements_.push_back(ei * 4 + 2);
        elements_.push_back(ei * 4 + 3);
        elements_.push_back(ei * 4 + 1);
        elements_.push_back(ei * 4 + 2);
    }

    // Draw background

    if (style.bgcolor[3] != 0)
    {
        gl_->UseProgram(programBg_);
        NVXIO_CHECK_GL_ERROR();

        gl_->BindVertexArray(bgVao_);
        NVXIO_CHECK_GL_ERROR();

        gl_->Uniform4f(0,
                    style.bgcolor[0] / 255.0f,
                    style.bgcolor[1] / 255.0f,
                    style.bgcolor[2] / 255.0f,
                    style.bgcolor[3] / 255.0f);
        NVXIO_CHECK_GL_ERROR();

        gl_->BindBuffer(GL_ARRAY_BUFFER, bgVbo_);
        NVXIO_CHECK_GL_ERROR();

        GLfloat offx = 7.0f / width, offy = 7.0f / height;
        GLfloat bg_box[4][2] = {
            { min_x - offx, min_y - offy },
            { max_x + offx, min_y - offy },
            { min_x - offx, max_y + offy },
            { max_x + offx, max_y + offy },
        };

        gl_->BufferSubData(GL_ARRAY_BUFFER, 0, sizeof(bg_box), &bg_box[0][0]);
        NVXIO_CHECK_GL_ERROR();

        gl_->DrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        NVXIO_CHECK_GL_ERROR();
    }

    // Draw Text

    gl_->UseProgram(program_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(vao_);
    NVXIO_CHECK_GL_ERROR();

    gl_->Uniform4f(0,
                style.color[0] / 255.0f,
                style.color[1] / 255.0f,
                style.color[2] / 255.0f,
                style.color[3] / 255.0f);
    NVXIO_CHECK_GL_ERROR();

    gl_->ActiveTexture(GL_TEXTURE0);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindTexture(GL_TEXTURE_2D, tex_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ARRAY_BUFFER, vbo_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboEA_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BufferSubData(GL_ARRAY_BUFFER, 0, points_.size() * 4 * sizeof(GLfloat), &points_[0]);
    NVXIO_CHECK_GL_ERROR();

    gl_->BufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, elements_.size() * sizeof(GLushort), &elements_[0]);
    NVXIO_CHECK_GL_ERROR();

    gl_->DrawElements(GL_TRIANGLES, static_cast<GLsizei>(elements_.size()), GL_UNSIGNED_SHORT, NULL);
    NVXIO_CHECK_GL_ERROR();

    // reset state
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();

    gl_->UseProgram(0);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindTexture(GL_TEXTURE_2D, 0);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
}

#ifndef __ANDROID__

//============================================================
// PointCloudRender
//============================================================

nvxio::PointCloudRender::PointCloudRender() :
    gl_(NULL), pointCloudProgram_(0), hPointCloudVBO_(0),
    hPointCloudVAO_(0), bufCapacity_(2000)
{
}

bool nvxio::PointCloudRender::init(std::shared_ptr<GLFunctions> _gl)
{
    gl_ = _gl;

     // Vertex array

    gl_->GenVertexArrays(1, &hPointCloudVAO_);
    NVXIO_CHECK_GL_ERROR();

    // Vertex buffer

    gl_->GenBuffers(1, &hPointCloudVBO_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(hPointCloudVAO_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ARRAY_BUFFER, hPointCloudVBO_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BufferData(GL_ARRAY_BUFFER, 3 * bufCapacity_ * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    NVXIO_CHECK_GL_ERROR();
    gl_->VertexAttribPointer((GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->EnableVertexAttribArray(0);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();

    // Shaders

    pointCloudProgram_ = gl_->CreateProgram();
    NVXIO_CHECK_GL_ERROR();

    if (gl_->IsProgram(pointCloudProgram_) == GL_FALSE)
    {
        NVXIO_PRINT("PointCloudRender: error creating shader program");
        return false;
    }

    if (!compileProgram(gl_, pointCloudProgram_,
                        point_cloud_render_shader_vs_code,
                        point_cloud_render_shader_fs_code))
        return false;

    return true;
}

void nvxio::PointCloudRender::release()
{
    if (!gl_)
        return;

    gl_->DeleteBuffers(1, &hPointCloudVBO_);
    NVXIO_CHECK_GL_ERROR();
    hPointCloudVBO_ = 0;

    gl_->DeleteVertexArrays(1, &hPointCloudVAO_);
    NVXIO_CHECK_GL_ERROR();
    hPointCloudVAO_ = 0;

    gl_->DeleteProgram(pointCloudProgram_);
    NVXIO_CHECK_GL_ERROR();
    pointCloudProgram_ = 0;
}

void nvxio::PointCloudRender::render(vx_array points, vx_matrix MVP, const nvxio::Render3D::PointCloudStyle& style)
{
    NVXIO_ASSERT(style.maxDistance >= style.minDistance);

    updateArray(MVP);
    renderArray(points, style);
}

void nvxio::PointCloudRender::updateArray(vx_matrix MVP)
{
    NVXIO_SAFE_CALL( vxReadMatrix(MVP, dataMVP_) );
}

void nvxio::PointCloudRender::renderArray(vx_array points, const nvxio::Render3D::PointCloudStyle& style)
{
    vx_enum type = 0;
    NVXIO_SAFE_CALL( vxQueryArray(points, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &type, sizeof(type)) );
    NVXIO_ASSERT(type == NVX_TYPE_POINT3F);

    gl_->Enable(GL_DEPTH_TEST);
    gl_->DepthFunc(GL_LESS);
#ifndef USE_GLES
    gl_->Enable(GL_VERTEX_PROGRAM_POINT_SIZE);
#endif

    gl_->UseProgram(pointCloudProgram_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(hPointCloudVAO_);
    NVXIO_CHECK_GL_ERROR();

    gl_->UniformMatrix4fv(0, 1, GL_FALSE, dataMVP_);
    NVXIO_CHECK_GL_ERROR();
    gl_->Uniform1f(1, style.maxDistance);
    NVXIO_CHECK_GL_ERROR();
    gl_->Uniform1f(2, style.minDistance);
    NVXIO_CHECK_GL_ERROR();
    gl_->Uniform1f(3, 1.0f / (style.maxDistance - style.minDistance));
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ARRAY_BUFFER, hPointCloudVBO_);
    NVXIO_CHECK_GL_ERROR();

    vx_size numOfPoints = 0;
    NVXIO_SAFE_CALL( vxQueryArray(points, VX_ARRAY_ATTRIBUTE_NUMITEMS, &numOfPoints, sizeof(numOfPoints)) );

    if (numOfPoints > 0)
    {
        vx_size stride = 0;
        void * ptr = NULL;
        NVXIO_SAFE_CALL( vxAccessArrayRange(points, 0, numOfPoints, &stride, &ptr, VX_READ_ONLY) );

        for (vx_size start_x = 0; start_x < numOfPoints; start_x += bufCapacity_)
        {
            vx_size end_x = std::min(start_x + bufCapacity_, numOfPoints);

            gl_->BufferSubData(GL_ARRAY_BUFFER, 0, (end_x - start_x) * 3 * sizeof(float), static_cast<vx_uint8 *>(ptr) + start_x * stride);
            NVXIO_CHECK_GL_ERROR();

            gl_->DrawArrays(GL_POINTS, 0, static_cast<GLsizei>(end_x - start_x));
            NVXIO_CHECK_GL_ERROR();
        }

        NVXIO_SAFE_CALL( vxCommitArrayRange(points, 0, numOfPoints, ptr) );
    }

    // Reset state
    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();
    gl_->UseProgram(0);
    NVXIO_CHECK_GL_ERROR();

    gl_->Disable(GL_DEPTH_TEST);
    NVXIO_CHECK_GL_ERROR();
#ifndef USE_GLES
    gl_->Disable(GL_VERTEX_PROGRAM_POINT_SIZE);
    NVXIO_CHECK_GL_ERROR();
#endif
}

//============================================================
// FencePlaneRender
//============================================================

nvxio::FencePlaneRender::FencePlaneRender() :
    gl_(NULL), fencePlaneProgram_(0), hFencePlaneVBO_(0),
    hFencePlaneEA_(0), hFencePlaneVAO_(0), bufCapacity_(2000)
{
    NVXIO_ASSERT(bufCapacity_ % 4 == 0);
}

bool nvxio::FencePlaneRender::init(std::shared_ptr<GLFunctions> _gl)
{
    gl_ = _gl;

    planes_vertices_.reserve(bufCapacity_);
    planes_elements_.reserve((bufCapacity_ >> 2) * 6);

     // Vertex array

    gl_->GenVertexArrays(1, &hFencePlaneVAO_);
    NVXIO_CHECK_GL_ERROR();

    // Vertex buffer

    gl_->GenBuffers(1, &hFencePlaneVBO_);
    NVXIO_CHECK_GL_ERROR();
    gl_->GenBuffers(1, &hFencePlaneEA_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(hFencePlaneVAO_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ARRAY_BUFFER, hFencePlaneVBO_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, hFencePlaneEA_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BufferData(GL_ARRAY_BUFFER, planes_vertices_.capacity() * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    NVXIO_CHECK_GL_ERROR();
    gl_->VertexAttribPointer((GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->EnableVertexAttribArray(0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BufferData(GL_ELEMENT_ARRAY_BUFFER, planes_elements_.capacity() * sizeof(GLushort), NULL, GL_DYNAMIC_DRAW);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();

    // Shaders

    fencePlaneProgram_ = gl_->CreateProgram();
    NVXIO_CHECK_GL_ERROR();

    if (gl_->IsProgram(fencePlaneProgram_) == GL_FALSE)
    {
        NVXIO_PRINT("FencePlaneRender: error creating shader program");
        return false;
    }

    if (!compileProgram(gl_, fencePlaneProgram_,
                        fence_plane_render_shader_vs_code,
                        fence_plane_render_shader_fs_code))
        return false;

    return true;
}

void nvxio::FencePlaneRender::release()
{
    if (!gl_)
        return;

    gl_->DeleteBuffers(1, &hFencePlaneVBO_);
    NVXIO_CHECK_GL_ERROR();
    hFencePlaneVBO_ = 0;

    gl_->DeleteBuffers(1, &hFencePlaneEA_);
    NVXIO_CHECK_GL_ERROR();
    hFencePlaneEA_ = 0;

    gl_->DeleteVertexArrays(1, &hFencePlaneVAO_);
    NVXIO_CHECK_GL_ERROR();
    hFencePlaneVAO_ = 0;

    gl_->DeleteProgram(fencePlaneProgram_);
    NVXIO_CHECK_GL_ERROR();
    fencePlaneProgram_ = 0;
}

void nvxio::FencePlaneRender::render(vx_array planes, vx_matrix MVP, const nvxio::Render3D::PlaneStyle& style)
{
    NVXIO_ASSERT(style.maxDistance >= style.minDistance);

    updateArray(planes, MVP);
    renderArray(style);
}

void nvxio::FencePlaneRender::updateArray(vx_array planes, vx_matrix MVP)
{
    NVXIO_SAFE_CALL( vxReadMatrix(MVP, dataMVP_) );

    vx_enum type = 0;
    NVXIO_SAFE_CALL( vxQueryArray(planes, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &type, sizeof(type)) );
    NVXIO_ASSERT(type == NVX_TYPE_POINT3F);

    void * ptr = 0;
    vx_size size = 0, stride = 0;
    NVXIO_SAFE_CALL( vxQueryArray(planes, VX_ARRAY_ATTRIBUTE_NUMITEMS, &size, sizeof(size)) );

    planes_vertices_.clear();
    planes_elements_.clear();

    if (size == 0)
        return;

    NVXIO_ASSERT(size % 4 == 0);

    size = std::min(size, bufCapacity_);

    planes_elements_.reserve((size >> 2) * 6);
    planes_vertices_.reserve(size);

    NVXIO_SAFE_CALL( vxAccessArrayRange(planes, 0, size, &stride, &ptr, VX_READ_ONLY) );

    for (GLushort i = 0; i < static_cast<GLushort>(size); i += 4)
    {
        nvx_point3f_t pt1 = vxArrayItem(nvx_point3f_t, ptr, i + 0, stride);
        nvx_point3f_t pt2 = vxArrayItem(nvx_point3f_t, ptr, i + 1, stride);
        nvx_point3f_t pt3 = vxArrayItem(nvx_point3f_t, ptr, i + 2, stride);
        nvx_point3f_t pt4 = vxArrayItem(nvx_point3f_t, ptr, i + 3, stride);

        planes_vertices_.push_back(pt1.x);
        planes_vertices_.push_back(pt1.y);
        planes_vertices_.push_back(pt1.z);

        planes_vertices_.push_back(pt2.x);
        planes_vertices_.push_back(pt2.y);
        planes_vertices_.push_back(pt2.z);

        planes_vertices_.push_back(pt3.x);
        planes_vertices_.push_back(pt3.y);
        planes_vertices_.push_back(pt3.z);

        planes_vertices_.push_back(pt4.x);
        planes_vertices_.push_back(pt4.y);
        planes_vertices_.push_back(pt4.z);

        planes_elements_.push_back(i);
        planes_elements_.push_back(i + 1);
        planes_elements_.push_back(i + 2);
        planes_elements_.push_back(i);
        planes_elements_.push_back(i + 3);
        planes_elements_.push_back(i + 2);
    }

    NVXIO_SAFE_CALL(vxCommitArrayRange(planes, 0, 0, ptr));
}

void nvxio::FencePlaneRender::renderArray(const nvxio::Render3D::PlaneStyle& style)
{
    gl_->Enable(GL_DEPTH_TEST);
    NVXIO_CHECK_GL_ERROR();
    gl_->DepthFunc(GL_LESS);
    NVXIO_CHECK_GL_ERROR();
    gl_->Enable(GL_BLEND);
    NVXIO_CHECK_GL_ERROR();
    gl_->BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    NVXIO_CHECK_GL_ERROR();

#ifndef USE_GLES
    gl_->Enable(GL_VERTEX_PROGRAM_POINT_SIZE);
    NVXIO_CHECK_GL_ERROR();
#endif

    gl_->UseProgram(fencePlaneProgram_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindVertexArray(hFencePlaneVAO_);
    NVXIO_CHECK_GL_ERROR();

    gl_->UniformMatrix4fv(0, 1, GL_FALSE, dataMVP_);
    NVXIO_CHECK_GL_ERROR();
    gl_->Uniform1f(1, style.maxDistance);
    NVXIO_CHECK_GL_ERROR();
    gl_->Uniform1f(2, style.minDistance);
    NVXIO_CHECK_GL_ERROR();
    gl_->Uniform1f(3, 1.0f / (style.maxDistance - style.minDistance));
    NVXIO_CHECK_GL_ERROR();

    size_t nPointCount = planes_elements_.size();

    gl_->BindBuffer(GL_ARRAY_BUFFER, hFencePlaneVBO_);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, hFencePlaneEA_);
    NVXIO_CHECK_GL_ERROR();

    gl_->BufferSubData(GL_ARRAY_BUFFER, 0, planes_vertices_.size() * sizeof(GLfloat), (void *)&planes_vertices_[0]);
    NVXIO_CHECK_GL_ERROR();
    gl_->BufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, nPointCount * sizeof(GLushort), (void *)&planes_elements_[0]);
    NVXIO_CHECK_GL_ERROR();

    gl_->DrawElements(GL_TRIANGLES, static_cast<GLsizei>(nPointCount), GL_UNSIGNED_SHORT, NULL);
    NVXIO_CHECK_GL_ERROR();

    gl_->BindBuffer(GL_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    NVXIO_CHECK_GL_ERROR();
    gl_->BindVertexArray(0);
    NVXIO_CHECK_GL_ERROR();
    gl_->UseProgram(0);
    NVXIO_CHECK_GL_ERROR();

    gl_->Disable(GL_DEPTH_TEST);
    NVXIO_CHECK_GL_ERROR();
    gl_->Disable(GL_BLEND);
    NVXIO_CHECK_GL_ERROR();
#ifndef USE_GLES
    gl_->Disable(GL_VERTEX_PROGRAM_POINT_SIZE);
    NVXIO_CHECK_GL_ERROR();
#endif
}

#endif

#endif // USE_GUI
