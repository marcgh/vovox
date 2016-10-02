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

#ifndef OPENGL_BASIC_RENDERS_HPP
#define OPENGL_BASIC_RENDERS_HPP

#include <vector>
#include <sstream>
#include <stdexcept>

#if USE_GUI
#include <NVXIO/OpenGL.hpp>
#endif

#include <ft2build.h>
FT_BEGIN_HEADER
#include FT_FREETYPE_H
FT_END_HEADER

#include <cuda_runtime.h>

#include <NVXIO/Render.hpp>

#ifndef __ANDROID__
#include <NVXIO/Render3D.hpp>
#endif

#include <NVXIO/Utility.hpp>

#include <VX/vx.h>

#include "Private/LogUtils.hpp"

#ifndef NDEBUG
    void __checkGlError(std::shared_ptr<nvxio::GLFunctions> gl_, const char* file, int line);

    #define NVXIO_CHECK_GL_ERROR() __checkGlError(gl_, __FILE__, __LINE__)
#else
    #define NVXIO_CHECK_GL_ERROR() /* nothing */
#endif

namespace nvxio {

class ImageRender
{
public:
    ImageRender();

    bool init(std::shared_ptr<GLFunctions> _gl, vx_uint32 wndWidth, vx_uint32 wndHeight);
    void release();

    void render(vx_image image, vx_uint32 imageWidth, vx_uint32 imageHeight);

private:
    void updateTexture(vx_image image, vx_uint32 imageWidth, vx_uint32 imageHeight);
    void renderTexture();

    std::shared_ptr<GLFunctions> gl_;
    GLuint wndWidth_, wndHeight_;

    GLuint tex_[3];
    cudaGraphicsResource_t res_[3];
    GLuint vao_;
    GLuint vbo_;

    GLuint pipeline_[3], fragmentProgram_[3];
    GLint index_;
    GLfloat scaleUniformX_, scaleUniformY_;
};

class RectangleRender
{
public:
    RectangleRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    void render(const vx_rectangle_t& location, const nvxio::Render::DetectedObjectStyle& style, vx_uint32 width, vx_uint32 height, vx_float32 scale);

private:
    void updateArray(const vx_rectangle_t& location, vx_uint32 width, vx_uint32 height, vx_float32 scale);
    void renderArray(const nvxio::Render::DetectedObjectStyle& style);

    std::shared_ptr<GLFunctions> gl_;

    GLuint vbo_;
    GLuint vao_;
    GLuint program_;
};

class FeaturesRender
{
public:
    FeaturesRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    void render(vx_array location, const nvxio::Render::FeatureStyle& style, vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio);
    void render(vx_array location, vx_array styles, vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio);

private:
    void updateArray(vx_size start_x, vx_size end_x, vx_array location, vx_array styles);
    void renderArray(vx_size num_items, vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio,
                     const nvxio::Render::FeatureStyle& style);

    std::shared_ptr<GLFunctions> gl_;

    size_t bufCapacity_;
    GLuint vbo_, vboStyles_;
    GLuint vao_;
    cudaGraphicsResource_t res_, resStyles_;
    GLuint pipeline_;
    GLuint vertexShaderPoints_, vertexShaderKeyPoints_;
    GLuint vertexShaderPointsPerFeature_, vertexShaderKeyPointsPerFeature_;
    GLuint fragmentShader_, fragmentShaderPerFeature_;

    vx_enum currentFeatureType_;
    bool perFeatureStyle_;
};

class LinesRender
{
public:
    LinesRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    void render(vx_array lines, const nvxio::Render::LineStyle& style, vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio);

private:
    void updateArray(vx_size start_x, vx_size end_x, vx_array lines);
    void renderArray(vx_size num_items, const nvxio::Render::LineStyle& style,
                     vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio);

    std::shared_ptr<GLFunctions> gl_;
    cudaGraphicsResource_t res_;
    size_t bufCapacity_;

    GLuint vbo_;
    GLuint vao_;
    GLuint program_;
};

class ArrowsRender
{
public:
    ArrowsRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    void render(vx_array old_points, vx_array new_points, const nvxio::Render::LineStyle& line_style,
                vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio);

private:
    void updateLinesArray(vx_size start_x, vx_size end_x,
                          vx_array old_points, vx_array new_points,
                          vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio);

    void renderArray(vx_size num_items, const nvxio::Render::LineStyle& style);

    std::shared_ptr<GLFunctions> gl_;
    cudaGraphicsResource_t resOld_, resNew_;
    size_t bufCapacity_;

    GLuint vbo_, ssboOld_, ssboNew_;
    GLuint vao_;
    GLuint program_,
        computeShaderProgramPoints_,
        computeShaderProgramVxKeyPoints_,
        computeShaderProgramNvxKeyPoints_;

    vx_enum featureType_;
};

class MotionFieldRender
{
public:
    MotionFieldRender();

    bool init(std::shared_ptr<GLFunctions> _gl, vx_uint32 width, vx_uint32 height);
    void release();

    void render(vx_image field, const nvxio::Render::MotionFieldStyle& style, vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio);

private:
    void updateArray(vx_image field, vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio);
    void renderArray(const nvxio::Render::MotionFieldStyle& style);

    std::shared_ptr<GLFunctions> gl_;

    size_t capacity_, numPoints_;
    GLuint ssbo_;
    GLuint vao_;
    cudaGraphicsResource_t res_;
    GLuint program_;

    GLuint computeShaderProgram_;
    GLuint ssboTex_;
};

class CirclesRender
{
public:
    CirclesRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    void render(vx_array circles, const nvxio::Render::CircleStyle& style, vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio);

private:
    void updateArray(vx_array circles);
    void renderArray(const nvxio::Render::CircleStyle& style,
                     vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio);

    std::shared_ptr<GLFunctions> gl_;

    std::vector<nvx_point4f_t> points_;
    size_t bufCapacity_;
    GLuint vbo_;
    GLuint vao_;
    GLuint program_;
};

class TextRender
{
public:
    TextRender();
    ~TextRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    void render(const std::string& text, const nvxio::Render::TextBoxStyle& style,
                vx_uint32 width, vx_uint32 height, vx_float32 scaleRatio);

private:
    std::shared_ptr<GLFunctions> gl_;

    FT_Library ft_;
    FT_Face face_;


    GLuint programBg_;

    GLuint program_;

    GLuint tex_;

    size_t bufCapacity_;
    GLuint vbo_;
    GLuint vboEA_;
    GLuint vao_;

    GLuint bgVbo_;
    GLuint bgVao_;

    struct CharacterInfo
    {
        float ax; // advance.x
        float ay; // advance.y

        float bw; // bitmap.width;
        float bh; // bitmap.rows;

        float bl; // bitmap_left;
        float bt; // bitmap_top;

        float tx; // x offset of glyph in texture coordinates
    } c[128];
    int atlasWidth_, atlasHeight_;

    std::vector<nvx_point4f_t> points_;
    std::vector<GLushort> elements_;
};

#ifndef __ANDROID__

class PointCloudRender
{
public:
    PointCloudRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    void render(vx_array points, vx_matrix MVP, const nvxio::Render3D::PointCloudStyle& style);

private:
    void updateArray(vx_matrix MVP);
    void renderArray(vx_array points, const nvxio::Render3D::PointCloudStyle& style);

    std::shared_ptr<GLFunctions> gl_;

    GLuint pointCloudProgram_;
    GLuint hPointCloudVBO_;
    GLuint hPointCloudVAO_;

    size_t bufCapacity_;

    float dataMVP_[4 * 4];
};

class FencePlaneRender
{
public:
    FencePlaneRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    void render(vx_array planes, vx_matrix MVP, const nvxio::Render3D::PlaneStyle & style);

private:
    void updateArray(vx_array planes, vx_matrix MVP);
    void renderArray(const nvxio::Render3D::PlaneStyle& style);

    std::shared_ptr<GLFunctions> gl_;

    GLuint fencePlaneProgram_;
    GLuint hFencePlaneVBO_;
    GLuint hFencePlaneEA_;
    GLuint hFencePlaneVAO_;

    size_t bufCapacity_;

    float dataMVP_[4 * 4];

    std::vector<GLfloat> planes_vertices_;
    std::vector<GLushort> planes_elements_;
};

#endif

}

#endif // OPENGL_BASIC_RENDERS_HPP
