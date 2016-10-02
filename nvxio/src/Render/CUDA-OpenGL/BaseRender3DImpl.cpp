/*
# Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
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

#include <Eigen/Dense>

#include <NVX/nvx.h>
#include <NVXIO/Utility.hpp>
#include <NVXIO/Application.hpp>

#include "Render/CUDA-OpenGL/BaseRender3DImpl.hpp"

//row-major storage order for compatibility with vx_matrix
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Matrix4f_rm;

#define MATRIX_4X4_FLOAT32_ASSERT(m) \
do \
{ \
    vx_enum type = 0; \
    NVXIO_SAFE_CALL( vxQueryMatrix(m, VX_MATRIX_ATTRIBUTE_TYPE, &type, sizeof(type)) ); \
    NVXIO_ASSERT(type == VX_TYPE_FLOAT32); \
    vx_size rows, cols; \
    NVXIO_SAFE_CALL( vxQueryMatrix(m, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows)) ); \
    NVXIO_SAFE_CALL( vxQueryMatrix(m, VX_MATRIX_ATTRIBUTE_COLUMNS, &cols, sizeof(cols)) ); \
    NVXIO_ASSERT(rows == 4 && cols == 4); \
} while (0)

//============================================================
// Utility stuff
//============================================================

static float toRadians(float degrees)
{
    return degrees * 0.017453292519f;
}

static void multiplyMatrix(vx_matrix m1, vx_matrix m2, vx_matrix res)
{
    MATRIX_4X4_FLOAT32_ASSERT(m1);
    MATRIX_4X4_FLOAT32_ASSERT(m2);
    MATRIX_4X4_FLOAT32_ASSERT(res);

    float m1Data[4*4];
    float m2Data[4*4];
    float resData[4*4];

    NVXIO_SAFE_CALL( vxReadMatrix(m1, m1Data) );
    NVXIO_SAFE_CALL( vxReadMatrix(m2, m2Data) );

    memset(resData, 0, sizeof(resData));

    for(int i=0; i<4; ++i)
        for(int j=0; j<4; ++j)
            for(int k=0; k<4; ++k)
            {
                resData[4*i+j] += m1Data[4*i+k] * m2Data[4*k+j];
            }

    NVXIO_SAFE_CALL( vxWriteMatrix(res, resData) );
}

static void calcProjectionMatrix(float fovY, float aspect, float zNear, float zFar, vx_matrix projection)
{
    MATRIX_4X4_FLOAT32_ASSERT(projection);

    float ctg = 1.f / tan( fovY/2 );

    float mat[4*4];
    memset(mat, 0, sizeof(mat));

    mat[0] = ctg / aspect;
    mat[5] = ctg;
    mat[10] = - (zFar + zNear) / (zFar - zNear);
    mat[11] = - 1.f;
    mat[14] = - 2.f * zFar * zNear / (zFar - zNear);

    NVXIO_SAFE_CALL( vxWriteMatrix(projection, mat) );
}

static Matrix4f_rm yawPitchRoll(float yaw, float pitch, float roll)
{
    float tmp_ch = cos(yaw);
    float tmp_sh = sin(yaw);
    float tmp_cp = cos(pitch);
    float tmp_sp = sin(pitch);
    float tmp_cb = cos(roll);
    float tmp_sb = sin(roll);

    Matrix4f_rm Result;
    Result(0,0) = tmp_ch * tmp_cb + tmp_sh * tmp_sp * tmp_sb;
    Result(0,1) = tmp_sb * tmp_cp;
    Result(0,2) = -tmp_sh * tmp_cb + tmp_ch * tmp_sp * tmp_sb;
    Result(0,3) = 0.0f;
    Result(1,0) = -tmp_ch * tmp_sb + tmp_sh * tmp_sp * tmp_cb;
    Result(1,1) = tmp_cb * tmp_cp;
    Result(1,2) = tmp_sb * tmp_sh + tmp_ch * tmp_sp * tmp_cb;
    Result(1,3) = 0.0f;
    Result(2,0) = tmp_sh * tmp_cp;
    Result(2,1) = -tmp_sp;
    Result(2,2) = tmp_ch * tmp_cp;
    Result(2,3) = 0.0f;
    Result(3,0) = 0.0f;
    Result(3,1) = 0.0f;
    Result(3,2) = 0.0f;
    Result(3,3) = 1.0f;
    return Result;
}

static void lookAt(vx_matrix view, const Eigen::Vector3f& eye, const Eigen::Vector3f& center, const Eigen::Vector3f& up)
{
    Eigen::Vector3f f = Eigen::Vector3f(center - eye).normalized();
    Eigen::Vector3f s = f.cross(up).normalized();
    Eigen::Vector3f u = s.cross(f);

    Matrix4f_rm result = Matrix4f_rm::Identity();
    result(0,0) = s(0);
    result(1,0) = s(1);
    result(2,0) = s(2);
    result(0,1) = u(0);
    result(1,1) = u(1);
    result(2,1) = u(2);
    result(0,2) = -f(0);
    result(1,2) = -f(1);
    result(2,2) = -f(2);
    result(3,0) = -s.dot(eye);
    result(3,1) = -u.dot(eye);
    result(3,2) = f.dot(eye);

    NVXIO_SAFE_CALL( vxWriteMatrix(view, result.data()) );
}

static void updateOrbitCamera(vx_matrix view, float xr, float yr, float distance, const Eigen::Vector3f & target)
{
    Matrix4f_rm R = yawPitchRoll(xr, yr, 0.0f);

    Eigen::Vector3f T(0.0f, 0.0f, -distance);

    Eigen::Vector4f T_ = R * Eigen::Vector4f(T(0), T(1), T(2), 0.0f);

    T = Eigen::Vector3f(T_(0), T_(1), T_(2));

    Eigen::Vector3f position = target + T;

    Eigen::Vector4f up_ = R*Eigen::Vector4f(0.0f, -1.0f, 0.0f, 0.0f);

    Eigen::Vector3f up(up_[0], up_[1], up_[2]);

    lookAt(view, position, target, up);
}

static void matrixSetEye(vx_matrix m)
{
    MATRIX_4X4_FLOAT32_ASSERT(m);

    static const float data[4*4] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    NVXIO_SAFE_CALL( vxWriteMatrix(m, data) );
}

//============================================================
// Callbacks and events
//============================================================

void nvxio::BaseRender3DImpl::enableDefaultKeyboardEventCallback()
{
    if (!useDefaultCallback_)
    {
        useDefaultCallback_ = true;

        fov_ = defaultFOV_;
        initMVP();
    }
}

void nvxio::BaseRender3DImpl::disableDefaultKeyboardEventCallback()
{
    if(useDefaultCallback_)
    {
        useDefaultCallback_ = false;

        fov_ = defaultFOV_;
        orbitCameraParams_.setDefault();
        initMVP();
    }
}

bool nvxio::BaseRender3DImpl::useDefaultKeyboardEventCallback()
{
    return useDefaultCallback_;
}

void nvxio::BaseRender3DImpl::setOnKeyboardEventCallback(nvxio::Render3D::OnKeyboardEventCallback callback, void* context)
{
    keyboardCallback_ = callback;
    keyboardCallbackContext_ = context;

    glfwSetKeyCallback(window_, keyboardCallback);
}

void nvxio::BaseRender3DImpl::keyboardCallbackDefault(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    nvxio::BaseRender3DImpl* impl = static_cast<nvxio::BaseRender3DImpl*>(glfwGetWindowUserPointer(window));

    static const float stepAngle = toRadians(4); // in radians
    static const float stepR = 1;

    if (action == GLFW_PRESS)
    {
        switch (key)
        {
            case GLFW_KEY_ESCAPE:
            {
                glfwSetWindowShouldClose(window, 1);
                break;
            }
            case GLFW_KEY_MINUS:
            {
                impl->orbitCameraParams_.R += stepR;
                break;
            }
            case GLFW_KEY_EQUAL:
            {
                impl->orbitCameraParams_.R -= stepR;
                break;
            }

            case GLFW_KEY_A:
            {
                impl->orbitCameraParams_.xr -= stepAngle;
                break;
            }
            case GLFW_KEY_D:
            {
                impl->orbitCameraParams_.xr += stepAngle;
                break;
            }
            case GLFW_KEY_W:
            {
                impl->orbitCameraParams_.yr += stepAngle;
                break;
            }
            case GLFW_KEY_S:
            {
                impl->orbitCameraParams_.yr -= stepAngle;
                break;
            }
        }
        impl->updateView();
    }
}

void nvxio::BaseRender3DImpl::keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    nvxio::BaseRender3DImpl* impl = static_cast<nvxio::BaseRender3DImpl*>(glfwGetWindowUserPointer(window));

    if (action == GLFW_PRESS)
    {
        double x, y;
        glfwGetCursorPos(window, &x, &y);

        if (key == GLFW_KEY_ESCAPE)
            key = 27;

        if(impl->keyboardCallback_)
            (impl->keyboardCallback_)(impl->keyboardCallbackContext_, tolower(key),
                                      static_cast<vx_uint32>(x),
                                      static_cast<vx_uint32>(y));

        if (impl->useDefaultKeyboardEventCallback())
        {
            keyboardCallbackDefault(window, key, scancode, action, mods);
        }
    }
}

void nvxio::BaseRender3DImpl::setOnMouseEventCallback(OnMouseEventCallback callback, void* context)
{
    mouseCallback_ = callback;
    mouseCallbackContext_ = context;

    glfwSetMouseButtonCallback(window_, mouse_button);
    glfwSetCursorPosCallback(window_, cursor_pos);
}

void nvxio::BaseRender3DImpl::mouse_button(GLFWwindow* window, int button, int action, int mods)
{
    (void)mods;
    BaseRender3DImpl* impl = static_cast<BaseRender3DImpl*>(glfwGetWindowUserPointer(window));

    if (impl->mouseCallback_)
    {
        Render3D::MouseButtonEvent event = Render3D::MouseMove;

        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            if (action == GLFW_RELEASE)
                event = Render3D::LeftButtonUp;
            else
                event = Render3D::LeftButtonDown;
        }
        if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            if (action == GLFW_RELEASE)
                event = Render3D::RightButtonUp;
            else
                event = Render3D::RightButtonDown;
        }
        if (button == GLFW_MOUSE_BUTTON_MIDDLE)
        {
            if (action == GLFW_RELEASE)
                event = Render3D::MiddleButtonUp;
            else
                event = Render3D::MiddleButtonDown;
        }

        double x, y;
        glfwGetCursorPos(window, &x, &y);
        (impl->mouseCallback_)(impl->mouseCallbackContext_, event,
                               static_cast<vx_uint32>(x),
                               static_cast<vx_uint32>(y));
    }
}

void nvxio::BaseRender3DImpl::cursor_pos(GLFWwindow* window, double x, double y)
{
    BaseRender3DImpl* impl = static_cast<BaseRender3DImpl*>(glfwGetWindowUserPointer(window));

    if (impl->mouseCallback_)
        (impl->mouseCallback_)(impl->mouseCallbackContext_, Render3D::MouseMove,
                               static_cast<vx_uint32>(x),
                               static_cast<vx_uint32>(y));
}

//============================================================
// Orbit Camera Params
//============================================================

nvxio::BaseRender3DImpl::OrbitCameraParams::OrbitCameraParams(float R_min_, float R_max_): R_min(R_min_), R_max(R_max_)
{
    setDefault();
}

void nvxio::BaseRender3DImpl::OrbitCameraParams::applyConstraints()
{
    if (R < R_min) R = R_min;
    else if (R > R_max) R = R_max;

    if (yr > 2 * nvxio::PI_F) yr -= 2 * nvxio::PI_F;
    else if (yr < 0) yr += 2 * nvxio::PI_F;

    if (xr > 2 * nvxio::PI_F) xr -= 2 * nvxio::PI_F;
    else if (xr < 0) xr += 2 * nvxio::PI_F;
}

void nvxio::BaseRender3DImpl::OrbitCameraParams::setDefault()
{
    R = R_min;
    xr = 0;
    yr = 0;
}

//============================================================
// Setters and getters
//============================================================

void nvxio::BaseRender3DImpl::setDefaultFOV(float fov)
{
    defaultFOV_ = std::max(std::min(fov, 180.0f), 0.0f);
}

void nvxio::BaseRender3DImpl::setViewMatrix(vx_matrix view)
{
    MATRIX_4X4_FLOAT32_ASSERT(view);

    float data[4*4];
    NVXIO_SAFE_CALL( vxReadMatrix(view, data) );
    NVXIO_SAFE_CALL( vxWriteMatrix(view_, data) );
}

void nvxio::BaseRender3DImpl::setProjectionMatrix(vx_matrix projection)
{
    MATRIX_4X4_FLOAT32_ASSERT(projection);

    float data[4*4];
    NVXIO_SAFE_CALL( vxReadMatrix(projection, data) );
    NVXIO_SAFE_CALL( vxWriteMatrix(projection_, data) );
}

void nvxio::BaseRender3DImpl::getViewMatrix(vx_matrix view) const
{
    MATRIX_4X4_FLOAT32_ASSERT(view);

    float data[4*4];
    NVXIO_SAFE_CALL( vxReadMatrix(view_, data) );
    NVXIO_SAFE_CALL( vxWriteMatrix(view, data) );
}

void nvxio::BaseRender3DImpl::getProjectionMatrix(vx_matrix projection) const
{
    MATRIX_4X4_FLOAT32_ASSERT(projection);

    float data[4*4];
    NVXIO_SAFE_CALL( vxReadMatrix(projection_, data) );
    NVXIO_SAFE_CALL( vxWriteMatrix(projection, data) );
}

void nvxio::BaseRender3DImpl::setModelMatrix(vx_matrix model)
{
    MATRIX_4X4_FLOAT32_ASSERT(model);

    float data[4*4];
    NVXIO_SAFE_CALL( vxReadMatrix(model, data) );
    NVXIO_SAFE_CALL( vxWriteMatrix(model_, data) );
}

//============================================================
// Put objects to OpenGL framebuffer
//============================================================

void nvxio::BaseRender3DImpl::putPointCloud(vx_array points, vx_matrix model, const PointCloudStyle& style)
{
    setModelMatrix(model);

    // MVP = model * view * projection
    vx_matrix MVP = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 4, 4);
    NVXIO_CHECK_REFERENCE(MVP);

    multiplyMatrix(model_, view_, MVP);
    multiplyMatrix(MVP, projection_,  MVP);

    // setup OpenGL context
    glfwMakeContextCurrent(window_);

    // invoke
    pointCloudRender_.render(points, MVP, style);

    NVXIO_SAFE_CALL( vxReleaseMatrix(&MVP) );
}

void nvxio::BaseRender3DImpl::putPlanes(vx_array planes, vx_matrix model, const nvxio::Render3D::PlaneStyle& style)
{
    setModelMatrix(model);

    // MVP = model * view * projection
    vx_matrix MVP = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 4, 4);
    NVXIO_CHECK_REFERENCE(MVP);

    multiplyMatrix(model_, view_, MVP);
    multiplyMatrix(MVP, projection_,  MVP);

    // Setup OpenGL context
    glfwMakeContextCurrent(window_);

    // Invoke
    fencePlaneRender_.render(planes, MVP, style);

    NVXIO_SAFE_CALL( vxReleaseMatrix(&MVP) );
}

void nvxio::BaseRender3DImpl::putImage(vx_image image)
{
    vx_uint32 imageWidth = 0, imageHeight = 0;

    NVXIO_SAFE_CALL( vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &imageWidth, sizeof(imageWidth)) );
    NVXIO_SAFE_CALL( vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &imageHeight, sizeof(imageHeight)) );

    // calculate actual ScaleRatio that will be applied to other primitives like lines, circles, etc.
    float widthRatio = static_cast<float>(windowWidth_) / imageWidth;
    float heightRatio = static_cast<float>(windowHeight_) / imageHeight;
    scaleRatio_ = std::min(widthRatio, heightRatio);

    glfwMakeContextCurrent(window_);
    imageRender_.render(image, imageWidth, imageHeight);
}

void nvxio::BaseRender3DImpl::putText(const std::string& text, const nvxio::Render::TextBoxStyle& style)
{
    glfwMakeContextCurrent(window_);
    textRender_.render(text, style, windowWidth_, windowHeight_, scaleRatio_);
}

//============================================================
// Initialize and deinitialize
//============================================================

nvxio::BaseRender3DImpl::BaseRender3DImpl(vx_context context):
    nvxio::Render3D(nvxio::Render3D::BASE_RENDER_3D, "BaseOpenGlRender3D"),
    context_(context),
    gl_(NULL),
    model_(0),
    view_(0),
    projection_(0),
    window_(0),
    keyboardCallback_(0),
    keyboardCallbackContext_(0),
    useDefaultCallback_(true),
    windowWidth_(0),
    windowHeight_(0),
    defaultFOV_(70),// in degrees
    Z_NEAR_(0.01f),
    Z_FAR_(500.0f),
    fov_(defaultFOV_),
    orbitCameraParams_(1e-6f, 100.f)
{
    model_ = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 4, 4);
    NVXIO_CHECK_REFERENCE( model_ );

    view_ = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 4, 4);
    NVXIO_CHECK_REFERENCE( view_ );

    projection_ = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 4, 4);
    NVXIO_CHECK_REFERENCE( projection_ );
}

void nvxio::BaseRender3DImpl::initMVP()
{
    matrixSetEye(model_);

    const static float viewData[4*4] = {1, 0, 0, 0,
                                        0, -1, 0, 0,
                                        0, 0, -1, 0,
                                        0, 0, 0, 1};

    NVXIO_SAFE_CALL( vxWriteMatrix(view_, viewData) );

    calcProjectionMatrix(toRadians(fov_), (float)windowWidth_ / windowHeight_, Z_NEAR_, Z_FAR_, projection_);
}

void nvxio::BaseRender3DImpl::updateView()
{
    orbitCameraParams_.applyConstraints();
    updateOrbitCamera(view_, orbitCameraParams_.xr, orbitCameraParams_.yr, orbitCameraParams_.R, Eigen::Vector3f(0, 0, 0));
}

bool nvxio::BaseRender3DImpl::open(int xPos, int yPos, vx_uint32 windowWidth, vx_uint32 windowHeight, const std::string& windowTitle)
{
    windowWidth_ = windowWidth;
    windowHeight_ = windowHeight;

    if (!nvxio::Application::get().initGui())
    {
        NVXIO_PRINT("Error: Failed to init GUI");
        return false;
    }

    bool result = initWindow(xPos, yPos, windowWidth, windowHeight, windowTitle.c_str());

    fov_ = defaultFOV_;
    initMVP();

    return result;
}

bool nvxio::BaseRender3DImpl::initWindow(int xpos, int ypos, vx_uint32 width, vx_uint32 height, const std::string& wintitle)
{
    int count = 0;
    GLFWmonitor ** monitors = glfwGetMonitors(&count);
    if (count == 0)
    {
        NVXIO_THROW_EXCEPTION("GLFW: no monitors found");
    }

    int maxPixels = 0;
    const GLFWvidmode* mode = NULL;

    for (int i = 0; i < count; ++i)
    {
        const GLFWvidmode* currentMode = glfwGetVideoMode(monitors[i]);
        int currentPixels = currentMode->width * currentMode->height;

        if (maxPixels < currentPixels)
        {
            mode = currentMode;
            maxPixels = currentPixels;
        }
    }

    vx_uint32 cur_width = 0;
    vx_uint32 cur_height = 0;
    if ((width <= (vx_uint32)mode->width) && (height <= (vx_uint32)mode->height))
    {
        cur_width = width;
        cur_height = height;
    }
    else
    {
        float widthRatio = static_cast<vx_float32>(mode->width) / width;
        float heightRatio = static_cast<vx_float32>(mode->height) / height;
        float scaleRatio = std::min(widthRatio, heightRatio);
        cur_width = static_cast<vx_uint32>(scaleRatio * width);
        cur_height = static_cast<vx_uint32>(scaleRatio * height);
    }

    glfwWindowHint(GLFW_RESIZABLE, 0);
#ifdef USE_GLES
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif
    window_ = glfwCreateWindow(cur_width, cur_height,
                               wintitle.c_str(),
                               NULL, NULL);
    if (!window_)
    {
        NVXIO_PRINT("Error: Failed to create GLFW window");
        return false;
    }

    glfwSetWindowUserPointer(window_, this);
    glfwSetWindowPos(window_, xpos, ypos);
    glfwSetInputMode(window_, GLFW_STICKY_KEYS, GL_TRUE);
    glfwMakeContextCurrent(window_);

    // Initialize OpenGL renders

    if (!gl_)
    {
        gl_ = std::make_shared<nvxio::GLFunctions>();
        nvxio::loadGLFunctions(gl_.get());
    }

    if (!imageRender_.init(gl_, width, height))
        return false;

    if (!textRender_.init(gl_))
        return false;

    if (!pointCloudRender_.init(gl_))
        return false;

    if (!fencePlaneRender_.init(gl_))
        return false;

    return true;
}

bool nvxio::BaseRender3DImpl::flush()
{
    glfwMakeContextCurrent(window_);

    if (glfwWindowShouldClose(window_))
        return false;

    glfwSwapBuffers(window_);
    glfwPollEvents();

    gl_->ClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    NVXIO_CHECK_GL_ERROR();
    gl_->Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    NVXIO_CHECK_GL_ERROR();

    return true;
}

void nvxio::BaseRender3DImpl::close()
{
    glfwMakeContextCurrent(window_);

    textRender_.release();
    imageRender_.release();
    pointCloudRender_.release();
    fencePlaneRender_.release();

    if (model_)
    {
        vxReleaseMatrix(&model_);
        model_ = NULL;
    }

    if (view_)
    {
        vxReleaseMatrix(&view_);
        view_ = NULL;
    }

    if (projection_)
    {
        vxReleaseMatrix(&projection_);
        projection_ = NULL;
    }

    if (window_)
    {
        glfwDestroyWindow(window_);
        window_ = NULL;
    }
}

nvxio::BaseRender3DImpl::~BaseRender3DImpl()
{
    close();
}

#endif
