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

#include <NVXIO/Utility.hpp>

#include "EventLogger.hpp"

#include <vector>
#include <string>
#include <algorithm>

#ifdef USE_OPENCV
# include <opencv2/imgproc/imgproc.hpp>
# include <opencv2/highgui/highgui.hpp>
#endif

#include "NVXIO/Utility.hpp"

static bool operator<(const nvx_point3f_t& a, const nvx_point3f_t& b)
{
    if (a.x == b.x)
    {
        if (a.y == b.y)
            return a.z < b.z;
        else
            return a.y < b.y;
    }
    else
        return a.x < b.x;
}

static bool operator<(const nvx_point4f_t& a, const nvx_point4f_t& b)
{
    if (a.x == b.x)
    {
        if (a.y == b.y)
        {
            if (a.z == b.z)
                return a.w < b.w;
            else
                return a.z < b.z;
        }
        else
            return a.y < b.y;
    }
    else
        return a.x < b.x;
}

namespace nvxio
{

EventLogger::EventLogger(bool _writeSrc):
    writeSrc(_writeSrc),
    handle(NULL),
    frameCounter(-1),
    keyBoardCallback(NULL),
    mouseCallback(NULL)
{
}

bool EventLogger::init(const std::string &path)
{
    if (handle != NULL)
    {
        // some log has been already opened
        return true;
    }

    size_t dot = path.find_last_of('.');
    std::string baseName = path.substr(0, dot);
    std::string ext = path.substr(dot, std::string::npos);

    handle = fopen(path.c_str(), "rt");
    if (handle != NULL)
    {
        // file with this name already exists that means that render was reopened
        int logNameIdx = 0;
        do
        {
            fclose(handle);
            logNameIdx++;
            handle = fopen((baseName+std::to_string(logNameIdx)+ext).c_str(), "rt");
        }
        while (handle);

        srcImageFilePattern = baseName + std::to_string(logNameIdx) + "_src_%05d.png";
        handle = fopen((baseName+std::to_string(logNameIdx)+ext).c_str(), "wt");
    }
    else
    {
        srcImageFilePattern = baseName + "_src_%05d.png";
        handle = fopen(path.c_str(), "wt");
    }

    frameCounter = 0;

    return handle != NULL;
}

void EventLogger::setEfficientRender(std::unique_ptr<Render> render)
{
    efficientRender = std::move(render);
    if (efficientRender)
    {
        efficientRender->setOnKeyboardEventCallback(keyboard, this);
        efficientRender->setOnMouseEventCallback(mouse, this);
    }
}

void EventLogger::final()
{
    if (handle != NULL)
        fclose(handle);

    frameCounter = -1;
}

EventLogger::~EventLogger()
{
    final();
}

void EventLogger::keyboard(void* context, vx_char key, vx_uint32 x, vx_uint32 y)
{
    EventLogger* self = (EventLogger*)context;
    if (context == NULL)
        return;
    if (self->handle != NULL)
        fprintf(self->handle, "%d: keyboard (%d,%d,%d)\n", self->frameCounter, key, x, y);

    if (self->keyBoardCallback)
        self->keyBoardCallback(self->keyboardCallbackContext, key, x, y);
}

void EventLogger::mouse(void* context, Render::MouseButtonEvent event, vx_uint32 x, vx_uint32 y)
{
    EventLogger* self = (EventLogger*)context;
    if (context == NULL)
        return;

    if (self->handle != NULL)
        fprintf(self->handle, "%d: mouse (%d,%d,%d)\n", self->frameCounter, (int)event, x, y);

    if (self->mouseCallback)
        self->mouseCallback(self->mouseCallbackContext, event, x, y);
}

void EventLogger::putTextViewport(const std::string &text, const Render::TextBoxStyle &style)
{
    if (handle != NULL)
    {
        std::string filtered = "";
        size_t curr_pos = 0;
        size_t prev_pos = 0;
        while((curr_pos = text.find("\n", prev_pos)) != std::string::npos)
        {
            filtered += text.substr(prev_pos, curr_pos-prev_pos);
            filtered += "\\n";
            prev_pos = curr_pos+1;
        }

        filtered += text.substr(prev_pos, std::string::npos);

        fprintf(handle, "%d: textBox(color(%d,%d,%d,%d), bkcolor(%d,%d,%d,%d), origin(%d,%d), \"%s\")\n",
                frameCounter,
                style.color[0], style.color[1], style.color[2], style.color[3],
                style.bgcolor[0], style.bgcolor[1], style.bgcolor[2], style.bgcolor[3],
                style.origin.x, style.origin.y,
                filtered.c_str()
               );
    }

    if (efficientRender != NULL)
        efficientRender->putTextViewport(text, style);
}

void EventLogger::putImage(vx_image image)
{
    if (handle != NULL)
    {
        vx_df_image format = 0;
        vx_uint32 width;
        vx_uint32 height;

        NVXIO_SAFE_CALL( vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
        NVXIO_SAFE_CALL( vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)) );
        NVXIO_SAFE_CALL( vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)) );

        fprintf(handle, "%d: image(%d, %dx%d)\n", frameCounter, format, width, height);

#ifdef USE_OPENCV
        if (writeSrc)
        {
            vx_imagepatch_addressing_t addr;
            void *dataPtr = NULL;
            vx_rectangle_t rect;
            int matType;

            rect.start_x = 0;
            rect.start_y = 0;
            rect.end_x = width;
            rect.end_y = height;

            if (format == VX_DF_IMAGE_RGBX)
            {
                matType = CV_8UC4;
            }
            else if (format == VX_DF_IMAGE_RGB)
            {
                matType = CV_8UC3;
            }
            else if (format == VX_DF_IMAGE_U8)
            {
                matType = CV_8UC1;
            }
            else
            {
                char sFormat[sizeof(format)+1];
                memcpy(sFormat, &format, sizeof(format));
                sFormat[sizeof(format)] = '\0';
                NVXIO_THROW_EXCEPTION( "Dumping frames in format " << sFormat << " is not supported" );
                return;
            }

            NVXIO_SAFE_CALL( vxAccessImagePatch(image, &rect, 0, &addr, &dataPtr, VX_READ_ONLY) );

            {
                cv::Mat srcFrame(addr.dim_y, addr.dim_x, matType, dataPtr, addr.stride_y);
                cv::Mat normalizedFrame;
                if (format == VX_DF_IMAGE_RGBX)
                {
                    cv::cvtColor(srcFrame, normalizedFrame, CV_RGBA2BGRA);
                }
                else if (format == VX_DF_IMAGE_RGB)
                {
                    cv::cvtColor(srcFrame, normalizedFrame, CV_RGB2BGR);
                }
                else
                {
                    normalizedFrame = srcFrame;
                }
                std::string name = cv::format(srcImageFilePattern.c_str(), frameCounter);

                if (!cv::imwrite(name, normalizedFrame))
                {
                    fprintf(stderr, "Cannot write frame to %s\n", name.c_str());
                }

            }

            NVXIO_SAFE_CALL( vxCommitImagePatch(image, NULL, 0, &addr, dataPtr) );
        }
#endif // USE_OPENCV
    }

    if (efficientRender != NULL)
        efficientRender->putImage(image);
}

void EventLogger::putObjectLocation(const vx_rectangle_t &location, const Render::DetectedObjectStyle &style)
{
    if (handle != NULL)
    {
        fprintf(handle, "%d: object(color(%d,%d,%d%d), location(%d,%d,%d,%d), \"%s\")\n",
                frameCounter,
                style.color[0], style.color[1], style.color[2], style.color[3],
                location.start_x, location.start_y, location.end_x, location.end_y,
                style.label.c_str()
                );
    }

    if (efficientRender)
        efficientRender->putObjectLocation(location, style);
}

void EventLogger::putFeatures(vx_array location, const Render::FeatureStyle &style)
{
    if (handle != NULL)
    {
        vx_enum item_type = 0;
        NVXIO_SAFE_CALL( vxQueryArray(location, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &item_type, sizeof(item_type)) );
        NVXIO_ASSERT( (item_type == VX_TYPE_KEYPOINT) || (item_type == NVX_TYPE_POINT2F) || (item_type == NVX_TYPE_KEYPOINTF) );

        vx_size size;
        NVXIO_SAFE_CALL( vxQueryArray(location, VX_ARRAY_ATTRIBUTE_NUMITEMS, &size, sizeof(size)) );
        fprintf(handle, "%d: features(color(%d,%d,%d,%d), %lu",
                frameCounter,
                style.color[0], style.color[1], style.color[2], style.color[3],
                static_cast<unsigned long>(size));

        if (size != 0)
        {
            vx_size stride;
            void * featureData = NULL;
            NVXIO_SAFE_CALL( vxAccessArrayRange(location, 0, size, &stride,
                                                (void**)&featureData, VX_READ_ONLY) );

            if (item_type == VX_TYPE_KEYPOINT)
            {
                for (vx_size i = 0; i < size; i++)
                {
                    vx_keypoint_t feature = vxArrayItem(vx_keypoint_t, featureData, i, stride);
                    fprintf(handle, ",ftr(%d,%d)", feature.x, feature.y);
                }
            }
            else if (item_type == NVX_TYPE_POINT2F)
            {
                for (vx_size i = 0; i < size; i++)
                {
                    nvx_point2f_t feature = vxArrayItem(nvx_point2f_t, featureData, i, stride);
                    fprintf(handle, ",ftr(%.1f,%.1f)", feature.x, feature.y);
                }
            }
            else if (item_type == NVX_TYPE_KEYPOINTF)
            {
                for (vx_size i = 0; i < size; i++)
                {
                    nvx_keypointf_t feature = vxArrayItem(nvx_keypointf_t, featureData, i, stride);
                    fprintf(handle, ",ftr(%.1f,%.1f)", feature.x, feature.y);
                }
            }

            NVXIO_SAFE_CALL( vxCommitArrayRange(location, 0, size, featureData) );
        }

        fprintf(handle, ")\n");
    }

    if (efficientRender)
        efficientRender->putFeatures(location, style);
}

void EventLogger::putFeatures(vx_array location, vx_array styles)
{
    if (handle != NULL)
    {
        vx_enum item_type = 0;
        NVXIO_SAFE_CALL( vxQueryArray(location, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &item_type, sizeof(item_type)) );
        NVXIO_ASSERT( (item_type == VX_TYPE_KEYPOINT) || (item_type == NVX_TYPE_POINT2F) || (item_type == NVX_TYPE_KEYPOINTF) );

        vx_size size = 0, styleSize = 0;
        NVXIO_SAFE_CALL( vxQueryArray(location, VX_ARRAY_ATTRIBUTE_NUMITEMS, &size, sizeof(size)) );
        NVXIO_SAFE_CALL( vxQueryArray(styles, VX_ARRAY_ATTRIBUTE_NUMITEMS, &styleSize, sizeof(styleSize)) );
        NVXIO_ASSERT( styleSize == size );

        fprintf(handle, "%d: features(%lu",
                frameCounter, static_cast<unsigned long>(size));

        if (size != 0)
        {
            vx_size stride = 0, styleStride = 0;
            void * featureData = NULL, * styleData = NULL;
            NVXIO_SAFE_CALL( vxAccessArrayRange(location, 0, size, &stride,
                                                (void**)&featureData, VX_READ_ONLY) );
            NVXIO_SAFE_CALL( vxAccessArrayRange(styles, 0, styleSize, &styleStride,
                                                (void**)&styleData, VX_READ_ONLY) );

            if (item_type == VX_TYPE_KEYPOINT)
            {
                for (vx_size i = 0; i < size; i++)
                {
                    vx_keypoint_t feature = vxArrayItem(vx_keypoint_t, featureData, i, stride);
                    Render::FeatureStyle style = vxArrayItem(Render::FeatureStyle, styleData, i, styleStride);

                    fprintf(handle, ",ftr(%d,%d,%d,%d,%d,%d)", feature.x, feature.y,
                            style.color[0], style.color[1], style.color[2], style.color[3]);
                }
            }
            else if (item_type == NVX_TYPE_POINT2F)
            {
                for (vx_size i = 0; i < size; i++)
                {
                    nvx_point2f_t feature = vxArrayItem(nvx_point2f_t, featureData, i, stride);
                    Render::FeatureStyle style = vxArrayItem(Render::FeatureStyle, styleData, i, styleStride);

                    fprintf(handle, ",ftr(%.1f,%.1f,%d,%d,%d,%d)", feature.x, feature.y,
                            style.color[0], style.color[1], style.color[2], style.color[3]);
                }
            }
            else if (item_type == NVX_TYPE_KEYPOINTF)
            {
                for (vx_size i = 0; i < size; i++)
                {
                    nvx_keypointf_t feature = vxArrayItem(nvx_keypointf_t, featureData, i, stride);
                    Render::FeatureStyle style = vxArrayItem(Render::FeatureStyle, styleData, i, styleStride);

                    fprintf(handle, ",ftr(%.1f,%.1f,%d,%d,%d,%d)", feature.x, feature.y,
                            style.color[0], style.color[1], style.color[2], style.color[3]);
                }
            }

            NVXIO_SAFE_CALL( vxCommitArrayRange(location, 0, size, featureData) );
        }

        fprintf(handle, ")\n");
    }

    if (efficientRender)
        efficientRender->putFeatures(location, styles);
}

void EventLogger::putLines(vx_array lines, const Render::LineStyle &style)
{
    if (handle != NULL)
    {
        vx_size size;
        NVXIO_SAFE_CALL( vxQueryArray(lines, VX_ARRAY_ATTRIBUTE_NUMITEMS, &size, sizeof(size)) );

        fprintf(handle, "%d: lines(color(%d,%d,%d,%d), thickness(%d), %lu",
                frameCounter,
                style.color[0], style.color[1], style.color[2], style.color[3],
                style.thickness,
                static_cast<unsigned long>(size));

        if (size != 0)
        {
            vx_size stride;
            void* linesData = NULL;
            NVXIO_SAFE_CALL( vxAccessArrayRange(lines, 0, size, &stride, &linesData, VX_READ_ONLY) );

            std::vector<nvx_point4f_t> sortedLines;
            sortedLines.reserve(size);

            for (vx_size i = 0; i < size; i++)
            {
                nvx_point4f_t line = vxArrayItem(nvx_point4f_t, linesData, i, stride);

                sortedLines.push_back(line);
            }

            std::sort(sortedLines.begin(), sortedLines.end());

            for (size_t i = 0; i < size; i++)
            {
                fprintf(handle, ",line(%d,%d,%d,%d)", (int)sortedLines[i].x, (int)sortedLines[i].y, (int)sortedLines[i].z, (int)sortedLines[i].w);
            }

            NVXIO_SAFE_CALL( vxCommitArrayRange(lines, 0, size, linesData) );
        }

        fprintf(handle, ")\n");
    }

    if (efficientRender != NULL)
        efficientRender->putLines(lines, style);
}

void EventLogger::putConvexPolygon(vx_array verticies, const LineStyle& style)
{
    if (handle != NULL)
    {
        vx_size size;
        NVXIO_SAFE_CALL( vxQueryArray(verticies, VX_ARRAY_ATTRIBUTE_NUMITEMS, &size, sizeof(size)) );

        fprintf(handle, "%d: poligon(color(%d,%d,%d,%d), thickness(%d), %lu",
                frameCounter,
                style.color[0], style.color[1], style.color[2], style.color[3],
                style.thickness,
                static_cast<unsigned long>(size));

        if (size != 0)
        {
            vx_size stride;
            void* verticiesData = NULL;
            NVXIO_SAFE_CALL( vxAccessArrayRange(verticies, 0, size, &stride, &verticiesData, VX_READ_ONLY) );

            for (vx_size i = 0; i < size; i++)
            {
                vx_coordinates2d_t item = vxArrayItem(vx_coordinates2d_t, verticiesData, i, stride);
                fprintf(handle, ",vertex(%d,%d)", (int)item.x, (int)item.y);
            }

            NVXIO_SAFE_CALL( vxCommitArrayRange(verticies, 0, size, verticiesData) );
        }

        fprintf(handle, ")\n");
    }

    if (efficientRender != NULL)
        efficientRender->putConvexPolygon(verticies, style);
}

void EventLogger::putMotionField(vx_image field, const Render::MotionFieldStyle &style)
{
    if (handle != NULL)
        fprintf(handle, "%d: motionField(color(%d,%d,%d,%d)",
                frameCounter, style.color[0], style.color[1], style.color[2], style.color[3]);

    vx_rectangle_t rect;
    NVXIO_SAFE_CALL( vxGetValidRegionImage(field, &rect) );

    void *mv_base = NULL;
    vx_imagepatch_addressing_t mv_addr;
    NVXIO_SAFE_CALL( vxAccessImagePatch(field, &rect, 0, &mv_addr, &mv_base, VX_READ_ONLY) );

    fprintf(handle, ",%dx%d", mv_addr.dim_x, mv_addr.dim_y);

    for (vx_uint32 y = 0u; y < mv_addr.dim_y; y++)
    {
        for (vx_uint32 x = 0u; x < mv_addr.dim_x; x++)
        {
            vx_float32 *mv_val = (vx_float32 *)vxFormatImagePatchAddress2d(mv_base, x, y, &mv_addr);
            fprintf(handle, ",%f,%f", mv_val[0], mv_val[1]);
        }
    }

    NVXIO_SAFE_CALL( vxCommitImagePatch(field, NULL, 0, &mv_addr, mv_base) );

    fprintf(handle, ")\n");

    if (efficientRender != NULL)
        efficientRender->putMotionField(field, style);
}

void EventLogger::putCircles(vx_array circles, const CircleStyle& style)
{
    if (handle != NULL)
    {
        vx_size num_items;
        NVXIO_SAFE_CALL( vxQueryArray(circles, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_items, sizeof(num_items)) );

        fprintf(handle, "%d: circles(color(%d,%d,%d,%d), thickness(%d), %lu",
                frameCounter,
                style.color[0], style.color[1], style.color[2], style.color[3],
                style.thickness,
                static_cast<unsigned long>(num_items));

        if (num_items != 0)
        {
            vx_size stride;
            void* ptr = NULL;
            std::vector<nvx_point3f_t> sortedCircles;
            sortedCircles.reserve(num_items);

            NVXIO_SAFE_CALL( vxAccessArrayRange(circles, 0, num_items, &stride, &ptr, VX_READ_ONLY) );

            for (vx_size i = 0; i < num_items; i++)
            {
                nvx_point3f_t circle = vxArrayItem(nvx_point3f_t, ptr, i, stride);
                sortedCircles.push_back(circle);
            }

            std::sort(sortedCircles.begin(), sortedCircles.end());

            for (vx_size i = 0; i < num_items; i++)
            {
                nvx_point3f_t& circle = sortedCircles[i];
                fprintf(handle, ",circle(%f,%f,%f)", circle.x, circle.y, circle.z);
            }

            NVXIO_SAFE_CALL( vxCommitArrayRange(circles, 0, num_items, ptr) );
        }

        fprintf(handle, ")\n");
    }

    if (efficientRender)
        efficientRender->putCircles(circles, style);
}

void EventLogger::putArrows(vx_array old_points, vx_array new_points,
                            const LineStyle& line_style)
{
    if (handle != NULL)
    {
        vx_size num_items = 0, old_size = 0, new_size = 0;

        NVXIO_SAFE_CALL( vxQueryArray(old_points, VX_ARRAY_ATTRIBUTE_NUMITEMS, &old_size, sizeof(old_size)) );
        NVXIO_SAFE_CALL( vxQueryArray(new_points, VX_ARRAY_ATTRIBUTE_NUMITEMS, &new_size, sizeof(new_size)) );

        num_items = std::min(old_size, new_size);

        fprintf(handle, "%d: arrows(color(%d,%d,%d,%d), thickness(%d), %lu)\n",
                frameCounter,
                line_style.color[0], line_style.color[1], line_style.color[2], line_style.color[3],
                line_style.thickness,
                static_cast<unsigned long>(num_items));
    }

    if (efficientRender)
        efficientRender->putArrows(old_points, new_points, line_style);
}

bool EventLogger::flush()
{
    ++frameCounter;

    if (handle)
        fflush(handle);

    if(efficientRender != NULL)
        return efficientRender->flush();
    else
        return true;
}

void EventLogger::close()
{
    frameCounter = -1;

    if (efficientRender != NULL)
        efficientRender->close();
}

void EventLogger::setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void *context)
{
    keyBoardCallback = callback;
    keyboardCallbackContext = context;
}

void EventLogger::setOnMouseEventCallback(OnMouseEventCallback callback, void *context)
{
    mouseCallback = callback;
    mouseCallbackContext = context;
}

}
