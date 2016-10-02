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

# OS info
OSLOWER := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

OS_ARCH := $(shell uname -m | sed -e "s/i386/i686/")

# Take command line flags that override any of these settings
ifeq ($(i386),1)
	OS_ARCH := i686
endif

ifeq ($(x86_64),1)
	OS_ARCH := x86_64
endif

ifeq ($(ARMv7),1)
	OS_ARCH := armv7l
endif

ifeq ($(ARMv8),1)
	OS_ARCH := aarch64
endif

CXXFLAGS += -std=c++0x

ifneq ($(VIBRANTE_TOOLCHAIN_SYSROOT),)
	CCFLAGS += --sysroot="$(VIBRANTE_TOOLCHAIN_SYSROOT)"
endif

# Configuration-specific build flags
ifeq ($(dbg),1)
	CCFLAGS += -g
	TARGET := debug
else
	CCFLAGS += -O3 -DNDEBUG
	TARGET := release
endif

# check visionworks availability
VISION_WORKS_EXISTS := $(shell pkg-config --exists visionworks && echo "1" || echo "0")
ifeq ($(VISION_WORKS_EXISTS), 0)
$(error You must put directory containing visionworks.pc to the PKG_CONFIG_PATH environment variable)
endif

VISION_WORKS_CFLAGS := $(shell pkg-config --cflags visionworks)
CUDA_CFLAGS := $(shell pkg-config --cflags cudart-7.0)

EIGEN_CFLAGS := -I../3rdparty/eigen
NVXIO_CFLAGS := -DCUDA_API_PER_THREAD_DEFAULT_STREAM -DUSE_GUI=1 -DUSE_GLES=1 -DUSE_GLFW=1  -DUSE_GSTREAMER_OMX=1 -DUSE_NVGSTCAMERA=1 -DUSE_GSTREAMER=1
NVXIO_CFLAGS += -I./include  -I../3rdparty/opengl $(shell pkg-config --cflags freetype2) -I../3rdparty/glfw3/include  -I../3rdparty/opengl $(shell pkg-config --cflags gstreamer-base-1.0 gstreamer-pbutils-1.0 gstreamer-app-1.0)

INCLUDES :=
INCLUDES += -Isrc/
INCLUDES += $(NVXIO_CFLAGS)
INCLUDES += $(CUDA_CFLAGS)
INCLUDES += $(VISION_WORKS_CFLAGS)
INCLUDES += $(EIGEN_CFLAGS)

VPATH=src/Private:src/Render/GStreamer:src/Render/CUDA-OpenGL:src/Render/OpenCV:src/Render:src/FrameSource/GStreamer:src/FrameSource/NvMedia/OV10640:src/FrameSource/NvMedia/OV10635:src/FrameSource/NvMedia:src/FrameSource/OpenCV:src/FrameSource:src:

CPP_FILES :=
CPP_FILES += $(wildcard src/Private/*.cpp)
CPP_FILES += $(wildcard src/Render/GStreamer/*.cpp)
CPP_FILES += $(wildcard src/Render/CUDA-OpenGL/*.cpp)
CPP_FILES += $(wildcard src/Render/OpenCV/*.cpp)
CPP_FILES += $(wildcard src/Render/*.cpp)
CPP_FILES += $(wildcard src/FrameSource/GStreamer/*.cpp)
CPP_FILES += $(wildcard src/FrameSource/NvMedia/OV10640/*.cpp)
CPP_FILES += $(wildcard src/FrameSource/NvMedia/OV10635/*.cpp)
CPP_FILES += $(wildcard src/FrameSource/NvMedia/*.cpp)
CPP_FILES += $(wildcard src/FrameSource/OpenCV/*.cpp)
CPP_FILES += $(wildcard src/FrameSource/*.cpp)
CPP_FILES += $(wildcard src/*.cpp)

C_FILES :=
C_FILES += $(wildcard src/Private/*.c)
C_FILES += $(wildcard src/Render/GStreamer/*.c)
C_FILES += $(wildcard src/Render/CUDA-OpenGL/*.c)
C_FILES += $(wildcard src/Render/OpenCV/*.c)
C_FILES += $(wildcard src/Render/*.c)
C_FILES += $(wildcard src/FrameSource/GStreamer/*.c)
C_FILES += $(wildcard src/FrameSource/NvMedia/OV10640/*.c)
C_FILES += $(wildcard src/FrameSource/NvMedia/OV10635/*.c)
C_FILES += $(wildcard src/FrameSource/NvMedia/*.c)
C_FILES += $(wildcard src/FrameSource/OpenCV/*.c)
C_FILES += $(wildcard src/FrameSource/*.c)
C_FILES += $(wildcard src/*.c)


OBJ_DIR := obj/$(TARGET)
OBJ_CPP_FILES := $(addprefix $(OBJ_DIR)/,$(notdir $(CPP_FILES:.cpp=.o)))
OBJ_C_FILES := $(addprefix $(OBJ_DIR)/,$(notdir $(C_FILES:.c=.o)))

OUTPUT_DIR := ../libs/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$(if $(abi),/$(abi))

################################################################################

# Target rules
all: $(OUTPUT_DIR)/libnvxio.a

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	$(CC) $(INCLUDES) $(CCFLAGS) -std=c99 -o $@ -c $<

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(INCLUDES) $(CCFLAGS) $(CXXFLAGS) -o $@ -c $<

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(OUTPUT_DIR)/libnvxio.a: $(OBJ_CPP_FILES) $(OBJ_C_FILES) | $(OUTPUT_DIR)
	$(AR) rsc -o $@ $^

clean:
	rm -f $(OBJ_CPP_FILES) $(OBJ_C_FILES)
	rm -f $(OUTPUT_DIR)/libnvxio.a
