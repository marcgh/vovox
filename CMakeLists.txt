cmake_minimum_required(VERSION 2.8)

#project (vovox)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#
# Let's find de required packages
#
find_package (Threads)
#CMAKE_THREAD_LIBS_INIT     - the thread library
#CMAKE_USE_SPROC_INIT       - are we using sproc?
#CMAKE_USE_WIN32_THREADS_INIT - using WIN32 threads?
#CMAKE_USE_PTHREADS_INIT    - are we using pthreads
#CMAKE_HP_PTHREADS_INIT     - are we using hp pthreads

#
#find_package(glfw3)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
include_directories( ${CMAKE_SOURCE_DIR}/nvxio/include )

add_subdirectory(src)

