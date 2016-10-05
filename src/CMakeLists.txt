
project(vovox)

link_directories(~/VisionWorks-1.4-Samples/libs/aarch64/linux/release)
link_directories(~/VisionWorks-1.4-Samples/3rdparty/glfw3/libs)
link_directories(/usr/lib/aarch64-linux-gnu/tegra-egl)
link_directories(/usr/local/cuda-7.0/lib64) 

add_executable (${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${PROJECT_NAME} nvxio freetype glfw3 )
#target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )
target_link_libraries(${PROJECT_NAME} /usr/lib/aarch64-linux-gnu/tegra-egl/libGLESv2.so.2)

target_link_libraries(${PROJECT_NAME} GLU) 

target_link_libraries(${PROJECT_NAME} /usr/lib/aarch64-linux-gnu/tegra-egl/libEGL.so)
target_link_libraries(${PROJECT_NAME} Xrandr Xi Xxf86vm X11)   

target_link_libraries(${PROJECT_NAME} gstpbutils-1.0 gstapp-1.0 gstbase-1.0 gstreamer-1.0) 
target_link_libraries(${PROJECT_NAME} gobject-2.0 glib-2.0)   

target_link_libraries(${PROJECT_NAME} /usr/lib/aarch64-linux-gnu/tegra/libcuda.so)   
target_link_libraries(${PROJECT_NAME} cudart)   

target_link_libraries(${PROJECT_NAME} visionworks)
target_link_libraries(${PROJECT_NAME} cudart)
 

# -lGLU -lGL -lSM -lICE -lX11 -lXext -lglut -lXmu
