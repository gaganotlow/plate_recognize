# 设置最低版本号
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
# 设置项目名称
project(rk3588-demo VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
# 输出系统信息
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

# 设置编译器
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置库架构
set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

#  rknn_api 文件夹路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
#  rknn_api include 路径
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)
#  rknn_api lib 路径
set(RKNN_API_LIB_PATH ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

# 寻找OpenCV库，使用自定义的OpenCV_DIR
set(3RDPARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)

set(RGA_DIR ${3RDPARTY_PATH}/rga)
set(RGA_LIB ${RGA_DIR}/libs/Linux/gcc-aarch64/librga.so)

# jpeg turbo
set(JPEG_DIR ${3RDPARTY_PATH}/jpeg_turbo)
set(JPEG_LIB ${JPEG_DIR}/Linux/${LIB_ARCH}/libturbojpeg.a)

set(STD_DIR ${3RDPARTY_PATH}/stb_image)

include_directories(${RGA_DIR}/include)
# set(OpenCV_DIR ${3RDPARTY_PATH}/opencv/opencv-linux-${LIB_ARCH}/share/OpenCV)
find_package(OpenCV REQUIRED)
# 输出OpenCV信息
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")


# 用来搜索头文件的目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_API_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${JPEG_DIR}/include
    ${STD_DIR}
)

# 构建预处理和后处理库
add_library(nn_process SHARED
            src/process/preprocess.cpp
            src/process/postprocess.cpp
)
# 链接库
target_link_libraries(nn_process
    ${OpenCV_LIBS}
    ${RGA_LIB}
)

# 构建自定义封装API库
add_library(rknn_engine SHARED src/engine/rknn_engine.cpp) 
# 链接库
target_link_libraries(rknn_engine
    ${RKNN_API_LIB_PATH}
)
# yolov8_lib
add_library(yolov8_lib SHARED src/task/yolov8_custom.cpp)
# 链接库
target_link_libraries(yolov8_lib
    rknn_engine
    nn_process
)

# plate_recognizer_lib
add_library(plate_recognizer_lib SHARED src/task/plate_recognizer.cpp)
# 链接库
target_link_libraries(plate_recognizer_lib
    rknn_engine
    nn_process
    ${OpenCV_LIBS}
)

# draw_lib
add_library(draw_lib SHARED src/draw/cv_draw.cpp)
# 链接库
target_link_libraries(draw_lib
    ${OpenCV_LIBS}
)

# 测试yolov8_img
add_executable(yolov8_pose_img src/yolov8_pose_img.cpp)

# 链接库
target_link_libraries(yolov8_pose_img
        draw_lib
        yolov8_lib
        plate_recognizer_lib
)
