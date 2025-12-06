# 1. 设置项目名称和所需的最低CMake版本
cmake_minimum_required(VERSION 3.22)
project(det_mot_cpp LANGUAGES CXX CUDA)

# 2. 启用C++11标准 (或其他你需要的标准，如C++17)
set(CMAKE_CXX_STANDARD 20)


find_package(OpenCV REQUIRED) # OpenCV
find_package(CUDA REQUIRED) # CUDA
find_package(yaml-cpp REQUIRED) # yaml-cpp
find_package(spdlog REQUIRED) # spdlog

# TensorRT
set(TRT_INCLUDE_DIRS /usr/local/TensorRT/include/)
set(TRT_LIBRARY_DIR /usr/local/TensorRT/lib/)
message(STATUS "TRT_LIBRARY_DIR path: ${TRT_LIBRARY_DIR}")
# 使用 find_library 查找具体的库文件 # 库文件的名字（libnvinfer.so）
find_library(TRT_LIBRARIES NAMES nvinfer PATHS ${TRT_LIBRARY_DIR} NO_DEFAULT_PATH)

# Eigen
# set(EIGEN_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include/Eigen)


# 3. 定义共享库目标
# --------------------------------------------------------------------------
# 3.1. 定义库的源文件
set(LIBRARY_SOURCES
    src/yolo_trt/yolov11_trt_detect.cpp
    src/yolo_trt/tensorrt_infer.cpp
)

# 3.2. 添加共享库 (SHARED)
add_library(yolodetect SHARED
    ${LIBRARY_SOURCES}
)

# 3.3. 指定库的头文件搜索路径
target_include_directories(yolodetect PUBLIC
    ${OpenCV_INCLUDE_DIRS} # 添加 OpenCV 头文件路径
    ${CUDA_INCLUDE_DIRS} # 添加 CUDA 头文件路径
    ${TRT_INCLUDE_DIRS} # 添加 TensorRT 头文件路径
    ${YOLO_TRT_INCLUDE_DIRS} # 添加个人的头文件路径
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo_trt
)

target_link_libraries(yolodetect PUBLIC
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES} # 链接 CUDA 运行时库 (cudart)
    ${TRT_LIBRARIES}
    yaml-cpp # yaml-cpp
    spdlog::spdlog
)

# 4. 定义可执行文件目标 (测试程序)
# --------------------------------------------------------------------------
# 4.1. 添加可执行文件
add_executable(
    image_test_yolo11  # 目标名称，即生成的可执行文件名
    src/test/image_test_yolo11.cpp
)

add_executable(
    video_test_yolo11  # 目标名称，即生成的可执行文件名
    src/test/video_test_yolo11.cpp
)

# 4.2. 将可执行文件链接到共享库
target_link_libraries(
    image_test_yolo11
    PRIVATE         # 这个目标只需要内部链接
    yolodetect    # 链接到我们上面定义的库目标
)

target_link_libraries(
    video_test_yolo11
    PRIVATE         # 这个目标只需要内部链接
    yolodetect    # 链接到我们上面定义的库目标
)


    
