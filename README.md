## 1、生成可执行文件
## 1.1、更新子仓
```shell
git submodule init
git submodule update
```
## 1.2、编译第三方文件
具体见[README.md](./3rd_party/README.md)


## 编译
```shell
mkdir build && cd build
rm -rf ./* && cmake .. && make -j4
## 或者使用新版本的cmake命令
cmake -B build -S .
cmake --build build
```
生成的可执行文件在build下，同时生成了libyolodetect.so
## 2、测试
| 测试功能      | cpp |
| ----------- | ----------- | 
| 单张图片的推理    | ✅[image_test_yolo11.cpp](./src/test/image_test_yolo11.cpp)  | 
| 单个视频的推理    | ✅[video_test_yolo11.cpp](./src/test/video_test_yolo11.cpp)  |   
| 多目标跟踪       | 🚧  | 

### 2.1、单张图片的推理 image_test_yolo11.cpp
- 这里的模型使用的是tensorrt的engine类型，
```shell
./image_test_yolo11 <image_path> <model_path> <output_path>
#                    图片路径        模型路径    处理后的保存路径
```
### 2.2、单个视频的推理 video_test_yolo11.cpp
- 这里的模型使用的是tensorrt的engine类型，
```shell
./video_test_yolo11 <ideo_path> <model_path> <output_path>
#                    视频路径        模型路径   处理后的保存路径
```

## 3、trt模型相关
> 要提前装好tensorRT
### 转模型 （端侧）
```shell
trtexec --onnx=model/yolo11n.onnx \
    --saveEngine=model/yolo11n.plan \
    --iterations=100 \
    --duration=10 \
    --fp16 
```

### 测模型耗时
```shell
trtexec --onnx=model/yolo11n.onnx  --fp16  --timingCacheFile=timing.cache
```

### 环境依赖
- 使用apt安装一些环境
```shell
sudo apt install  libopencv-dev libyaml-cpp-dev install libspdlog-dev libeigen3-dev
```
- 安装tensorRT
根据cuda版本[下载](https://developer.nvidia.com/tensorrt/download)
```shell
# 解压到指定目录
tar -zxvf TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz -C /usr/local/
# 软连接
cd /usr/local && ln -s TensorRT-10.14.1.48 TensorRT
```



