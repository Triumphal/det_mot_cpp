## 0、编译命令

```shell
mkdir build && cd build
rm -rf ./* && cmake .. && make -j4

```
生成的可执行文件在build下，同时生成了libyolodetect.so
## 测试
### 图片 yolo11_test_image.cpp
- 这里的模型使用的是tensorrt的engine类型，
```shell
./image_test_yolo11 <image_path> <model_path> <output_path>
#                    图片路径        模型路径    处理后的保存路径
```
