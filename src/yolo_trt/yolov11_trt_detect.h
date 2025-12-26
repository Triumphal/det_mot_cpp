

#ifndef __YOLOV11_TRT_DETECT_H__
#define __YOLOV11_TRT_DETECT_H__
#include <array>
#include <opencv2/opencv.hpp>

#include "tensorrt_infer.h"

using namespace std;

// 图像尺寸结构体
typedef struct _ImageSize {
    int64_t width;
    int64_t height;
} ImageSize;

// 检测框结构体
typedef struct _DetectionBox {
    double xmin, ymin, xmax, ymax;  // 左上角和右下角坐标
    int class_id;                // 类别ID
    float confidence;            // 置信度
} DetectionBox;

// yolo检测配置的
typedef struct _YoloDetectConfig {
    float score_threshold;  // 置信度阈值= 0.25f;
    float nms_threshold;    // NMS 阈值= 0.45f;
} YoloDetectConfig;

// 图像Pad的结构体 padding 一般是对称的，只需要记录左和上
typedef struct _ImagePad {
    int64_t left;
    int64_t top;
} ImagePad;

class YoloDetect {
private:
    // 模型相关参数
    string m_model_path;           // 模型路径
    ImageSize m_image_shape;       // 图像输入宽高 [W,H]
    array<int, 4> m_input_shape;   // 输入形状 [B,C,H,W]
    array<int, 3> m_output_shape;  // 输出形状 [B,N,M]
    size_t m_input_dtype_size;     // 输入的数据类型字节大小
    size_t m_output_dtype_size;    // 输出的数据类型字节大小
    size_t m_input_bytes;          // 输入占用的字节数
    size_t m_output_bytes;         // 输出占用的字节数
    float* m_hInputBuffer;         // 输入缓冲区
    float* m_hOutputBuffer;        // 输出缓冲区

    // TensorRT 推理需要的对象
    cudaStream_t m_stream;
    TensorRTInfer m_trt_infer;
    SpdlogLogger m_logger;

    // 预处理和后处理需要的参数
    float m_resize_scale = 1.0f;  // 图片resize的比例
    ImagePad m_image_pad{};       // 图片pad的尺寸
    YoloDetectConfig m_detect_config{0.25, 0.45};

private:
    void remap_bbox2ori(cv::Rect& bbox);  // 坐标映射会原图

public:
    inline vector<int> getWH() { return {m_input_shape[3], m_input_shape[2]}; }
    YoloDetect(const string& model_path, SpdlogLogger& logger);
    void preprocess(const cv::Mat& img);
    void detectInfer();
    vector<DetectionBox> postprocess();
    ~YoloDetect();
};

#endif  // __YOLOV11_TRT_DETECT_H__
