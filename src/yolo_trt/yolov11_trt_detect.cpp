

#include "yolov11_trt_detect.h"

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/dnn.hpp>  // NMS
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "timing.h"

using namespace std;

YoloDetect::YoloDetect(const string& model_path, SpdlogLogger& logger)
    : m_model_path(model_path), m_logger(logger), m_trt_infer(model_path, logger) {
    m_detect_config.score_threshold = 0.25f;
    m_detect_config.nms_threshold = 0.45f;

    // 初始化推理
    cudaStreamCreate(&m_stream);                                               // 创建 CUDA 流
    m_trt_infer.inferWarmUp(m_stream);                                         // 推理预热，减少第一次推理的耗时
    m_trt_infer.getModelIOShape(m_input_shape, m_output_shape);                // 获取模型输入输出形状
    m_trt_infer.getModelIODtypeSize(m_input_dtype_size, m_output_dtype_size);  // 获取模型输入输出数据类型大小
    m_input_bytes =
        m_input_shape[0] * m_input_shape[1] * m_input_shape[2] * m_input_shape[3] * sizeof(m_input_dtype_size);
    m_output_bytes = m_output_shape[0] * m_output_shape[1] * m_output_shape[2] * sizeof(m_output_dtype_size);
    // 分配空间
    cudaMallocHost((void**)&m_hInputBuffer, m_input_bytes);
    cudaMallocHost((void**)&m_hOutputBuffer, m_output_bytes);
}

/// 预处理函数：BGR->RGB，Resize，归一化，HWC->NCHW
void YoloDetect::preprocess(const cv::Mat& img) {
    FunctionTimer preprocess_timer(__FILE__, __LINE__, __func__);

    cv::Mat processed_img;
    // 判断输入的图像是否为彩色图像
    if (img.channels() == 3) {
        cv::cvtColor(img, processed_img, cv::COLOR_BGR2RGB);
    } else {
        cv::cvtColor(img, processed_img, cv::COLOR_GRAY2BGR);
    }

    // 1、获取原始图像形状和模型输入的形状
    m_image_shape.width = img.cols;
    m_image_shape.height = img.rows;

    int CHANNELS = m_input_shape[1];
    int INPUT_H = m_input_shape[2];
    int INPUT_W = m_input_shape[3];

    // 2、计算缩放比例，保持图像宽高比
    m_resize_scale =
        std::min(static_cast<float>(INPUT_W) / m_image_shape.width, static_cast<float>(INPUT_H) / m_image_shape.height);
    const int new_w = static_cast<int>(m_image_shape.width * m_resize_scale);
    const int new_h = static_cast<int>(m_image_shape.height * m_resize_scale);

    // 3、调整图像大小
    cv::resize(processed_img, processed_img, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);

    // 4、计算填充大小
    const int pad_w = INPUT_W - new_w;
    const int pad_h = INPUT_H - new_h;

    // 5、填充图像 上下左右居中填充
    m_image_pad.top = pad_h / 2;
    m_image_pad.left = pad_w / 2;
    cv::copyMakeBorder(processed_img, processed_img, m_image_pad.top, m_image_pad.top, m_image_pad.left,
                       m_image_pad.left, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // vector<float> input_data(CHANNELS * INPUT_H * INPUT_W);
    // HWC -> CHW
    for (int c = 0; c < CHANNELS; ++c) {  // 通道
        for (int h = 0; h < INPUT_H; ++h) {
            for (int w = 0; w < INPUT_W; ++w) {
                // 计算当前像素在 HWC 图像中的位置 OpenCV 图像是 8U 类型，需要转换为 float
                unsigned char pixel_value = processed_img.at<cv::Vec3b>(h, w)[c];
                // 计算当前元素在目标 NCHW 缓冲区中的索引 index = c * H * W + h * W + w
                size_t index = c * (INPUT_H * INPUT_W) + h * INPUT_W + w;
                // 写入并归一化 (直接写入 hInputBuffer，消除了 memcpy)
                m_hInputBuffer[index] = static_cast<float>(pixel_value) / 255.0f;
            }
        }
    }
}

void YoloDetect::detectInfer() {
    FunctionTimer infer_timer(__FILE__, __LINE__, __func__);
    // 这里调用 TensorRT 推理接口，代码省略
    m_trt_infer.infer((const void*)m_hInputBuffer, m_hOutputBuffer, m_stream);
    cudaStreamSynchronize(m_stream);  // 等待GPU完成异步操作
}

/// 后处理函数：解码、置信度过滤、NMS
vector<DetectionBox> YoloDetect::postprocess() {
    FunctionTimer postprocess_timer(__FILE__, __LINE__, __func__);
    // 临时容器，用于存储通过置信度过滤的检测结果，供 NMS 使用
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    // Step 1: 遍历、解码和置信度过滤 这里要要注意输出的是[1,84,8400]
    int num_boxes = m_output_shape[2];  // NUM_BOXES
    int element = m_output_shape[1];    // 4 + NUM_CLASSES

    for (int i = 0; i < num_boxes; ++i) {
        // 1. 获取坐标信息 (前 4 个元素)
        float cx = m_hOutputBuffer[i + 0 * num_boxes];  // center_x
        float cy = m_hOutputBuffer[i + 1 * num_boxes];  // center_y
        float w = m_hOutputBuffer[i + 2 * num_boxes];   // width
        float h = m_hOutputBuffer[i + 3 * num_boxes];   // height

        // 3. 找到最高分数和对应的类别
        float max_score = 0;
        int max_class_id = -1;

        for (int j = 4; j < element; ++j) {
            if (m_hOutputBuffer[i + j * num_boxes] > max_score) {
                max_score = m_hOutputBuffer[i + j * num_boxes];
                max_class_id = j - 4;
            }
        }

        // 4. 应用置信度阈值过滤
        if (max_score >= m_detect_config.score_threshold) {
            // 坐标转换：将 center/wh 转换为 xmin/ymin/xmax/ymax (像素坐标)
            // 输出通常是归一化坐标或 模型输入 尺寸下的坐标。
            // 这里的 cx, cy, w, h 是 模型输入 尺寸下的像素值。
            int xmin = static_cast<int>(cx - w / 2.0f);
            int ymin = static_cast<int>(cy - h / 2.0f);
            int xmax = static_cast<int>(cx + w / 2.0f);
            int ymax = static_cast<int>(cy + h / 2.0f);

            // 存储到临时容器
            bboxes.emplace_back(xmin, ymin, w, h);  // OpenCV Rect 需要 (x, y, width, height)
            scores.push_back(max_score);
            class_ids.push_back(max_class_id);
        }
    }

    // ----------------------------------------------------
    // Step 2: NMS (非极大值抑制)
    std::vector<int> indices;
    // 使用 OpenCV 的高效 NMS 函数
    cv::dnn::NMSBoxes(bboxes, scores,
                      m_detect_config.score_threshold,  // NMSBoxes 内部也会再次使用置信度阈值
                      m_detect_config.nms_threshold, indices);

    // ----------------------------------------------------
    // Step 3: 构造最终结果 并将坐标映射会原始图像
    std::vector<DetectionBox> final_detections;
    spdlog::debug("检测到{}个框", indices.size());
    for (int idx : indices) {
        // 从 NMS 筛选出的索引中，获取最终信息
        cv::Rect bbox = bboxes[idx];
        remap_bbox2ori(bbox);
        spdlog::debug("第{}个:[{}, {}, {}, {}]", idx, bbox.x, bbox.y, bbox.width, bbox.height);
        // 坐标映射
        final_detections.push_back(
            {(double)bbox.x, (double)bbox.y, (double)(bbox.x + bbox.width), (double)(bbox.y + bbox.height), class_ids[idx], scores[idx]});
    }

    return final_detections;
}

void YoloDetect::remap_bbox2ori(cv::Rect& bbox) {
    bbox.x = (bbox.x - m_image_pad.left + 1) / m_resize_scale;  // 由于长度和坐标的差异这里要+1,否则会出现<0的情况
    bbox.y = (bbox.y - m_image_pad.top + 1) / m_resize_scale;
    bbox.width /= m_resize_scale;
    bbox.height /= m_resize_scale;
}

YoloDetect::~YoloDetect() {
    // 释放资源
    cudaFreeHost(m_hInputBuffer);
    cudaFreeHost(m_hOutputBuffer);
    cudaStreamDestroy(m_stream);
}
