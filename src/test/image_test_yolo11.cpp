

#include <iostream>
#include <array>
#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include "yolov11_trt_detect.h"


void infer_trt(int argc, char** argv){
    std::string image_path= "../../data/bus.jpg";
    std::string model_path= "../../model/yolo11n.plan";
    std::string output_path = "./yolo11_trt_result.jpg";
    // 初始化日志
    spdlog::set_level(spdlog::level::info);  // 设置日志等级
    spdlog::info("spdlog 系统初始化完成.");

    if (argc > 1) {
        image_path = argv[1];
        spdlog::info("使用自定义图像路径: {}",image_path);
    } else {
        spdlog::info("使用默认图像路径: ../../data/bus.jpg");
    }
    if (argc > 2) {
        model_path = argv[2];
        spdlog::info("使用自定义模型路径: " ,model_path );
    } else {
        spdlog::info("使用默认模型路径: ../../model/yolo11n.plan");
    }
    if (argc > 3) {
        output_path = argv[3];
        spdlog::info("使用自定义输出路径: " ,output_path);
    } else {
        spdlog::info( "使用默认输出路径: ./yolo11_trt_result.jpg");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // 读取图像
    cv::Mat img = cv::imread(image_path);

    // 创建 YoloDetect 对象
    SpdlogLogger logger(ILogger::Severity::kINFO);
    YoloDetect yolo_detector(model_path, logger);

    // 前处理
    yolo_detector.preprocess(img);

    // 推理
    yolo_detector.detectInfer();

    // 后处理
    auto detections = yolo_detector.postprocess();
    spdlog::info("检测到的目标数量: {}", detections.size() );

    auto postprocess_time = std::chrono::high_resolution_clock::now();
    
    // 绘制结果
    // cv::resize(img, img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_AREA);
    for (const auto& box : detections) {
        cv::rectangle(img, cv::Point(box.xmin, box.ymin), cv::Point(box.xmax, box.ymax), cv::Scalar(0, 0, 125), 2);
        std::string label = "cls_id:" + std::to_string(box.class_id) + " score:" + std::to_string(box.confidence);
        cv::putText(img, label, cv::Point(box.xmin, box.ymin - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("./yolo11_trt_result.jpg", img);
}

int main(int argc, char** argv) {
    infer_trt(argc, argv);
    return 0;
}