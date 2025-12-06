

#include <iostream>
#include <array>
#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include "yolov11_trt_detect.h"
#include "tqdm.h"
#include "yaml-cpp/yaml.h"


void infer_trt(int argc, char** argv){
    std::string vido_path= "../../data/pedestrain.mp4";
    std::string model_path= "../../model/yolo11n.plan";
    std::string output_path = "../../output/videos/pedestrain_cpp.mp4";
    // 初始化日志
    spdlog::set_level(spdlog::level::err);  // 设置日志等级
    spdlog::info("spdlog 系统初始化完成.");

    if (argc > 1) {
        vido_path = argv[1];
        spdlog::info("使用自定义图像路径: {}",vido_path);
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


    // 读取视频
    auto cap = cv::VideoCapture(vido_path);
    if (!cap.isOpened()){
        spdlog::info("Error opening video stream or file");
        return;
    }
    auto frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    auto frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    auto fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    auto frame_counts = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    spdlog::warn("");
    auto fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter video_writer;
    // 初始化 VideoWriter
    video_writer.open(output_path, fourcc, fps, cv::Size(frame_width, frame_height));
    if (!video_writer.isOpened()) {
        spdlog::error("无法创建或初始化 VideoWriter");
        return;
    }

    // 创建 YoloDetect 对象
    SpdlogLogger logger(ILogger::Severity::kINFO);
    YoloDetect yolo_detector(model_path, logger);

    for (auto i:tqdm::trange(frame_counts)){
        cv::Mat frame;
        auto ret = cap.read(frame);
        if (!ret){ 
            spdlog::error("Don't read video");
            break;
        }
        // 前处理
        yolo_detector.preprocess(frame);
        // 推理
        yolo_detector.detectInfer();

        // 后处理
        auto detections = yolo_detector.postprocess() ;
        // spdlog::info("检测到的目标数量: {}", detections.size());
        auto postprocess_time = std::chrono::high_resolution_clock::now();
    
        // 绘制结果
        // cv::resize(img, img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_AREA);
        for (const auto& box : detections) {
            cv::rectangle(frame, cv::Point(box.xmin, box.ymin), cv::Point(box.xmax, box.ymax), cv::Scalar(0, 0, 125), 2);
            std::string label = "cls_id:" + std::to_string(box.class_id) + " score:" + std::to_string(box.confidence);
            cv::putText(frame, label, cv::Point(box.xmin, box.ymin - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0), 2);
        }
        video_writer.write(frame);
        // cv::imwrite("./yolo11_trt_result.jpg", frame);
    }
}

int main(int argc, char** argv) {
    infer_trt(argc, argv);
    return 0;
}