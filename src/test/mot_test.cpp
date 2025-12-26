

#include <spdlog/spdlog.h>

#include <array>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "Sort.h"
#include "tqdm.h"
#include "yaml-cpp/yaml.h"
#include "yolov11_trt_detect.h"

void convert2BBox(const vector<DetectionBox>& detections, vector<BBox>& detections_new) {
    for (auto& det : detections) {
        BBox tmp;
        tmp << det.xmin, det.ymin, det.xmax, det.ymax;
        detections_new.push_back(tmp);
    }
}

void infer_trt(int argc, char** argv) {
    std::string vido_path = "../data/pedestrain_1.mp4";
    std::string model_path = "../model/yolo11n.plan";
    std::string output_path = "../output/videos/pedestrain_1_cpp_mot.mp4";
    // 初始化日志
    spdlog::set_level(spdlog::level::err);  // 设置日志等级
    spdlog::info("spdlog 系统初始化完成.");

    if (argc > 1) {
        vido_path = argv[1];
        spdlog::info("使用自定义图像路径: {}", vido_path);
    } else {
        spdlog::info("使用默认图像路径: ../data/bus.jpg");
    }
    if (argc > 2) {
        model_path = argv[2];
        spdlog::info("使用自定义模型路径: {}", model_path);
    } else {
        spdlog::info("使用默认模型路径: ../model/yolo11n.plan");
    }
    if (argc > 3) {
        output_path = argv[3];
        spdlog::info("使用自定义输出路径: {}", output_path);
    } else {
        spdlog::info("使用默认输出路径: {}", output_path);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // 读取视频
    auto cap = cv::VideoCapture(vido_path);
    if (!cap.isOpened()) {
        spdlog::info("Error opening video stream or file");
        return;
    }
    auto frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    auto frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    auto fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    auto frame_counts = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
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

    // 初始化跟踪器
    SortParam sort_param = {.iou_threshold = 0.3, .max_missed = 10, .min_hits = 3};
    KalmanFilterParam kf_param;
    kf_param.std_dev_process = 1;  // 控制过程噪声Q标准差
    kf_param.std_dev_measure = 0.1;   // 控制测量噪声R标准差
    kf_param.initial_P_value = 1000;  // 误差协方差矩阵初始化值
    SortTracker sort_tracker(sort_param, kf_param);

    for (auto i : tqdm::trange(frame_counts)) {
        cv::Mat frame;
        auto ret = cap.read(frame);
        if (!ret) {
            spdlog::error("Don't read video");
            break;
        }
        // 前处理
        yolo_detector.preprocess(frame);
        // 推理
        yolo_detector.detectInfer();

        // 后处理
        auto detections = yolo_detector.postprocess();
        // spdlog::info("检测到的目标数量: {}", detections.size());
        auto postprocess_time = std::chrono::high_resolution_clock::now();

        // 跟踪
        vector<BBox> detections_new;
        convert2BBox(detections, detections_new);
        auto track_results = sort_tracker.update(detections_new);

        // 绘制跟踪结果
        // cv::resize(img, img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_AREA);
        for (const auto& [id, box] : track_results) {
            cv::rectangle(frame, cv::Point(box(0), box(1)), cv::Point(box(2), box(3)), cv::Scalar(0, 0, 125), 2);
            std::string label = "track_id:" + std::to_string(id);
            cv::putText(frame, label, cv::Point(box(0), box(1) - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
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