
// 使用spdlog初始化TRT日志

#ifndef __GLOGGING_H__
#define __GLOGGING_H__

#include <NvInferRuntime.h>
#include <string>
#include "spdlog/spdlog.h"

using namespace std;
using Severity = nvinfer1::ILogger::Severity;

// 继承 TensorRT 的 ILogger 接口
class SpdlogLogger : public nvinfer1::ILogger {
public:
    // 设置日志打印的等级
    explicit SpdlogLogger(Severity severity = Severity::kWARNING) : mReportableSeverity(severity) {}
    void log(Severity severity, const char* msg) noexcept override {
        // 根据 TensorRT 的 Severity 级别映射到 glog 的级别
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                spdlog::critical("[TRT] [F] {}", msg);
                break;
            case Severity::kERROR:
                spdlog::error("[TRT] [E] {}", msg);
                break;
            case Severity::kWARNING:
                spdlog::warn("[TRT] [W] {}", msg);
                break;
            case Severity::kINFO:
                spdlog::info("[TRT] [I] {}", msg);
                break;
            case Severity::kVERBOSE:
                spdlog::debug("[TRT] [V] {}", msg);
                break;
            default:
                // 未知级别，作为警告处理
                spdlog::warn("[TRT] [Unknown Severity] {}", msg);
                break;
        }
    }

private:
    Severity mReportableSeverity;
    string m_func_time;
};

#endif // __GLOGGING_H__