// 时间相关的函数

#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

// 辅助函数：执行查找逻辑
inline const char* get_unix_filename(const char* path) {
    if (!path) return "";
    const char* separator = std::strrchr(path, '/');// 查找最后一个 '/' 的位置
    return separator ? separator + 1 : path;
}

/**
 * @brief 使用 RAII 机制统计代码块（函数）耗时的计时器类。
 */
class FunctionTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    FunctionTimer(const string& file_path, const int& line, const string& func_name)
        : m_file_name(get_unix_filename(file_path.c_str())), m_line(line), m_func_name(func_name), m_start_time(Clock::now()) {
    }

    // 析构函数：释放资源（停止计时并打印结果）
    ~FunctionTimer() {
        TimePoint end_time = Clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - m_start_time);
        // 打印结果
        double ms = static_cast<double>(duration.count()) / 1000.0;
        stringstream ss;
        ss << "[TIMER] [" << m_file_name << ":" << m_line << ":" << m_func_name << "]: " << "cost time: " << ms << " ms";
        spdlog::debug(ss.str().c_str()); // 耗时设置为debug级别
    }

    // 禁用拷贝和赋值，确保对象生命周期唯一性
    FunctionTimer(const FunctionTimer&) = delete;
    FunctionTimer& operator=(const FunctionTimer&) = delete;

private:
    string m_file_name;  // 文件名称
    int m_line;          // 行号
    string m_func_name;  // 函数名
    TimePoint m_start_time;
};

#endif  // __TIMER_HPP__