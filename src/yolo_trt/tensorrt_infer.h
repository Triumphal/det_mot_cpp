// TensorRT 相关的头文件

#ifndef __TENSORRT_INFER_H__
#define __TENSORRT_INFER_H__

// 抑制弃用接口的告警，后面可能不再需要
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <NvInfer.h>

#include <sstream>
#include <string>
#include <vector>
#include "spdlogging.h"

// #include "trt_logging.h"

using namespace nvinfer1;
using namespace std;

/// @brief 模型输入输出信息结构体
struct ModelIOInfo {
    string input_name;                // 模型输入名称
    string output_name;               // 模型输出名称
    array<int, 4> input_shape;        // 输入形状 [B,C,H,W]
    array<int, 3> output_shape;       // 输出形状 [B,N,M]
    size_t input_bytes;               // 输入占用的字节数
    size_t output_bytes;              // 输出占用的字节数
    nvinfer1::DataType input_dtype;   // 输入的数据类型
    nvinfer1::DataType output_dtype;  // 输出的数据类型
};

/// @brief TensorRTInfer 推理类
class TensorRTInfer {
   private:
    string m_model_path;             // 输入的模型路径，TensorRT 引擎文件
    std::vector<char> m_model_data;  // 模型数据缓冲区
    SpdlogLogger m_logger;                 // TensorRT 日志记录器
    IRuntime* m_runtime;               // 用于反序列化引擎
    ICudaEngine* m_engine;             // TensorRT 引擎
    IExecutionContext* m_context;      // 执行上下文
    void* m_dInputBuffer;              // 输入的 GPU 缓存地址
    void* m_dOutputBuffer;             // 输出的 GPU 缓存地址
    ModelIOInfo m_model_io_info;     // 模型输入和输出的相关信息
    stringstream ss;                 // 字符串数据流用来组装打印信息
   private:
    void initialize();        // 初始化
    void readModelFromFile();  // 从文件读取模型数据
    void buildEngine();        // 构建 TensorRT 引擎
    void getIOInfo();          // 获取模型输入输出信息

   public:
    void inferWarmUp(const cudaStream_t& stream);  // 推理之前热启动
    size_t get_element_size(nvinfer1::DataType type);
    TensorRTInfer(const std::string model_path, const SpdlogLogger& logger);
    ~TensorRTInfer();

    // 对Host上的hInputBuffer数据进行推理，结果保存在hOutputBuffer中
    void infer(const void* hInputBuffer, float* hOutputBuffer, const cudaStream_t& stream);

    inline void getModelIOShape(array<int, 4>& input_shape, array<int, 3>& output_shape) {  // 获取模型输入输出形状
        input_shape = m_model_io_info.input_shape;
        output_shape = m_model_io_info.output_shape;
    }

    inline void getModelIODtypeSize(size_t& input_dtype_size, size_t& output_dtype_size) {  // 获取模型输入输出形状
        input_dtype_size = get_element_size(m_model_io_info.input_dtype);
        output_dtype_size = get_element_size(m_model_io_info.output_dtype);
    }
    
};

#endif  // __TENSORRT_INFER_H__
