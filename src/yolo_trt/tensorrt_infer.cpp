

#include "tensorrt_infer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

// 辅助函数：根据 TensorRT 的 DataType 获取每个元素的字节数
size_t TensorRTInfer::get_element_size(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return 4;  // FP32
        case nvinfer1::DataType::kHALF:
            return 2;  // FP16
        case nvinfer1::DataType::kINT32:
            return 4;  // INT32
        case nvinfer1::DataType::kINT8:
            return 1;  // INT8
        case nvinfer1::DataType::kBOOL:
            return 1;  // BOOL
        // case nvinfer1::DataType::kBF16: return 2; // BF16 (需TRT 8.5+ & 对应硬件)
        default:
            return 0;  // 或抛出异常
    }
}

TensorRTInfer::TensorRTInfer(const std::string model_path, const SpdlogLogger& logger)
    : m_model_path(model_path),
      m_logger(logger),
      m_runtime(nullptr),
      m_engine(nullptr),
      m_context(nullptr),
      m_dInputBuffer(nullptr),
      m_dOutputBuffer(nullptr) {
    initialize();
};

void TensorRTInfer::initialize() {
    // 1、读取TensorRT模型文件
    readModelFromFile();
    // 2、构建引擎
    buildEngine();
    // 3、获取输入输出信息
    getIOInfo();
    // 4、分配空间
    cudaMalloc(&(m_dInputBuffer), m_model_io_info.input_bytes);
    cudaMalloc(&(m_dOutputBuffer), m_model_io_info.output_bytes);
    // 5、绑定输入和输出的地址
    m_context->setTensorAddress(m_model_io_info.input_name.c_str(), m_dInputBuffer);
    m_context->setTensorAddress(m_model_io_info.output_name.c_str(), m_dOutputBuffer);
}

void TensorRTInfer::readModelFromFile() {
    std::ifstream mode_file(m_model_path, std::ios::binary);
    if (!mode_file.good()) {
        ss.str("");
        ss << "read " << m_model_path << " error!";
        m_logger.log(ILogger::Severity::kERROR, ss.str().c_str());
        assert(false);  // 断言错误，终止程序
    }
    // 读取引擎数据
    mode_file.seekg(0, mode_file.end);
    size_t size = mode_file.tellg();
    mode_file.seekg(0, std::ios::beg);  // 确保从文件头读取

    m_model_data.resize(size);  // 提前分配空间 这里要用resize，不能使用reserve
    // 读取数据 这种方式比下面的在读取大文件的时候更快更稳定
    if (!mode_file.read(m_model_data.data(), size)) {
        m_logger.log(ILogger::Severity::kERROR, "Error: Failed to read all data from model file.");
        m_model_data.clear();  // 如果读取失败，应该处理错误，清空 vector
    }
    mode_file.close();
}

void TensorRTInfer::buildEngine() {
    m_runtime = createInferRuntime(m_logger);
    assert(m_runtime);
    m_engine = m_runtime->deserializeCudaEngine(m_model_data.data(), m_model_data.size());
    assert(m_engine);
    m_context = m_engine->createExecutionContext();
    assert(m_context);
}

void TensorRTInfer::getIOInfo() {
    // 获取输入和输出的名称（YOLOv11只有1个输入、1个输出）
    size_t nbTensors = m_engine->getNbIOTensors();
    for (int i = 0; i < nbTensors; ++i) {
        const char* name = m_engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = m_engine->getTensorIOMode(name);
        nvinfer1::DataType dtype = m_engine->getTensorDataType(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            m_model_io_info.input_name = name;
            m_model_io_info.input_dtype = dtype;
        } else {
            m_model_io_info.output_name = name;
            m_model_io_info.output_dtype = dtype;
        }
        ss.str("");
        ss << (mode == TensorIOMode::kINPUT ? "Input name: " : "Output name: ") << name
           << (mode == TensorIOMode::kINPUT ? " Input dtype: " : " Output dtype: ") << static_cast<int>(dtype);
        m_logger.log(ILogger::Severity::kINFO, ss.str().c_str());
    }

    // 获取输入的形状 [B,C,H,W]
    Dims input_dims = m_engine->getTensorShape(m_model_io_info.input_name.c_str());
    for (int i = 0; i < input_dims.nbDims; ++i) {  // 这里 input_dims.nbDims == 4
        m_model_io_info.input_shape[i] = input_dims.d[i];
    }
    m_model_io_info.input_bytes = m_model_io_info.input_shape[0] * m_model_io_info.input_shape[1] *
                                  m_model_io_info.input_shape[2] * m_model_io_info.input_shape[3] *
                                  sizeof(get_element_size(m_model_io_info.input_dtype));
    ss.str("");
    ss << "input_shape [B, C, H, W]:[" << m_model_io_info.input_shape[0] << ", " << m_model_io_info.input_shape[1]
       << ", " << m_model_io_info.input_shape[2] << ", " << m_model_io_info.input_shape[3] << "]" << ", input_bytes: "
       << m_model_io_info.input_bytes;

    m_logger.log(ILogger::Severity::kINFO, ss.str().c_str());
    

    // 获取输出的形状 [B,N,M]
    Dims output_dims = m_engine->getTensorShape(m_model_io_info.output_name.c_str());
    for (int i = 0; i < output_dims.nbDims; ++i) {  // 这里 output_dims.nbDims == 3
        m_model_io_info.output_shape[i] = output_dims.d[i];
    }
    m_model_io_info.output_bytes = m_model_io_info.output_shape[0] * m_model_io_info.output_shape[1] *
                                   m_model_io_info.output_shape[2] *
                                   sizeof(get_element_size(m_model_io_info.output_dtype));
    ss.str("");
    ss << "output_shape [B, N, M]:[" << m_model_io_info.output_shape[0] << ", " << m_model_io_info.output_shape[1]
       << ", " << m_model_io_info.output_shape[2] << "]" << ", output_bytes: " << m_model_io_info.output_bytes;
    m_logger.log(ILogger::Severity::kINFO, ss.str().c_str());
    
}

void TensorRTInfer::inferWarmUp(const cudaStream_t& stream) {
    m_logger.log(ILogger::Severity::kINFO, "Begin inference warm up");
    // 0、生成随即数据
    std::vector<float> hInputBuffer(m_model_io_info.input_bytes, 0.5f);
    // 1、异步拷贝数据 (Host-> Device)
    cudaError_t err = cudaMemcpyAsync(m_dInputBuffer, hInputBuffer.data(), m_model_io_info.input_bytes,
                                      cudaMemcpyHostToDevice, stream);

    for (int i = 0; i < 10; ++i) {
        m_context->enqueueV3(stream);
    }
    err = cudaStreamSynchronize(stream);

    if (err != cudaSuccess) {
        ss.str("");
        ss << "Warmup synchronization failed: " << cudaGetErrorString(err);
        m_logger.log(ILogger::Severity::kERROR, ss.str().c_str());
    } else {
        m_logger.log(ILogger::Severity::kINFO, "TensorRT Warmup complete with 10 times.");
    }
}
void TensorRTInfer::infer(const void* hInputBuffer, float* hOutputBuffer, const cudaStream_t& stream) {
    // 1、异步拷贝数据 (Host-> Device)
    cudaError_t err = cudaMemcpyAsync(m_dInputBuffer,               // GPU
                                      hInputBuffer,                 // CPU
                                      m_model_io_info.input_bytes,  // 拷贝的字节数
                                      cudaMemcpyHostToDevice,       // 拷贝方向
                                      stream);                      // 在同一 CUDA 流上执行
    if (err != cudaSuccess) {
        m_logger.log(ILogger::Severity::kERROR, "CUDA error during memcpy.");
        assert(false);
    } else {
        m_logger.log(ILogger::Severity::kINFO, "Successfully copy result Host-> Device.");
    }

    // 2、推理
    // auto start = std::chrono::system_clock::now();
    m_context->enqueueV3(stream);
    // auto end = std::chrono::system_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // ss.str("");
    // ss << "inference cost time: " << duration << " us";
    // m_logger.log(ILogger::Severity::kINFO, ss.str().c_str());

    // 3、从GPU拷贝到CPU (Device-> Host)
    err = cudaMemcpyAsync(hOutputBuffer,                 // CPU
                          m_dOutputBuffer,               // GPU
                          m_model_io_info.output_bytes,  // 拷贝的字节数
                          cudaMemcpyDeviceToHost,        // 拷贝方向
                          stream);                       // 在同一 CUDA 流上执行
    if (err != cudaSuccess) {
        m_logger.log(ILogger::Severity::kERROR, "CUDA error during memcpy.");
        assert(false);
    } else {
        m_logger.log(ILogger::Severity::kINFO, "Successfully copy result Device-> Host.");
    }
}

TensorRTInfer::~TensorRTInfer() {
    // 按照顺序释放资源 否则会报错 相互之间有依赖关系
    if (m_context != nullptr) {
        delete m_context;
        m_context = nullptr;
        m_logger.log(ILogger::Severity::kINFO, "TensorRT Context destroyed.");
    }
    if (m_engine != nullptr) {
        delete m_engine;
        m_engine = nullptr;
        m_logger.log(ILogger::Severity::kINFO, "TensorRT Engine destroyed.");
    }
    if (m_runtime != nullptr) {
        delete m_runtime;
        m_runtime = nullptr;
        m_logger.log(ILogger::Severity::kINFO, "TensorRT Runtime destroyed.");
    }
    if (m_dInputBuffer != nullptr) {
        cudaFree(m_dInputBuffer);
        m_logger.log(ILogger::Severity::kINFO, "Release cuda m_dInputBuffer.");
    }
    if (m_dOutputBuffer != nullptr) {
        cudaFree(m_dOutputBuffer);
        m_logger.log(ILogger::Severity::kINFO, "Release cuda m_dOutputBuffer.");
    }
}
