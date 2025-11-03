#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <memory>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstring>

// TensorRT includes
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "cuda_runtime_api.h"

// CUDA 错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// TensorRT 错误检查宏
#define CHECK_TRT(call) \
    do { \
        if (!(call)) { \
            std::cerr << "TensorRT error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// 简单的 Logger
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

// Half 精度转换函数
float half2float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;

    if (exponent == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            float f = mantissa / 1024.0f;
            f = f / 16384.0f;
            return sign ? -f : f;
        }
    } else if (exponent == 31) {
        if (mantissa == 0) {
            return sign ? -INFINITY : INFINITY;
        } else {
            return NAN;
        }
    }

    float f = 1.0f + (mantissa / 1024.0f);
    int exp = exponent - 15;
    f = f * std::pow(2.0f, exp);

    return sign ? -f : f;
}

class SimpleAlexNetInfer {
private:
    Logger logger;
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;

    // 网络参数
    static const int INPUT_N = 1;
    static const int INPUT_C = 3;
    static const int INPUT_H = 224;
    static const int INPUT_W = 224;
    static const int OUTPUT_SIZE = 1000;

    const std::string INPUT_BLOB_NAME = "alexnet_input";
    const std::string OUTPUT_BLOB_NAME = "alexnet_output";

    // 权重映射
    std::map<std::string, std::vector<float>> weightMap;

    // GPU 内存
    void* inputBuffer = nullptr;
    void* outputBuffer = nullptr;
    void* buffers[2] = {nullptr, nullptr};

    // 输入输出尺寸
    size_t inputSize;
    size_t outputSize;

public:
    SimpleAlexNetInfer() {
        // 使用FP16输入，所以是2字节而不是4字节
        inputSize = INPUT_N * INPUT_C * INPUT_H * INPUT_W * sizeof(uint16_t);
        outputSize = OUTPUT_SIZE * sizeof(float);

        // 分配 GPU 内存
        CHECK_CUDA(cudaMalloc(&inputBuffer, inputSize));
        CHECK_CUDA(cudaMalloc(&outputBuffer, outputSize));

        buffers[0] = inputBuffer;
        buffers[1] = outputBuffer;
    }

    ~SimpleAlexNetInfer() {
        if (inputBuffer) cudaFree(inputBuffer);
        if (outputBuffer) cudaFree(outputBuffer);
        if (context) context->destroy();
        if (engine) engine->destroy();
        if (runtime) runtime->destroy();
    }

    // 加载权重文件
    bool loadWeights(const std::string& filename) {
        std::cout << "加载权重文件: " << filename << std::endl;

        std::ifstream input(filename);
        if (!input.is_open()) {
            std::cerr << "错误: 无法打开权重文件 " << filename << std::endl;
            return false;
        }

        int32_t count;
        input >> count;

        if (count <= 0) {
            std::cerr << "错误: 无效的权重数量 " << count << std::endl;
            return false;
        }

        std::cout << "加载 " << count << " 个权重层..." << std::endl;

        while (count--) {
            std::string name;
            uint32_t size;
            input >> name >> std::dec >> size;

            std::cout << "加载层: " << name << " (" << size << " 参数)" << std::endl;

            std::vector<float> weights(size);

            // 读取十六进制权重数据并转换为float (统一FP16格式)
            for (uint32_t i = 0; i < size; ++i) {
                uint32_t hex_val;
                input >> std::hex >> hex_val;

                // FP16权重：从16位hex转换
                uint16_t fp16_val = static_cast<uint16_t>(hex_val);
                weights[i] = half2float(fp16_val);
            }

            weightMap[name] = std::move(weights);
        }

        std::cout << "权重加载完成!" << std::endl;
        return true;
    }

    // 构建网络
    bool buildNetwork() {
        std::cout << "构建 AlexNet 网络..." << std::endl;

        auto builder = nvinfer1::createInferBuilder(logger);
        if (!builder) {
            std::cerr << "错误: 创建 builder 失败" << std::endl;
            return false;
        }

        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(explicitBatch);
        if (!network) {
            std::cerr << "错误: 创建 network 失败" << std::endl;
            builder->destroy();
            return false;
        }

        // 构建简化的 AlexNet 架构
        if (!buildAlexNet(network)) {
            std::cerr << "错误: 构建 AlexNet 失败" << std::endl;
            network->destroy();
            builder->destroy();
            return false;
        }

        // 创建引擎配置
        auto config = builder->createBuilderConfig();
        if (!config) {
            std::cerr << "错误: 创建 config 失败" << std::endl;
            network->destroy();
            builder->destroy();
            return false;
        }

        // 设置最大工作空间大小
        config->setMaxWorkspaceSize(1U << 30); // 1GB

        // 启用FP16精度
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "启用 FP16 精度推理" << std::endl;

        // 构建引擎
        auto enginePlan = builder->buildSerializedNetwork(*network, *config);
        if (!enginePlan) {
            std::cerr << "错误: 构建引擎失败" << std::endl;
            config->destroy();
            network->destroy();
            builder->destroy();
            return false;
        }

        // 创建运行时
        runtime = nvinfer1::createInferRuntime(logger);
        if (!runtime) {
            std::cerr << "错误: 创建 runtime 失败" << std::endl;
            return false;
        }

        // 反序列化引擎
        engine = runtime->deserializeCudaEngine(enginePlan->data(), enginePlan->size());
        if (!engine) {
            std::cerr << "错误: 反序列化引擎失败" << std::endl;
            return false;
        }

        // 创建执行上下文
        context = engine->createExecutionContext();
        if (!context) {
            std::cerr << "错误: 创建 execution context 失败" << std::endl;
            return false;
        }

        // 清理临时对象
        enginePlan->destroy();
        config->destroy();
        network->destroy();
        builder->destroy();

        std::cout << "网络构建完成!" << std::endl;
        return true;
    }


    bool buildAlexNet(nvinfer1::INetworkDefinition* network) {
    

        auto input = network->addInput(INPUT_BLOB_NAME.c_str(), nvinfer1::DataType::kHALF,
                                     nvinfer1::Dims4{INPUT_N, INPUT_C, INPUT_H, INPUT_W});
        if (!input) {
            std::cerr << "错误: 添加输入层失败" << std::endl;
            return false;
        }
        std::cout << "✅ 创建FP16输入层: [1, 3, 224, 224] (NCHW)" << std::endl;

        // Conv1: features.0 - Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        std::cout << "添加 Conv1: 3->64, kernel=11, stride=4, pad=2" << std::endl;
        auto conv1 = addConvBNRelu(network, *input, 64, 11, 4, 2, "features.0", "features.1");
        if (!conv1) return false;

        // MaxPool1: features.2 - MaxPool2d(kernel_size=3, stride=2)
        std::cout << "添加 MaxPool1: kernel=3, stride=2" << std::endl;
        auto pool1 = network->addPoolingNd(*conv1, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{3, 3});
        pool1->setStrideNd(nvinfer1::DimsHW{2, 2});

        // Conv2: features.3 - Conv2d(64, 192, kernel_size=5, padding=2)
        std::cout << "添加 Conv2: 64->192, kernel=5, stride=1, pad=2" << std::endl;
        auto conv2 = addConvBNRelu(network, *pool1->getOutput(0), 192, 5, 1, 2, "features.3", "features.4");
        if (!conv2) return false;

        // MaxPool2: features.5 - MaxPool2d(kernel_size=3, stride=2)
        std::cout << "添加 MaxPool2: kernel=3, stride=2" << std::endl;
        auto pool2 = network->addPoolingNd(*conv2, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{3, 3});
        pool2->setStrideNd(nvinfer1::DimsHW{2, 2});

        // Conv3: features.6 - Conv2d(192, 384, kernel_size=3, padding=1)
        std::cout << "添加 Conv3: 192->384, kernel=3, stride=1, pad=1" << std::endl;
        auto conv3 = addConvRelu(network, *pool2->getOutput(0), 384, 3, 1, 1, "features.6");
        if (!conv3) return false;

        // Conv4: features.8 - Conv2d(384, 256, kernel_size=3, padding=1)
        std::cout << "添加 Conv4: 384->256, kernel=3, stride=1, pad=1" << std::endl;
        auto conv4 = addConvRelu(network, *conv3, 256, 3, 1, 1, "features.8");
        if (!conv4) return false;

        // Conv5: features.10 - Conv2d(256, 256, kernel_size=3, padding=1)
        std::cout << "添加 Conv5: 256->256, kernel=3, stride=1, pad=1" << std::endl;
        auto conv5 = addConvRelu(network, *conv4, 256, 3, 1, 1, "features.10");
        if (!conv5) return false;

        // MaxPool3: features.12 - MaxPool2d(kernel_size=3, stride=2)
        std::cout << "添加 MaxPool3: kernel=3, stride=2" << std::endl;
        auto pool3 = network->addPoolingNd(*conv5, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{3, 3});
        pool3->setStrideNd(nvinfer1::DimsHW{2, 2});

        // Flatten - 从 [1, 256, 6, 6] 展平为 [1, 9216]
        std::cout << "添加 Flatten: [1, 256, 6, 6] -> [1, 9216]" << std::endl;
        auto flatten = network->addShuffle(*pool3->getOutput(0));
        flatten->setReshapeDimensions(nvinfer1::Dims4{1, 256 * 6 * 6, 1, 1});

        // FC1: classifier.1 - Linear(9216, 4096)
        std::cout << "添加 FC1: 9216->4096" << std::endl;
        auto fc1 = addLinearRelu(network, *flatten->getOutput(0), 4096, "classifier.1");
        if (!fc1) return false;

        // FC2: classifier.4 - Linear(4096, 4096)
        std::cout << "添加 FC2: 4096->4096" << std::endl;
        auto fc2 = addLinearRelu(network, *fc1, 4096, "classifier.4");
        if (!fc2) return false;

        // FC3: classifier.6 - Linear(4096, 1000) - 输出层
        std::cout << "添加 FC3: 4096->1000 (输出层)" << std::endl;
        auto fc3 = addLinear(network, *fc2, 1000, "classifier.6");
        if (!fc3) return false;

        auto softmax = network->addSoftMax(*fc3);
        if (!softmax) return false;

        // 标记输出
        network->markOutput(*softmax->getOutput(0));
        softmax->getOutput(0)->setName(OUTPUT_BLOB_NAME.c_str());

        std::cout << "✅ AlexNet网络架构构建完成!" << std::endl;
        return true;
    }

    nvinfer1::ITensor* addConvBNRelu(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                   int outCh, int ksize, int stride, int pad,
                                   const std::string& convName, const std::string& bnName) {
        auto conv = addConv(network, input, outCh, ksize, stride, pad, convName);
        if (!conv) return nullptr;

        auto relu = network->addActivation(*conv, nvinfer1::ActivationType::kRELU);
        return relu->getOutput(0);
    }

    // 添加卷积+ReLU层
    nvinfer1::ITensor* addConvRelu(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                 int outCh, int ksize, int stride, int pad, const std::string& name) {
        auto conv = addConv(network, input, outCh, ksize, stride, pad, name);
        if (!conv) return nullptr;

        auto relu = network->addActivation(*conv, nvinfer1::ActivationType::kRELU);
        return relu->getOutput(0);
    }

    // 添加卷积层
    nvinfer1::ITensor* addConv(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                             int outCh, int ksize, int stride, int pad, const std::string& name) {
        auto weightKey = name + ".weight";
        auto biasKey = name + ".bias";

        if (weightMap.find(weightKey) == weightMap.end() || weightMap.find(biasKey) == weightMap.end()) {
            std::cerr << "错误: 未找到权重 " << weightKey << " 或 " << biasKey << std::endl;
            return nullptr;
        }

        // int inCh = input.getDimensions().d[1];  // 未使用，注释掉
        auto& weights = weightMap[weightKey];
        auto& bias = weightMap[biasKey];

        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, weights.data(), (int64_t)weights.size()};
        nvinfer1::Weights bt{nvinfer1::DataType::kFLOAT, bias.data(), (int64_t)bias.size()};

        auto conv = network->addConvolutionNd(input, outCh, nvinfer1::DimsHW{ksize, ksize}, wt, bt);
        conv->setStrideNd(nvinfer1::DimsHW{stride, stride});
        conv->setPaddingNd(nvinfer1::DimsHW{pad, pad});

        return conv->getOutput(0);
    }

    // 添加批归一化层
    nvinfer1::ITensor* addBatchNorm(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input, const std::string& name) {
        auto weightKey = name + ".weight";
        auto biasKey = name + ".bias";
        auto meanKey = name + ".running_mean";
        auto varKey = name + ".running_var";

        if (weightMap.find(weightKey) == weightMap.end()) {
            return &input; // 如果没有BN权重，直接返回输入
        }

        auto& gamma = weightMap[weightKey];
        auto& beta = weightMap[biasKey];
        auto& mean = weightMap[meanKey];
        auto& var = weightMap[varKey];

        // 计算 scale 和 shift
        std::vector<float> scale(gamma.size());
        std::vector<float> shift(beta.size());
        const float eps = 1e-5;

        for (size_t i = 0; i < gamma.size(); ++i) {
            scale[i] = gamma[i] / std::sqrt(var[i] + eps);
            shift[i] = beta[i] - mean[i] * scale[i];
        }

        nvinfer1::Weights scaleWt{nvinfer1::DataType::kFLOAT, scale.data(), (int64_t)scale.size()};
        nvinfer1::Weights shiftWt{nvinfer1::DataType::kFLOAT, shift.data(), (int64_t)shift.size()};
        nvinfer1::Weights powerWt{nvinfer1::DataType::kFLOAT, nullptr, 0};

        auto bn = network->addScaleNd(input, nvinfer1::ScaleMode::kCHANNEL, shiftWt, scaleWt, powerWt, 1);
        return bn->getOutput(0);
    }

    // 添加全连接+ReLU层
    nvinfer1::ITensor* addLinearRelu(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                   int outSize, const std::string& name) {
        auto fc = addLinear(network, input, outSize, name);
        if (!fc) return nullptr;

        auto relu = network->addActivation(*fc, nvinfer1::ActivationType::kRELU);
        return relu->getOutput(0);
    }

    // 添加全连接层
    nvinfer1::ITensor* addLinear(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                               int outSize, const std::string& name) {
        auto weightKey = name + ".weight";
        auto biasKey = name + ".bias";

        if (weightMap.find(weightKey) == weightMap.end() || weightMap.find(biasKey) == weightMap.end()) {
            std::cerr << "错误: 未找到权重 " << weightKey << " 或 " << biasKey << std::endl;
            return nullptr;
        }

        auto& weights = weightMap[weightKey];
        auto& bias = weightMap[biasKey];

        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, weights.data(), (int64_t)weights.size()};
        nvinfer1::Weights bt{nvinfer1::DataType::kFLOAT, bias.data(), (int64_t)bias.size()};

        auto fc = network->addFullyConnected(input, outSize, wt, bt);
        return fc->getOutput(0);
    }

    // 加载输入数据 (支持FP16)
    bool loadInput(const std::string& filename) {
        std::cout << "加载输入文件: " << filename << std::endl;

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "错误: 无法打开输入文件 " << filename << std::endl;
            return false;
        }

        // 根据文件名判断数据类型
        bool isFP16 = filename.find("fp16") != std::string::npos;

        if (isFP16) {
            // FP16输入数据
            std::vector<uint16_t> inputDataFP16(INPUT_N * INPUT_C * INPUT_H * INPUT_W);
            file.read(reinterpret_cast<char*>(inputDataFP16.data()), inputSize);
            file.close();

            // 将FP16数据复制到GPU
            CHECK_CUDA(cudaMemcpy(inputBuffer, inputDataFP16.data(), inputSize, cudaMemcpyHostToDevice));
            std::cout << "✅ FP16输入数据加载完成!" << std::endl;
        } else {
            // FP32输入数据，需要转换为FP16
            std::vector<float> inputDataFP32(INPUT_N * INPUT_C * INPUT_H * INPUT_W);
            file.read(reinterpret_cast<char*>(inputDataFP32.data()), inputSize);
            file.close();

            // 转换为FP16
            std::vector<uint16_t> inputDataFP16(INPUT_N * INPUT_C * INPUT_H * INPUT_W);
            for (size_t i = 0; i < inputDataFP32.size(); ++i) {
                // 简单的FP32到FP16转换
                float val = inputDataFP32[i];
                uint16_t fp16_val = 0;

                // 提取符号位
                uint32_t sign = (*(uint32_t*)&val) >> 31;
                uint32_t exp = ((*(uint32_t*)&val) >> 23) & 0xFF;
                uint32_t mantissa = (*(uint32_t*)&val) & 0x7FFFFF;

                if (exp == 0) {
                    fp16_val = sign << 15;
                } else if (exp == 255) {
                    fp16_val = (sign << 15) | 0x7C00;
                } else {
                    int new_exp = exp - 127 + 15;
                    if (new_exp <= 0) {
                        fp16_val = sign << 15;
                    } else if (new_exp >= 31) {
                        fp16_val = (sign << 15) | 0x7C00;
                    } else {
                        fp16_val = (sign << 15) | (new_exp << 10) | (mantissa >> 13);
                    }
                }

                inputDataFP16[i] = fp16_val;
            }

            // 将FP16数据复制到GPU
            CHECK_CUDA(cudaMemcpy(inputBuffer, inputDataFP16.data(), inputSize, cudaMemcpyHostToDevice));
            std::cout << "✅ FP32->FP16输入数据转换完成!" << std::endl;
        }

        return true;
    }

    // 执行推理
    bool doInference(std::vector<float>& output) {
        std::cout << "执行推理..." << std::endl;

        // 执行推理
        auto start = std::chrono::high_resolution_clock::now();
        bool success = context->executeV2(buffers);
        auto end = std::chrono::high_resolution_clock::now();

        if (!success) {
            std::cerr << "错误: 推理执行失败" << std::endl;
            return false;
        }

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "推理耗时: " << duration.count() << " ms" << std::endl;

        // 将结果从 GPU 复制到 CPU
        output.resize(OUTPUT_SIZE);
        CHECK_CUDA(cudaMemcpy(output.data(), outputBuffer, outputSize, cudaMemcpyDeviceToHost));

        return true;
    }

    // 性能统计结构
    struct PerfStats {
        double min_time = 1e9;
        double max_time = 0.0;
        double total_time = 0.0;
        double avg_time = 0.0;
        int iterations = 0;
        bool valid = false;

        void addSample(double time_ms) {
            min_time = std::min(min_time, time_ms);
            max_time = std::max(max_time, time_ms);
            total_time += time_ms;
            iterations++;
            avg_time = total_time / iterations;
            valid = true;
        }

        void printStats(const std::string& name) {
            std::cout << "\n=== " << name << " 性能统计 ===" << std::endl;
            if (!valid) {
                std::cout << "⚠️ 测试失败，无有效数据" << std::endl;
                return;
            }
            std::cout << "执行次数: " << iterations << std::endl;
            std::cout << "平均时间: " << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
            std::cout << "最小时间: " << std::fixed << std::setprecision(3) << min_time << " ms" << std::endl;
            std::cout << "最大时间: " << std::fixed << std::setprecision(3) << max_time << " ms" << std::endl;
            std::cout << "总计时间: " << std::fixed << std::setprecision(3) << total_time << " ms" << std::endl;
            std::cout << "吞吐量: " << std::fixed << std::setprecision(2) << (1000.0 / avg_time) << " FPS" << std::endl;
        }
    };

    // 计算TOPS性能指标
    void calculateTOPS(const PerfStats& stats) {
        if (!stats.valid) {
            std::cout << "⚠️ 无法计算TOPS，性能数据无效" << std::endl;
            return;
        }

        // AlexNet模型操作数统计 (基于FP16精度)
        const double alexnet_ops = 71600000.0;  // AlexNet总操作数
        const double fp16_ops = alexnet_ops * 2;  // FP16需要2倍操作

        // 转换为秒
        double time_seconds = stats.avg_time / 1000.0;

        // 计算实际TOPS
        double actual_tops = fp16_ops / (time_seconds * 1e12);

        // 计算理论峰值TOPS (假设GPU规格)
        double theoretical_peak = 0.0;
        std::string gpu_info = "未知GPU";

        // 尝试获取GPU信息
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            gpu_info = prop.name;
            // 基于GPU规格估算理论峰值
            if (strstr(prop.name, "RTX") || strstr(prop.name, "GTX")) {
                theoretical_peak = 10.0;  // 假设10 TOPS理论峰值
            } else if (strstr(prop.name, "Tesla")) {
                theoretical_peak = 20.0;  // 假设20 TOPS理论峰值
            } else {
                theoretical_peak = 5.0;   // 默认5 TOPS
            }
        }

        std::cout << "\n=== TOPS性能分析 ===" << std::endl;
        std::cout << "GPU设备: " << gpu_info << std::endl;
        std::cout << "模型: AlexNet (FP16精度)" << std::endl;
        std::cout << "总操作数: " << std::scientific << std::setprecision(2) << fp16_ops << std::endl;
        std::cout << "平均推理时间: " << std::fixed << std::setprecision(3) << stats.avg_time << " ms" << std::endl;
        std::cout << "实际TOPS: " << std::fixed << std::setprecision(6) << actual_tops << " TOPS" << std::endl;
        std::cout << "理论峰值: " << std::fixed << std::setprecision(2) << theoretical_peak << " TOPS" << std::endl;

        if (theoretical_peak > 0) {
            double efficiency = (actual_tops / theoretical_peak) * 100.0;
            std::cout << "硬件效率: " << std::fixed << std::setprecision(2) << efficiency << "%" << std::endl;
        }

        // 计算不同精度下的TOPS
        double int8_tops = actual_tops * 2.0;  // INT8精度下TOPS翻倍
        double fp32_tops = actual_tops * 0.5;  // FP32精度下TOPS减半

        std::cout << "\n=== 不同精度TOPS预估 ===" << std::endl;
        std::cout << "INT8精度预估: " << std::fixed << std::setprecision(6) << int8_tops << " TOPS" << std::endl;
        std::cout << "FP16精度实测: " << std::fixed << std::setprecision(6) << actual_tops << " TOPS" << std::endl;
        std::cout << "FP32精度预估: " << std::fixed << std::setprecision(6) << fp32_tops << " TOPS" << std::endl;

        // 性能建议
        std::cout << "\n=== 性能优化建议 ===" << std::endl;
        if (actual_tops < 1.0) {
            std::cout << "🔧 当前TOPS较低，建议:" << std::endl;
            std::cout << "  - 检查GPU利用率是否充分" << std::endl;
            std::cout << "  - 考虑使用INT8量化提升性能" << std::endl;
            std::cout << "  - 优化内存访问模式" << std::endl;
        } else if (actual_tops < 5.0) {
            std::cout << "⚡ 性能中等，可进一步优化:" << std::endl;
            std::cout << "  - 尝试TensorRT优化策略" << std::endl;
            std::cout << "  - 考虑模型剪枝或量化" << std::endl;
        } else {
            std::cout << "🚀 性能良好!" << std::endl;
        }
    }

    // 执行性能测试
    bool doPerformanceTest(int iterations = 100, int warmup = 10) {
        std::cout << "\n=== 开始性能测试 ===" << std::endl;
        std::cout << "预热次数: " << warmup << std::endl;
        std::cout << "测试次数: " << iterations << std::endl;

        PerfStats stats;

        // 预热阶段
        std::cout << "预热中..." << std::flush;
        for (int i = 0; i < warmup; ++i) {
            std::vector<float> dummy_output;
            if (!doInference(dummy_output)) {
                std::cerr << "预热失败" << std::endl;
                return false;
            }
            if (i % 2 == 1) std::cout << "." << std::flush;
        }
        std::cout << " 完成" << std::endl;

        // 性能测试
        std::cout << "性能测试中..." << std::flush;
        for (int i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            std::vector<float> output;
            if (!doInference(output)) {
                std::cerr << "推理失败 at iteration " << i << std::endl;
                return false;
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double time_ms = duration.count() / 1000.0;
            stats.addSample(time_ms);

            if (i % 10 == 9) std::cout << "." << std::flush;
        }
        std::cout << " 完成" << std::endl;

        // 打印性能统计
        stats.printStats("AlexNet FP16推理");

        // 计算TOPS
        calculateTOPS(stats);

        return true;
    }

    // 打印结果
    void printResults(const std::vector<float>& output) {
        std::cout << "\n=== 推理结果 ===" << std::endl;

        // 找到最大值的索引
        auto maxIt = std::max_element(output.begin(), output.end());
        int maxIndex = std::distance(output.begin(), maxIt);

        std::cout << "预测类别: " << maxIndex << std::endl;
        std::cout << "置信度: " << *maxIt << std::endl;

        // 打印 Top 5 结果
        std::vector<std::pair<float, int>> scoreIndex;
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            scoreIndex.push_back(std::make_pair(output[i], i));
        }
        std::sort(scoreIndex.begin(), scoreIndex.end(), std::greater<std::pair<float, int>>());

        std::cout << "\nTop 5 预测结果:" << std::endl;
        for (int i = 0; i < 5; ++i) {
            std::cout << "  " << (i+1) << ". 类别 " << scoreIndex[i].second
                     << ": " << std::fixed << std::setprecision(6) << scoreIndex[i].first << std::endl;
        }
    }
};

int main() {
    std::cout << "=== AlexNet TensorRT 完整FP16推理 + TOPS性能分析 ===" << std::endl;
    std::cout << "✅ 初始化FP16推理引擎" << std::endl;
    std::cout << "  输入大小: " << (1 * 3 * 224 * 224 * sizeof(float)) << " bytes (" << (1 * 3 * 224 * 224 * sizeof(float) / 1024) << " KB)" << std::endl;
    std::cout << "  输出大小: " << (1000 * sizeof(float)) << " bytes (" << (1000 * sizeof(float) / 1024) << " KB)" << std::endl;

    SimpleAlexNetInfer infer;

    // 加载FP16权重
    std::cout << "\n📁 加载FP16权重文件: alexnet-tensorrt-fp16.wts" << std::endl;
    if (!infer.loadWeights("alexnet-tensorrt-fp16.wts")) {
        std::cerr << "❌ 权重加载失败" << std::endl;
        return -1;
    }
    std::cout << "✅ 权重加载完成!" << std::endl;

    // 构建网络
    std::cout << "\n🔧 构建 FP16 AlexNet 网络..." << std::endl;
    if (!infer.buildNetwork()) {
        std::cerr << "❌ 网络构建失败" << std::endl;
        return -1;
    }
    std::cout << "✅ FP16网络构建完成!" << std::endl;

    // 加载输入 - 使用NCHW格式的输入文件
    std::cout << "\n📥 加载输入文件: input_fp32_nchw.bin" << std::endl;
    if (!infer.loadInput("input_fp32_nchw.bin")) {
        std::cerr << "❌ 输入加载失败" << std::endl;
        return -1;
    }
    std::cout << "✅ 输入数据加载完成!" << std::endl;

    // 执行单次推理验证
    std::cout << "\n🔍 执行单次推理验证..." << std::endl;
    std::vector<float> output;
    if (!infer.doInference(output)) {
        std::cerr << "❌ 推理执行失败" << std::endl;
        return -1;
    }

    // 打印推理结果
    std::cout << "\n=== 推理结果验证 ===" << std::endl;
    infer.printResults(output);

    // 执行性能测试和TOPS计算
    std::cout << "\n🚀 开始性能测试和TOPS分析..." << std::endl;
    if (!infer.doPerformanceTest(100, 10)) {  // 100次测试，10次预热
        std::cerr << "❌ 性能测试失败" << std::endl;
        return -1;
    }

    // 显示GPU信息
    std::cout << "\n=== 硬件信息 ===" << std::endl;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        std::cout << "GPU设备: " << prop.name << std::endl;
        std::cout << "计算能力: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "显存大小: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "多处理器数量: " << prop.multiProcessorCount << std::endl;
        std::cout << "最大线程数: " << prop.maxThreadsPerBlock << std::endl;
    } else {
        std::cout << "⚠️ 无法获取GPU信息" << std::endl;
    }

    std::cout << "\n🎉 AlexNet FP16推理 + TOPS分析完成!" << std::endl;
    std::cout << "📊 性能分析报告已生成，请查看上述TOPS计算结果" << std::endl;

    return 0;
}