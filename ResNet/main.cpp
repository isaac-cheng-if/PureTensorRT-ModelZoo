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

class ResNet50TensorRT {
private:
    Logger logger;
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    nvinfer1::IBuilder* builder;
    nvinfer1::INetworkDefinition* network;

    // 网络参数 - 固定Batch=16, 使用FP16
    static const int INPUT_N = 16;  // 固定batch size
    static const int INPUT_C = 3;
    static const int INPUT_H = 224;
    static const int INPUT_W = 224;
    static const int OUTPUT_SIZE = 1000;
    
    // 使用FP16数据类型
    static const nvinfer1::DataType INPUT_DATA_TYPE = nvinfer1::DataType::kHALF;
    static const nvinfer1::DataType OUTPUT_DATA_TYPE = nvinfer1::DataType::kHALF;

    const std::string INPUT_BLOB_NAME = "resnet50_input";
    const std::string OUTPUT_BLOB_NAME = "resnet50_output";

    // 权重映射 - 使用FP16
    std::map<std::string, std::vector<uint16_t>> weightMap;

    // 转换后的float权重存储 - 确保Weights对象的生命周期
    std::map<std::string, std::vector<float>> floatWeightMap;

    // GPU 内存
    void* inputBuffer = nullptr;
    void* outputBuffer = nullptr;

public:
    ResNet50TensorRT() : runtime(nullptr), engine(nullptr), context(nullptr), builder(nullptr), network(nullptr) {
        // 初始化CUDA
        CHECK_CUDA(cudaSetDevice(0));
        
        // 创建TensorRT运行时
        runtime = nvinfer1::createInferRuntime(logger);
        CHECK_TRT(runtime != nullptr);
        
        // 创建构建器
        builder = nvinfer1::createInferBuilder(logger);
        CHECK_TRT(builder != nullptr);
        
        // 创建网络定义
        network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        CHECK_TRT(network != nullptr);
    }

    ~ResNet50TensorRT() {
        // 清理GPU内存
        if (inputBuffer) {
            cudaFree(inputBuffer);
            inputBuffer = nullptr;
        }
        if (outputBuffer) {
            cudaFree(outputBuffer);
            outputBuffer = nullptr;
        }
        
        // 清理TensorRT对象
        if (context) {
            context->destroy();
            context = nullptr;
        }
        if (engine) {
            engine->destroy();
            engine = nullptr;
        }
        if (runtime) {
            runtime->destroy();
            runtime = nullptr;
        }
        if (network) {
            network->destroy();
            network = nullptr;
        }
        if (builder) {
            builder->destroy();
            builder = nullptr;
        }
    }

    // 加载FP16权重
    void LoadWeights(const std::string& file) {
        std::cout << "Loading FP16 weights: " << file << std::endl;

        std::ifstream input(file);
        if (!input.is_open()) {
            std::cerr << "ERROR: Unable to load weight file: " << file << std::endl;
            exit(1);
        }

        // 读取权重数量
        int32_t count;
        input >> count;
        if (count <= 0) {
            std::cerr << "ERROR: Invalid weight count: " << count << std::endl;
            exit(1);
        }

        std::cout << "Loading " << count << " weight tensors..." << std::endl;

        while (count--) {
            std::string name;
            uint32_t size;
            input >> name >> std::dec >> size;

            // 加载FP16权重
            std::vector<uint16_t> val(size);
            for (uint32_t x = 0; x < size; ++x) {
                uint32_t temp;
                input >> std::hex >> temp;
                val[x] = static_cast<uint16_t>(temp & 0xFFFF);
            }

            weightMap[name] = val;
        }

        std::cout << "Successfully loaded " << weightMap.size() << " weight tensors" << std::endl;
    }

    // 创建卷积层（不带bias，ResNet标准实现）
    nvinfer1::IConvolutionLayer* AddConvLayer(nvinfer1::ITensor* input,
                                             const std::string& weightName,
                                             int outChannels, int kernelSize,
                                             nvinfer1::DimsHW stride, nvinfer1::DimsHW padding) {
        // 获取权重
        auto weightIt = weightMap.find(weightName);

        if (weightIt == weightMap.end()) {
            std::cerr << "ERROR: Weight not found: " << weightName << std::endl;
            exit(1);
        }

        // 将FP16权重转换为float用于TensorRT，并存储在map中以确保生命周期
        std::string weightKey = weightName + "_float";

        // 只在第一次调用时转换并存储
        if (floatWeightMap.find(weightKey) == floatWeightMap.end()) {
            std::vector<float> floatWeights;
            for (auto w : weightIt->second) {
                floatWeights.push_back(half2float(w));
            }
            floatWeightMap[weightKey] = std::move(floatWeights);
        }

        // 创建Weights对象，使用存储在map中的数据
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, floatWeightMap[weightKey].data(), static_cast<int64_t>(floatWeightMap[weightKey].size())};

        // ResNet的Conv层没有bias，使用空的Weights
        nvinfer1::Weights emptyBias{nvinfer1::DataType::kFLOAT, nullptr, 0};

        // 创建卷积层
        auto conv = network->addConvolutionNd(*input, outChannels, nvinfer1::DimsHW{kernelSize, kernelSize}, wt, emptyBias);
        conv->setStrideNd(stride);
        conv->setPaddingNd(padding);
        conv->setName(("conv_" + weightName).c_str());

        return conv;
    }

    // 创建BatchNorm层
    nvinfer1::IScaleLayer* AddBatchNormLayer(nvinfer1::ITensor* input,
                                            const std::string& weightName,
                                            const std::string& biasName,
                                            const std::string& meanName,
                                            const std::string& varName,
                                            float epsilon = 1e-5) {
        // 获取参数
        auto weightIt = weightMap.find(weightName);
        auto biasIt = weightMap.find(biasName);
        auto meanIt = weightMap.find(meanName);
        auto varIt = weightMap.find(varName);

        if (weightIt == weightMap.end() || biasIt == weightMap.end() ||
            meanIt == weightMap.end() || varIt == weightMap.end()) {
            std::cerr << "ERROR: BatchNorm parameters not found" << std::endl;
            exit(1);
        }

        // 生成存储键
        std::string scaleKey = weightName + "_scale";
        std::string shiftKey = weightName + "_shift";

        // 只在第一次调用时转换并计算
        if (floatWeightMap.find(scaleKey) == floatWeightMap.end()) {
            // 将FP16参数转换为float
            std::vector<float> gamma, beta, mean, var;
            for (auto w : weightIt->second) gamma.push_back(half2float(w));
            for (auto b : biasIt->second) beta.push_back(half2float(b));
            for (auto m : meanIt->second) mean.push_back(half2float(m));
            for (auto v : varIt->second) var.push_back(half2float(v));

            // 计算 scale 和 shift: y = scale * x + shift
            // BatchNorm公式: y = gamma * (x - mean) / sqrt(var + eps) + beta
            // 转换为: y = (gamma / sqrt(var + eps)) * x + (beta - mean * gamma / sqrt(var + eps))
            std::vector<float> scale(gamma.size());
            std::vector<float> shift(beta.size());

            for (size_t i = 0; i < gamma.size(); ++i) {
                scale[i] = gamma[i] / std::sqrt(var[i] + epsilon);
                shift[i] = beta[i] - mean[i] * scale[i];
            }

            // 存储计算结果
            floatWeightMap[scaleKey] = std::move(scale);
            floatWeightMap[shiftKey] = std::move(shift);
        }

        // 创建Weights对象，使用存储在map中的数据
        nvinfer1::Weights scaleWt{nvinfer1::DataType::kFLOAT, floatWeightMap[scaleKey].data(), static_cast<int64_t>(floatWeightMap[scaleKey].size())};
        nvinfer1::Weights shiftWt{nvinfer1::DataType::kFLOAT, floatWeightMap[shiftKey].data(), static_cast<int64_t>(floatWeightMap[shiftKey].size())};
        nvinfer1::Weights powerWt{nvinfer1::DataType::kFLOAT, nullptr, 0};

        // 创建Scale层实现BatchNorm
        auto scaleLayer = network->addScaleNd(*input, nvinfer1::ScaleMode::kCHANNEL, shiftWt, scaleWt, powerWt, 1);
        scaleLayer->setName(("bn_" + weightName).c_str());

        return scaleLayer;
    }

    // 创建Bottleneck块
    nvinfer1::ITensor* AddBottleneckBlock(nvinfer1::ITensor* input,
                                        const std::string& prefix,
                                        int inChannels, int outChannels,
                                        int stride = 1) {
        // conv1: 1x1卷积
        auto conv1 = AddConvLayer(input, prefix + "conv1.weight",
                                outChannels, 1, nvinfer1::DimsHW{1, 1}, nvinfer1::DimsHW{0, 0});
        auto bn1 = AddBatchNormLayer(conv1->getOutput(0), prefix + "bn1.weight", prefix + "bn1.bias",
                                   prefix + "bn1.running_mean", prefix + "bn1.running_var");
        auto relu1 = network->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);

        // conv2: 3x3卷积
        auto conv2 = AddConvLayer(relu1->getOutput(0), prefix + "conv2.weight",
                                outChannels, 3, nvinfer1::DimsHW{stride, stride}, nvinfer1::DimsHW{1, 1});
        auto bn2 = AddBatchNormLayer(conv2->getOutput(0), prefix + "bn2.weight", prefix + "bn2.bias",
                                   prefix + "bn2.running_mean", prefix + "bn2.running_var");
        auto relu2 = network->addActivation(*bn2->getOutput(0), nvinfer1::ActivationType::kRELU);

        // conv3: 1x1卷积
        auto conv3 = AddConvLayer(relu2->getOutput(0), prefix + "conv3.weight",
                                outChannels * 4, 1, nvinfer1::DimsHW{1, 1}, nvinfer1::DimsHW{0, 0});
        auto bn3 = AddBatchNormLayer(conv3->getOutput(0), prefix + "bn3.weight", prefix + "bn3.bias",
                                   prefix + "bn3.running_mean", prefix + "bn3.running_var");

        // 残差连接
        nvinfer1::ITensor* residual = input;
        if (stride != 1 || inChannels != outChannels * 4) {
            // 下采样层
            auto downsample_conv = AddConvLayer(input, prefix + "downsample.0.weight",
                                              outChannels * 4, 1, nvinfer1::DimsHW{stride, stride}, nvinfer1::DimsHW{0, 0});
            auto downsample_bn = AddBatchNormLayer(downsample_conv->getOutput(0), prefix + "downsample.1.weight", prefix + "downsample.1.bias",
                                                 prefix + "downsample.1.running_mean", prefix + "downsample.1.running_var");
            residual = downsample_bn->getOutput(0);
        }

        // 相加
        auto add = network->addElementWise(*residual, *bn3->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        auto relu3 = network->addActivation(*add->getOutput(0), nvinfer1::ActivationType::kRELU);

        return relu3->getOutput(0);
    }

    // 构建ResNet50网络
    void BuildResNet50() {
        std::cout << "Building ResNet50 network..." << std::endl;

        // 输入层 - 使用FP16
        auto input = network->addInput(INPUT_BLOB_NAME.c_str(), INPUT_DATA_TYPE, 
                                      nvinfer1::Dims4{INPUT_N, INPUT_C, INPUT_H, INPUT_W});

        // conv1: 7x7卷积，stride=2
        auto conv1 = AddConvLayer(input, "conv1.weight", 64, 7,
                                 nvinfer1::DimsHW{2, 2}, nvinfer1::DimsHW{3, 3});
        auto bn1 = AddBatchNormLayer(conv1->getOutput(0), "bn1.weight", "bn1.bias",
                                   "bn1.running_mean", "bn1.running_var");
        auto relu1 = network->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);

        // maxpool: 3x3, stride=2
        auto maxpool = network->addPoolingNd(*relu1->getOutput(0), nvinfer1::PoolingType::kMAX, 
                                           nvinfer1::DimsHW{3, 3});
        maxpool->setStrideNd(nvinfer1::DimsHW{2, 2});
        maxpool->setPaddingNd(nvinfer1::DimsHW{1, 1});

        // layer1: 3个bottleneck块
        auto x = AddBottleneckBlock(maxpool->getOutput(0), "layer1.0.", 64, 64, 1);
        x = AddBottleneckBlock(x, "layer1.1.", 256, 64, 1);
        x = AddBottleneckBlock(x, "layer1.2.", 256, 64, 1);

        // layer2: 4个bottleneck块，第一个stride=2
        x = AddBottleneckBlock(x, "layer2.0.", 256, 128, 2);
        x = AddBottleneckBlock(x, "layer2.1.", 512, 128, 1);
        x = AddBottleneckBlock(x, "layer2.2.", 512, 128, 1);
        x = AddBottleneckBlock(x, "layer2.3.", 512, 128, 1);

        // layer3: 6个bottleneck块，第一个stride=2
        x = AddBottleneckBlock(x, "layer3.0.", 512, 256, 2);
        x = AddBottleneckBlock(x, "layer3.1.", 1024, 256, 1);
        x = AddBottleneckBlock(x, "layer3.2.", 1024, 256, 1);
        x = AddBottleneckBlock(x, "layer3.3.", 1024, 256, 1);
        x = AddBottleneckBlock(x, "layer3.4.", 1024, 256, 1);
        x = AddBottleneckBlock(x, "layer3.5.", 1024, 256, 1);

        // layer4: 3个bottleneck块，第一个stride=2
        x = AddBottleneckBlock(x, "layer4.0.", 1024, 512, 2);
        x = AddBottleneckBlock(x, "layer4.1.", 2048, 512, 1);
        x = AddBottleneckBlock(x, "layer4.2.", 2048, 512, 1);

        // 全局平均池化 (Global Average Pooling，输出1x1)
        auto avgpool = network->addPoolingNd(*x, nvinfer1::PoolingType::kAVERAGE,
                                           nvinfer1::DimsHW{7, 7});
        avgpool->setStrideNd(nvinfer1::DimsHW{1, 1});  // 设置stride为1，使kernel覆盖整个7x7区域

        // 全连接层 - 将权重存储在map中以确保生命周期
        std::string fc_weight_key = "fc.weight_float";
        std::string fc_bias_key = "fc.bias_float";

        if (floatWeightMap.find(fc_weight_key) == floatWeightMap.end()) {
            auto fc_weight = weightMap["fc.weight"];
            std::vector<float> fc_weight_float;
            for (auto w : fc_weight) fc_weight_float.push_back(half2float(w));
            floatWeightMap[fc_weight_key] = std::move(fc_weight_float);
        }

        if (floatWeightMap.find(fc_bias_key) == floatWeightMap.end()) {
            auto fc_bias = weightMap["fc.bias"];
            std::vector<float> fc_bias_float;
            for (auto b : fc_bias) fc_bias_float.push_back(half2float(b));
            floatWeightMap[fc_bias_key] = std::move(fc_bias_float);
        }

        nvinfer1::Weights fc_w{nvinfer1::DataType::kFLOAT, floatWeightMap[fc_weight_key].data(), static_cast<int64_t>(floatWeightMap[fc_weight_key].size())};
        nvinfer1::Weights fc_b{nvinfer1::DataType::kFLOAT, floatWeightMap[fc_bias_key].data(), static_cast<int64_t>(floatWeightMap[fc_bias_key].size())};

        auto fc = network->addFullyConnected(*avgpool->getOutput(0), OUTPUT_SIZE, fc_w, fc_b);
        fc->setName("fc");

        // 添加Softmax层
        auto softmax = network->addSoftMax(*fc->getOutput(0));
        softmax->setName("softmax");

        // 标记输出 - 使用FP16
        softmax->getOutput(0)->setName(OUTPUT_BLOB_NAME.c_str());
        softmax->getOutput(0)->setType(OUTPUT_DATA_TYPE);
        network->markOutput(*softmax->getOutput(0));

        std::cout << "ResNet50 network built successfully!" << std::endl;
    }

    // 构建引擎
    void BuildEngine() {
        std::cout << "Building TensorRT engine..." << std::endl;

        // 创建构建配置
        auto config = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1 << 30); // 1GB
        config->setFlag(nvinfer1::BuilderFlag::kFP16); // 启用FP16

        // 构建引擎
        engine = builder->buildEngineWithConfig(*network, *config);
        CHECK_TRT(engine != nullptr);

        // 创建执行上下文
        context = engine->createExecutionContext();
        CHECK_TRT(context != nullptr);

        std::cout << "TensorRT engine built successfully!" << std::endl;

        // 清理
        config->destroy();
    }

    // 分配GPU内存
    void AllocateBuffers() {
        std::cout << "Allocating GPU memory..." << std::endl;

        // 分配输入缓冲区 - FP16
        size_t inputSize = INPUT_N * INPUT_C * INPUT_H * INPUT_W * sizeof(uint16_t);
        CHECK_CUDA(cudaMalloc(&inputBuffer, inputSize));

        // 分配输出缓冲区 - FP16
        size_t outputSize = INPUT_N * OUTPUT_SIZE * sizeof(uint16_t);
        CHECK_CUDA(cudaMalloc(&outputBuffer, outputSize));

        std::cout << "GPU memory allocated successfully!" << std::endl;
    }

    // 推理
    void Infer(const std::string& inputFile, const std::string& outputFile) {
        std::cout << "Running inference..." << std::endl;

        // 加载输入数据
        std::ifstream input(inputFile, std::ios::binary);
        if (!input) {
            std::cerr << "ERROR: Cannot open input file: " << inputFile << std::endl;
            exit(1);
        }

        size_t inputSize = INPUT_N * INPUT_C * INPUT_H * INPUT_W * sizeof(float);
        input.read(reinterpret_cast<char*>(inputBuffer), inputSize);
        input.close();

        // 执行推理
        void* bindings[] = {inputBuffer, outputBuffer};
        bool status = context->executeV2(bindings);
        if (!status) {
            std::cerr << "ERROR: Inference execution failed!" << std::endl;
            exit(1);
        }

        // 保存输出
        std::ofstream output(outputFile, std::ios::binary);
        if (!output) {
            std::cerr << "ERROR: Cannot create output file: " << outputFile << std::endl;
            exit(1);
        }

        size_t outputSize = INPUT_N * OUTPUT_SIZE * sizeof(float);
        output.write(reinterpret_cast<char*>(outputBuffer), outputSize);
        output.close();

        std::cout << "Inference completed successfully!" << std::endl;
        std::cout << "Output saved to: " << outputFile << std::endl;
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
            std::cout << "吞吐量: " << std::fixed << std::setprecision(2) << (1000.0 / avg_time) * INPUT_N << " FPS" << std::endl;
        }
    };

    // 计算TOPS性能指标 (与mytestCodefork/main.cpp保持一致)
    void CalculateTOPS(const PerfStats& stats) {
        if (!stats.valid) {
            std::cout << "⚠️ 无法计算TOPS，性能数据无效" << std::endl;
            return;
        }

        // ResNet50模型操作数统计 (基于FP16精度，Batch=16)
        // ResNet50大约有 4.1 GFLOPS (multiply-adds counted as 2 operations)
        const double resnet50_ops = 4100000000.0;  // ResNet50总操作数 (约4.1 GFLOPS)
        const double fp16_ops = resnet50_ops * 2;  // FP16需要2倍操作
        const double batch_fp16_ops = fp16_ops * INPUT_N;  // Batch=16的总操作数

        // 转换为秒 (使用纯推理时间，与mytestCodefork/main.cpp保持一致)
        double time_seconds = stats.avg_time / 1000.0;

        // 计算实际TOPS (基于纯推理时间，更公平)
        double actual_tops = batch_fp16_ops / (time_seconds * 1e12);

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
            } else if (strstr(prop.name, "Orin")) {
                theoretical_peak = 5.0;   // Jetson Orin约5 TOPS
            } else {
                theoretical_peak = 5.0;   // 默认5 TOPS
            }
        }

        std::cout << "\n=== TOPS性能分析 (基于纯推理时间) ===" << std::endl;
        std::cout << "GPU设备: " << gpu_info << std::endl;
        std::cout << "模型: ResNet50 (FP16精度, Batch=" << INPUT_N << ")" << std::endl;
        std::cout << "单次操作数: " << std::scientific << std::setprecision(2) << fp16_ops << std::endl;
        std::cout << "批处理操作数: " << std::scientific << std::setprecision(2) << batch_fp16_ops << std::endl;
        std::cout << "平均推理时间: " << std::fixed << std::setprecision(3) << stats.avg_time << " ms" << std::endl;
        std::cout << "实际TOPS (纯推理): " << std::fixed << std::setprecision(6) << actual_tops << " TOPS" << std::endl;
        std::cout << "理论峰值: " << std::fixed << std::setprecision(2) << theoretical_peak << " TOPS" << std::endl;

        if (theoretical_peak > 0) {
            double efficiency = (actual_tops / theoretical_peak) * 100.0;
            std::cout << "硬件效率: " << std::fixed << std::setprecision(2) << efficiency << "%" << std::endl;
        }

        // 计算不同精度下的TOPS
        double int8_tops = actual_tops * 2.0;  // INT8精度下TOPS翻倍
        double fp32_tops = actual_tops * 0.5;  // FP32精度下TOPS减半
        
        // 计算单张图片等效TOPS
        double per_image_tops = actual_tops / INPUT_N;

        std::cout << "\n=== 不同精度TOPS预估 ===" << std::endl;
        std::cout << "INT8精度预估: " << std::fixed << std::setprecision(6) << int8_tops << " TOPS" << std::endl;
        std::cout << "FP16精度实测: " << std::fixed << std::setprecision(6) << actual_tops << " TOPS" << std::endl;
        std::cout << "FP32精度预估: " << std::fixed << std::setprecision(6) << fp32_tops << " TOPS" << std::endl;
        std::cout << "单张图片等效TOPS: " << std::fixed << std::setprecision(6) << per_image_tops << " TOPS" << std::endl;

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

    // 执行单次推理并返回结果 (Batch=16, FP16)
    bool DoInference(std::vector<float>& output) {
        // 检查缓冲区是否有效
        if (!inputBuffer || !outputBuffer) {
            std::cerr << "ERROR: GPU buffers not allocated" << std::endl;
            return false;
        }
        
        // 执行推理
        void* bindings[] = {inputBuffer, outputBuffer};
        bool status = context->executeV2(bindings);
        if (!status) {
            std::cerr << "ERROR: Inference execution failed" << std::endl;
            return false;
        }

        // 将FP16结果从 GPU 复制到 CPU
        std::vector<uint16_t> outputFP16(INPUT_N * OUTPUT_SIZE);
        size_t outputSize = INPUT_N * OUTPUT_SIZE * sizeof(uint16_t);
        
        cudaError_t err = cudaMemcpy(outputFP16.data(), outputBuffer, outputSize, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "ERROR: cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        // 将FP16转换为FP32用于显示
        output.resize(INPUT_N * OUTPUT_SIZE);
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] = half2float(outputFP16[i]);
        }

        return true;
    }

    // 执行纯推理（不包含内存拷贝）
    bool DoInferenceOnly() {
        // 检查缓冲区是否有效
        if (!inputBuffer || !outputBuffer) {
            std::cerr << "ERROR: GPU buffers not allocated" << std::endl;
            return false;
        }
        
        // 执行推理
        void* bindings[] = {inputBuffer, outputBuffer};
        bool status = context->executeV2(bindings);
        if (!status) {
            std::cerr << "ERROR: Inference execution failed" << std::endl;
            return false;
        }
        return true;
    }

    // 执行性能测试
    bool DoPerformanceTest(int iterations = 100, int warmup = 10) {
        std::cout << "\n=== 开始性能测试 ===" << std::endl;
        std::cout << "预热次数: " << warmup << std::endl;
        std::cout << "测试次数: " << iterations << std::endl;

        PerfStats inference_stats;  // 纯推理统计
        PerfStats total_stats;      // 总执行统计
        PerfStats memory_stats;     // 内存拷贝统计

        // 预热阶段
        std::cout << "预热中..." << std::flush;
        for (int i = 0; i < warmup; ++i) {
            std::vector<float> dummy_output;
            if (!DoInference(dummy_output)) {
                std::cerr << "预热失败" << std::endl;
                return false;
            }
            // 确保CUDA操作完成
            cudaDeviceSynchronize();
            if (i % 2 == 1) std::cout << "." << std::flush;
        }
        std::cout << " 完成" << std::endl;

        // 预分配输出缓冲区，避免重复分配 (FP16)
        std::vector<uint16_t> output_buffer_fp16(INPUT_N * OUTPUT_SIZE);
        size_t outputSize = INPUT_N * OUTPUT_SIZE * sizeof(uint16_t);
        
        // 性能测试
        std::cout << "性能测试中..." << std::flush;
        for (int i = 0; i < iterations; ++i) {
            // 测试总执行时间（推理+内存拷贝）
            auto total_start = std::chrono::high_resolution_clock::now();
            
            // 测试纯推理时间
            auto inference_start = std::chrono::high_resolution_clock::now();
            if (!DoInferenceOnly()) {
                std::cerr << "推理失败 at iteration " << i << std::endl;
                return false;
            }
            auto inference_end = std::chrono::high_resolution_clock::now();
            
            // 测试内存拷贝时间 (FP16)
            auto memory_start = std::chrono::high_resolution_clock::now();
            CHECK_CUDA(cudaMemcpy(output_buffer_fp16.data(), outputBuffer, outputSize, cudaMemcpyDeviceToHost));
            auto memory_end = std::chrono::high_resolution_clock::now();
            
            auto total_end = std::chrono::high_resolution_clock::now();

            // 计算各阶段时间
            auto inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start);
            auto memory_duration = std::chrono::duration_cast<std::chrono::microseconds>(memory_end - memory_start);
            auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);

            double inference_time_ms = inference_duration.count() / 1000.0;
            double memory_time_ms = memory_duration.count() / 1000.0;
            double total_time_ms = total_duration.count() / 1000.0;

            inference_stats.addSample(inference_time_ms);
            memory_stats.addSample(memory_time_ms);
            total_stats.addSample(total_time_ms);

            if (i % 10 == 9) std::cout << "." << std::flush;
        }
        std::cout << " 完成" << std::endl;

        // 详细性能统计输出
        std::cout << "\n=== Inference Only 性能统计 ===" << std::endl;
        std::cout << "Iterations: " << inference_stats.iterations << std::endl;
        std::cout << "Average time: " << std::fixed << std::setprecision(3) << inference_stats.avg_time << " ms" << std::endl;
        std::cout << "Min time: " << std::fixed << std::setprecision(3) << inference_stats.min_time << " ms" << std::endl;
        std::cout << "Max time: " << std::fixed << std::setprecision(3) << inference_stats.max_time << " ms" << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(3) << inference_stats.total_time << " ms" << std::endl;
        if (inference_stats.avg_time > 0) {
            double fps = (1000.0 / inference_stats.avg_time) * INPUT_N;
            std::cout << "Throughput: " << std::fixed << std::setprecision(2) << fps << " FPS" << std::endl;
        }
        std::cout << "=============================" << std::endl;

        std::cout << "\n=== Memory Copy (D2H) 性能统计 ===" << std::endl;
        std::cout << "Iterations: " << memory_stats.iterations << std::endl;
        std::cout << "Average time: " << std::fixed << std::setprecision(3) << memory_stats.avg_time << " ms" << std::endl;
        std::cout << "Min time: " << std::fixed << std::setprecision(3) << memory_stats.min_time << " ms" << std::endl;
        std::cout << "Max time: " << std::fixed << std::setprecision(3) << memory_stats.max_time << " ms" << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(3) << memory_stats.total_time << " ms" << std::endl;
        if (memory_stats.avg_time > 0) {
            double fps = (1000.0 / memory_stats.avg_time) * INPUT_N;
            std::cout << "Throughput: " << std::fixed << std::setprecision(2) << fps << " FPS" << std::endl;
        }
        std::cout << "=============================" << std::endl;

        std::cout << "\n=== Total Execution 性能统计 ===" << std::endl;
        std::cout << "Iterations: " << total_stats.iterations << std::endl;
        std::cout << "Average time: " << std::fixed << std::setprecision(3) << total_stats.avg_time << " ms" << std::endl;
        std::cout << "Min time: " << std::fixed << std::setprecision(3) << total_stats.min_time << " ms" << std::endl;
        std::cout << "Max time: " << std::fixed << std::setprecision(3) << total_stats.max_time << " ms" << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(3) << total_stats.total_time << " ms" << std::endl;
        if (total_stats.avg_time > 0) {
            double fps = (1000.0 / total_stats.avg_time) * INPUT_N;
            std::cout << "Throughput: " << std::fixed << std::setprecision(2) << fps << " FPS" << std::endl;
        }
        std::cout << "=============================" << std::endl;

        // 计算TOPS（基于纯推理时间）
        CalculateTOPS(inference_stats);

        // 额外性能分析
        std::cout << "\n=== 效率分析 ===" << std::endl;
        double inference_ratio = (inference_stats.avg_time / total_stats.avg_time) * 100.0;
        double memory_ratio = (memory_stats.avg_time / total_stats.avg_time) * 100.0;
        std::cout << "推理时间占比: " << std::fixed << std::setprecision(2) << inference_ratio << "%" << std::endl;
        std::cout << "内存拷贝占比: " << std::fixed << std::setprecision(2) << memory_ratio << "%" << std::endl;
        
        // 计算每张图片的处理时间 (基于纯推理时间，更公平)
        double per_image_inference_time = inference_stats.avg_time / INPUT_N;
        double per_image_total_time = total_stats.avg_time / INPUT_N;
        std::cout << "单张图片推理时间: " << std::fixed << std::setprecision(3) << per_image_inference_time << " ms" << std::endl;
        std::cout << "单张图片总时间: " << std::fixed << std::setprecision(3) << per_image_total_time << " ms" << std::endl;
        
        // 计算理论最大吞吐量 (基于纯推理时间，与mytestCodefork/main.cpp保持一致)
        double theoretical_max_fps = (1000.0 / inference_stats.avg_time) * INPUT_N;
        std::cout << "理论最大吞吐量 (纯推理): " << std::fixed << std::setprecision(2) << theoretical_max_fps << " FPS" << std::endl;
        
        // 计算实际端到端吞吐量
        double end_to_end_fps = (1000.0 / total_stats.avg_time) * INPUT_N;
        std::cout << "端到端吞吐量 (含内存拷贝): " << std::fixed << std::setprecision(2) << end_to_end_fps << " FPS" << std::endl;

        // 内存带宽分析 (FP16)
        size_t input_size = INPUT_N * INPUT_C * INPUT_H * INPUT_W * sizeof(uint16_t);
        size_t output_size = INPUT_N * OUTPUT_SIZE * sizeof(uint16_t);
        double input_mb = input_size / (1024.0 * 1024.0);
        double output_mb = output_size / (1024.0 * 1024.0);
        
        std::cout << "\n=== 内存分析 ===" << std::endl;
        std::cout << "输入大小: " << std::fixed << std::setprecision(2) << input_mb << " MB" << std::endl;
        std::cout << "输出大小: " << std::fixed << std::setprecision(2) << output_mb << " MB" << std::endl;
        
        if (memory_stats.avg_time > 0) {
            double memory_bandwidth = (output_mb / (memory_stats.avg_time / 1000.0));
            std::cout << "内存带宽 (D2H): " << std::fixed << std::setprecision(2) << memory_bandwidth << " MB/s" << std::endl;
        }
        
        double total_data_mb = input_mb + output_mb;
        std::cout << "处理数据总量: " << std::fixed << std::setprecision(2) << total_data_mb << " MB" << std::endl;
        std::cout << "总处理数据量: " << std::fixed << std::setprecision(2) << total_data_mb * inference_stats.iterations << " MB" << std::endl;

        // 清理临时缓冲区
        output_buffer_fp16.clear();
        output_buffer_fp16.shrink_to_fit();

        return true;
    }

    // 打印结果 (Batch=16)
    void PrintResults(const std::vector<float>& output) {
        std::cout << "\n=== ResNet50 Batch=" << INPUT_N << " 推理结果 ===" << std::endl;

        // 为每个样本显示结果
        for (int batch_idx = 0; batch_idx < INPUT_N; ++batch_idx) {
            std::cout << "\n--- 样本 " << (batch_idx + 1) << "/" << INPUT_N << " ---" << std::endl;
            
            // 获取当前样本的输出
            std::vector<float> sample_output(OUTPUT_SIZE);
            for (int i = 0; i < OUTPUT_SIZE; ++i) {
                sample_output[i] = output[batch_idx * OUTPUT_SIZE + i];
            }

            // 找到最大值的索引
            auto maxIt = std::max_element(sample_output.begin(), sample_output.end());
            int maxIndex = std::distance(sample_output.begin(), maxIt);

            std::cout << "预测类别: " << maxIndex << std::endl;
            std::cout << "置信度: " << std::fixed << std::setprecision(6) << *maxIt << std::endl;

            // 打印 Top 5 结果
            std::vector<std::pair<float, int>> scoreIndex;
            for (int i = 0; i < OUTPUT_SIZE; ++i) {
                scoreIndex.push_back(std::make_pair(sample_output[i], i));
            }
            std::sort(scoreIndex.begin(), scoreIndex.end(), std::greater<std::pair<float, int>>());

            std::cout << "Top 5 预测结果:" << std::endl;
            for (int i = 0; i < 5; ++i) {
                std::cout << "  " << (i+1) << ". 类别 " << scoreIndex[i].second
                         << ": " << std::fixed << std::setprecision(6) << scoreIndex[i].first << std::endl;
            }
        }
    }

    // 加载输入数据到GPU
    bool LoadInput(const std::string& inputFile) {
        std::cout << "Loading input file: " << inputFile << std::endl;

        std::ifstream input(inputFile, std::ios::binary);
        if (!input) {
            std::cerr << "ERROR: Cannot open input file: " << inputFile << std::endl;
            return false;
        }

        // 直接读取FP16数据 (Batch=16, FP16格式)
        size_t inputSize = INPUT_N * INPUT_C * INPUT_H * INPUT_W * sizeof(uint16_t);  // FP16 = 2 bytes
        std::vector<uint16_t> hostInputFP16(INPUT_N * INPUT_C * INPUT_H * INPUT_W);
        input.read(reinterpret_cast<char*>(hostInputFP16.data()), inputSize);
        input.close();

        // 直接从CPU拷贝FP16数据到GPU
        CHECK_CUDA(cudaMemcpy(inputBuffer, hostInputFP16.data(), inputSize, cudaMemcpyHostToDevice));

        // 调试：打印输入数据信息
        std::cout << "Input data loaded successfully! (Batch=" << INPUT_N << ", FP16)" << std::endl;
        std::cout << "FP16 input size: " << inputSize << " bytes" << std::endl;
        
        // 检查不同样本的输入数据是否相同
        std::cout << "Checking input data consistency..." << std::endl;
        bool all_samples_same = true;
        size_t sample_size = INPUT_C * INPUT_H * INPUT_W;
        
        for (int batch_idx = 1; batch_idx < INPUT_N; ++batch_idx) {
            for (size_t i = 0; i < sample_size; ++i) {
                if (hostInputFP16[i] != hostInputFP16[batch_idx * sample_size + i]) {
                    all_samples_same = false;
                    break;
                }
            }
            if (!all_samples_same) break;
        }
        
        if (all_samples_same) {
            std::cout << "⚠️ 警告: 所有样本的输入数据相同 (这是预期的，因为使用同一张图片)" << std::endl;
        } else {
            std::cout << "✅ 不同样本的输入数据不同" << std::endl;
        }
        
        // 打印前10个输入值 (FP16)
        std::cout << "First 10 input values (FP16):" << std::endl;
        for (size_t i = 0; i < 10 && i < hostInputFP16.size(); ++i) {
            std::cout << "  [" << i << "] FP16: " << hostInputFP16[i] << std::endl;
        }
        
        // 显示数值范围
        uint16_t min_val = *std::min_element(hostInputFP16.begin(), hostInputFP16.end());
        uint16_t max_val = *std::max_element(hostInputFP16.begin(), hostInputFP16.end());
        std::cout << "Input value range: [" << min_val << ", " << max_val << "]" << std::endl;
        return true;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "=== ResNet50 TensorRT FP16推理 + TOPS性能分析 (Batch=16, FP16) ===" << std::endl;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <weights_file> <input_file> [output_file]" << std::endl;
        std::cerr << "Example: " << argv[0] << " resnet50-tensorrt-fp16.wts input_fp16_batch16_nchw.bin output.bin" << std::endl;
        std::cerr << "Note: 本程序仅支持 Batch=16 的 FP16 推理" << std::endl;
        return 1;
    }

    std::string weightsFile = argv[1];
    std::string inputFile = argv[2];
    std::string outputFile = (argc > 3) ? argv[3] : "output.bin";

    try {
        // 创建ResNet50实例 (Batch=16)
        std::cout << "\n✅ 初始化FP16推理引擎 (Batch=16)" << std::endl;
        ResNet50TensorRT resnet50;

        // 加载权重
        std::cout << "\n📁 加载FP16权重文件..." << std::endl;
        resnet50.LoadWeights(weightsFile);
        std::cout << "✅ 权重加载完成!" << std::endl;

        // 构建网络
        std::cout << "\n🔧 构建 FP16 ResNet50 网络..." << std::endl;
        resnet50.BuildResNet50();
        std::cout << "✅ FP16网络构建完成!" << std::endl;

        // 构建引擎
        std::cout << "\n⚙️  构建TensorRT引擎..." << std::endl;
        resnet50.BuildEngine();
        std::cout << "✅ TensorRT引擎构建完成!" << std::endl;

        // 分配内存
        std::cout << "\n💾 分配GPU内存..." << std::endl;
        resnet50.AllocateBuffers();
        std::cout << "✅ GPU内存分配完成!" << std::endl;

        // 加载输入数据
        std::cout << "\n📥 加载输入文件: " << inputFile << std::endl;
        if (!resnet50.LoadInput(inputFile)) {
            std::cerr << "❌ 输入加载失败" << std::endl;
            return 1;
        }

        // 执行单次推理验证
        std::cout << "\n🔍 执行单次推理验证..." << std::endl;
        std::vector<float> output;
        if (!resnet50.DoInference(output)) {
            std::cerr << "❌ 推理执行失败" << std::endl;
            return 1;
        }

        // 打印推理结果
        std::cout << "\n=== 推理结果验证 ===" << std::endl;
        resnet50.PrintResults(output);

        // 保存输出到文件
        std::ofstream outFile(outputFile, std::ios::binary);
        if (outFile) {
            outFile.write(reinterpret_cast<char*>(output.data()), output.size() * sizeof(float));
            outFile.close();
            std::cout << "\n✅ 输出已保存到: " << outputFile << std::endl;
        }

        // 执行性能测试和TOPS计算
        std::cout << "\n🚀 开始性能测试和TOPS分析..." << std::endl;
        if (!resnet50.DoPerformanceTest(100, 10)) {  // 100次测试，10次预热
            std::cerr << "❌ 性能测试失败" << std::endl;
            return 1;
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

        std::cout << "\n🎉 ResNet50 FP16推理 + TOPS分析完成! (Batch=16, FP16)" << std::endl;
        std::cout << "📊 性能分析报告已生成，请查看上述TOPS计算结果" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
