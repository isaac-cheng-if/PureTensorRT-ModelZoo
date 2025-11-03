/*
 * ViT Network Builder
 * Handles network construction, weight loading, and engine serialization
 */

#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <cassert>
#include <string>
#include <cstring>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

// ==================== Configuration ====================
namespace Config {
    const int BATCH_SIZE = 16;  // 修改为 batch16
    const int NUM_CLASS = 1000;
    const int INPUT_H = 224;
    const int INPUT_W = 224;
    const int PATCH_SIZE = 16;
    const int NUM_PATCHES = (INPUT_H / PATCH_SIZE) * (INPUT_W / PATCH_SIZE);
    const char* INPUT_BLOB_NAME = "images";
    const char* OUTPUT_BLOB_NAME = "output";
    const int GPU_ID = 0;
}

// ==================== CUDA Error Checking ====================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// ==================== TensorRT Logger ====================
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// ==================== 常量管理器（优化：避免重复创建常量）====================
class ConstantManager {
private:
    std::map<float, std::unique_ptr<float[]>> constants;
    
public:
    nvinfer1::Weights getConstant(float value) {
        if (constants.find(value) == constants.end()) {
            constants[value] = std::unique_ptr<float[]>(new float[1]{value});
        }
        return nvinfer1::Weights{nvinfer1::DataType::kFLOAT, constants[value].get(), 1};
    }
    
    ~ConstantManager() = default;
};

ConstantManager g_ConstantManager;

// ==================== Weight Loading ====================
std::map<std::string, nvinfer1::Weights> LoadWeights(const std::string& file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap;

    std::ifstream input(file);
    if (!input.is_open()) {
        std::cerr << "ERROR: Unable to load weight file: " << file << std::endl;
        exit(1);
    }

    int32_t count;
    input >> count;
    if (count <= 0) {
        std::cerr << "ERROR: Invalid weight count" << std::endl;
        exit(1);
    }

    while (count--) {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;
        std::string name;
        input >> name >> std::dec >> size;

        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));
        for (uint32_t x = 0; x < size; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }

    std::cout << "Loaded " << weightMap.size() << " weight tensors" << std::endl;
    return weightMap;
}

// ==================== ViT Network Building Blocks ====================

// Helper: Fully Connected wrapper that normalizes tensor ranks for TensorRT
// FC(x) = x @ W^T + b
// For 3D input [B, N, D_in], weight [D_out, D_in], output [B, N, D_out]
nvinfer1::ITensor* FullyConnected3D(nvinfer1::INetworkDefinition* network,
                                    nvinfer1::ITensor& input,
                                    int output_size,
                                    nvinfer1::Weights weight,
                                    nvinfer1::Weights bias) {
    auto input_dims = input.getDimensions();
    int input_size = input_dims.d[input_dims.nbDims - 1];
    
    // For 2D input [B, D_in]
    if (input_dims.nbDims == 2) {
        auto fc = network->addFullyConnected(input, output_size, weight, bias);
        assert(fc);
        return fc->getOutput(0);
    }
    // For 3D input [B, N, D_in]
    else if (input_dims.nbDims == 3) {
        int batch = input_dims.d[0];
        int seq_len = input_dims.d[1];
        
        // Reshape input to [B*N, D_in]
        auto input_reshape = network->addShuffle(input);
        nvinfer1::Dims flatten_dims;
        flatten_dims.nbDims = 2;
        flatten_dims.d[0] = batch * seq_len;
        flatten_dims.d[1] = input_size;
        input_reshape->setReshapeDimensions(flatten_dims);

        auto fc = network->addFullyConnected(*input_reshape->getOutput(0), output_size, weight, bias);
        assert(fc);
        
        // Reshape back to [B, N, D_out]
        auto output_reshape = network->addShuffle(*fc->getOutput(0));
        nvinfer1::Dims output_dims;
        output_dims.nbDims = 3;
        output_dims.d[0] = batch;
        output_dims.d[1] = seq_len;
        output_dims.d[2] = output_size;
        output_reshape->setReshapeDimensions(output_dims);
        
        return output_reshape->getOutput(0);
    } else {
        // For 4D+ use standard FC
        auto fc = network->addFullyConnected(input, output_size, weight, bias);
        assert(fc);
        return fc->getOutput(0);
    }
}

// Patch Embedding Layer
nvinfer1::IConvolutionLayer* PatchEmbed(nvinfer1::INetworkDefinition* network,
                                       std::map<std::string, nvinfer1::Weights>& weightMap,
                                       nvinfer1::ITensor& input,
                                       int embed_dim,
                                       const std::string& lname) {
    auto conv = network->addConvolutionNd(input, embed_dim, nvinfer1::DimsHW{Config::PATCH_SIZE, Config::PATCH_SIZE},
                                          weightMap[lname + ".proj.weight"], weightMap[lname + ".proj.bias"]);
    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{Config::PATCH_SIZE, Config::PATCH_SIZE});
    conv->setPaddingNd(nvinfer1::DimsHW{0, 0});
    return conv;
}

// GELU Activation（可选使用 TensorRT 内置，需要编译时定义 ENABLE_TRT_GELU）
nvinfer1::ITensor* GELU(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input) {
#if defined(ENABLE_TRT_GELU)
    auto gelu = network->addActivation(input, nvinfer1::ActivationType::kGELU);
    assert(gelu);
    return gelu->getOutput(0);
#else
    // Constants
    const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/π)
    const float coeff = 0.044715f;
    
    // x³
    auto x_pow2 = network->addElementWise(input, input, nvinfer1::ElementWiseOperation::kPROD);
    auto x_pow3 = network->addElementWise(*x_pow2->getOutput(0), input, nvinfer1::ElementWiseOperation::kPROD);
    
    // 0.044715 * x³
    auto coeff_const = network->addConstant(nvinfer1::Dims3{1, 1, 1}, g_ConstantManager.getConstant(coeff));
    auto coeff_mul = network->addElementWise(*x_pow3->getOutput(0), *coeff_const->getOutput(0), 
                                             nvinfer1::ElementWiseOperation::kPROD);
    
    // x + 0.044715 * x³
    auto sum1 = network->addElementWise(input, *coeff_mul->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    
    // sqrt(2/π) * (x + 0.044715 * x³)
    auto sqrt_const = network->addConstant(nvinfer1::Dims3{1, 1, 1}, g_ConstantManager.getConstant(sqrt_2_over_pi));
    auto sqrt_mul = network->addElementWise(*sum1->getOutput(0), *sqrt_const->getOutput(0),
                                            nvinfer1::ElementWiseOperation::kPROD);
    
    // tanh(...)
    auto tanh_layer = network->addActivation(*sqrt_mul->getOutput(0), nvinfer1::ActivationType::kTANH);
    
    // 1 + tanh(...)
    auto one_const = network->addConstant(nvinfer1::Dims3{1, 1, 1}, g_ConstantManager.getConstant(1.0f));
    auto add_one = network->addElementWise(*tanh_layer->getOutput(0), *one_const->getOutput(0),
                                           nvinfer1::ElementWiseOperation::kSUM);
    
    // x * (1 + tanh(...))
    auto mul1 = network->addElementWise(input, *add_one->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    
    // 0.5 * x * (1 + tanh(...))
    auto half_const = network->addConstant(nvinfer1::Dims3{1, 1, 1}, g_ConstantManager.getConstant(0.5f));
    auto final_mul = network->addElementWise(*mul1->getOutput(0), *half_const->getOutput(0),
                                             nvinfer1::ElementWiseOperation::kPROD);
    
    return final_mul->getOutput(0);
#endif
}

// Layer Normalization (优化版本：使用常量管理器)
// 完整实现: (x - mean) / sqrt(variance + eps) * gamma + beta
nvinfer1::ILayer* LayerNorm(nvinfer1::INetworkDefinition* network,
                            std::map<std::string, nvinfer1::Weights>& weightMap,
                            nvinfer1::ITensor& input,
                            int dim,
                            const std::string& lname,
                            float eps = 1e-6) {
    auto input_dims = input.getDimensions();
    assert(input_dims.nbDims == 3);
    int feature_dim = input_dims.d[2];
    if (feature_dim != -1) {
        assert(feature_dim == dim);
    }

    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    
    // 1. 计算均值 (沿着最后一个维度, 即 dim 维度)
    uint32_t reduceAxes = 1 << (input_dims.nbDims - 1);
    auto mean_layer = network->addReduce(input, nvinfer1::ReduceOperation::kAVG, reduceAxes, true);
    assert(mean_layer);
    
    // 2. x - mean
    auto x_sub_mean = network->addElementWise(input, *mean_layer->getOutput(0), 
                                              nvinfer1::ElementWiseOperation::kSUB);
    
    // 3. (x - mean)^2
    auto x_sub_mean_sq = network->addElementWise(*x_sub_mean->getOutput(0), 
                                                  *x_sub_mean->getOutput(0), 
                                                  nvinfer1::ElementWiseOperation::kPROD);
    
    // 4. 计算方差 variance = mean((x - mean)^2)
    auto variance = network->addReduce(*x_sub_mean_sq->getOutput(0), 
                                      nvinfer1::ReduceOperation::kAVG, reduceAxes, true);
    
    // 5. variance + eps (使用常量管理器)
    auto eps_layer = network->addConstant(nvinfer1::Dims3{1, 1, 1}, g_ConstantManager.getConstant(eps));
    auto var_plus_eps = network->addElementWise(*variance->getOutput(0), 
                                                *eps_layer->getOutput(0), 
                                                nvinfer1::ElementWiseOperation::kSUM);
    
    // 6. sqrt(variance + eps)
    auto std_dev = network->addUnary(*var_plus_eps->getOutput(0), nvinfer1::UnaryOperation::kSQRT);
    
    // 7. (x - mean) / sqrt(variance + eps)
    auto normalized = network->addElementWise(*x_sub_mean->getOutput(0), 
                                              *std_dev->getOutput(0), 
                                              nvinfer1::ElementWiseOperation::kDIV);
    
    // 8. 应用仿射变换: gamma * normalized + beta
    nvinfer1::Weights gamma_weights{nvinfer1::DataType::kFLOAT, gamma, dim};
    nvinfer1::Weights beta_weights{nvinfer1::DataType::kFLOAT, beta, dim};
    
    nvinfer1::Dims affine_dims;
    affine_dims.nbDims = 3;
    affine_dims.d[0] = 1;
    affine_dims.d[1] = 1;
    affine_dims.d[2] = feature_dim;
    
    auto gamma_const = network->addConstant(affine_dims, gamma_weights);
    auto beta_const = network->addConstant(affine_dims, beta_weights);
    
    // gamma * normalized
    auto scaled = network->addElementWise(*normalized->getOutput(0), 
                                         *gamma_const->getOutput(0), 
                                         nvinfer1::ElementWiseOperation::kPROD);
    
    // scaled + beta
    auto output = network->addElementWise(*scaled->getOutput(0), 
                                         *beta_const->getOutput(0), 
                                         nvinfer1::ElementWiseOperation::kSUM);
    
    return output;
}

// Multi-Head Attention
nvinfer1::ITensor* MultiHeadAttention(nvinfer1::INetworkDefinition* network,
                                     std::map<std::string, nvinfer1::Weights>& weightMap,
                                     nvinfer1::ITensor& input,
                                     int dim, int num_heads,
                                     const std::string& lname) {
    int head_dim = dim / num_heads;
    
    // QKV Linear transformation
    auto qkv_output = FullyConnected3D(network, input, dim * 3,
                                       weightMap[lname + ".qkv.weight"], 
                                       weightMap[lname + ".qkv.bias"]);
    
    // Reshape QKV: [B, N, 3*D] -> [B, N, 3, num_heads, head_dim]
    auto qkv_shuffle = network->addShuffle(*qkv_output);
    nvinfer1::Dims qkv_dims;
    qkv_dims.nbDims = 5;
    qkv_dims.d[0] = Config::BATCH_SIZE;
    qkv_dims.d[1] = Config::NUM_PATCHES + 1;
    qkv_dims.d[2] = 3;
    qkv_dims.d[3] = num_heads;
    qkv_dims.d[4] = head_dim;
    qkv_shuffle->setReshapeDimensions(qkv_dims);

    nvinfer1::Dims start, size, stride;
    start.nbDims = 5;
    size.nbDims = 5;
    stride.nbDims = 5;
    for (int i = 0; i < 5; ++i) {
        start.d[i] = 0;
        size.d[i] = qkv_dims.d[i];
        stride.d[i] = 1;
    }

    // Slice Q
    size.d[2] = 1;
    auto q_slice = network->addSlice(*qkv_shuffle->getOutput(0), start, size, stride);
    assert(q_slice);

    // Slice K
    start.d[2] = 1;
    auto k_slice = network->addSlice(*qkv_shuffle->getOutput(0), start, size, stride);
    assert(k_slice);

    // Slice V
    start.d[2] = 2;
    auto v_slice = network->addSlice(*qkv_shuffle->getOutput(0), start, size, stride);
    assert(v_slice);

    auto to_heads = [&](nvinfer1::ILayer* slice) -> nvinfer1::IShuffleLayer* {
        auto shuffle = network->addShuffle(*slice->getOutput(0));
        assert(shuffle);
        shuffle->setFirstTranspose(nvinfer1::Permutation{0, 3, 1, 4, 2});
        shuffle->setReshapeDimensions(nvinfer1::Dims3{Config::BATCH_SIZE * num_heads,
                                                      Config::NUM_PATCHES + 1,
                                                      head_dim});
        return shuffle;
    };

    auto q_reshape = to_heads(q_slice);
    auto k_reshape = to_heads(k_slice);
    auto v_reshape = to_heads(v_slice);
    
    // Q @ K^T
    auto qk_matmul = network->addMatrixMultiply(*q_reshape->getOutput(0), 
                                               nvinfer1::MatrixOperation::kNONE,
                                               *k_reshape->getOutput(0), 
                                               nvinfer1::MatrixOperation::kTRANSPOSE);
    assert(qk_matmul);
    
    // Softmax
    auto softmax = network->addSoftMax(*qk_matmul->getOutput(0));
    softmax->setAxes(1 << 2); // Softmax on last dimension
    
    // Attention @ V
    auto attn_v_matmul = network->addMatrixMultiply(*softmax->getOutput(0),
                                                   nvinfer1::MatrixOperation::kNONE,
                                                   *v_reshape->getOutput(0),
                                                   nvinfer1::MatrixOperation::kNONE);
    assert(attn_v_matmul);
    
    // Reshape back: [B*num_heads, N, head_dim] -> [B, num_heads, N, head_dim]
    auto attn_reshape = network->addShuffle(*attn_v_matmul->getOutput(0));
    attn_reshape->setReshapeDimensions(nvinfer1::Dims4{Config::BATCH_SIZE, num_heads, Config::NUM_PATCHES + 1, head_dim});
    
    // Transpose: [B, num_heads, N, head_dim] -> [B, N, num_heads, head_dim]
    auto attn_transpose = network->addShuffle(*attn_reshape->getOutput(0));
    attn_transpose->setFirstTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    
    // Reshape: [B, N, num_heads, head_dim] -> [B, N, dim]
    auto attn_final_reshape = network->addShuffle(*attn_transpose->getOutput(0));
    attn_final_reshape->setReshapeDimensions(nvinfer1::Dims3{Config::BATCH_SIZE, Config::NUM_PATCHES + 1, dim});
    
    // Output projection
    auto out_proj = FullyConnected3D(network, *attn_final_reshape->getOutput(0), dim,
                                     weightMap[lname + ".proj.weight"],
                                     weightMap[lname + ".proj.bias"]);
    
    return out_proj;
}

// MLP (Feed Forward Network)
nvinfer1::ITensor* MLP(nvinfer1::INetworkDefinition* network,
                       std::map<std::string, nvinfer1::Weights>& weightMap,
                       nvinfer1::ITensor& input,
                       int dim, int mlp_ratio,
                       const std::string& lname) {
    int hidden_dim = dim * mlp_ratio;
    
    // First linear layer
    auto fc1_output = FullyConnected3D(network, input, hidden_dim,
                                       weightMap[lname + ".fc1.weight"],
                                       weightMap[lname + ".fc1.bias"]);
    
    // GELU activation (TensorRT 内置)
    auto gelu_output = GELU(network, *fc1_output);
    assert(gelu_output);
    
    // Second linear layer
    auto fc2_output = FullyConnected3D(network, *gelu_output, dim,
                                       weightMap[lname + ".fc2.weight"],
                                       weightMap[lname + ".fc2.bias"]);
    
    return fc2_output;
}

// Transformer Block
nvinfer1::ITensor* TransformerBlock(nvinfer1::INetworkDefinition* network,
                                   std::map<std::string, nvinfer1::Weights>& weightMap,
                                   nvinfer1::ITensor& input,
                                   int dim, int num_heads, int mlp_ratio,
                                   const std::string& lname) {
    // Layer norm 1
    auto norm1 = LayerNorm(network, weightMap, input, dim, lname + ".norm1");
    
    // Multi-head attention
    auto attn = MultiHeadAttention(network, weightMap, *norm1->getOutput(0), dim, num_heads, lname + ".attn");
    
    // Residual connection 1
    auto residual1 = network->addElementWise(input, *attn, nvinfer1::ElementWiseOperation::kSUM);
    assert(residual1);
    
    // Layer norm 2
    auto norm2 = LayerNorm(network, weightMap, *residual1->getOutput(0), dim, lname + ".norm2");
    
    // MLP
    auto mlp = MLP(network, weightMap, *norm2->getOutput(0), dim, mlp_ratio, lname + ".mlp");
    
    // Residual connection 2
    auto residual2 = network->addElementWise(*residual1->getOutput(0), *mlp, nvinfer1::ElementWiseOperation::kSUM);
    assert(residual2);
    
    return residual2->getOutput(0);
}

// ==================== ViT Network Builder Class ====================
class ViTBuilder {
private:
    nvinfer1::IBuilder* builder;
    nvinfer1::IBuilderConfig* config;
    nvinfer1::INetworkDefinition* network;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IRuntime* runtime;

    std::map<std::string, nvinfer1::Weights> weightMap;
    int embed_dim, depth, num_heads, mlp_ratio;

public:
    ViTBuilder() : builder(nullptr), config(nullptr), network(nullptr),
                   engine(nullptr), runtime(nullptr),
                   embed_dim(768), depth(12), num_heads(12), mlp_ratio(4) {}

    ~ViTBuilder() {
        if (engine) engine->destroy();
        if (network) network->destroy();
        if (config) config->destroy();
        if (builder) builder->destroy();
        if (runtime) runtime->destroy();

        for (auto& mem : weightMap) {
            free((void*)(mem.second.values));
        }
    }

    void SetModelConfig(const std::string& model_type) {
        if (model_type == "large") {
            embed_dim = 1024;
            depth = 24;
            num_heads = 16;
            mlp_ratio = 4;
        } else { // base
            embed_dim = 768;
            depth = 12;
            num_heads = 12;
            mlp_ratio = 4;
        }
        std::cout << "Model: ViT-" << model_type << " (embed_dim=" << embed_dim 
                  << ", depth=" << depth << ", num_heads=" << num_heads << ")" << std::endl;
    }

    void Build(const std::string& wts_file, const std::string& model_type) {
        auto total_start = std::chrono::high_resolution_clock::now();
        
        weightMap = LoadWeights(wts_file);
        SetModelConfig(model_type);
        FuseAttentionScale();

        builder = nvinfer1::createInferBuilder(gLogger);
        network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

        std::cout << "\n=== Building network layers ===" << std::endl;
        auto network_start = std::chrono::high_resolution_clock::now();
        BuildViTNetwork();
        auto network_end = std::chrono::high_resolution_clock::now();
        auto network_duration = std::chrono::duration_cast<std::chrono::milliseconds>(network_end - network_start);
        std::cout << "Network construction time: " << network_duration.count() << " ms" << std::endl;

        config = builder->createBuilderConfig();
        // 优化：为 batch16 增加 workspace 大小
        config->setMaxWorkspaceSize(2ULL << 30);  // 2GB (对于 batch16 更合适)
        
        // 优化：启用更多优化选项
        if (builder->platformHasFastFp16()) {
            std::cout << "✓ Enabling FP16 precision" << std::endl;
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        
        // 优化：启用 GPU fallback 以提高兼容性
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        
        // 优化：设置 profiling verbosity
        config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

        std::cout << "\n=== Optimizing and building engine ===" << std::endl;
        std::cout << "This may take several minutes for batch16..." << std::endl;
        auto build_start = std::chrono::high_resolution_clock::now();
        auto serialized_model = builder->buildSerializedNetwork(*network, *config);
        auto build_end = std::chrono::high_resolution_clock::now();
        auto build_duration = std::chrono::duration_cast<std::chrono::seconds>(build_end - build_start);
        std::cout << "Engine optimization & build time: " << build_duration.count() << " seconds" << std::endl;

        if (!serialized_model) {
            std::cerr << "ERROR: buildSerializedNetwork failed" << std::endl;
            return;
        }

        runtime = nvinfer1::createInferRuntime(gLogger);
        if (!runtime) {
            std::cerr << "ERROR: createInferRuntime failed" << std::endl;
            delete serialized_model;
            return;
        }

        engine = runtime->deserializeCudaEngine(serialized_model->data(), serialized_model->size());
        if (!engine) {
            std::cerr << "ERROR: deserializeCudaEngine failed" << std::endl;
            delete serialized_model;
            return;
        }

        delete serialized_model;
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
        
        std::cout << "\n✓ Engine built successfully!" << std::endl;
        std::cout << "Total time: " << total_duration.count() << " seconds" << std::endl;
        std::cout << "Engine size: " << (engine->getDeviceMemorySize() / (1024.0 * 1024.0)) << " MB" << std::endl;
    }

    void Serialize(const std::string& engine_file) {
        if (!engine) {
            std::cerr << "ERROR: No engine to serialize" << std::endl;
            return;
        }

        std::cout << "\n=== Serializing engine ===" << std::endl;
        auto serialize_start = std::chrono::high_resolution_clock::now();
        
        auto serialized = engine->serialize();
        if (!serialized) {
            std::cerr << "ERROR: engine serialize failed" << std::endl;
            return;
        }

        std::ofstream out(engine_file, std::ios::binary);
        if (!out.is_open()) {
            std::cerr << "ERROR: Cannot open file for writing: " << engine_file << std::endl;
            delete serialized;
            return;
        }
        
        out.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
        out.close();
        
        auto serialize_end = std::chrono::high_resolution_clock::now();
        auto serialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serialize_end - serialize_start);
        
        double size_mb = serialized->size() / (1024.0 * 1024.0);
        delete serialized;
        
        std::cout << "✓ Engine saved to: " << engine_file << std::endl;
        std::cout << "  File size: " << std::fixed << std::setprecision(2) << size_mb << " MB" << std::endl;
        std::cout << "  Serialization time: " << serialize_duration.count() << " ms" << std::endl;
    }

private:
    void BuildViTNetwork() {
        std::cout << "Building ViT network..." << std::endl;
        PrepareCLSToken();

        // Input
        auto data = network->addInput(Config::INPUT_BLOB_NAME, nvinfer1::DataType::kFLOAT,
                                      nvinfer1::Dims4{Config::BATCH_SIZE, 3, Config::INPUT_H, Config::INPUT_W});

        // Patch Embedding
        auto patch_embed = PatchEmbed(network, weightMap, *data, embed_dim, "patch_embed");

        // Reshape: [B, embed_dim, H/P, W/P] -> [B, embed_dim, num_patches]
        auto patch_reshape = network->addShuffle(*patch_embed->getOutput(0));
        patch_reshape->setReshapeDimensions(nvinfer1::Dims3{Config::BATCH_SIZE, embed_dim, Config::NUM_PATCHES});

        // Transpose: [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        auto patch_transpose = network->addShuffle(*patch_reshape->getOutput(0));
        patch_transpose->setFirstTranspose(nvinfer1::Permutation{0, 2, 1});

        // Add CLS token 常量：直接提供 [B, 1, embed_dim]
        auto cls_token_constant = network->addConstant(nvinfer1::Dims3{Config::BATCH_SIZE, 1, embed_dim},
                                                      weightMap["cls_token_batched"]);
        assert(cls_token_constant);

        // Concatenate CLS token with patches: [B, 1, D] + [B, N, D] -> [B, N+1, D]
        nvinfer1::ITensor* concat_inputs[] = {cls_token_constant->getOutput(0), patch_transpose->getOutput(0)};
        auto concat = network->addConcatenation(concat_inputs, 2);
        concat->setAxis(1); // Concatenate along sequence dimension

        // Add positional embedding
        // pos_embed 权重是 [1, NUM_PATCHES+1, embed_dim]，可以直接 broadcast
        auto pos_embed_constant = network->addConstant(nvinfer1::Dims3{1, Config::NUM_PATCHES + 1, embed_dim}, 
                                                     weightMap["pos_embed"]);
        assert(pos_embed_constant);

        auto pos_embed_add = network->addElementWise(*concat->getOutput(0), *pos_embed_constant->getOutput(0), 
                                                    nvinfer1::ElementWiseOperation::kSUM);
        assert(pos_embed_add);

        // Transformer blocks
        auto x = pos_embed_add->getOutput(0);
        for (int i = 0; i < depth; i++) {
            x = TransformerBlock(network, weightMap, *x, embed_dim, num_heads, mlp_ratio, 
                                "blocks." + std::to_string(i));
        }

        // Final layer norm
        auto final_norm = LayerNorm(network, weightMap, *x, embed_dim, "norm");

        // Extract CLS token (first token)
        auto cls_slice = network->addSlice(*final_norm->getOutput(0),
                                          nvinfer1::Dims3{0, 0, 0},
                                          nvinfer1::Dims3{Config::BATCH_SIZE, 1, embed_dim},
                                          nvinfer1::Dims3{1, 1, 1});

        // Reshape CLS token: [B, 1, embed_dim] -> [B, embed_dim]
        auto cls_reshape = network->addShuffle(*cls_slice->getOutput(0));
        cls_reshape->setReshapeDimensions(nvinfer1::Dims2{Config::BATCH_SIZE, embed_dim});

        // Classification head
        auto head_output = FullyConnected3D(network, *cls_reshape->getOutput(0), Config::NUM_CLASS,
                                           weightMap["head.weight"], weightMap["head.bias"]);

        // Output
        head_output->setName(Config::OUTPUT_BLOB_NAME);
        network->markOutput(*head_output);

        std::cout << "ViT network built successfully! Output: ["
                  << Config::BATCH_SIZE << ", " << Config::NUM_CLASS << "]" << std::endl;
    }

    void FuseAttentionScale() {
        int head_dim = embed_dim / num_heads;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (int i = 0; i < depth; ++i) {
            std::string base = "blocks." + std::to_string(i) + ".attn.qkv.";
            auto weight_iter = weightMap.find(base + "weight");
            auto bias_iter = weightMap.find(base + "bias");
            if (weight_iter == weightMap.end() || bias_iter == weightMap.end()) {
                continue;
            }

            auto& weight = weight_iter->second;
            auto& bias = bias_iter->second;
            size_t q_elements = static_cast<size_t>(embed_dim) * static_cast<size_t>(embed_dim);
            if (weight.count < static_cast<int64_t>(q_elements) || bias.count < embed_dim) {
                continue;
            }

            float* weight_data = reinterpret_cast<float*>(const_cast<void*>(weight.values));
            float* bias_data = reinterpret_cast<float*>(const_cast<void*>(bias.values));

            for (size_t idx = 0; idx < q_elements; ++idx) {
                weight_data[idx] *= scale;
            }
            for (int idx = 0; idx < embed_dim; ++idx) {
                bias_data[idx] *= scale;
            }
        }
    }

    void PrepareCLSToken() {
        if (weightMap.count("cls_token_batched")) {
            return;
        }
        auto it = weightMap.find("cls_token");
        if (it == weightMap.end()) {
            return;
        }

        const float* cls_src = reinterpret_cast<const float*>(it->second.values);
        size_t dim = it->second.count;
        size_t total = static_cast<size_t>(Config::BATCH_SIZE) * dim;
        float* cls_tiled = reinterpret_cast<float*>(malloc(sizeof(float) * total));
        for (int b = 0; b < Config::BATCH_SIZE; ++b) {
            std::memcpy(cls_tiled + b * dim, cls_src, sizeof(float) * dim);
        }
        weightMap["cls_token_batched"] = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cls_tiled, static_cast<int64_t>(total)};
    }
};

// ==================== Main Function for Builder ====================
int main(int argc, char** argv) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "       ViT TensorRT Engine Builder (Optimized for Batch 16)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  • Batch size: " << Config::BATCH_SIZE << std::endl;
    std::cout << "  • Input shape: [" << Config::BATCH_SIZE << ", 3, " 
              << Config::INPUT_H << ", " << Config::INPUT_W << "]" << std::endl;
    std::cout << "  • Number of classes: " << Config::NUM_CLASS << std::endl;
    std::cout << "  • Patch size: " << Config::PATCH_SIZE << "x" << Config::PATCH_SIZE << std::endl;
    std::cout << "  • Number of patches: " << Config::NUM_PATCHES << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;

    if (argc < 4) {
        std::cout << "Usage:" << std::endl;
        std::cout << "  ./vit_builder <wts_file> <engine_file> <model_type>" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  ./vit_builder vit_base.wts vit_base_batch16.engine base" << std::endl;
        std::cout << "  ./vit_builder vit_large.wts vit_large_batch16.engine large" << std::endl;
        std::cout << "\nModel types:" << std::endl;
        std::cout << "  • base  - ViT-Base (768 dim, 12 layers, 12 heads)" << std::endl;
        std::cout << "  • large - ViT-Large (1024 dim, 24 layers, 16 heads)" << std::endl;
        return -1;
    }

    // 设置 GPU 设备
    cudaSetDevice(Config::GPU_ID);
    
    // 打印 CUDA 设备信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, Config::GPU_ID);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total Memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) 
              << " GB\n" << std::endl;

    ViTBuilder builder;
    std::cout << "=== Starting Engine Build Process ===" << std::endl;
    builder.Build(argv[1], argv[3]);
    builder.Serialize(argv[2]);
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "✓ Build process completed successfully!" << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;

    return 0;
}
