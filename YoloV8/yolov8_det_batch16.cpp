/*
 * YOLOv8 Detection - Unified Single File Implementation
 * Based on yolov8_single_file_b1.cpp architecture
 * Builds TensorRT engine directly from .wts files (no ONNX, no custom plugins)
 *
 * Usage:
 *   Build engine: ./yolov8_det_unified -s yolov8n.wts yolov8n.engine n
 *   Run inference: ./yolov8_det_unified -d yolov8n.engine ./images/
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
#include <sstream>
#include <numeric>
#include <algorithm>
#include <dirent.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

// ==================== Configuration ====================
namespace Config {
    const int BATCH_SIZE = 16;
    const int NUM_CLASS = 80;
    const int INPUT_H = 640;
    const int INPUT_W = 640;
    const float CONF_THRESH = 0.25f;
    const float NMS_THRESH = 0.45f;
    const int MAX_OUTPUT_BBOX = 1000;
    const int MAX_INPUT_IMAGE_SIZE = 3000 * 3000;
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

// ==================== Structures ====================
struct Detection {
    float bbox[4];  // x1, y1, x2, y2
    float conf;
    int class_id;
};

struct PerfStats {
    int iterations;
    double avg_time;
    double min_time;
    double max_time;
    double total_time;
    double throughput;      // FPS for inference, GB/s for memory copy
    bool is_bandwidth;      // true if throughput represents bandwidth

    PerfStats() : iterations(0), avg_time(0), min_time(0), max_time(0),
                  total_time(0), throughput(0), is_bandwidth(false) {}

    void print(const std::string& title) const {
        std::cout << "\n=== " << title << " ===\n";
        std::cout << "Iterations: " << iterations << "\n";
        std::cout << "Average time: " << std::fixed << std::setprecision(3) << avg_time << " ms\n";
        std::cout << "Min time: " << std::fixed << std::setprecision(3) << min_time << " ms\n";
        std::cout << "Max time: " << std::fixed << std::setprecision(3) << max_time << " ms\n";
        std::cout << "Total time: " << std::fixed << std::setprecision(3) << total_time << " ms\n";

        if (is_bandwidth) {
            std::cout << "Bandwidth: " << std::fixed << std::setprecision(2) << throughput << " GB/s\n";
        } else {
            std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughput << " FPS\n";
        }
        std::cout << "=============================\n";
    }
};

// ==================== COCO Class Names ====================
static const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

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

// ==================== Utility Functions ====================
int GetWidth(int x, float gw, int max_channels, int divisor = 8) {
    auto channel = int(ceil((x * gw) / divisor)) * divisor;
    return channel >= max_channels ? max_channels : channel;
}

int GetDepth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) --r;
    return std::max<int>(r, 1);
}

int ReadFilesInDir(const char* p_dir_name, std::vector<std::string>& file_names) {
    DIR* p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 && strcmp(p_file->d_name, "..") != 0) {
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

// ==================== Network Building Blocks ====================
nvinfer1::IScaleLayer* AddBatchNorm2d(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights>& weightMap,
                                      nvinfer1::ITensor& input,
                                      const std::string& lname,
                                      float eps = 1e-3) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scval, len};

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, shval, len};

    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;

    auto output = network->addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(output);
    return output;
}

nvinfer1::IElementWiseLayer* ConvBnSiLU(nvinfer1::INetworkDefinition* network,
                                        std::map<std::string, nvinfer1::Weights>& weightMap,
                                        nvinfer1::ITensor& input,
                                        int ch, int k, int s, int p,
                                        const std::string& lname) {
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto conv = network->addConvolutionNd(input, ch, nvinfer1::DimsHW{k, k},
                                          weightMap[lname + ".conv.weight"], bias_empty);
    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    conv->setPaddingNd(nvinfer1::DimsHW{p, p});

    auto bn = AddBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);

    auto sigmoid = network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    auto silu = network->addElementWise(*bn->getOutput(0), *sigmoid->getOutput(0),
                                        nvinfer1::ElementWiseOperation::kPROD);
    assert(silu);
    return silu;
}

nvinfer1::ILayer* Bottleneck(nvinfer1::INetworkDefinition* network,
                             std::map<std::string, nvinfer1::Weights>& weightMap,
                             nvinfer1::ITensor& input,
                             int c1, int c2, bool shortcut, float /*e*/,
                             const std::string& lname) {
    auto cv1 = ConvBnSiLU(network, weightMap, input, c2, 3, 1, 1, lname + ".cv1");
    auto cv2 = ConvBnSiLU(network, weightMap, *cv1->getOutput(0), c2, 3, 1, 1, lname + ".cv2");

    if (shortcut && c1 == c2) {
        auto ew = network->addElementWise(input, *cv2->getOutput(0),
                                          nvinfer1::ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

nvinfer1::IElementWiseLayer* C2F(nvinfer1::INetworkDefinition* network,
                                 std::map<std::string, nvinfer1::Weights>& weightMap,
                                 nvinfer1::ITensor& input,
                                 int /*c1*/, int c2, int n, bool shortcut, float e,
                                 const std::string& lname) {
    int c_ = int(c2 * e);
    auto cv1 = ConvBnSiLU(network, weightMap, input, 2 * c_, 1, 1, 0, lname + ".cv1");

    nvinfer1::Dims dims = cv1->getOutput(0)->getDimensions();
    auto split1 = network->addSlice(*cv1->getOutput(0),
                                    nvinfer1::Dims4{0, 0, 0, 0},
                                    nvinfer1::Dims4{dims.d[0], c_, dims.d[2], dims.d[3]},
                                    nvinfer1::Dims4{1, 1, 1, 1});
    auto split2 = network->addSlice(*cv1->getOutput(0),
                                    nvinfer1::Dims4{0, c_, 0, 0},
                                    nvinfer1::Dims4{dims.d[0], c_, dims.d[2], dims.d[3]},
                                    nvinfer1::Dims4{1, 1, 1, 1});

    std::vector<nvinfer1::ITensor*> concat_inputs;
    concat_inputs.push_back(split1->getOutput(0));
    concat_inputs.push_back(split2->getOutput(0));

    auto y = split2->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = Bottleneck(network, weightMap, *y, c_, c_, shortcut, 1.0,
                           lname + ".m." + std::to_string(i));
        y = b->getOutput(0);
        concat_inputs.push_back(y);
    }

    auto cat = network->addConcatenation(concat_inputs.data(), concat_inputs.size());
    cat->setAxis(1);

    auto cv2 = ConvBnSiLU(network, weightMap, *cat->getOutput(0), c2, 1, 1, 0, lname + ".cv2");
    return cv2;
}

nvinfer1::IElementWiseLayer* SPPF(nvinfer1::INetworkDefinition* network,
                                  std::map<std::string, nvinfer1::Weights>& weightMap,
                                  nvinfer1::ITensor& input,
                                  int c1, int c2, int k,
                                  const std::string& lname) {
    int c_ = c1 / 2;
    auto cv1 = ConvBnSiLU(network, weightMap, input, c_, 1, 1, 0, lname + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), nvinfer1::PoolingType::kMAX,
                                       nvinfer1::DimsHW{k, k});
    pool1->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool1->setPaddingNd(nvinfer1::DimsHW{k/2, k/2});

    auto pool2 = network->addPoolingNd(*pool1->getOutput(0), nvinfer1::PoolingType::kMAX,
                                       nvinfer1::DimsHW{k, k});
    pool2->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool2->setPaddingNd(nvinfer1::DimsHW{k/2, k/2});

    auto pool3 = network->addPoolingNd(*pool2->getOutput(0), nvinfer1::PoolingType::kMAX,
                                       nvinfer1::DimsHW{k, k});
    pool3->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool3->setPaddingNd(nvinfer1::DimsHW{k/2, k/2});

    nvinfer1::ITensor* inputTensors[] = {
        cv1->getOutput(0),
        pool1->getOutput(0),
        pool2->getOutput(0),
        pool3->getOutput(0)
    };
    auto cat = network->addConcatenation(inputTensors, 4);
    cat->setAxis(1);

    auto cv2 = ConvBnSiLU(network, weightMap, *cat->getOutput(0), c2, 1, 1, 0, lname + ".cv2");
    return cv2;
}

nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network,
                             std::map<std::string, nvinfer1::Weights>& weightMap,
                             nvinfer1::ITensor& input,
                             int /*ch*/, int grid, int /*k*/, int /*s*/, int /*p*/,
                             const std::string& lname) {
    // Input: [batch, 64, grid] -> Reshape to [batch, 4, 16, grid]
    auto shuffle1 = network->addShuffle(input);
    shuffle1->setReshapeDimensions(nvinfer1::Dims4{Config::BATCH_SIZE, 4, 16, grid});
    
    // Transpose [batch, 4, 16, grid] -> [batch, 16, 4, grid]
    auto shuffle_transpose = network->addShuffle(*shuffle1->getOutput(0));
    shuffle_transpose->setFirstTranspose(nvinfer1::Permutation{0, 2, 1, 3});

    // Softmax on dimension 1 (the 16)
    auto softmax = network->addSoftMax(*shuffle_transpose->getOutput(0));
    softmax->setAxes(1 << 1);

    // Create scale weights [0, 1, 2, ..., 15] for DFL
    float* binScale = reinterpret_cast<float*>(malloc(sizeof(float) * 16));
    for (int i = 0; i < 16; ++i) binScale[i] = static_cast<float>(i);
    nvinfer1::Weights wScale{nvinfer1::DataType::kFLOAT, binScale, 16};
    nvinfer1::Weights wZero{nvinfer1::DataType::kFLOAT, nullptr, 0};
    weightMap[lname + std::string(".dfl_scale")] = wScale;

    // Apply scale: multiply each of the 16 channels by its corresponding weight
    auto scale = network->addScale(*softmax->getOutput(0), nvinfer1::ScaleMode::kCHANNEL, wZero, wScale, wZero);
    
    // Reduce sum along dimension 1 (the 16) to get [batch, 4, grid]
    auto reduce = network->addReduce(*scale->getOutput(0), nvinfer1::ReduceOperation::kSUM, 1 << 1, false);

    // Output shape is [batch, 4, grid]
    auto shuffle_final = network->addShuffle(*reduce->getOutput(0));
    shuffle_final->setReshapeDimensions(nvinfer1::Dims3{Config::BATCH_SIZE, 4, grid});

    return shuffle_final;
}

// ==================== Image Preprocessing ====================
cv::Mat PreProcessImage(const cv::Mat& img, bool debug = false) {
    float r_w = Config::INPUT_W / (img.cols * 1.0);
    float r_h = Config::INPUT_H / (img.rows * 1.0);

    int resize_w, resize_h, pad_x, pad_y;

    if (r_h > r_w) {
        resize_w = Config::INPUT_W;
        resize_h = r_w * img.rows;
        pad_x = 0;
        pad_y = (Config::INPUT_H - resize_h) / 2;
    } else {
        resize_w = r_h * img.cols;
        resize_h = Config::INPUT_H;
        pad_x = (Config::INPUT_W - resize_w) / 2;
        pad_y = 0;
    }

    cv::Mat resized(resize_h, resize_w, CV_8UC3);
    cv::resize(img, resized, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat out(Config::INPUT_H, Config::INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    resized.copyTo(out(cv::Rect(pad_x, pad_y, resized.cols, resized.rows)));

    // Convert BGR to RGB (YOLOv8 expects RGB)
    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);

    cv::Mat blob;
    out.convertTo(blob, CV_32FC3, 1.0 / 255.0);

    cv::Mat channels[3];
    cv::split(blob, channels);

    // NCHW format: R, G, B order (already converted from BGR to RGB above)
    // Use row-wise copy to ensure correct memory layout
    cv::Mat result(1, 3 * Config::INPUT_H * Config::INPUT_W, CV_32FC1);
    float* result_ptr = (float*)result.data;

    // Copy R channel
    for (int i = 0; i < Config::INPUT_H; i++) {
        memcpy(result_ptr + i * Config::INPUT_W,
               channels[0].ptr<float>(i),
               Config::INPUT_W * sizeof(float));
    }

    // Copy G channel
    for (int i = 0; i < Config::INPUT_H; i++) {
        memcpy(result_ptr + Config::INPUT_H * Config::INPUT_W + i * Config::INPUT_W,
               channels[1].ptr<float>(i),
               Config::INPUT_W * sizeof(float));
    }

    // Copy B channel
    for (int i = 0; i < Config::INPUT_H; i++) {
        memcpy(result_ptr + 2 * Config::INPUT_H * Config::INPUT_W + i * Config::INPUT_W,
               channels[2].ptr<float>(i),
               Config::INPUT_W * sizeof(float));
    }

    return result;
}

// ==================== NMS ====================
void NMS(std::vector<Detection>& dets, float thresh) {
    if (dets.empty()) return;

    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.conf > b.conf;
    });

    std::vector<bool> suppressed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;

        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            if (dets[i].class_id != dets[j].class_id) continue;

            float x1 = std::max(dets[i].bbox[0], dets[j].bbox[0]);
            float y1 = std::max(dets[i].bbox[1], dets[j].bbox[1]);
            float x2 = std::min(dets[i].bbox[2], dets[j].bbox[2]);
            float y2 = std::min(dets[i].bbox[3], dets[j].bbox[3]);

            if (x2 <= x1 || y2 <= y1) continue;

            float inter = (x2 - x1) * (y2 - y1);
            float area1 = (dets[i].bbox[2] - dets[i].bbox[0]) * (dets[i].bbox[3] - dets[i].bbox[1]);
            float area2 = (dets[j].bbox[2] - dets[j].bbox[0]) * (dets[j].bbox[3] - dets[j].bbox[1]);

            if (area1 <= 0 || area2 <= 0) continue;

            float union_area = area1 + area2 - inter;
            if (union_area <= 0) continue;

            float iou = inter / union_area;

            if (iou > thresh) {
                suppressed[j] = true;
            }
        }
    }

    std::vector<Detection> result;
    for (size_t i = 0; i < dets.size(); i++) {
        if (!suppressed[i]) {
            result.push_back(dets[i]);
        }
    }

    dets = result;
}

// ==================== Postprocessing ====================
void PostProcess(float* output, std::vector<Detection>& dets, int orig_w, int orig_h) {
    float r_w = Config::INPUT_W / (orig_w * 1.0);
    float r_h = Config::INPUT_H / (orig_h * 1.0);

    const int strides[3] = {8, 16, 32};
    const int grid_sizes[3][2] = {{80, 80}, {40, 40}, {20, 20}};
    const int start_indices[3] = {0, 6400, 8000};

    // YOLOv8 output format: [84, 8400] (Feature-Major)
    // output[feature_idx * 8400 + anchor_idx]
    for (int i = 0; i < 8400; i++) {
        int scale_idx = 0;
        if (i >= 8000) scale_idx = 2;
        else if (i >= 6400) scale_idx = 1;
        else scale_idx = 0;

        int stride = strides[scale_idx];
        int grid_w = grid_sizes[scale_idx][1];
        int anchor_idx = i - start_indices[scale_idx];
        int row = anchor_idx / grid_w;
        int col = anchor_idx % grid_w;

        // Get bbox distances (Feature-Major: [feature][anchor])
        float dist_left = output[0 * 8400 + i];
        float dist_top = output[1 * 8400 + i];
        float dist_right = output[2 * 8400 + i];
        float dist_bottom = output[3 * 8400 + i];

        // Get class scores
        int class_id = 0;
        float max_score = 0.0f;
        for (int c = 0; c < Config::NUM_CLASS; c++) {
            float score = 1.0f / (1.0f + std::exp(-output[(4 + c) * 8400 + i]));
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score > Config::CONF_THRESH) {
            Detection det;

            // Decode bbox coordinates
            float x1 = (col + 0.5f - dist_left) * stride;
            float y1 = (row + 0.5f - dist_top) * stride;
            float x2 = (col + 0.5f + dist_right) * stride;
            float y2 = (row + 0.5f + dist_bottom) * stride;

            // Map back to original image
            float l, r, t, b;
            if (r_h > r_w) {
                l = x1;
                r = x2;
                t = y1 - (Config::INPUT_H - r_w * orig_h) / 2;
                b = y2 - (Config::INPUT_H - r_w * orig_h) / 2;
                l = l / r_w;
                r = r / r_w;
                t = t / r_w;
                b = b / r_w;
            } else {
                float pad_x = (Config::INPUT_W - r_h * orig_w) / 2;
                l = x1 - pad_x;
                r = x2 - pad_x;
                t = y1;
                b = y2;
                l = l / r_h;
                r = r / r_h;
                t = t / r_h;
                b = b / r_h;
            }

            l = std::max(0.0f, std::min(l, float(orig_w)));
            t = std::max(0.0f, std::min(t, float(orig_h)));
            r = std::max(0.0f, std::min(r, float(orig_w)));
            b = std::max(0.0f, std::min(b, float(orig_h)));

            det.bbox[0] = l;
            det.bbox[1] = t;
            det.bbox[2] = r;
            det.bbox[3] = b;
            det.conf = max_score;
            det.class_id = class_id;

            dets.push_back(det);
        }
    }

    NMS(dets, Config::NMS_THRESH);
}

// ==================== Visualization ====================
void DrawBboxes(cv::Mat& img, const std::vector<Detection>& dets) {
    for (const auto& det : dets) {
        cv::rectangle(img,
                     cv::Point(det.bbox[0], det.bbox[1]),
                     cv::Point(det.bbox[2], det.bbox[3]),
                     cv::Scalar(0, 255, 0), 2);

        std::string label = CLASS_NAMES[det.class_id] + " " +
                           std::to_string(int(det.conf * 100)) + "%";

        cv::putText(img, label,
                   cv::Point(det.bbox[0], det.bbox[1] - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}

// ==================== YOLOv8 TensorRT Class ====================
class YOLOv8TensorRT {
private:
    nvinfer1::IBuilder* builder;
    nvinfer1::IBuilderConfig* config;
    nvinfer1::INetworkDefinition* network;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    nvinfer1::IRuntime* runtime;

    std::map<std::string, nvinfer1::Weights> weightMap;
    float gd, gw;
    int max_channels;

public:
    YOLOv8TensorRT() : builder(nullptr), config(nullptr), network(nullptr),
                       engine(nullptr), context(nullptr), runtime(nullptr),
                       gd(0.0f), gw(0.0f), max_channels(0) {}

    ~YOLOv8TensorRT() {
        if (context) context->destroy();
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
        char size = model_type[0];
        if (size == 'n') {
            gd = 0.33; gw = 0.25; max_channels = 1024;
        } else if (size == 's') {
            gd = 0.33; gw = 0.50; max_channels = 1024;
        } else if (size == 'm') {
            gd = 0.67; gw = 0.75; max_channels = 576;
        } else if (size == 'l') {
            gd = 1.0; gw = 1.0; max_channels = 512;
        } else if (size == 'x') {
            gd = 1.0; gw = 1.25; max_channels = 640;
        } else {
            std::cerr << "Invalid model type: " << model_type << std::endl;
            exit(1);
        }
        std::cout << "Model: YOLOv8" << model_type << " (gd=" << gd << ", gw=" << gw << ")" << std::endl;
    }

    void Build(const std::string& wts_file, const std::string& model_type) {
        weightMap = LoadWeights(wts_file);
        SetModelConfig(model_type);

        builder = nvinfer1::createInferBuilder(gLogger);
        network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

        BuildYOLOv8DetNetwork();

        config = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(size_t(2500) * (1 << 20));  // 改为 2500MB   // 16MB

        // Enable FP16 if available
        if (builder->platformHasFastFp16()) {
            std::cout << "Using FP16 precision" << std::endl;
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        std::cout << "Building engine... This may take a while." << std::endl;
        auto build_start = std::chrono::high_resolution_clock::now();
        auto serialized_model = builder->buildSerializedNetwork(*network, *config);
        auto build_end = std::chrono::high_resolution_clock::now();
        auto build_duration = std::chrono::duration_cast<std::chrono::seconds>(build_end - build_start);
        std::cout << "Engine build time: " << build_duration.count() << " seconds" << std::endl;

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

        context = engine->createExecutionContext();
        if (!context) {
            std::cerr << "ERROR: createExecutionContext failed" << std::endl;
            delete serialized_model;
            return;
        }

        delete serialized_model;
        std::cout << "Engine built successfully!" << std::endl;
    }

    void Serialize(const std::string& engine_file) {
        if (!engine) {
            std::cerr << "ERROR: No engine to serialize" << std::endl;
            return;
        }

        auto serialized = engine->serialize();
        if (!serialized) {
            std::cerr << "ERROR: engine serialize failed" << std::endl;
            return;
        }

        std::ofstream out(engine_file, std::ios::binary);
        out.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
        out.close();
        delete serialized;
        std::cout << "Engine saved to: " << engine_file << std::endl;
    }

    void Deserialize(const std::string& engine_file) {
        std::ifstream file(engine_file, std::ios::binary);
        if (!file) {
            std::cerr << "ERROR: Cannot open engine file: " << engine_file << std::endl;
            return;
        }

        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);

        char* buffer = new char[size];
        file.read(buffer, size);
        file.close();

        runtime = nvinfer1::createInferRuntime(gLogger);
        if (!runtime) {
            std::cerr << "ERROR: createInferRuntime failed" << std::endl;
            delete[] buffer;
            return;
        }

        engine = runtime->deserializeCudaEngine(buffer, size);
        if (!engine) {
            std::cerr << "ERROR: deserializeCudaEngine failed" << std::endl;
            delete[] buffer;
            return;
        }

        context = engine->createExecutionContext();
        if (!context) {
            std::cerr << "ERROR: createExecutionContext failed" << std::endl;
            delete[] buffer;
            return;
        }

        delete[] buffer;
        std::cout << "Engine loaded from: " << engine_file << std::endl;
    }

    void Infer(const std::vector<cv::Mat>& imgs, std::vector<std::vector<Detection>>& results) {
        if (imgs.size() != Config::BATCH_SIZE) {
            std::cerr << "ERROR: Batch size must be " << Config::BATCH_SIZE << std::endl;
            return;
        }

        // Allocate input buffer for batch=16
        const int single_img_size = 3 * Config::INPUT_H * Config::INPUT_W;
        const int batch_input_size = Config::BATCH_SIZE * single_img_size;
        float* input_data = new float[batch_input_size];

        // Preprocess all images in the batch
        for (int b = 0; b < Config::BATCH_SIZE; b++) {
            cv::Mat pr_img = PreProcessImage(imgs[b]);
            memcpy(input_data + b * single_img_size, pr_img.data, single_img_size * sizeof(float));
        }

        void* buffers[2];
        const int input_index = engine->getBindingIndex(Config::INPUT_BLOB_NAME);
        const int output_index = engine->getBindingIndex(Config::OUTPUT_BLOB_NAME);

        // Allocate GPU memory for batch=16
        CUDA_CHECK(cudaMalloc(&buffers[input_index], batch_input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[output_index], Config::BATCH_SIZE * 8400 * 84 * sizeof(float)));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        CUDA_CHECK(cudaMemcpyAsync(buffers[input_index], input_data,
                                   batch_input_size * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));

        context->enqueueV2(buffers, stream, nullptr);

        float* output_data = new float[Config::BATCH_SIZE * 8400 * 84];
        CUDA_CHECK(cudaMemcpyAsync(output_data, buffers[output_index],
                                   Config::BATCH_SIZE * 8400 * 84 * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Postprocess all images in the batch
        results.resize(Config::BATCH_SIZE);
        for (int b = 0; b < Config::BATCH_SIZE; b++) {
            PostProcess(output_data + b * 8400 * 84, results[b], imgs[b].cols, imgs[b].rows);
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(buffers[input_index]));
        CUDA_CHECK(cudaFree(buffers[output_index]));
        delete[] input_data;
        delete[] output_data;
    }

    // Performance testing method with detailed timing
    void PerformanceTest(const std::vector<cv::Mat>& imgs, int iterations = 100) {
        if (imgs.size() != Config::BATCH_SIZE) {
            std::cerr << "ERROR: Batch size must be " << Config::BATCH_SIZE << std::endl;
            return;
        }

        // Allocate and prepare input data
        const int single_img_size = 3 * Config::INPUT_H * Config::INPUT_W;
        const int batch_input_size = Config::BATCH_SIZE * single_img_size;
        float* input_data = new float[batch_input_size];

        for (int b = 0; b < Config::BATCH_SIZE; b++) {
            cv::Mat pr_img = PreProcessImage(imgs[b]);
            memcpy(input_data + b * single_img_size, pr_img.data, single_img_size * sizeof(float));
        }

        void* buffers[2];
        const int input_index = engine->getBindingIndex(Config::INPUT_BLOB_NAME);
        const int output_index = engine->getBindingIndex(Config::OUTPUT_BLOB_NAME);

        const size_t output_size = Config::BATCH_SIZE * 8400 * 84 * sizeof(float);

        CUDA_CHECK(cudaMalloc(&buffers[input_index], batch_input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[output_index], output_size));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Copy input to device (H2D - only once)
        CUDA_CHECK(cudaMemcpyAsync(buffers[input_index], input_data,
                                   batch_input_size * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float* output_data = new float[Config::BATCH_SIZE * 8400 * 84];

        // Warmup
        std::cout << "Warming up..." << std::endl;
        for (int i = 0; i < 10; i++) {
            context->enqueueV2(buffers, stream, nullptr);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // Statistics for Inference Only (with synchronization)
        std::vector<double> inference_times;

        // Statistics for Memory Copy (D2H)
        std::vector<double> memcpy_times;

        // Statistics for Total Execution
        std::vector<double> total_times;

        std::cout << "Running " << iterations << " iterations for performance testing (with sync)..." << std::endl;

        for (int i = 0; i < iterations; i++) {
            // Total execution timing
            auto total_start = std::chrono::high_resolution_clock::now();

            // Inference timing
            auto infer_start = std::chrono::high_resolution_clock::now();
            context->enqueueV2(buffers, stream, nullptr);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto infer_end = std::chrono::high_resolution_clock::now();

            // Memory copy (D2H) timing
            auto memcpy_start = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaMemcpyAsync(output_data, buffers[output_index],
                                       output_size, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto memcpy_end = std::chrono::high_resolution_clock::now();

            auto total_end = std::chrono::high_resolution_clock::now();

            inference_times.push_back(std::chrono::duration<double, std::milli>(infer_end - infer_start).count());
            memcpy_times.push_back(std::chrono::duration<double, std::milli>(memcpy_end - memcpy_start).count());
            total_times.push_back(std::chrono::duration<double, std::milli>(total_end - total_start).count());
        }

        // ========== Pipeline Mode Test (Maximum Throughput) ==========
        std::cout << "\nRunning pipeline mode test for maximum GPU throughput..." << std::endl;

        // Warmup for pipeline
        for (int i = 0; i < 10; i++) {
            context->enqueueV2(buffers, stream, nullptr);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto pipeline_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            context->enqueueV2(buffers, stream, nullptr);
            // No sync here - let GPU keep working!
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));  // Only sync at the end
        auto pipeline_end = std::chrono::high_resolution_clock::now();

        double pipeline_total_time = std::chrono::duration<double, std::milli>(pipeline_end - pipeline_start).count();

        // Calculate statistics
        auto calc_throughput_stats = [](const std::vector<double>& times, int iters) -> PerfStats {
            PerfStats stats;
            stats.iterations = iters;
            stats.min_time = *std::min_element(times.begin(), times.end());
            stats.max_time = *std::max_element(times.begin(), times.end());
            stats.total_time = std::accumulate(times.begin(), times.end(), 0.0);
            stats.avg_time = stats.total_time / iters;
            stats.throughput = (Config::BATCH_SIZE * 1000.0 * iters) / stats.total_time;
            stats.is_bandwidth = false;
            return stats;
        };

        auto calc_bandwidth_stats = [output_size](const std::vector<double>& times, int iters) -> PerfStats {
            PerfStats stats;
            stats.iterations = iters;
            stats.min_time = *std::min_element(times.begin(), times.end());
            stats.max_time = *std::max_element(times.begin(), times.end());
            stats.total_time = std::accumulate(times.begin(), times.end(), 0.0);
            stats.avg_time = stats.total_time / iters;
            // Bandwidth = (total_bytes * iterations) / (total_time_ms / 1000) / (1024^3)
            stats.throughput = (output_size * iters) / (stats.total_time / 1000.0) / (1024.0 * 1024.0 * 1024.0);
            stats.is_bandwidth = true;
            return stats;
        };

        PerfStats inference_stats = calc_throughput_stats(inference_times, iterations);
        PerfStats memcpy_stats = calc_bandwidth_stats(memcpy_times, iterations);
        PerfStats total_stats = calc_throughput_stats(total_times, iterations);

        // Pipeline stats
        PerfStats pipeline_stats;
        pipeline_stats.iterations = iterations;
        pipeline_stats.total_time = pipeline_total_time;
        pipeline_stats.avg_time = pipeline_total_time / iterations;
        pipeline_stats.min_time = pipeline_stats.avg_time;  // Can't measure individual times in pipeline
        pipeline_stats.max_time = pipeline_stats.avg_time;
        pipeline_stats.throughput = (Config::BATCH_SIZE * 1000.0 * iterations) / pipeline_total_time;
        pipeline_stats.is_bandwidth = false;

        // Print results
        inference_stats.print("Inference Only 性能统计");
        memcpy_stats.print("Memory Copy (D2H) 性能统计");
        total_stats.print("Total Execution 性能统计");
        pipeline_stats.print("GPU Peak Throughput (Pipeline) 性能统计");

        // Cleanup
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(buffers[input_index]));
        CUDA_CHECK(cudaFree(buffers[output_index]));
        delete[] input_data;
        delete[] output_data;
    }

private:
    void BuildYOLOv8DetNetwork() {
        std::cout << "Building YOLOv8 Detection network..." << std::endl;

        auto data = network->addInput(Config::INPUT_BLOB_NAME, nvinfer1::DataType::kFLOAT,
                                      nvinfer1::Dims4{Config::BATCH_SIZE, 3, Config::INPUT_H, Config::INPUT_W});

        // Backbone
        auto conv0 = ConvBnSiLU(network, weightMap, *data, GetWidth(64, gw, max_channels), 3, 2, 1, "model.0");
        auto conv1 = ConvBnSiLU(network, weightMap, *conv0->getOutput(0), GetWidth(128, gw, max_channels), 3, 2, 1, "model.1");
        auto conv2 = C2F(network, weightMap, *conv1->getOutput(0), GetWidth(128, gw, max_channels),
                        GetWidth(128, gw, max_channels), GetDepth(3, gd), true, 0.5, "model.2");
        auto conv3 = ConvBnSiLU(network, weightMap, *conv2->getOutput(0), GetWidth(256, gw, max_channels), 3, 2, 1, "model.3");
        auto conv4 = C2F(network, weightMap, *conv3->getOutput(0), GetWidth(256, gw, max_channels),
                        GetWidth(256, gw, max_channels), GetDepth(6, gd), true, 0.5, "model.4");
        auto conv5 = ConvBnSiLU(network, weightMap, *conv4->getOutput(0), GetWidth(512, gw, max_channels), 3, 2, 1, "model.5");
        auto conv6 = C2F(network, weightMap, *conv5->getOutput(0), GetWidth(512, gw, max_channels),
                        GetWidth(512, gw, max_channels), GetDepth(6, gd), true, 0.5, "model.6");
        auto conv7 = ConvBnSiLU(network, weightMap, *conv6->getOutput(0), GetWidth(1024, gw, max_channels), 3, 2, 1, "model.7");
        auto conv8 = C2F(network, weightMap, *conv7->getOutput(0), GetWidth(1024, gw, max_channels),
                        GetWidth(1024, gw, max_channels), GetDepth(3, gd), true, 0.5, "model.8");
        auto conv9 = SPPF(network, weightMap, *conv8->getOutput(0), GetWidth(1024, gw, max_channels),
                         GetWidth(1024, gw, max_channels), 5, "model.9");

        // Neck
        float scale_nchw[] = {1.0f, 1.0f, 2.0f, 2.0f};
        auto up10 = network->addResize(*conv9->getOutput(0));
        up10->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
        up10->setScales(scale_nchw, 4);

        nvinfer1::ITensor* cat11_inputs[] = {up10->getOutput(0), conv6->getOutput(0)};
        auto cat11 = network->addConcatenation(cat11_inputs, 2);
        cat11->setAxis(1);
        auto conv12 = C2F(network, weightMap, *cat11->getOutput(0),
                         GetWidth(1024, gw, max_channels) + GetWidth(512, gw, max_channels),
                         GetWidth(512, gw, max_channels), GetDepth(3, gd), false, 0.5, "model.12");

        auto up13 = network->addResize(*conv12->getOutput(0));
        up13->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
        up13->setScales(scale_nchw, 4);

        nvinfer1::ITensor* cat14_inputs[] = {up13->getOutput(0), conv4->getOutput(0)};
        auto cat14 = network->addConcatenation(cat14_inputs, 2);
        cat14->setAxis(1);
        auto conv15 = C2F(network, weightMap, *cat14->getOutput(0),
                         GetWidth(512, gw, max_channels) + GetWidth(256, gw, max_channels),
                         GetWidth(256, gw, max_channels), GetDepth(3, gd), false, 0.5, "model.15");

        auto conv16 = ConvBnSiLU(network, weightMap, *conv15->getOutput(0), GetWidth(256, gw, max_channels), 3, 2, 1, "model.16");
        nvinfer1::ITensor* cat17_inputs[] = {conv16->getOutput(0), conv12->getOutput(0)};
        auto cat17 = network->addConcatenation(cat17_inputs, 2);
        cat17->setAxis(1);
        auto conv18 = C2F(network, weightMap, *cat17->getOutput(0),
                         GetWidth(256, gw, max_channels) + GetWidth(512, gw, max_channels),
                         GetWidth(512, gw, max_channels), GetDepth(3, gd), false, 0.5, "model.18");

        auto conv19 = ConvBnSiLU(network, weightMap, *conv18->getOutput(0), GetWidth(512, gw, max_channels), 3, 2, 1, "model.19");
        nvinfer1::ITensor* cat20_inputs[] = {conv19->getOutput(0), conv9->getOutput(0)};
        auto cat20 = network->addConcatenation(cat20_inputs, 2);
        cat20->setAxis(1);
        auto conv21 = C2F(network, weightMap, *cat20->getOutput(0),
                         GetWidth(512, gw, max_channels) + GetWidth(1024, gw, max_channels),
                         GetWidth(1024, gw, max_channels), GetDepth(3, gd), false, 0.5, "model.21");

        // Detection Head
        int base_in = (gw == 1.25) ? 80 : 64;
        int base_out = (gw == 0.25) ? std::max(64, std::min(Config::NUM_CLASS, 100)) : GetWidth(256, gw, max_channels);

        nvinfer1::Dims dims15 = conv15->getOutput(0)->getDimensions();
        nvinfer1::Dims dims18 = conv18->getOutput(0)->getDimensions();
        nvinfer1::Dims dims21 = conv21->getOutput(0)->getDimensions();

        int stride_0 = Config::INPUT_H / dims15.d[2];
        int stride_1 = Config::INPUT_H / dims18.d[2];
        int stride_2 = Config::INPUT_H / dims21.d[2];
        int grid_0 = (Config::INPUT_H / stride_0) * (Config::INPUT_W / stride_0);
        int grid_1 = (Config::INPUT_H / stride_1) * (Config::INPUT_W / stride_1);
        int grid_2 = (Config::INPUT_H / stride_2) * (Config::INPUT_W / stride_2);

        // Build detection heads for 3 scales (similar to yolov8_single_file_b1.cpp)
        // Scale 0 (80x80)
        auto cv2_0_0 = ConvBnSiLU(network, weightMap, *conv15->getOutput(0), base_in, 3, 1, 1, "model.22.cv2.0.0");
        auto cv2_0_1 = ConvBnSiLU(network, weightMap, *cv2_0_0->getOutput(0), base_in, 3, 1, 1, "model.22.cv2.0.1");
        auto cv2_0_2 = network->addConvolutionNd(*cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                                weightMap["model.22.cv2.0.2.weight"], weightMap["model.22.cv2.0.2.bias"]);
        cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});

        auto cv3_0_0 = ConvBnSiLU(network, weightMap, *conv15->getOutput(0), base_out, 3, 1, 1, "model.22.cv3.0.0");
        auto cv3_0_1 = ConvBnSiLU(network, weightMap, *cv3_0_0->getOutput(0), base_out, 3, 1, 1, "model.22.cv3.0.1");
        auto cv3_0_2 = network->addConvolutionNd(*cv3_0_1->getOutput(0), Config::NUM_CLASS, nvinfer1::DimsHW{1, 1},
                                                weightMap["model.22.cv3.0.2.weight"], weightMap["model.22.cv3.0.2.bias"]);
        cv3_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});

        nvinfer1::ITensor* cat22_0_inputs[] = {cv2_0_2->getOutput(0), cv3_0_2->getOutput(0)};
        auto cat22_0 = network->addConcatenation(cat22_0_inputs, 2);
        cat22_0->setAxis(1);

        auto shuffle_0 = network->addShuffle(*cat22_0->getOutput(0));
        shuffle_0->setReshapeDimensions(nvinfer1::Dims3{Config::BATCH_SIZE, 64 + Config::NUM_CLASS, grid_0});

        auto split_0_bbox = network->addSlice(*shuffle_0->getOutput(0),
                                             nvinfer1::Dims3{0, 0, 0},
                                             nvinfer1::Dims3{Config::BATCH_SIZE, 64, grid_0},
                                             nvinfer1::Dims3{1, 1, 1});
        auto split_0_cls = network->addSlice(*shuffle_0->getOutput(0),
                                            nvinfer1::Dims3{0, 64, 0},
                                            nvinfer1::Dims3{Config::BATCH_SIZE, Config::NUM_CLASS, grid_0},
                                            nvinfer1::Dims3{1, 1, 1});

        auto dfl_0 = DFL(network, weightMap, *split_0_bbox->getOutput(0), 4, grid_0,
                        1, 1, 0, "model.22.dfl.conv.weight");

        nvinfer1::ITensor* det_0_inputs[] = {dfl_0->getOutput(0), split_0_cls->getOutput(0)};
        auto det_0 = network->addConcatenation(det_0_inputs, 2);
        det_0->setAxis(1);  // [batch, 4, grid] + [batch, 80, grid] -> [batch, 84, grid]

        // Scale 1 (40x40) - similar pattern
        auto cv2_1_0 = ConvBnSiLU(network, weightMap, *conv18->getOutput(0), base_in, 3, 1, 1, "model.22.cv2.1.0");
        auto cv2_1_1 = ConvBnSiLU(network, weightMap, *cv2_1_0->getOutput(0), base_in, 3, 1, 1, "model.22.cv2.1.1");
        auto cv2_1_2 = network->addConvolutionNd(*cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                                weightMap["model.22.cv2.1.2.weight"], weightMap["model.22.cv2.1.2.bias"]);
        cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});

        auto cv3_1_0 = ConvBnSiLU(network, weightMap, *conv18->getOutput(0), base_out, 3, 1, 1, "model.22.cv3.1.0");
        auto cv3_1_1 = ConvBnSiLU(network, weightMap, *cv3_1_0->getOutput(0), base_out, 3, 1, 1, "model.22.cv3.1.1");
        auto cv3_1_2 = network->addConvolutionNd(*cv3_1_1->getOutput(0), Config::NUM_CLASS, nvinfer1::DimsHW{1, 1},
                                                weightMap["model.22.cv3.1.2.weight"], weightMap["model.22.cv3.1.2.bias"]);
        cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});

        nvinfer1::ITensor* cat22_1_inputs[] = {cv2_1_2->getOutput(0), cv3_1_2->getOutput(0)};
        auto cat22_1 = network->addConcatenation(cat22_1_inputs, 2);
        cat22_1->setAxis(1);

        auto shuffle_1 = network->addShuffle(*cat22_1->getOutput(0));
        shuffle_1->setReshapeDimensions(nvinfer1::Dims3{Config::BATCH_SIZE, 64 + Config::NUM_CLASS, grid_1});

        auto split_1_bbox = network->addSlice(*shuffle_1->getOutput(0),
                                             nvinfer1::Dims3{0, 0, 0},
                                             nvinfer1::Dims3{Config::BATCH_SIZE, 64, grid_1},
                                             nvinfer1::Dims3{1, 1, 1});
        auto split_1_cls = network->addSlice(*shuffle_1->getOutput(0),
                                            nvinfer1::Dims3{0, 64, 0},
                                            nvinfer1::Dims3{Config::BATCH_SIZE, Config::NUM_CLASS, grid_1},
                                            nvinfer1::Dims3{1, 1, 1});

        auto dfl_1 = DFL(network, weightMap, *split_1_bbox->getOutput(0), 4, grid_1,
                        1, 1, 0, "model.22.dfl.conv.weight");

        nvinfer1::ITensor* det_1_inputs[] = {dfl_1->getOutput(0), split_1_cls->getOutput(0)};
        auto det_1 = network->addConcatenation(det_1_inputs, 2);
        det_1->setAxis(1);  // [batch, 4, grid] + [batch, 80, grid] -> [batch, 84, grid]

        // Scale 2 (20x20) - similar pattern
        auto cv2_2_0 = ConvBnSiLU(network, weightMap, *conv21->getOutput(0), base_in, 3, 1, 1, "model.22.cv2.2.0");
        auto cv2_2_1 = ConvBnSiLU(network, weightMap, *cv2_2_0->getOutput(0), base_in, 3, 1, 1, "model.22.cv2.2.1");
        auto cv2_2_2 = network->addConvolutionNd(*cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                                weightMap["model.22.cv2.2.2.weight"], weightMap["model.22.cv2.2.2.bias"]);
        cv2_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});

        auto cv3_2_0 = ConvBnSiLU(network, weightMap, *conv21->getOutput(0), base_out, 3, 1, 1, "model.22.cv3.2.0");
        auto cv3_2_1 = ConvBnSiLU(network, weightMap, *cv3_2_0->getOutput(0), base_out, 3, 1, 1, "model.22.cv3.2.1");
        auto cv3_2_2 = network->addConvolutionNd(*cv3_2_1->getOutput(0), Config::NUM_CLASS, nvinfer1::DimsHW{1, 1},
                                                weightMap["model.22.cv3.2.2.weight"], weightMap["model.22.cv3.2.2.bias"]);
        cv3_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});

        nvinfer1::ITensor* cat22_2_inputs[] = {cv2_2_2->getOutput(0), cv3_2_2->getOutput(0)};
        auto cat22_2 = network->addConcatenation(cat22_2_inputs, 2);
        cat22_2->setAxis(1);

        auto shuffle_2 = network->addShuffle(*cat22_2->getOutput(0));
        shuffle_2->setReshapeDimensions(nvinfer1::Dims3{Config::BATCH_SIZE, 64 + Config::NUM_CLASS, grid_2});

        auto split_2_bbox = network->addSlice(*shuffle_2->getOutput(0),
                                             nvinfer1::Dims3{0, 0, 0},
                                             nvinfer1::Dims3{Config::BATCH_SIZE, 64, grid_2},
                                             nvinfer1::Dims3{1, 1, 1});
        auto split_2_cls = network->addSlice(*shuffle_2->getOutput(0),
                                            nvinfer1::Dims3{0, 64, 0},
                                            nvinfer1::Dims3{Config::BATCH_SIZE, Config::NUM_CLASS, grid_2},
                                            nvinfer1::Dims3{1, 1, 1});

        auto dfl_2 = DFL(network, weightMap, *split_2_bbox->getOutput(0), 4, grid_2,
                        1, 1, 0, "model.22.dfl.conv.weight");

        nvinfer1::ITensor* det_2_inputs[] = {dfl_2->getOutput(0), split_2_cls->getOutput(0)};
        auto det_2 = network->addConcatenation(det_2_inputs, 2);
        det_2->setAxis(1);  // [batch, 4, grid] + [batch, 80, grid] -> [batch, 84, grid]

        // Concatenate all 3 scales
        nvinfer1::ITensor* final_inputs[] = {det_0->getOutput(0), det_1->getOutput(0), det_2->getOutput(0)};
        auto final_concat = network->addConcatenation(final_inputs, 3);
        final_concat->setAxis(2);  // [16,84,6400] + [16,84,1600] + [16,84,400] -> [16,84,8400]

        // Output shape: [batch, 84, 8400] (Feature-Major format)
        // No transpose needed - postprocess expects Feature-Major layout
        final_concat->getOutput(0)->setName(Config::OUTPUT_BLOB_NAME);
        network->markOutput(*final_concat->getOutput(0));

        std::cout << "YOLOv8 Detection network built successfully! Output: ["
                  << Config::BATCH_SIZE << ", 84, 8400] (Batch-Feature-Anchor)" << std::endl;
    }
};

// ==================== Main Function ====================
int main(int argc, char** argv) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "YOLOv8 Detection - Batch 16 with Performance Testing" << std::endl;
    std::cout << "Based on yolov8_single_file_b1.cpp architecture" << std::endl;
    std::cout << "Batch size: " << Config::BATCH_SIZE << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;

    if (argc < 3) {
        std::cout << "Usage:" << std::endl;
        std::cout << "  Serialize:  ./yolov8_det_batch16 -s <wts> <engine> <model>" << std::endl;
        std::cout << "  Infer:      ./yolov8_det_batch16 -d <engine> <image_dir>" << std::endl;
        std::cout << "  Perf Test:  ./yolov8_det_batch16 -p <engine> <image_path> [iterations]" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  ./yolov8_det_batch16 -s yolov8n.wts yolov8n.engine n" << std::endl;
        std::cout << "  ./yolov8_det_batch16 -d yolov8n.engine ./images/" << std::endl;
        std::cout << "  ./yolov8_det_batch16 -p yolov8n.engine ./image.jpg 100" << std::endl;
        std::cout << "\nModel types: n/s/m/l/x" << std::endl;
        return -1;
    }

    cudaSetDevice(Config::GPU_ID);

    YOLOv8TensorRT yolo;
    std::string mode = argv[1];

    if (mode == "-s") {
        if (argc < 5) {
            std::cerr << "ERROR: -s requires <wts> <engine> <model>" << std::endl;
            return -1;
        }
        std::cout << "=== Building and Serializing Engine ===" << std::endl;
        yolo.Build(argv[2], argv[4]);
        yolo.Serialize(argv[3]);
        std::cout << "Done!" << std::endl;

    } else if (mode == "-p") {
        // Performance testing mode
        if (argc < 4) {
            std::cerr << "ERROR: -p requires <engine> <image_path> [iterations]" << std::endl;
            return -1;
        }

        std::cout << "=== Performance Testing Mode ===" << std::endl;
        yolo.Deserialize(argv[2]);

        std::string img_path = argv[3];
        int iterations = 100;
        if (argc >= 5) {
            iterations = std::atoi(argv[4]);
        }

        cv::Mat img = cv::imread(img_path);
        if (img.empty()) {
            std::cerr << "ERROR: Failed to load image: " << img_path << std::endl;
            return -1;
        }

        std::cout << "Image: " << img_path << " (" << img.cols << "x" << img.rows << ")" << std::endl;
        std::cout << "Batch size: " << Config::BATCH_SIZE << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;

        // Prepare batch of images (replicate the same image)
        std::vector<cv::Mat> imgs(Config::BATCH_SIZE);
        for (int i = 0; i < Config::BATCH_SIZE; i++) {
            imgs[i] = img.clone();
        }

        yolo.PerformanceTest(imgs, iterations);

    } else if (mode == "-d") {
        if (argc < 4) {
            std::cerr << "ERROR: -d requires <engine> <image_dir>" << std::endl;
            return -1;
        }

        std::cout << "=== Loading Engine and Running Inference ===" << std::endl;
        yolo.Deserialize(argv[2]);

        std::string image_dir = argv[3];
        std::vector<std::string> image_files;

        if (ReadFilesInDir(image_dir.c_str(), image_files) < 0) {
            std::cerr << "ERROR: Cannot read directory: " << image_dir << std::endl;
            return -1;
        }

        if (image_files.empty()) {
            std::cout << "No images found in " << image_dir << std::endl;
            return -1;
        }

        std::cout << "Found " << image_files.size() << " images" << std::endl;

        // Process images in batches of BATCH_SIZE
        for (size_t i = 0; i < image_files.size(); i += Config::BATCH_SIZE) {
            std::vector<cv::Mat> imgs;
            std::vector<std::string> batch_files;

            // Load batch
            for (int b = 0; b < Config::BATCH_SIZE; b++) {
                size_t idx = i + b;
                if (idx >= image_files.size()) {
                    // Pad with last image if batch is incomplete
                    imgs.push_back(imgs.back().clone());
                    batch_files.push_back(batch_files.back());
                } else {
                    std::string img_path = image_dir + "/" + image_files[idx];
                    cv::Mat img = cv::imread(img_path);
                    if (img.empty()) {
                        std::cerr << "Failed to load: " << img_path << std::endl;
                        continue;
                    }
                    imgs.push_back(img);
                    batch_files.push_back(image_files[idx]);
                }
            }

            if (imgs.size() != Config::BATCH_SIZE) continue;

            std::cout << "\n=== Processing batch " << (i / Config::BATCH_SIZE + 1) << " ===" << std::endl;

            auto start = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<Detection>> results;
            yolo.Infer(imgs, results);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::cout << "Batch inference time: " << duration.count() << " ms" << std::endl;

            // Process each image in the batch
            for (int b = 0; b < Config::BATCH_SIZE && (i + b) < image_files.size(); b++) {
                std::cout << "\nImage " << (i + b + 1) << ": " << batch_files[b] << std::endl;
                std::cout << "Detected " << results[b].size() << " objects" << std::endl;

                for (const auto& det : results[b]) {
                    std::cout << "  " << CLASS_NAMES[det.class_id]
                             << " (" << std::fixed << std::setprecision(2) << (det.conf * 100) << "%)"
                             << " bbox=[" << int(det.bbox[0]) << "," << int(det.bbox[1]) << ","
                             << int(det.bbox[2]) << "," << int(det.bbox[3]) << "]" << std::endl;
                }

                DrawBboxes(imgs[b], results[b]);

                std::string output_name = "output_" + batch_files[b];
                cv::imwrite(output_name, imgs[b]);
                std::cout << "Result saved to: " << output_name << std::endl;
            }
        }

        std::cout << "\n=== All inference completed ===" << std::endl;
    } else {
        std::cerr << "Invalid mode: " << mode << std::endl;
        return -1;
    }

    return 0;
}
