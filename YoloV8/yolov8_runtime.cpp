/*
 * YOLOv8 Runtime Inference
 * Handles engine loading, inference, postprocessing, and performance testing
 */

/*
 * YOLOv8 Common Headers and Structures
 * Shared definitions for both builder and runtime
 */

 #ifndef YOLOV8_COMMON_H
 #define YOLOV8_COMMON_H
 
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
 
 // ==================== Utility Functions ====================
 int GetWidth(int x, float gw, int max_channels, int divisor = 8);
 int GetDepth(int x, float gd);
 int ReadFilesInDir(const char* p_dir_name, std::vector<std::string>& file_names);
 
 // ==================== Weight Loading ====================
 std::map<std::string, nvinfer1::Weights> LoadWeights(const std::string& file);
 
 // ==================== Image Preprocessing ====================
 cv::Mat PreProcessImage(const cv::Mat& img, bool debug = false);
 
 // ==================== Postprocessing ====================
 void PostProcess(float* output, std::vector<Detection>& dets, int orig_w, int orig_h);
 void NMS(std::vector<Detection>& dets, float thresh);
 
 // ==================== Visualization ====================
 void DrawBboxes(cv::Mat& img, const std::vector<Detection>& dets);
 
 #endif // YOLOV8_COMMON_H
 

// ==================== Utility Functions ====================
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

// ==================== Image Preprocessing ====================
cv::Mat PreProcessImage(const cv::Mat& img, bool debug) {
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

// ==================== YOLOv8 Runtime Class ====================
class YOLOv8Runtime {
private:
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    nvinfer1::IRuntime* runtime;

public:
    YOLOv8Runtime() : engine(nullptr), context(nullptr), runtime(nullptr) {}

    ~YOLOv8Runtime() {
        if (context) context->destroy();
        if (engine) engine->destroy();
        if (runtime) runtime->destroy();
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
};

// ==================== Main Function for Runtime ====================
int main(int argc, char** argv) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "YOLOv8 Runtime Inference - Batch 16" << std::endl;
    std::cout << "Handles inference, postprocessing, and performance testing" << std::endl;
    std::cout << "Batch size: " << Config::BATCH_SIZE << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;

    if (argc < 3) {
        std::cout << "Usage:" << std::endl;
        std::cout << "  Infer:      ./yolov8_runtime -d <engine> <image_dir>" << std::endl;
        std::cout << "  Perf Test:  ./yolov8_runtime -p <engine> <image_path> [iterations]" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  ./yolov8_runtime -d yolov8n.engine ./images/" << std::endl;
        std::cout << "  ./yolov8_runtime -p yolov8n.engine ./image.jpg 100" << std::endl;
        return -1;
    }

    cudaSetDevice(Config::GPU_ID);

    YOLOv8Runtime runtime;
    std::string mode = argv[1];

    if (mode == "-p") {
        // Performance testing mode
        if (argc < 4) {
            std::cerr << "ERROR: -p requires <engine> <image_path> [iterations]" << std::endl;
            return -1;
        }

        std::cout << "=== Performance Testing Mode ===" << std::endl;
        runtime.Deserialize(argv[2]);

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

        runtime.PerformanceTest(imgs, iterations);

    } else if (mode == "-d") {
        if (argc < 4) {
            std::cerr << "ERROR: -d requires <engine> <image_dir>" << std::endl;
            return -1;
        }

        std::cout << "=== Loading Engine and Running Inference ===" << std::endl;
        runtime.Deserialize(argv[2]);

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
            runtime.Infer(imgs, results);
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
