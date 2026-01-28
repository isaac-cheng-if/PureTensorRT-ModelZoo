<div class="lang-container">
  <input type="radio" name="lang" id="lang-en" checked>
  <label for="lang-en">English</label>
  <input type="radio" name="lang" id="lang-zh">
  <label for="lang-zh">中文</label>

  <div class="lang-block lang-en">

# TensorRT-From-Scratch

> **Pure C++ TensorRT Implementations | No ONNX | No Plugins | Build Engines Directly from Weights**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.x-76B900?logo=nvidia)](https://developer.nvidia.com/tensorrt)
[![CUDA](https://img.shields.io/badge/CUDA-11+-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Jetson](https://img.shields.io/badge/Jetson-Orin-76B900?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-orin)

---

## Why This Repository?

95% of TensorRT projects rely on ONNX conversion. This repository provides **pure C++ TensorRT API implementations**—building inference engines directly from `.wts` weight files, giving you complete control over network architecture.

| Feature | This Project | ONNX Approach |
|---------|--------------|---------------|
| Network Construction | Layer-by-layer C++ API | Black-box conversion |
| Custom Layer Support | Full control | Limited by export support |
| Debug Friendliness | Inspect each layer | Hard to locate issues |
| Dependencies | TensorRT + CUDA only | Requires ONNX Runtime |
| Learning Value | Deep TensorRT understanding | Surface-level usage |

---

## Supported Models

| Model | Task | Input Size | Precision | Lines of Code |
|-------|------|------------|-----------|---------------|
| **YOLOv8** | Object Detection | 640×640 | FP32/FP16 | 2,800+ |
| **Vision Transformer (ViT)** | Image Classification | 224×224 | FP32 | 1,450+ |
| **ResNet-50** | Image Classification | 224×224 | FP16 | 1,000+ |
| **AlexNet** | Image Classification | 224×224 | FP32/FP16 | 780+ |
| **BERT** | Text Processing | ≤128 tokens | FP32 | 1,700+ |

> Covers **both CV and NLP** — from classic CNNs to Transformer architectures

---

## Quick Start

### Requirements

- TensorRT 8.x + CUDA 11+ (tested on Ubuntu 20.04 + CUDA 11.8 + TensorRT 8.6)
- cuDNN, OpenCV
- CMake or g++/nvcc

### Three Steps to Run

```bash
# 1. Convert weights
python YoloV8/gen_wts.py --weights yolov8n.pt --output yolov8n.wts

# 2. Build
cd YoloV8 && make builder && make runtime

# 3. Create engine and run inference
./yolov8_builder yolov8n.wts yolov8n.engine n
./yolov8_runtime -d yolov8n.engine ./images/
```

---

## Project Structure

```
├── YoloV8/          # Complete YOLOv8 (detect/classify/segment/pose)
├── Vit/             # Vision Transformer (custom LayerNorm, GELU, etc.)
├── ResNet/          # ResNet-50 (Builder/Runtime separation example)
├── Alexnet/         # AlexNet (simplest beginner example)
└── bert_trt/        # BERT (NLP Transformer implementation)
```

---

## Who Is This For?

- Developers who want to **deeply learn TensorRT C++ API**
- Engineers deploying models on **Jetson and embedded devices**
- Researchers who need **fine-grained control** over network structure
- Job seekers preparing for **NVIDIA technical interviews**

---

## License

MIT License. Use it however you like.

If this project helps you, a ⭐️ would be appreciated!

  </div>

  <div class="lang-block lang-zh">

# TensorRT-From-Scratch

> **纯C++ TensorRT推理实现 | 无ONNX依赖 | 无插件 | 从权重直接构建引擎**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.x-76B900?logo=nvidia)](https://developer.nvidia.com/tensorrt)
[![CUDA](https://img.shields.io/badge/CUDA-11+-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Jetson](https://img.shields.io/badge/Jetson-Orin-76B900?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-orin)

---

## 为什么选择这个仓库？

市面上 95% 的 TensorRT 项目都依赖 ONNX 转换，而这个仓库提供的是**纯 C++ TensorRT API 实现**——直接从 `.wts` 权重文件构建推理引擎，让你完全掌控网络结构。

| 特性 | 本项目 | ONNX 方案 |
|------|--------|-----------|
| 网络构建方式 | C++ API 逐层构建 | 黑盒转换 |
| 自定义层支持 | 完全可控 | 受限于导出支持 |
| 调试友好度 | 可逐层检查 | 难以定位问题 |
| 依赖项 | 仅 TensorRT + CUDA | 需要 ONNX Runtime |
| 学习价值 | 深入理解 TensorRT | 仅了解调用流程 |

---

## 支持的模型

| 模型 | 任务 | 输入尺寸 | 精度 | 代码行数 |
|------|------|----------|------|----------|
| **YOLOv8** | 物体检测 | 640×640 | FP32/FP16 | 2,800+ |
| **Vision Transformer (ViT)** | 图像分类 | 224×224 | FP32 | 1,450+ |
| **ResNet-50** | 图像分类 | 224×224 | FP16 | 1,000+ |
| **AlexNet** | 图像分类 | 224×224 | FP32/FP16 | 780+ |
| **BERT** | 文本处理 | ≤128 tokens | FP32 | 1,700+ |

> 涵盖 **CV + NLP** 双领域，从经典 CNN 到 Transformer 架构全覆盖

---

## 快速开始

### 环境要求

- TensorRT 8.x + CUDA 11+（测试环境：Ubuntu 20.04 + CUDA 11.8 + TensorRT 8.6）
- cuDNN、OpenCV
- CMake 或 g++/nvcc

### 三步运行

```bash
# 1. 转换权重
python YoloV8/gen_wts.py --weights yolov8n.pt --output yolov8n.wts

# 2. 编译
cd YoloV8 && make builder && make runtime

# 3. 构建引擎并推理
./yolov8_builder yolov8n.wts yolov8n.engine n
./yolov8_runtime -d yolov8n.engine ./images/
```

---

## 项目结构

```
├── YoloV8/          # YOLOv8 完整实现（检测/分类/分割/姿态）
├── Vit/             # Vision Transformer（含 LayerNorm、GELU 等自定义实现）
├── ResNet/          # ResNet-50（Builder/Runtime 分离设计示例）
├── Alexnet/         # AlexNet（最简单的入门示例）
└── bert_trt/        # BERT（NLP Transformer 实现）
```

---

## 适合谁？

- 想**深入学习 TensorRT C++ API** 的开发者
- 需要在 **Jetson 等嵌入式设备**上部署模型的工程师
- 对 ONNX 转换结果不满意，想**精细控制网络结构**的研究者
- 正在准备 **NVIDIA 相关技术面试**的求职者

---

## 许可证

MIT License，随意使用。

如果这个项目对你有帮助，欢迎点个 ⭐️ 支持一下！

  </div>
</div>

<style>
.lang-container {
  text-align: right;
}
.lang-container label {
  margin-left: 0.5rem;
  cursor: pointer;
  font-weight: bold;
}
.lang-container input[type="radio"] {
  display: none;
}
.lang-block {
  display: none;
  text-align: left;
  margin-top: 1rem;
}
#lang-zh:checked ~ label[for="lang-zh"] {
  color: #1f6feb;
}
#lang-en:checked ~ label[for="lang-en"] {
  color: #1f6feb;
}
#lang-zh:checked ~ .lang-zh,
#lang-en:checked ~ .lang-en {
  display: block;
}
</style>
