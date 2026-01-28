# TensorRT-From-Scratch

> **纯C++ TensorRT推理实现 | 无ONNX依赖 | 无插件 | 从权重直接构建引擎**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.x-76B900?logo=nvidia)](https://developer.nvidia.com/tensorrt)
[![CUDA](https://img.shields.io/badge/CUDA-11+-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Jetson](https://img.shields.io/badge/Jetson-Orin-76B900?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-orin)

[English](README.en.md) | 中文

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

## 详细使用说明

### YOLOv8

```bash
cd YoloV8
python gen_wts.py --weights yolov8n.pt --output yolov8n.wts
make builder && make runtime
./yolov8_builder yolov8n.wts yolov8n.engine n

# 检测整个文件夹
./yolov8_runtime -d yolov8n.engine ./images/

# 性能测试
./yolov8_runtime -p yolov8n.engine ./image.jpg 100
```

### Vision Transformer (ViT)

```bash
cd Vit
python gen_wts.py --weights vit.pth --output vit.wts
make builder && make runtime
./vit_builder vit.wts vit.engine
./vit_runtime vit.engine ./image.jpg
```

### ResNet / AlexNet

```bash
cd ResNet   # 或 Alexnet
python gen_wts.py --weights resnet50.pth --output resnet50.wts
make builder && make runtime
./resnet_builder resnet50.wts resnet50.engine
./resnet_runtime resnet50.engine ./image.jpg
```

### BERT

```bash
cd bert_trt
python gen_bert_wts.py --model bert-base-uncased --output bert.wts
make builder && make runtime
./bert_builder bert.wts bert.engine
./bert_runtime bert.engine
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
