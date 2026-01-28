# TensorRT-From-Scratch

> **Pure C++ TensorRT Implementations | No ONNX | No Plugins | Build Engines Directly from Weights**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.x-76B900?logo=nvidia)](https://developer.nvidia.com/tensorrt)
[![CUDA](https://img.shields.io/badge/CUDA-11+-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Jetson](https://img.shields.io/badge/Jetson-Orin-76B900?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-orin)

English | [中文](README.zh.md)

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

## Detailed Usage

### YOLOv8

```bash
cd YoloV8
python gen_wts.py --weights yolov8n.pt --output yolov8n.wts
make builder && make runtime
./yolov8_builder yolov8n.wts yolov8n.engine n

# Run detection on a folder
./yolov8_runtime -d yolov8n.engine ./images/

# Benchmark
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
cd ResNet   # or Alexnet
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

## Who Is This For?

- Developers who want to **deeply learn TensorRT C++ API**
- Engineers deploying models on **Jetson and embedded devices**
- Researchers who need **fine-grained control** over network structure
- Job seekers preparing for **NVIDIA technical interviews**

---

## License

MIT License. Use it however you like.

If this project helps you, a ⭐️ would be appreciated!
