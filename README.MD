# PureTensorRT-ModelZoo
*Minimal pure C++ TensorRT implementations for multiple deep learning models — no ONNX, no plugins.*

---

## 🧠 简介 | Overview

**PureTensorRT-ModelZoo** 是一个完全基于 **TensorRT C++ API** 的模型集合项目，  
展示了如何在 **不依赖 ONNX 或任何外部插件** 的情况下，直接从 `.wts` 权重文件构建高性能推理引擎。

每个模型均以 **单个 C++ 文件** 实现构图逻辑，结构清晰，  
既可作为 **学习 TensorRT 底层机制的参考实现**，  
也可作为 **迁移至任意 NPU / DSP / TPU 后端的模板**。

> 🚀 *From PyTorch weights to TensorRT engine — portable, minimal, and educational.*

---

## ✨ 特性 | Features

- ⚙️ **纯 TensorRT 实现**：无 ONNX / 无插件依赖。  
- 📦 **多模型支持**：YOLOv8、ViT、ResNet、AlexNet 等。  
- 🧩 **结构极简**：每个模型仅一个 `.cpp` 文件，代码可读性强。  
- 🔧 **易于迁移**：代码层次清晰，可快速适配任意 NPU 后端。  
- 📊 **性能测试模块**：支持同步与 Pipeline 模式性能统计。  
- 🧠 **教学友好**：非常适合开发者、研究者学习 TensorRT 构图原理。

---

## 🗂️ 目录结构 | Repository Structure

