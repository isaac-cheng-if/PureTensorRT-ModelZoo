<div class="lang-container">
  <input type="radio" name="lang" id="lang-zh" checked>
  <label for="lang-zh">中文</label>
  <input type="radio" name="lang" id="lang-en">
  <label for="lang-en">English</label>

  <div class="lang-block lang-zh">

# PureTensorRT-ModelZoo

> 如果上方语言切换按钮无法使用（例如某些平台禁用了样式），请直接查看 `README.zh.md` 或 `README.en.md`。

我把自己写的几个纯 C++ TensorRT Demo 放在这个仓库里，全部都是直接用 `.wts` 权重起引擎，不走 ONNX、也没有插件。下面就是怎么跑。

---

## 环境准备

- TensorRT 8.x + CUDA 11 以上（我在 Ubuntu 20.04 + CUDA 11.8 + TensorRT 8.6 上测试）
- cuDNN、OpenCV、CMake 或直接用 g++/nvcc
- 把 CUDA 和 TensorRT 的 `include` / `lib` 路径加到编译命令里

---

## 权重转换

每个模型目录都带了一个把 PyTorch 权重转成 `.wts` 的脚本，命令大同小异：

```bash
python <模型目录>/gen_wts.py --weights your_model.pth --output your_model.wts
```

YOLOv8 还可以用 `convert_for_tensorrt.py` 直接把官方的 `.pt` 转好。

---

## 编译与使用

所有目录都放了 `Makefile` 或者编译命令，基本套路如下：

1. 进入模型目录
2. 编译 builder：`make builder`（或者 `make`，看目录里的提示）
3. 编译 runtime：`make runtime`
4. 用 builder 把 `.wts` 变成 `.engine`
5. 用 runtime 跑推理或者性能测试

下面给几个常用命令：

### YOLOv8
```bash
cd YoloV8
python gen_wts.py --weights yolov8n.pt --output yolov8n.wts
make builder && make runtime
./yolov8_builder yolov8n.wts yolov8n.engine n
# 检测整张图
./yolov8_runtime -d yolov8n.engine ./images/
# 跑性能
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

输出一般会打印 Top-1/Top-5 或者把检测结果存到 `results/`，具体看终端提示。

---

## 目录说明

- `Alexnet/`：单文件版 AlexNet，适合看最基础的 API 使用。
- `ResNet/`：ResNet-50 的 builder/runtime 拆分示例。
- `Vit/`：Vision Transformer，里面有 LayerNorm、GELU 等写法。
- `YoloV8/`：YOLOv8 全流程，含批量推理和性能统计脚本。

我之后会慢慢把别的模型也搬进来，更新都会写在各自目录的 README 里。

---

## 许可证

MIT License，随意使用，记得自测。

如果这些代码对你有帮助，欢迎点个 ⭐️。

  </div>

  <div class="lang-block lang-en">

# PureTensorRT-ModelZoo

> If the language toggle above does not work (for example, when styles are stripped), open `README.en.md` or `README.zh.md` instead.

A set of pure C++ TensorRT demos I wrote for myself. Everything builds engines straight from `.wts` files—no ONNX export, no plugins. Here’s how to run them.

---

## Setup

- TensorRT 8.x with CUDA 11 or newer (I test on Ubuntu 20.04 + CUDA 11.8 + TensorRT 8.6)
- cuDNN, OpenCV, and either CMake or plain g++/nvcc
- Make sure the CUDA and TensorRT `include` / `lib` paths are added to your build commands

---

## Convert weights

Each model folder has a helper that turns PyTorch checkpoints into `.wts` files:

```bash
python <model-folder>/gen_wts.py --weights your_model.pth --output your_model.wts
```

For YOLOv8 you can also run `convert_for_tensorrt.py` on the official `.pt` file.

---

## Build and run

Every directory ships with a `Makefile` or explicit commands. The routine is always:

1. `cd` into the model folder
2. Build the builder binary: `make builder` (or just `make`, follow the local notes)
3. Build the runtime binary: `make runtime`
4. Run the builder to turn `.wts` into `.engine`
5. Use the runtime binary for inference or benchmarking

Quick examples:

### YOLOv8
```bash
cd YoloV8
python gen_wts.py --weights yolov8n.pt --output yolov8n.wts
make builder && make runtime
./yolov8_builder yolov8n.wts yolov8n.engine n
# run detection on a folder
./yolov8_runtime -d yolov8n.engine ./images/
# benchmark loop
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

The runtime prints Top-1/Top-5 numbers or saves detection images in `results/` depending on the model.

---

## Folders

- `Alexnet/`: single-file TensorRT example for the basics
- `ResNet/`: builder/runtime split for ResNet-50
- `Vit/`: Vision Transformer layers (LayerNorm, GELU, etc.) written out in TensorRT
- `YoloV8/`: full YOLOv8 flow with batch inference and perf tools

I’ll keep adding more models over time. Check each folder’s README for details.

---

## License

MIT License. Use it however you like—just test on your own setup.

If it helps you, a ⭐️ would be appreciated.

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
