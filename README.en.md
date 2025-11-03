# PureTensorRT-ModelZoo

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
