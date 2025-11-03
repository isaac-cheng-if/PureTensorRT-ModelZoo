# YOLOv8 代码分离说明

## 文件结构

现在代码已经分离成以下文件：

1. **yolov8_common.h** - 共享头文件，包含公共结构体和函数声明
2. **yolov8_builder.cpp** - 构图器，负责网络构建和引擎序列化
3. **yolov8_runtime.cpp** - 运行时，负责推理、后处理和性能测试

## 编译方法

```bash
# 编译构图器
g++ -o yolov8_builder yolov8_builder.cpp -I/usr/local/cuda/include -I/usr/local/include/opencv4 -L/usr/local/cuda/lib64 -L/usr/local/lib -lnvinfer -lnvparsers -lnvinfer_plugin -lcudart -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui

# 编译运行时
g++ -o yolov8_runtime yolov8_runtime.cpp -I/usr/local/cuda/include -I/usr/local/include/opencv4 -L/usr/local/cuda/lib64 -L/usr/local/lib -lnvinfer -lnvparsers -lnvinfer_plugin -lcudart -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
```

## 使用方法

### 1. 构建引擎 (yolov8_builder)

```bash
./yolov8_builder <wts_file> <engine_file> <model_type>
```

**示例：**
```bash
./yolov8_builder yolov8n.wts yolov8n.engine n
```

**参数说明：**
- `wts_file`: 权重文件路径
- `engine_file`: 输出的引擎文件路径  
- `model_type`: 模型类型 (n/s/m/l/x)

### 2. 运行推理 (yolov8_runtime)

#### 推理模式
```bash
./yolov8_runtime -d <engine_file> <image_dir>
```

**示例：**
```bash
./yolov8_runtime -d yolov8n.engine ./images/
```

#### 性能测试模式
```bash
./yolov8_runtime -p <engine_file> <image_path> [iterations]
```

**示例：**
```bash
./yolov8_runtime -p yolov8n.engine ./image.jpg 100
```

## 主要变化

1. **分离关注点**: 构图和运行时逻辑完全分离
2. **独立编译**: 每个程序可以独立编译和运行
3. **简化参数**: 构图器不需要 `-s` 参数，直接按位置传递参数
4. **保持功能**: 所有原有功能都得到保留

## 注意事项

- 确保有正确的 TensorRT 和 OpenCV 环境
- 构图器只需要运行一次来生成引擎文件
- 运行时可以重复使用生成的引擎文件进行推理
- 批处理大小固定为 16，在 Config 命名空间中定义
