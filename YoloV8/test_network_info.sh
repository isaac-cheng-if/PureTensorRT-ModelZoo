#!/bin/bash

# YOLOv8 TensorRT 网络信息测试脚本

echo "=========================================="
echo "YOLOv8 TensorRT 网络信息功能测试"
echo "=========================================="

# 检查编译后的可执行文件
EXECUTABLE="./yolov8_det_batch16"
if [ ! -f "$EXECUTABLE" ]; then
    echo "错误: 找不到可执行文件 $EXECUTABLE"
    echo "请先编译程序:"
    echo "  g++ -o yolov8_det_batch16 yolov8_det_batch16\ copy.cpp \\"
    echo "      -I/usr/local/cuda/include \\"
    echo "      -I/usr/local/TensorRT/include \\"
    echo "      -I/usr/include/opencv4 \\"
    echo "      -L/usr/local/cuda/lib64 \\"
    echo "      -L/usr/local/TensorRT/lib \\"
    echo "      -lnvinfer -lnvonnxparser -lcudart -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui"
    exit 1
fi

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <engine_file> [verbose]"
    echo "示例: $0 yolov8n.engine"
    echo "示例: $0 yolov8n.engine verbose"
    exit 1
fi

ENGINE_FILE="$1"
VERBOSE="$2"

if [ ! -f "$ENGINE_FILE" ]; then
    echo "错误: 找不到引擎文件 $ENGINE_FILE"
    echo "请先构建引擎:"
    echo "  ./yolov8_det_batch16 -s yolov8n.wts yolov8n.engine n"
    exit 1
fi

echo "引擎文件: $ENGINE_FILE"
echo "详细模式: ${VERBOSE:-否}"
echo ""

# 运行网络信息打印
if [ "$VERBOSE" = "verbose" ]; then
    echo "运行详细网络信息分析..."
    $EXECUTABLE -i "$ENGINE_FILE" verbose
else
    echo "运行基本网络信息分析..."
    $EXECUTABLE -i "$ENGINE_FILE"
fi

echo ""
echo "=========================================="
echo "网络信息分析完成"
echo "=========================================="
