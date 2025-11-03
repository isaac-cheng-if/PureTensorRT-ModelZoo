# 简化的 AlexNet TensorRT Makefile - 专为 Orin 优化

# 编译器设置
CXX = g++
TARGET = AlexNetInfer

# 编译标志
CXXFLAGS = -std=c++14 -O3 -DNDEBUG -Wall -Wextra -march=armv8-a -Wno-deprecated-declarations

# Orin 平台路径
INCLUDES = -I/usr/include/aarch64-linux-gnu \
           -I/usr/local/cuda-11.4/include \
           -I/usr/local/cuda/include

# Orin 库路径
LDFLAGS = -L/usr/lib/aarch64-linux-gnu \
          -L/usr/local/cuda-11.4/targets/sbsa-linux/lib \
          -L/usr/local/cuda-11.4/lib64 \
          -L/usr/local/cuda/lib64

# 链接库
LIBS = -lnvinfer -lnvonnxparser -lcudart -lcublas -lcurand -lm -lpthread -lstdc++

# 为 Orin 平台添加额外的链接标志 - 让编译器自动处理库依赖
EXTRA_LDFLAGS =

# 源文件
SOURCES = main.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# 默认目标
all: check_deps $(TARGET)

# 检查依赖
check_deps:
	@echo "检查 Orin 平台依赖..."
	@command -v nvcc >/dev/null 2>&1 || { echo "错误: 未找到 nvcc"; echo "请运行: export PATH=/usr/local/cuda-11.4/bin:\$$PATH"; exit 1; }
	@test -f /usr/include/aarch64-linux-gnu/NvInfer.h || { echo "错误: 未找到 TensorRT 头文件"; exit 1; }
	@test -f /usr/lib/aarch64-linux-gnu/libnvinfer.so || { echo "错误: 未找到 TensorRT 库"; exit 1; }
	@echo "检查 GCC 可用性..."
	@gcc --version >/dev/null 2>&1 || { echo "错误: GCC 不可用"; exit 1; }
	@test -f alexnet-owt-7be5be79.wts || { echo "错误: 未找到权重文件"; exit 1; }
	@test -f input_fp16_nchw.bin || { echo "错误: 未找到输入文件 input_fp16_nchw.bin"; exit 1; }
	@echo "✅ 所有依赖检查通过"

# 构建主程序
$(TARGET): $(OBJECTS)
	@echo "链接 $(TARGET)..."
	$(CXX) $(OBJECTS) $(LDFLAGS) -Wl,--no-as-needed $(LIBS) -o $(TARGET)
	@echo "✅ 构建完成: $(TARGET)"

# 编译 C++ 文件
%.o: %.cpp
	@echo "编译 $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 运行程序 (FP16)
run: $(TARGET)
	@echo "运行 $(TARGET) (FP16)..."
	@echo "设置环境变量..."
	@export PATH=/usr/local/cuda-11.4/bin:$$PATH && \
	 export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/usr/lib/aarch64-linux-gnu:$$LD_LIBRARY_PATH && \
	 ./$(TARGET)

# 生成FP16权重
generate_fp16:
	@echo "生成FP16权重文件..."
	python3 generate_fp16_weights.py

# 清理
clean:
	@echo "清理构建文件..."
	rm -f $(OBJECTS) $(TARGET) *.engine
	@echo "清理完成"

# 显示帮助
help:
	@echo "简化版 AlexNet TensorRT Makefile"
	@echo ""
	@echo "可用目标:"
	@echo "  all          - 构建项目 (默认)"
	@echo "  run          - 运行FP16推理程序"
	@echo "  generate_fp16- 生成FP16权重文件"
	@echo "  clean        - 清理构建文件"
	@echo "  help         - 显示此帮助"
	@echo ""
	@echo "使用说明 (FP16优化版本):"
	@echo "1. 生成FP16权重: make generate_fp16"
	@echo "2. 确保输入文件存在: input_fp16_nchw.bin"
	@echo "3. 运行: make && make run"

# 调试信息
debug-info:
	@echo "=== Orin 构建配置 ==="
	@echo "编译器: $(CXX)"
	@echo "目标: $(TARGET)"
	@echo "编译标志: $(CXXFLAGS)"
	@echo "包含路径: $(INCLUDES)"
	@echo "库路径: $(LDFLAGS)"
	@echo "链接库: $(LIBS)"
	@echo "源文件: $(SOURCES)"
	@echo "========================"

.PHONY: all check_deps run clean help debug-info