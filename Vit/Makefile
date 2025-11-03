# ViT TensorRT - 简化版 Makefile (仅编译)

CXX = g++
TARGET_BUILDER = vit_builder
TARGET_RUNTIME = vit_runtime

# 编译器标志
CXXFLAGS = -std=c++14 -O3 -DNDEBUG -Wall -Wextra -Wno-deprecated-declarations

# 头文件路径
INCLUDES = -I/usr/include/aarch64-linux-gnu \
           -I/usr/local/cuda/include \
           -I/usr/include/opencv4

# 库文件路径
LDFLAGS = -L/usr/lib/aarch64-linux-gnu \
          -L/usr/local/cuda/lib64

# 链接库
LIBS = -lnvinfer \
       -lcudart \
       -lopencv_core \
       -lopencv_imgproc \
       -lopencv_imgcodecs \
       -lopencv_highgui \
       -lopencv_dnn \
       -lstdc++ \
       -lpthread

# 源文件
BUILDER_SOURCE = vit_builder.cpp
RUNTIME_SOURCE = vit_runtime.cpp

# 默认目标：编译两个可执行文件
all: $(TARGET_BUILDER) $(TARGET_RUNTIME)
	@echo "✅ 编译完成！"

# 编译 builder
$(TARGET_BUILDER): $(BUILDER_SOURCE)
	@echo "=== 正在编译 $(TARGET_BUILDER) ==="
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(BUILDER_SOURCE) $(LDFLAGS) $(LIBS) -o $(TARGET_BUILDER)
	@echo "✅ $(TARGET_BUILDER) 编译完成"

# 编译 runtime
$(TARGET_RUNTIME): $(RUNTIME_SOURCE)
	@echo "=== 正在编译 $(TARGET_RUNTIME) ==="
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(RUNTIME_SOURCE) $(LDFLAGS) $(LIBS) -o $(TARGET_RUNTIME)
	@echo "✅ $(TARGET_RUNTIME) 编译完成"

# 清理编译产物
clean:
	@echo "=== 清理编译文件 ==="
	rm -f $(TARGET_BUILDER) $(TARGET_RUNTIME)
	@echo "✅ 清理完成"

# 帮助信息
help:
	@echo "======================================================================"
	@echo "ViT TensorRT - 简化版 Makefile"
	@echo "======================================================================"
	@echo ""
	@echo "📋 可用目标："
	@echo "  make         - 编译 vit_builder 和 vit_runtime (默认)"
	@echo "  make clean   - 清理编译产物"
	@echo "  make help    - 显示此帮助信息"
	@echo ""
	@echo "🔧 编译输出："
	@echo "  vit_builder  - 用于构建 TensorRT 引擎"
	@echo "  vit_runtime  - 用于运行推理"
	@echo ""
	@echo "📝 使用示例："
	@echo "  1. 编译：make"
	@echo "  2. 构建引擎：./vit_builder <wts_file> <engine_file> <type>"
	@echo "  3. 运行推理：./vit_runtime <engine_file> <image_file>"
	@echo ""
	@echo "======================================================================"

.PHONY: all clean help
