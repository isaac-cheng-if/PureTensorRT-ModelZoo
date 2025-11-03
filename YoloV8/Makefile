# YOLOv8 Detection - Unified Single File Makefile (No Plugin Required)

CXX = g++
TARGET = yolov8_det_unified

# Compiler flags for Orin
CXXFLAGS = -std=c++14 -O3 -DNDEBUG -Wall -Wextra -march=armv8-a -Wno-deprecated-declarations

# Include paths for Orin
INCLUDES = -I/usr/include/aarch64-linux-gnu \
           -I/usr/local/cuda/include \
           -I/usr/include/opencv4

# Library paths for Orin
LDFLAGS = -L/usr/lib/aarch64-linux-gnu \
          -L/usr/local/cuda/lib64

# Libraries
LIBS = -lnvinfer \
       -lcudart \
       -lopencv_core \
       -lopencv_imgproc \
       -lopencv_imgcodecs \
       -lopencv_highgui \
       -lstdc++ \
       -lpthread

# Source file
SOURCE = yolov8_det_batch16.cpp

# Default target
all: check_deps $(TARGET)

# Check dependencies
check_deps:
	@echo "=== Checking Orin Platform Dependencies ==="
	@command -v nvcc >/dev/null 2>&1 || { echo "❌ Error: nvcc not found"; exit 1; }
	@test -f /usr/include/aarch64-linux-gnu/NvInfer.h || { echo "❌ Error: TensorRT headers not found"; exit 1; }
	@test -f /usr/lib/aarch64-linux-gnu/libnvinfer.so || { echo "❌ Error: TensorRT libraries not found"; exit 1; }
	@echo "Checking GCC..."
	@gcc --version >/dev/null 2>&1 || { echo "❌ Error: GCC not available"; exit 1; }
	@echo "Checking OpenCV..."
	@pkg-config --exists opencv4 2>/dev/null || { echo "⚠️  Warning: OpenCV pkg-config not found"; }
	@echo "✅ All dependencies OK"

# Build the executable
$(TARGET): $(SOURCE)
	@echo "=== Compiling $(TARGET) ==="
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCE) $(LDFLAGS) $(LIBS) -o $(TARGET)
	@echo ""
	@echo "✅ Build complete: $(TARGET)"
	@echo ""
	@echo "Usage:"
	@echo "  1. Generate .wts file from PyTorch model:"
	@echo "     python3 gen_wts.py -w yolov8n.pt -o yolov8n.wts -t detect"
	@echo ""
	@echo "  2. Build TensorRT engine:"
	@echo "     ./$(TARGET) -s yolov8n.wts yolov8n.engine n"
	@echo ""
	@echo "  3. Run inference:"
	@echo "     ./$(TARGET) -d yolov8n.engine ./images/"
	@echo ""

# Generate .wts file
gen_wts:
	@echo "=== Generating .wts file ==="
	@if [ -z "$(MODEL)" ]; then \
		echo "❌ Error: Please specify MODEL"; \
		echo "Usage: make gen_wts MODEL=yolov8n.pt"; \
		exit 1; \
	fi
	@if [ ! -f $(MODEL) ]; then \
		echo "❌ Error: Model file not found: $(MODEL)"; \
		exit 1; \
	fi
	@if [ ! -f gen_wts.py ]; then \
		echo "❌ Error: gen_wts.py not found"; \
		exit 1; \
	fi
	@echo "Generating .wts from $(MODEL)..."
	python3 gen_wts.py -w $(MODEL) -o $(MODEL:.pt=.wts) -t detect
	@echo "✅ .wts file generated: $(MODEL:.pt=.wts)"

# Build engine from .wts
build_engine: $(TARGET)
	@echo "=== Building TensorRT Engine ==="
	@if [ -z "$(WTS)" ]; then \
		echo "❌ Error: Please specify WTS file"; \
		echo "Usage: make build_engine WTS=yolov8n.wts MODEL=n"; \
		exit 1; \
	fi
	@if [ -z "$(MODEL)" ]; then \
		echo "❌ Error: Please specify MODEL type (n/s/m/l/x)"; \
		echo "Usage: make build_engine WTS=yolov8n.wts MODEL=n"; \
		exit 1; \
	fi
	@if [ ! -f $(WTS) ]; then \
		echo "❌ Error: .wts file not found: $(WTS)"; \
		exit 1; \
	fi
	@export PATH=/usr/local/cuda/bin:$$PATH && \
	 export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:$$LD_LIBRARY_PATH && \
	 ./$(TARGET) -s $(WTS) $(WTS:.wts=.engine) $(MODEL)
	@echo "✅ Engine built: $(WTS:.wts=.engine)"

# Run inference
run: $(TARGET)
	@echo "=== Running YOLOv8 Inference ==="
	@if [ -z "$(ENGINE)" ]; then \
		echo "❌ Error: Please specify ENGINE file"; \
		echo "Usage: make run ENGINE=yolov8n.engine IMAGES=./images/"; \
		exit 1; \
	fi
	@if [ -z "$(IMAGES)" ]; then \
		echo "❌ Error: Please specify IMAGES directory"; \
		echo "Usage: make run ENGINE=yolov8n.engine IMAGES=./images/"; \
		exit 1; \
	fi
	@if [ ! -f $(ENGINE) ]; then \
		echo "❌ Error: Engine file not found: $(ENGINE)"; \
		exit 1; \
	fi
	@if [ ! -d $(IMAGES) ]; then \
		echo "❌ Error: Image directory not found: $(IMAGES)"; \
		exit 1; \
	fi
	@export PATH=/usr/local/cuda/bin:$$PATH && \
	 export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:$$LD_LIBRARY_PATH && \
	 ./$(TARGET) -d $(ENGINE) $(IMAGES)

# Complete workflow
workflow:
	@echo "=== YOLOv8 Complete Workflow ==="
	@echo ""
	@echo "Step 1: Check dependencies"
	@make check_deps
	@echo ""
	@echo "Step 2: Compile program"
	@make all
	@echo ""
	@echo "Step 3: Generate .wts file (if needed)"
	@echo "  make gen_wts MODEL=yolov8n.pt"
	@echo ""
	@echo "Step 4: Build engine"
	@echo "  make build_engine WTS=yolov8n.wts MODEL=n"
	@echo ""
	@echo "Step 5: Run inference"
	@echo "  make run ENGINE=yolov8n.engine IMAGES=./images/"
	@echo ""

# Clean
clean:
	@echo "=== Cleaning ==="
	rm -f $(TARGET)
	@echo "Clean complete"

# Deep clean
clean-all: clean
	@echo "=== Deep cleaning ==="
	rm -f *.engine output_*.jpg output_*.png
	@echo "Deep clean complete"

# Help
help:
	@echo "======================================================================"
	@echo "YOLOv8 Detection - Unified Single File Makefile"
	@echo "======================================================================"
	@echo ""
	@echo "📋 Available targets:"
	@echo "  all          - Compile the program (default)"
	@echo "  check_deps   - Check system dependencies"
	@echo "  gen_wts      - Generate .wts file from PyTorch model"
	@echo "  build_engine - Build TensorRT engine from .wts"
	@echo "  run          - Run inference"
	@echo "  workflow     - Show complete workflow"
	@echo "  clean        - Clean build files"
	@echo "  clean-all    - Deep clean (including engines)"
	@echo "  help         - Show this help"
	@echo ""
	@echo "📖 Complete workflow:"
	@echo ""
	@echo "1. Download YOLOv8 model:"
	@echo "   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
	@echo ""
	@echo "2. Generate .wts file:"
	@echo "   make gen_wts MODEL=yolov8n.pt"
	@echo ""
	@echo "3. Compile program:"
	@echo "   make"
	@echo ""
	@echo "4. Build TensorRT engine:"
	@echo "   make build_engine WTS=yolov8n.wts MODEL=n"
	@echo ""
	@echo "5. Prepare test images:"
	@echo "   mkdir images"
	@echo "   # Copy .jpg/.png files to images/"
	@echo ""
	@echo "6. Run inference:"
	@echo "   make run ENGINE=yolov8n.engine IMAGES=./images/"
	@echo ""
	@echo "⚙️  Model types:"
	@echo "  n - YOLOv8n (fastest, lowest accuracy)"
	@echo "  s - YOLOv8s"
	@echo "  m - YOLOv8m (recommended, balanced)"
	@echo "  l - YOLOv8l"
	@echo "  x - YOLOv8x (slowest, highest accuracy)"
	@echo ""
	@echo "✨ Features:"
	@echo "  • Single file implementation"
	@echo "  • No custom plugins required"
	@echo "  • Direct .wts to TensorRT engine"
	@echo "  • FP16 precision support"
	@echo "  • Batch size = 1"
	@echo ""
	@echo "======================================================================"

.PHONY: all check_deps gen_wts build_engine run workflow clean clean-all help
