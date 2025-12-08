# Simple wrapper Makefile so you can just run `make` in the project root.
# It configures CMake to use the nvcc wrapper script, same as scripts/build.sh.

CUDA_ARCH ?= 50
BUILD_DIR ?= build
CMAKE     ?= cmake

# Absolute path to the nvcc wrapper
CUDA_WRAPPER := $(CURDIR)/scripts/nvcc-wrap.sh

all: $(BUILD_DIR)/CMakeCache.txt
	$(CMAKE) --build $(BUILD_DIR) -- -j

# Configure CMake once (or when CMakeLists.txt changes)
$(BUILD_DIR)/CMakeCache.txt: CMakeLists.txt
	@mkdir -p $(BUILD_DIR)
	$(CMAKE) -S . -B $(BUILD_DIR) \
	    -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) \
	    -DCMAKE_CUDA_COMPILER=$(CUDA_WRAPPER) \
	    -DCMAKE_BUILD_TYPE=Release

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean

