# Makefile for GGUF Shard Suite

# Configuration
CMAKE_BUILD_TYPE ?= Release
CUDA_ENABLED ?= ON
BUILD_DIR = build
INSTALL_PREFIX ?= /usr/local

# Detect CUDA
CUDA_AVAILABLE := $(shell command -v nvcc 2> /dev/null)
ifdef CUDA_AVAILABLE
    CUDA_FLAG = -DGGUF_SHARD_CUDA=ON
else
    CUDA_FLAG = -DGGUF_SHARD_CUDA=OFF
    $(warning CUDA not found - building without GPU support)
endif

# Default target
.PHONY: all
all: build

# Build the project
.PHONY: build
build: configure
	@echo "🔨 Building GGUF Shard Suite..."
	cd $(BUILD_DIR) && make -j$(shell nproc)

# Configure CMake
.PHONY: configure
configure:
	@echo "Configuring build..."
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		$(CUDA_FLAG) \
		-DGGUF_SHARD_TESTS=ON \
		-DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX)

# Clean build directory
.PHONY: clean
clean:
	@echo "🧹 Cleaning build directory..."
	rm -rf $(BUILD_DIR)

# Install
.PHONY: install
install: build
	@echo "📦 Installing GGUF Shard Suite..."
	cd $(BUILD_DIR) && make install

# Run tests
.PHONY: test
test: build
	@echo "🧪 Running tests..."
	cd $(BUILD_DIR) && ctest --output-on-failure --parallel
	python3 -m pytest tests/ -v

# Run quick test
.PHONY: test-quick
test-quick:
	@echo "Running quick tests..."
	python3 tests/test_suite.py --quick

# Build Docker image
.PHONY: docker
docker:
	@echo "🐳 Building Docker image..."
	docker build -t gguf-shard:latest .

# Run Docker container
.PHONY: docker-run
docker-run: docker
	@echo "Running Docker container..."
	docker run --rm -it \
		--gpus all \
		-v $(PWD)/test_data:/app/data \
		gguf-shard:latest

# Format code
.PHONY: format
format:
	@echo "🎨 Formatting code..."
	find . -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" | \
		xargs clang-format -i -style=file
	black forge/ trainer/ tests/

# Lint code
.PHONY: lint
lint:
	@echo "🔍 Linting code..."
	flake8 forge/ trainer/ tests/ --max-line-length=100
	pylint forge/ trainer/ tests/ --disable=C0114,C0115,C0116

# Security scan
.PHONY: security
security:
	@echo "🔒 Running security scan..."
	bandit -r forge/ trainer/ -f json -o security-report.json
	safety check -r requirements.txt

# Documentation
.PHONY: docs
docs: configure
	@echo "📚 Building documentation..."
	cd $(BUILD_DIR) && make docs

# Benchmark
.PHONY: benchmark
benchmark: build
	@echo "Running benchmarks..."
	cd $(BUILD_DIR) && ./bin/gguf_shard_benchmark

# Create test data
.PHONY: test-data
test-data:
	@echo "📋 Creating test data..."
	mkdir -p test_data
	python3 -c "
import os
# Create dummy GGUF files for testing
for size, name in [(16384, 'small'), (1048576, 'medium'), (16777216, 'large')]:
    with open(f'test_data/{name}.gguf', 'wb') as f:
        f.write(b'GGUF')  # Magic
        f.write(b'\x03\x00\x00\x00')  # Version
        f.write(b'\x00' * 16)  # Headers
        f.write(b'TEST_DATA_' * (size // 10))[:size-24]
print('SUCCESS: Test data created')
"

# Development setup
.PHONY: dev-setup
dev-setup:
	@echo "Setting up development environment..."
	pip3 install -r requirements.txt
	pip3 install black flake8 pylint bandit safety
	pre-commit install || echo "pre-commit not available"

# Package release
.PHONY: package
package: build
	@echo "📦 Creating package..."
	cd $(BUILD_DIR) && cpack

# Help
.PHONY: help
help:
	@echo "GGUF Shard Suite - Available targets:"
	@echo ""
	@echo "  build         - Build the project"
	@echo "  clean         - Clean build directory"
	@echo "  install       - Install to system"
	@echo "  test          - Run all tests"
	@echo "  test-quick    - Run quick tests only"
	@echo "  docker        - Build Docker image"
	@echo "  docker-run    - Run Docker container"
	@echo "  format        - Format source code"
	@echo "  lint          - Lint source code"
	@echo "  security      - Run security scan"
	@echo "  docs          - Build documentation"
	@echo "  benchmark     - Run performance benchmarks"
	@echo "  test-data     - Create test data files"
	@echo "  dev-setup     - Set up development environment"
	@echo "  package       - Create release package"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  CMAKE_BUILD_TYPE  - Build type (Release, Debug)"
	@echo "  CUDA_ENABLED      - Enable CUDA support (ON, OFF)"
	@echo "  INSTALL_PREFIX    - Installation prefix"
