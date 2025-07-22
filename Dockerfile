# Multi-stage build for production deployment
FROM nvidia/cuda:11.8-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy source code
COPY . .

# Build the CUDA components
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGUF_SHARD_CUDA=ON \
        -DCUDA_ARCHITECTURES="70;75;80;86" && \
    make -j$(nproc)

# Build Python tools
RUN pip3 install --no-cache-dir -r requirements.txt

# Production stage
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r gguf && useradd -r -g gguf -d /app -s /bin/bash gguf

# Set working directory
WORKDIR /app

# Copy built binaries and tools
COPY --from=builder /build/build/bin/* /usr/local/bin/
COPY --from=builder /build/forge/ ./forge/
COPY --from=builder /build/trainer/ ./trainer/
COPY --from=builder /build/tests/ ./tests/
COPY --from=builder /build/patches/ ./patches/

# Copy Python requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create data directories
RUN mkdir -p /app/data /app/shards /app/deltas /app/logs && \
    chown -R gguf:gguf /app

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Environment variables
ENV GGUF_SHARD_DATA_DIR=/app/data
ENV GGUF_SHARD_OUTPUT_DIR=/app/shards
ENV GGUF_SHARD_DELTA_DIR=/app/deltas
ENV GGUF_SHARD_LOG_LEVEL=INFO
ENV GGUF_SHARD_RESIDENT_PAGES=512
ENV GGUF_SHARD_PREFETCH_DISTANCE=8

# Expose ports for monitoring
EXPOSE 8080 8090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Switch to non-root user
USER gguf

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["--help"]
