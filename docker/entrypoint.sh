#!/bin/bash
set -e

# GGUF Shard Docker Entrypoint

# Function to print usage
show_help() {
    cat << EOF
GGUF Shard Suite - Docker Container

Usage: docker run gguf-shard [COMMAND] [OPTIONS]

Commands:
  shard     Create shards from GGUF file
  delta     Create delta updates
  test      Run test suite
  verify    Verify shard integrity
  help      Show this help

Examples:
  docker run -v /data:/app/data gguf-shard shard /app/data/model.gguf
  docker run -v /data:/app/data gguf-shard delta --base base.gguf --target new.gguf
  docker run gguf-shard test

Environment Variables:
  GGUF_SHARD_DATA_DIR       - Data directory (default: /app/data)
  GGUF_SHARD_OUTPUT_DIR     - Output directory (default: /app/shards)  
  GGUF_SHARD_RESIDENT_PAGES - Memory pages to keep resident (default: 512)
  GGUF_SHARD_LOG_LEVEL      - Logging level (default: INFO)

EOF
}

# Set up logging
setup_logging() {
    export PYTHONUNBUFFERED=1
    
    case "${GGUF_SHARD_LOG_LEVEL:-INFO}" in
        DEBUG) export LOG_LEVEL=10 ;;
        INFO)  export LOG_LEVEL=20 ;;
        WARN)  export LOG_LEVEL=30 ;;
        ERROR) export LOG_LEVEL=40 ;;
    esac
}

# Check CUDA availability
check_cuda() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "CUDA detected: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1)"
        nvidia-smi -L
    else
        echo "WARNING: CUDA not available - running in CPU mode"
    fi
}

# Validate environment
validate_env() {
    # Check data directory
    if [[ ! -d "$GGUF_SHARD_DATA_DIR" ]]; then
        echo "WARNING: Data directory $GGUF_SHARD_DATA_DIR does not exist"
        mkdir -p "$GGUF_SHARD_DATA_DIR"
    fi
    
    # Check output directory
    if [[ ! -d "$GGUF_SHARD_OUTPUT_DIR" ]]; then
        echo "📁 Creating output directory $GGUF_SHARD_OUTPUT_DIR"
        mkdir -p "$GGUF_SHARD_OUTPUT_DIR"
    fi
    
    # Check write permissions
    if [[ ! -w "$GGUF_SHARD_OUTPUT_DIR" ]]; then
        echo "ERROR: No write permission to $GGUF_SHARD_OUTPUT_DIR"
        exit 1
    fi
}

# Main command dispatcher
main() {
    setup_logging
    check_cuda
    validate_env
    
    case "${1:-help}" in
        shard)
            shift
            echo "🔨 Starting shard creation..."
            python3 /app/forge/shard.py shard "$@"
            ;;
        delta)
            shift
            echo "🔄 Starting delta creation..."
            python3 /app/trainer/delta_trainer.py "$@"
            ;;
        test)
            shift
            echo "🧪 Running test suite..."
            cd /app
            python3 /app/tests/test_suite.py "$@"
            ;;
        verify)
            shift
            echo "🔍 Verifying shard integrity..."
            # Add verification tool here
            echo "Verification not yet implemented"
            ;;
        monitor)
            echo "Starting monitoring dashboard..."
            # Add monitoring dashboard here  
            python3 -m http.server 8080 --directory /app/logs
            ;;
        bash|sh)
            echo "🐚 Starting interactive shell..."
            exec /bin/bash
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "ERROR: Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
