#!/usr/bin/env bash

# ============================================
# SGLang Server Launch Script
# ============================================
# This script launches an SGLang server with optimized settings
# for AMD MI355X GPUs running FP4 models
#
# Required Environment Variables:
#   MODEL: Model path (e.g., amd/DeepSeek-R1-0528-MXFP4-Preview)
#   PORT: Server port (default: 8888)
#   TP: Tensor parallel size (e.g., 4, 8)
#   ISL: Input sequence length (for PREFILL_SIZE optimization)
#   OSL: Output sequence length (for PREFILL_SIZE optimization)
#   CONC: Concurrency level (for PREFILL_SIZE optimization)
#
# Optional Environment Variables:
#   MEM_FRACTION: Memory fraction for static allocation (default: 0.8)
#   CUDA_GRAPH_MAX_BS: CUDA graph max batch size (default: 128)
#   NUM_CONTINUOUS_DECODE_STEPS: Number of continuous decode steps (default: 4)
#   DISABLE_RADIX_CACHE: Set to "true" to disable radix cache (default: true)
#   SERVER_LOG: Path to server log file (auto-generated if not set)

# ============================================
# Validate Required Environment Variables
# ============================================

if [[ -z "$MODEL" ]]; then
    echo "ERROR: MODEL environment variable is required"
    echo "Example: export MODEL=amd/DeepSeek-R1-0528-MXFP4-Preview"
    exit 1
fi

if [[ -z "$PORT" ]]; then
    echo "WARNING: PORT not set, using default: 8888"
    PORT=8888
fi

if [[ -z "$TP" ]]; then
    echo "ERROR: TP (tensor parallel size) environment variable is required"
    echo "Example: export TP=8"
    exit 1
fi

# ============================================
# Set Default Values
# ============================================

MEM_FRACTION=${MEM_FRACTION:-0.8}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-128}
NUM_CONTINUOUS_DECODE_STEPS=${NUM_CONTINUOUS_DECODE_STEPS:-4}
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-true}
WAIT_FOR_READY=${WAIT_FOR_READY:-false}

# Enable AMD-specific optimizations
export SGLANG_USE_AITER=1

# ============================================
# Calculate Optimal PREFILL_SIZE
# ============================================

PREFILL_SIZE=196608  # Default value

# Optimize PREFILL_SIZE based on workload characteristics
if [[ -n "$ISL" && -n "$OSL" && -n "$CONC" ]]; then
    echo "INFO: Optimizing PREFILL_SIZE for ISL=$ISL, OSL=$OSL, CONC=$CONC"
    
    # For long input + short output + high concurrency, use smaller prefill size
    if [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
        if [[ "$CONC" -gt "32" ]]; then
            PREFILL_SIZE=32768
            echo "INFO: Using reduced PREFILL_SIZE=$PREFILL_SIZE for high concurrency long input scenario"
        fi
    fi
else
    echo "INFO: ISL/OSL/CONC not set, using default PREFILL_SIZE=$PREFILL_SIZE"
fi

# ============================================
# Create Server Log File
# ============================================

if [[ -z "$SERVER_LOG" ]]; then
    SERVER_LOG=$(mktemp /tmp/sglang-server-XXXXXX.log)
    echo "INFO: Server log file: $SERVER_LOG"
fi

# ============================================
# Build SGLang Server Command
# ============================================

SGLANG_CMD="python3 -m sglang.launch_server --model-path=$MODEL --trust-remote-code \
--host=0.0.0.0 --port=$PORT \
--tensor-parallel-size=$TP \
--chunked-prefill-size=$PREFILL_SIZE \
--mem-fraction-static=$MEM_FRACTION \
--num-continuous-decode-steps=$NUM_CONTINUOUS_DECODE_STEPS \
--max-prefill-tokens=$PREFILL_SIZE \
--cuda-graph-max-bs=$CUDA_GRAPH_MAX_BS"

# Add optional flags
if [[ "$DISABLE_RADIX_CACHE" == "true" ]]; then
    SGLANG_CMD="$SGLANG_CMD --disable-radix-cache"
fi

# ============================================
# Launch SGLang Server
# ============================================

echo "============================================"
echo "Launching SGLang Server"
echo "============================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Tensor Parallel: $TP"
echo "Prefill Size: $PREFILL_SIZE"
echo "Memory Fraction: $MEM_FRACTION"
echo "CUDA Graph Max BS: $CUDA_GRAPH_MAX_BS"
echo "Continuous Decode Steps: $NUM_CONTINUOUS_DECODE_STEPS"
echo "Radix Cache: $([ "$DISABLE_RADIX_CACHE" == "true" ] && echo "Disabled" || echo "Enabled")"
echo "Log File: $SERVER_LOG"
echo "============================================"

echo ""
echo "⚠️  NOTE: First launch may take 20+ minutes due to JIT compilation of kernels"
echo "⚠️  Server will run in FOREGROUND. Press Ctrl+C to stop."
echo "⚠️  Server is ready when you see: 'Uvicorn running on http://0.0.0.0:$PORT'"
echo ""
echo "============================================"
echo "Starting Server..."
echo "============================================"
echo ""

set -x
eval "$SGLANG_CMD"
set +x

