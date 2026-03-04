#!/usr/bin/env bash

# ============================================
# Atom Server Launch Script for GPT-OSS FP4
# ============================================
# This script launches an Atom OpenAI-compatible server with optimized settings
# for AMD MI355X GPUs running GPT-OSS 120B FP4.
#
# Image: rocm/atom:rocm7.1.1-ubuntu24.04-pytorch2.9-atom0.1.1-MI350x
#
# Required Environment Variables:
#   MODEL: Model path (e.g., openai/gpt-oss-120b)
#   PORT: Server port (default: 8888)
#   TP: Tensor parallel size (e.g., 1, 8)
#
# Optional Environment Variables:
#   EP_SIZE: Expert parallel size (default: 1); use >1 for --enable-expert-parallel
#   BLOCK_SIZE: Block size (default: 16)
#   ISL, OSL: For 8192/1024 (only supported case) we add --max-model-len 10240
#   SERVER_LOG: Path to server log file (auto-generated if not set)

# ============================================
# Validate Required Environment Variables
# ============================================

if [[ -z "$MODEL" ]]; then
    echo "ERROR: MODEL environment variable is required"
    echo "Example: export MODEL=openai/gpt-oss-120b"
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
# ROCm / device visibility
# ============================================
if command -v rocm-smi &>/dev/null; then
  version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
  if [[ -z "$version" || "$version" -lt 177 ]]; then
    export HSA_NO_SCRATCH_RECLAIM=1
  fi
fi
if [[ -n "$ROCR_VISIBLE_DEVICES" ]]; then
  export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# ============================================
# Set Defaults (from gptoss_fp4_mi355x_atom.sh)
# ============================================

export OMP_NUM_THREADS=1
BLOCK_SIZE=${BLOCK_SIZE:-16}
EP_SIZE=${EP_SIZE:-1}

CALCULATED_MAX_MODEL_LEN=" --max-model-len 10240 "

if [[ "$EP_SIZE" -gt 1 ]]; then
    EP=" --enable-expert-parallel"
else
    EP=" "
fi

# GPT-OSS model flag (Atom)
export ATOM_GPT_OSS_MODEL=1

# ============================================
# Create Server Log File
# ============================================

if [[ -z "$SERVER_LOG" ]]; then
    SERVER_LOG=$(mktemp /tmp/atom-server-XXXXXX.log)
    echo "INFO: Server log file: $SERVER_LOG"
fi

# ============================================
# Launch Atom Server
# ============================================

echo "============================================"
echo "Launching Atom Server (GPT-OSS FP4)"
echo "============================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Tensor Parallel: $TP"
echo "Block Size: $BLOCK_SIZE"
echo "EP_SIZE: $EP_SIZE"
echo "Log File: $SERVER_LOG"
echo "============================================"
echo ""
echo "Server will run in FOREGROUND. Press Ctrl+C to stop."
echo "Server is ready when health endpoint responds: http://0.0.0.0:$PORT/health"
echo "============================================"
echo ""

set -x
python3 -m atom.entrypoints.openai_server \
    --model "$MODEL" \
    --server-port "$PORT" \
    -tp "$TP" \
    --kv_cache_dtype fp8 $CALCULATED_MAX_MODEL_LEN $EP \
    --block-size "$BLOCK_SIZE" > "$SERVER_LOG" 2>&1
set +x
