#!/usr/bin/env bash

# ============================================
# ATOM (MTP) Server Launch Script
# ============================================
# This script launches an ATOM OpenAI-compatible server with MTP
# for AMD MI355X GPUs running DeepSeek-R1 MXFP4.
#
# Image: rocm/atom:rocm7.2.0-ubuntu24.04-pytorch2.9-atom0.1.1
# Model: amd/DeepSeek-R1-0528-MXFP4
#
# Required Environment Variables:
#   MODEL: Model path (e.g., amd/DeepSeek-R1-0528-MXFP4)
#   PORT: Server port (default: 8888)
#   TP: Tensor parallel size (e.g. 8)
#
# Optional Environment Variables:
#   ISL: Input sequence length (for --max-model-len; if not 1024/1024 with OSL, uses 10240)
#   OSL: Output sequence length (for --max-model-len)
#   EP_SIZE: Expert parallel size (default: 1; use >1 for --enable-expert-parallel)
#   DP_ATTENTION: (reserved for future use)
#   SERVER_LOG: Path to server log file (default: /tmp/atom-server-XXXXXX.log)

# ============================================
# Validate Required Environment Variables
# ============================================

if [[ -z "$MODEL" ]]; then
    echo "ERROR: MODEL environment variable is required"
    echo "Example: export MODEL=amd/DeepSeek-R1-0528-MXFP4"
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
# Set Defaults
# ============================================

EP_SIZE=${EP_SIZE:-1}
export OMP_NUM_THREADS=1
export AMDGCN_USE_BUFFER_OPS=1

# Calculate max-model-len based on ISL and OSL (match InferenceX dsr1_fp4_mi355x_atom_mtp.sh)
if [[ -n "$ISL" && -n "$OSL" && "$ISL" = "1024" && "$OSL" = "1024" ]]; then
    CALCULATED_MAX_MODEL_LEN=""
else
    CALCULATED_MAX_MODEL_LEN=" --max-model-len 10240 "
fi

if [[ "${EP_SIZE:-1}" -gt 1 ]]; then
    EP_FLAG=" --enable-expert-parallel"
else
    EP_FLAG=""
fi

# ============================================
# Create Server Log File
# ============================================

if [[ -z "$SERVER_LOG" ]]; then
    SERVER_LOG=$(mktemp /tmp/atom-server-XXXXXX.log)
    echo "INFO: Server log file: $SERVER_LOG"
fi

# ============================================
# Build ATOM Server Command
# ============================================

ATOM_CMD="python3 -m atom.entrypoints.openai_server \
    --model $MODEL \
    --server-port $PORT \
    -tp $TP \
    --kv_cache_dtype fp8 ${CALCULATED_MAX_MODEL_LEN} ${EP_FLAG} \
    --method mtp"

# ============================================
# Launch ATOM Server
# ============================================

echo "============================================"
echo "Launching ATOM (MTP) Server"
echo "============================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Tensor Parallel: $TP"
echo "KV cache dtype: fp8"
echo "Method: mtp"
echo "Max model len: ${CALCULATED_MAX_MODEL_LEN:-default}"
echo "Expert parallel: $EP_FLAG"
echo "Log File: $SERVER_LOG"
echo "============================================"
echo ""
echo "Server will run in FOREGROUND. Press Ctrl+C to stop."
echo "Server is ready when health check passes: curl http://0.0.0.0:$PORT/health"
echo ""
echo "============================================"
echo "Starting Server..."
echo "============================================"
echo ""

set -x
eval "$ATOM_CMD" 2>&1 | tee "$SERVER_LOG"
set +x
