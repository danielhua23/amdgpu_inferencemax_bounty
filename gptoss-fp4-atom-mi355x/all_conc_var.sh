#!/bin/bash

# ============================================
# Multi-Concurrency Environment Variables
# GPT-OSS FP4 Benchmark on Atom - All CONC Values
# ============================================
# This script sets environment variables for multi-concurrency testing.
# Use with: ./gptoss_benchmark submit <team> -isl <value> -osl <value>
#
# Usage:
#   source all_conc_var.sh
#   bash launch_atom_server.sh  # Start server first (in another terminal)
#   ./gptoss_benchmark submit "YourTeam" -isl 8192 -osl 1024

# ============================================
# Server Configuration (Atom)
# ============================================

export MODEL="openai/gpt-oss-120b"
export PORT=8888
export TP=8

# Atom-specific
export EP_SIZE=1
export BLOCK_SIZE=16
export DP_ATTENTION=0

# ============================================
# Model / Seq-Len Configuration
# ============================================
# Only ISL=8192, OSL=1024 (8k/1k) is supported

export ISL=8192
export OSL=1024
export RANDOM_RANGE_RATIO=1.0

# ============================================
# Team Configuration (Optional)
# ============================================

# export TEAM_NAME_ENV="YourTeam"

# ============================================
# Summary
# ============================================

echo "============================================"
echo "Multi-Concurrency Environment Variables Set (Atom)"
echo "============================================"
echo "MODEL:        $MODEL"
echo "PORT:         $PORT"
echo "TP:           $TP"
echo "EP_SIZE:      $EP_SIZE"
echo "BLOCK_SIZE:   $BLOCK_SIZE"
echo "ISL:          $ISL"
echo "OSL:          $OSL"
echo "RANDOM_RANGE_RATIO: $RANDOM_RANGE_RATIO"
echo ""
echo "Next steps:"
echo "1. bash launch_atom_server.sh   # Start Atom server (in one terminal)"
echo "2. Wait for server ready (health at http://0.0.0.0:$PORT/health)"
echo "3. In another terminal: source all_conc_var.sh"
echo "4. ./gptoss_benchmark submit \"YourTeam\" -isl 8192 -osl 1024"
echo "============================================"
