#!/bin/bash

# ============================================
# Multi-Concurrency Environment Variables
# kimi int4 Benchmark - All CONC Values
# ============================================
# This script sets environment variables for multi-concurrency testing
# Use with: ./kimi_benchmark submit <team> -isl <value> -osl <value>
#
# Usage:
#   source all_conc_var.sh
#   bash launch_vllm_server.sh  # Start server first
#   ./kimi_benchmark submit "YourTeam" -isl 1024 -osl 1024
#   ./kimi_benchmark submit "YourTeam" -isl 1024 -osl 8192
#   ./kimi_benchmark submit "YourTeam" -isl 8192 -osl 1024

# ============================================
# Server Configuration
# ============================================

export MODEL="moonshotai/Kimi-K2.5"
export PORT=8888
export TP=8

# ============================================
# Model Configuration
# ============================================

export MAX_MODEL_LEN=16384

# ============================================
# Testing Configuration
# ============================================

# Note: ISL, OSL, CONC will be set by the benchmark script based on -isl/-osl flags
# The script will automatically test the following CONC values:
#   - ISL=1024, OSL=1024: CONC=4,8,16 (3 configurations)
#   - ISL=1024, OSL=8192: CONC=4,8,16 (3 configurations)
#   - ISL=8192, OSL=1024: CONC=4,8 (2 configurations)

export RANDOM_RANGE_RATIO=1.0

# ============================================
# Team Configuration (Optional)
# ============================================

# Set your team name here (optional, can override with command line)
# export TEAM_NAME_ENV="YourTeam"

# ============================================
# Summary
# ============================================

echo "============================================"
echo "Multi-Concurrency Environment Variables Set"
echo "============================================"
echo "MODEL:        $MODEL"
echo "PORT:         $PORT"
echo "TP:           $TP"
echo "MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "RANDOM_RANGE_RATIO: $RANDOM_RANGE_RATIO"
echo ""
echo "Next steps:"
echo "1. bash launch_vllm_server.sh  # Start server"
echo "2. Wait for server to be ready (see 'Uvicorn running...')"
echo "3. Run multi-concurrency tests:"
echo "   ./kimi_benchmark submit \"YourTeam\" -isl 1024 -osl 1024"
echo "   ./kimi_benchmark submit \"YourTeam\" -isl 1024 -osl 8192"
echo "   ./kimi_benchmark submit \"YourTeam\" -isl 8192 -osl 1024"
echo "============================================"

