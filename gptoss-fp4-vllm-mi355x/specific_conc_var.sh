#!/bin/bash

# ============================================
# Single Configuration Environment Variables
# GPT-OSS FP4 Benchmark - Specific CONC Value
# ============================================
# This script sets environment variables for single configuration testing
# Use with: ./gptoss_benchmark submit <team>
#
# Usage:
#   source specific_conc_var.sh
#   bash launch_vllm_server.sh  # Start server first
#   ./gptoss_benchmark submit "YourTeam"

# ============================================
# Server Configuration
# ============================================

export MODEL="openai/gpt-oss-120b"
export PORT=8878
export TP=8

# ============================================
# Model Configuration
# ============================================

export MAX_MODEL_LEN=16384

# ============================================
# Testing Configuration - Single Configuration
# ============================================

# Input/Output Sequence Lengths
export ISL=1024
export OSL=1024

# Concurrency Level
# we have 4,8,16,32,64
export CONC=8

# Random Range Ratio
export RANDOM_RANGE_RATIO=1.0

# Number of Prompts (GPT-OSS: CONC * 10)
export NUM_PROMPTS=$(( CONC * 10 ))

# Result Filename
export RESULT_FILENAME="result_isl${ISL}_osl${OSL}_conc${CONC}"

# ============================================
# Team Configuration (Optional)
# ============================================

# Set your team name here (optional, can override with command line)
# export TEAM_NAME_ENV="YourTeam"

# ============================================
# Summary
# ============================================

echo "============================================"
echo "Single Configuration Environment Variables Set"
echo "============================================"
echo "MODEL:        $MODEL"
echo "PORT:         $PORT"
echo "TP:           $TP"
echo "MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo ""
echo "ISL:          $ISL"
echo "OSL:          $OSL"
echo "CONC:         $CONC"
echo "NUM_PROMPTS:  $NUM_PROMPTS"
echo "RESULT_FILE:  $RESULT_FILENAME.json"
echo ""
echo "Next steps:"
echo "1. bash launch_vllm_server.sh  # Start server"
echo "2. Wait for server to be ready (see 'Uvicorn running...')"
echo "3. Run test: ./gptoss_benchmark submit \"YourTeam\""
echo "============================================"

