#!/bin/bash

# ============================================
# Single Configuration Environment Variables
# GPT-OSS FP4 Benchmark on Atom - Specific CONC Value
# ============================================
# This script sets environment variables for single configuration testing.
# Use with: ./gptoss_benchmark submit <team>
#
# Usage:
#   source specific_conc_var.sh
#   bash launch_atom_server.sh  # Start server first (in another terminal)
#   ./gptoss_benchmark submit "YourTeam"

# ============================================
# Server Configuration (Atom)
# ============================================

export MODEL="openai/gpt-oss-120b"
export PORT=8878
export TP=1

# Atom-specific
export EP_SIZE=1
export BLOCK_SIZE=16
export DP_ATTENTION=0

# ============================================
# Testing Configuration - Single Configuration
# ============================================

export ISL=8192
export OSL=1024
export CONC=4

export RANDOM_RANGE_RATIO=1.0
export NUM_PROMPTS=$(( CONC * 10 ))
export RESULT_FILENAME="result_isl${ISL}_osl${OSL}_conc${CONC}"

# ============================================
# Team Configuration (Optional)
# ============================================

# export TEAM_NAME_ENV="YourTeam"

# ============================================
# Summary
# ============================================

echo "============================================"
echo "Single Configuration Environment Variables Set (Atom)"
echo "============================================"
echo "MODEL:        $MODEL"
echo "PORT:         $PORT"
echo "TP:           $TP"
echo "EP_SIZE:      $EP_SIZE"
echo "BLOCK_SIZE:   $BLOCK_SIZE"
echo "ISL:          $ISL"
echo "OSL:          $OSL"
echo "CONC:         $CONC"
echo "NUM_PROMPTS:  $NUM_PROMPTS"
echo "RESULT_FILE:  $RESULT_FILENAME.json"
echo ""
echo "Next steps:"
echo "1. bash launch_atom_server.sh   # Start Atom server (in one terminal)"
echo "2. Wait for server ready (health at http://0.0.0.0:$PORT/health)"
echo "3. In another terminal: source specific_conc_var.sh"
echo "4. ./gptoss_benchmark submit \"YourTeam\""
echo "============================================"
