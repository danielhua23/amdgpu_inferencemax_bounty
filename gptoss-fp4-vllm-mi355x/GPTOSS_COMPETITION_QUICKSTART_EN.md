# 🏆 GPT-OSS Competition Quick Start Guide

## 📑 Table of Contents

- [Objective](#objective)
- [📌 Important Notice](#-important-notice)
- [Core Files](#core-files)
- [Quick Start (5 Steps)](#quick-start-5-steps)
  - [1️⃣ Prepare Working Directory (on Host Machine)](#1️⃣-prepare-working-directory-on-host-machine)
  - [2️⃣ Launch Development Container](#2️⃣-launch-development-container)
  - [3️⃣ Install Latest Editable vLLM in Container](#3️⃣-install-latest-editable-vllm-in-container)
  - [4️⃣ Example: How to Recompile After Code Modifications](#4️⃣-example-how-to-recompile-after-code-modifications)
  - [5️⃣ Test Optimization Results](#5️⃣-test-optimization-results)
- [Test Mode Comparison](#test-mode-comparison)
- [Two Testing Approaches Comparison](#two-testing-approaches-comparison)
- [Evaluation Criteria](#evaluation-criteria)
  - [Performance Metrics (Primary)](#performance-metrics-primary)
  - [Accuracy Requirements (Must Meet)](#accuracy-requirements-must-meet)
  - [B200 Baseline Comparison 📊](#b200-baseline-comparison-)
- [Optimization Directions](#optimization-directions)
- [Development Tips](#development-tips)
- [FAQ](#faq)
- [Recommended Workflow](#recommended-workflow)
- [Resource Links](#resource-links)

---

## Objective

Optimize vLLM inference performance for GPT-OSS 120B FP4 model on AMD MI355X GPUs while maintaining model accuracy.

## 📌 Important Notice

This competition's benchmark **aligns with the [InferenceMAX](https://github.com/InferenceMAX/InferenceMAX)** repository's AMD MI355X test configuration and will be synchronized with InferenceMAX updates.

**Model Specifics**:
- **Model**: `openai/gpt-oss-120b` (FP4 quantized)
- **Framework**: vLLM
- **Features**: Uses AMD AITER optimized MoE and attention kernels

## Core Files

| File | Purpose |
|------|------|
| `amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x/launch_vllm_server.sh` | Launch vLLM server |
| `amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x/gptoss_benchmark` | Run tests and submit results (binary file)|
| `amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x/all_conc_var.sh` | Multi-concurrency test environment variables |
| `amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x/specific_conc_var.sh` | Single configuration test environment variables |

## Quick Start (5 Steps)

### 1️⃣ Prepare Working Directory (on Host Machine)

```bash
# Create working directory on host machine
mkdir -p ~/competition
cd ~/competition

# Clone vLLM (you will optimize based on this)
git clone https://github.com/vllm-project/vllm.git

# Clone AITER (AMD GPU operator library)
git clone --recursive https://github.com/ROCm/aiter.git

# Clone scripts repository
git clone https://github.com/danielhua23/amdgpu_inferencemax_bounty.git
```

### 2️⃣ Launch Development Container

**Note**: Replace `HF_TOKEN` with your Hugging Face Token.

```bash
docker run -it \
  --name vllm-dev \
  --ipc=host --shm-size=16g --network=host \
  --privileged --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /nfsdata/hf_hub_cache-1/:/root/.cache/huggingface \
  -v ~/competition:/workspace \
  -v ~/competition/aiter:/workspace/aiter \
  -v ~/competition/vllm:/workspace/vllm \
  -e HF_TOKEN=your_huggingface_token_here \
  rocm/7.0:rocm7.0_ubuntu_22.04_vllm_0.10.1_instinct_20250927_rc1 \
  /bin/bash
```

**Mount Instructions**:
- Host `~/competition/*` → Container `/workspace/*`
- Code modifications on host machine take effect immediately in container (and vice versa)
- Test scripts are located in `/workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x/` directory

### 3️⃣ Install Latest Editable vLLM in Container

> refer to https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#build-wheel-from-source

```bash
# Uninstall existing libraries in container
pip uninstall -y aiter vllm

# Install AITER
cd /workspace/aiter
python3 setup.py develop
```

Verify AITER installation:
```bash
root@mi355:/workspace# pip list | grep aiter
aiter                             0.1.7.post3.dev39+g1f5b378dc        /workspace/aiter
```

Install vLLM:
```bash
# Enter vLLM directory
cd /workspace/vllm

# Upgrade pip
pip install --upgrade pip

# Install vLLM (editable mode)
pip install --upgrade numba \
    scipy \
    huggingface-hub[cli,hf_transfer] \
    setuptools_scm
pip install "numpy<2"
# Install dependencies
pip install -r requirements/rocm.txt
# Build vLLM for MI GPU
export PYTORCH_ROCM_ARCH="gfx942;gfx950"
python3 setup.py develop

# Verify
python -c "import vllm; print(vllm.__file__)"
# Expected output: /workspace/vllm/vllm/__init__.py
```

>note: you might meet some **error** like : ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. transformers 4.56.2 requires huggingface-hub<1.0,>=0.34.0, but you have huggingface-hub 1.1.7 which is incompatible..**Just ignore it**


### 4️⃣ Example: How to Recompile After Code Modifications

```bash
# Edit in container or host machine (VS Code)
# Example: Optimize scheduler
cd /workspace/vllm
vim vllm/engine/llm_engine.py

# If you modified Python code, no recompile needed (editable mode auto-applies)

# If you modified C++/CUDA/HIP extensions, need to recompile:
cd /workspace/vllm
rm -r ./build
rm -r ./vllm.egg-info
pip uninstall -y vllm
python3 setup.py clean
python3 setup.py develop
```

### 5️⃣ Test Optimization Results

#### Recommended Workflow ⭐

```
Development Phase (Rapid Iteration)
  ↓
1. Single Config Test & Submit (Approach 1)
   - Use submit mode to test single CONC config (~15-20 mins)
   - Auto-submit to Leaderboard, view ranking in real-time
  ↓
2. Multi-Concurrency Batch Test & Submit (Approach 2)
   - Use submit mode to test all CONC configs(~1-2 hours/ISL-OSL)
   - Auto-submit all results
  ↓
Done! View Leaderboard rankings in real-time 🎉
```

**Why use submit mode directly?**
- ✅ **All-in-one**: submit = accuracy test + performance test + auto-submit
- ✅ **Real-time feedback**: See Leaderboard ranking immediately, rapid iteration
- ✅ **Save time**: No need to run perf then submit, just submit directly

---

#### Approach 1: Single Config Test (Quick Validation) ⚡

**Use Case**: Quickly validate single configuration performance during development

```bash
cd /workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x

# 1. Load environment variables (no manual export needed)
source specific_conc_var.sh

# 2. Launch vLLM server (first launch needs 20+ mins for JIT compilation)
bash launch_vllm_server.sh

# Wait for server ready (see "application startup..."）, then open a new window to run tests
# 3. go into new window and reload env var
docker exec -ti vllm-dev bash
source specific_conc_var.sh

# 4. Recommended: Test and submit directly (~15-20 mins) ⭐
./gptoss_benchmark submit "YourTeam"

# Optional: Quick accuracy validation only (~5-10 mins)
./gptoss_benchmark acc

# Optional: Test performance without submitting (~15-20 mins)
./gptoss_benchmark perf
```

**Environment Variables**: `specific_conc_var.sh` sets:
- `MODEL`, `PORT`, `TP` (server configuration)
- `ISL`, `OSL`, `CONC` (test configuration)
- `MAX_MODEL_LEN`, `RANDOM_RANGE_RATIO`, `NUM_PROMPTS`, `RESULT_FILENAME` (test parameters)

**Tip**: All `.sh` scripts are located in `/workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x/` directory

---

#### Approach 2: Multi-Concurrency Batch Testing (Test All CONC with One Command) 🚀

**Use Case**: Batch test all CONC values and submit to Leaderboard

**Only 3 commands to auto-test all configurations and submit! ⭐**

```bash
cd /workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x

# 1. Load environment variables (no manual export needed)
source all_conc_var.sh

# 2. Launch vLLM server (first launch needs 20+ mins for JIT compilation)
bash launch_vllm_server.sh

# ========== Recommended: Test and submit directly (all-in-one) ========== 
# Wait for server ready (see "application startup..."）, then open a new window to run tests
# 3. go into new window and reload env var
docker exec -ti sglang-dev bash
source specific_conc_var.sh

# 4.
# Submit all results for ISL=1024, OSL=1024 (auto-run CONC=4,8,16, ~1 hour)
./gptoss_benchmark submit "YourTeam" -isl 1024 -osl 1024

# Submit all results for ISL=1024, OSL=8192 (auto-run CONC=4,8,16, ~1 hour)
./gptoss_benchmark submit "YourTeam" -isl 1024 -osl 8192

# Submit all results for ISL=8192, OSL=1024 (auto-run CONC=4,8, ~40 mins)
./gptoss_benchmark submit "YourTeam" -isl 8192 -osl 1024

# ========== Optional: Test without submitting, use perf mode ========== 

# Test ISL=1024, OSL=1024 (no submit, ~1 hour)
./gptoss_benchmark perf -isl 1024 -osl 1024

# Test ISL=1024, OSL=8192 (no submit, ~1 hour)
./gptoss_benchmark perf -isl 1024 -osl 8192

# Test ISL=8192, OSL=1024 (no submit, ~40 mins)
./gptoss_benchmark perf -isl 8192 -osl 1024
```

**Results auto-submit to corresponding Leaderboards**:
- ISL=1024, OSL=1024 → https://daniehua-gptoss-fp4-vllm-isl1024osl1024.hf.space
- ISL=1024, OSL=8192 → https://daniehua-gptoss-fp4-vllm-isl1024osl8192.hf.space
- ISL=8192, OSL=1024 → https://daniehua-gptoss-fp4-vllm-isl8192osl1024.hf.space

**Submission Content**: Each CONC configuration submits independently, including:
- Team name + CONC value
- **MI355X vs B200 Direct Comparison**: E2E, throughput, performance ratios
- Accuracy metrics: bits_per_byte, byte_perplexity, word_perplexity

**CONC Range Explanation**:
- ISL=1024, OSL=1024: CONC=4,8,16 (3 configs)
- ISL=1024, OSL=8192: CONC=4,8,16 (3 configs)
- ISL=8192, OSL=1024: CONC=4,8 (2 configs) ⚠️ Note: 8192-1024 only tests CONC=4,8

## Test Mode Comparison

| Mode | Command Example | What Runs | Time (Single Config)| Use Case |
|------|---------|---------|-------------|---------|
| **submit** ⭐ | `./gptoss_benchmark submit "Team"` | Accuracy + Performance + Submit | ~15-20 mins | **Recommended: All-in-one, view ranking real-time** |
| **submit -isl -osl** ⭐ | `./gptoss_benchmark submit "Team" -isl 1024 -osl 1024` | Auto-test 3 CONC + Submit | ~1 hour | **Recommended: Batch test and submit** |
| **acc** | `./gptoss_benchmark acc` | Accuracy test only | ~5-10 mins | Optional: Quick accuracy validation |
| **perf** | `./gptoss_benchmark perf` | Accuracy + Performance (no submit) | ~15-20 mins | Optional: Test performance without submitting |
| **perf -isl -osl** | `./gptoss_benchmark perf -isl 1024 -osl 1024` | Auto-test 3 CONC (no submit) | ~1 hour | Optional: Batch test without submitting |

## Two Testing Approaches Comparison

| Approach | Recommended Command | Configs | Time Estimate | Recommended Scenario |
|------|---------|-------|---------|---------|
| **Approach 1: Single Config** ⭐ | `./gptoss_benchmark submit "Team"` | 1 | ~15-20 mins | **Development phase rapid iteration** |
| **Approach 2: Multi-Concurrency** ⭐ | `./gptoss_benchmark submit "Team" -isl 1024 -osl 1024` | 3 | ~1 hour | **Batch test all CONC** |

**Recommended Workflow** 🎯:
1. **Development Phase**: Use **Approach 1** (single config + submit) for rapid iteration, view Leaderboard real-time
2. **Batch Submission**: Use **Approach 2** (multi-conc + submit) to test and submit all configurations at once

**Why use submit directly?**
- ✅ submit = accuracy test + performance test + auto-submit (all-in-one)
- ✅ View Leaderboard ranking real-time, immediately know optimization effects
- ✅ Save time, no need to run perf then submit

## Evaluation Criteria

### Performance Metrics (Primary)

- **Throughput per GPU** (`tput_per_gpu`) - Highest weight 🏅
  - Single GPU normalized throughput = `total_token_throughput / 8`
  - Direct comparison with B200 baseline
- **E2E (median)** (ms) - End-to-end latency median
  - Direct comparison with B200 baseline

### Accuracy Requirements (Must Meet)

All metrics must be within baseline ± 3% range:
- bits_per_byte ≤ 2.0558 × 1.03 = **2.1175**
- byte_perplexity ≤ 4.1577 × 1.03 = **4.2824**
- word_perplexity ≤ 222.7893 × 1.03 = **229.4730**

❌ Exceeding range will immediately terminate testing, performance benchmark will not run

### B200 Baseline Comparison 📊

**Auto-comparison feature**: Each result JSON automatically includes NVIDIA B200 (periodically synced with InferenceMAX B200 performance data) baseline data and performance ratios!

**Performance Ratio Interpretation**:
- `tput_per_gpu_ratio_vs_b200_1126 > 1.0` = MI355X has higher throughput ✅
- `median_e2e_ratio_vs_b200_1126 < 1.0` = MI355X has lower latency ✅

See `b200_baseline_nv1126` field in result JSON for details.

## Optimization Directions

### 1. Kernel Optimization ⚡
- Attention kernel (Flash Attention, PagedAttention)
- MoE (Mixture of Experts) kernel - Critical for GPT-OSS!
- Quantization kernel (FP4/MXFP4)

### 2. Scheduling Optimization 📊
- Continuous batching
- Prefill/decode switching strategy
- KV cache management

### 3. Memory Optimization 💾
- Memory allocation strategy
- Paged attention
- CUDA graph / compilation config optimization

### 4. ROCm-Specific Optimization 🔧
- Leverage AMD GPU features
- HIP/ROCm API optimization
- AITER async iterator

### 5. vLLM-Specific Optimization 🚀
- Async scheduling
- Block manager
- Compilation config tuning

## Development Tips

### View Logs

```bash
# View server logs in real-time
tail -f /tmp/vllm-server-*.log

# Filter errors
tail -f /tmp/vllm-server-*.log | grep -i error
```

### Multi-Concurrency Batch Testing (Recommended) ⭐

```bash
# 1. Load environment variables
cd /workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x
source all_conc_var.sh

# 2. Launch vLLM server
bash launch_vllm_server.sh

# ========== Recommended: Test and submit directly (all-in-one) ========== 
# Wait for server ready (see "application startup..."）, then open a new window to run tests
# 3. go into new window and reload env var
docker exec -ti sglang-dev bash
source specific_conc_var.sh

# Submit all results for ISL=1024, OSL=1024 (auto-test CONC=4,8,16, ~1 hour)
./gptoss_benchmark submit "YourTeam" -isl 1024 -osl 1024

# Submit all results for ISL=1024, OSL=8192 (auto-test CONC=4,8,16, ~1 hour)
./gptoss_benchmark submit "YourTeam" -isl 1024 -osl 8192

# Submit all results for ISL=8192, OSL=1024 (auto-test CONC=4,8, ~40 mins)
./gptoss_benchmark submit "YourTeam" -isl 8192 -osl 1024
```

**Each command will automatically**:
- ✅ Test corresponding CONC values (1024-1024 and 1024-8192: 3 each, 8192-1024: 2)
- ✅ Run accuracy + performance tests
- ✅ Auto-submit to corresponding ISL-OSL Leaderboard
- ✅ Save all results to independent directory
- ✅ Generate summary report

**Leaderboard Auto-Routing**:
- `ISL=1024, OSL=1024` → https://daniehua-gptoss-fp4-vllm-isl1024osl1024.hf.space
- `ISL=1024, OSL=8192` → https://daniehua-gptoss-fp4-vllm-isl1024osl8192.hf.space
- `ISL=8192, OSL=1024` → https://daniehua-gptoss-fp4-vllm-isl8192osl1024.hf.space

**Result Output Example**:
```
============================================
Multi-Concurrency Testing Mode
============================================
ISL: 1024
OSL: 1024
Mode: submit
CONC values: 4, 8, 16
Team: YourTeam
Leaderboard: https://daniehua-gptoss-fp4-vllm-isl1024osl1024.hf.space
============================================

Results directory: batch_isl1024_osl1024_20251127_150000

============================================
Testing CONC=4
============================================
... (running tests) ...
✓ CONC=4: PASSED (180s)

============================================
Testing CONC=8
============================================
... (continue testing other CONC values) ...

============================================
Multi-Concurrency Test Complete!
============================================
Total tests: 3
Passed: 3
Failed: 0

Results saved in: batch_isl1024_osl1024_20251127_150000/
============================================
```

**Development Phase Quick Validation**:
```bash
# Recommended: Test and submit directly (all-in-one) ⭐
./gptoss_benchmark submit "YourTeam" -isl 1024 -osl 1024

# Optional: Accuracy test only (quick validation)
./gptoss_benchmark acc -isl 1024 -osl 1024

# Optional: Full test without submitting
./gptoss_benchmark perf -isl 1024 -osl 1024
```

## FAQ

### Q: What if accuracy validation fails?

```
ERROR: Accuracy validation FAILED!
bits_per_byte: 6.5000 > 5.1500
```

**Solution**: Your optimization affected model quality, need to adjust algorithm or parameters

### Q: How to launch server only without running tests?

```bash
cd /workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x

# Load environment variables
source all_conc_var.sh

# Launch server
bash launch_vllm_server.sh
```

Server will run in foreground, logs output directly to terminal.

### Q: C++ code modifications not taking effect?

Need to recompile:

```bash
cd /workspace/vllm
rm -rf build/
pip uninstall -y vllm
VLLM_TARGET_DEVICE=rocm python3 setup.py develop
```

### Q: What if multi-concurrency test fails midway?

Test will continue with remaining CONC configurations, generating complete report at the end. Failed configurations will be marked as "FAILED".

View failure reasons:
```bash
# View summary
cat batch_isl*_osl*/summary.txt

# View server logs
tail -f /tmp/vllm-server-*.log
```

### Q: How to test specific CONC value only?

Use single configuration mode:

```bash
cd /workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x

# 1. Edit specific_conc_var.sh to modify CONC value
vim specific_conc_var.sh  # Modify CONC=16

# 2. Load environment variables
source specific_conc_var.sh

# 3. Recommended: Test and submit directly ⭐
./gptoss_benchmark submit "YourTeam"
```

Or set manually:
```bash
cd /workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x
source specific_conc_var.sh
export CONC=16  # Override default, test CONC=16 only
export NUM_PROMPTS=160  # GPT-OSS: CONC * 10

# Recommended: Submit directly
./gptoss_benchmark submit "YourTeam"

# Optional: Test without submitting
./gptoss_benchmark perf
```

### Q: How long do tests take?

**Single Configuration Test**:
- **submit mode**: ~15-20 mins ⭐ **Recommended: All-in-one**
- **acc mode**: ~5-10 mins (Optional: Accuracy validation only)
- **perf mode**: ~15-20 mins (Optional: Test without submitting)

**Multi-Concurrency Test (per ISL-OSL combination)**:
- **ISL=1024, OSL=1024 (3 CONC)**: ~15-20 mins/CONC × 3 = **~1 hour** ⭐
- **ISL=1024, OSL=8192 (3 CONC)**: ~15-20 mins/CONC × 3 = **~1 hour** ⭐
- **ISL=8192, OSL=1024 (2 CONC)**: ~15-20 mins/CONC × 2 = **~40 mins** ⭐

**All 3 ISL-OSL Combinations** (8 configs):
- **submit mode**: ~1 hour + ~1 hour + ~40 mins = **~2.5-3 hours** ⭐

**Recommended Workflow** 🎯:
1. **Development Phase**: Single config `submit "YourTeam"` rapid iteration (~15-20 mins/time)
   - See Leaderboard ranking immediately, quickly validate optimization effects
2. **Batch Submission**: Multi-conc `submit "YourTeam" -isl -osl` submit all configs (~1 hour/combination)
   - Complete testing and submission at once, can run overnight

💡 **Why use submit directly?**
- ✅ All-in-one, no need to run perf then submit
- ✅ View ranking real-time, immediately know optimization effects
- ✅ Save time, avoid redundant runs

### Q: What's the difference between GPT-OSS and DeepSeek-R1?

**Main Differences**:

| Feature | GPT-OSS | DeepSeek-R1 |
|------|---------|------------|
| Model Size | 120B | ~670B |
| Architecture | MoE (Mixture of Experts) | Dense |
| Framework | vLLM | SGLang |
| CONC Range | 4-16 (8192-1024: 4-8) | 4-64 |
| NUM_PROMPTS | CONC × 10 | CONC × 50 (1024-1024/8192-1024) / CONC × 20 (1024-8192) |
| Optimization Focus | MoE kernel, vLLM scheduling | Long context, chunked prefill |

**Optimization Suggestions**:
- GPT-OSS: Focus on optimizing MoE kernel (A16W4 fused MoE provided by AITER)
- Adjust compile_sizes and cudagraph_capture_sizes in compilation config


## Recommended Workflow

```
Round 1: Familiarize with Baseline
  ├─ Run baseline test: ./gptoss_benchmark submit "YourTeam"
  ├─ Understand vLLM architecture
  └─ View Leaderboard baseline performance

Round 2: Low-Risk Optimization
  ├─ Adjust compilation config
  ├─ Optimize GPU memory utilization
  └─ Quick validation: ./gptoss_benchmark submit "YourTeam" (~15-20 mins)

Round 3: AMD GPU Kernel Optimization
  ├─ Profile to find bottlenecks
  ├─ Optimize MoE kernel (Critical!)
  └─ Real-time comparison: ./gptoss_benchmark submit "YourTeam", view Leaderboard

Round 4: System Optimization
  ├─ Async scheduling
  ├─ Block manager
  └─ End-to-end tuning, submit for validation after each optimization

Round 5: Batch Submission
  ├─ Test all ISL-OSL combinations
  ├─ ./gptoss_benchmark submit "YourTeam" -isl 1024 -osl 1024
  ├─ ./gptoss_benchmark submit "YourTeam" -isl 1024 -osl 8192
  └─ ./gptoss_benchmark submit "YourTeam" -isl 8192 -osl 1024
```

**Key Advantage**: Submit directly after each optimization, view Leaderboard ranking real-time, rapid iteration!

## Resource Links

- 📖 [InferenceMAX Official Repository](https://github.com/InferenceMAX/InferenceMAX) - Benchmark reference
- 🔧 [vLLM GitHub](https://github.com/vllm-project/vllm) - Inference framework
- 🔧 [AITER GitHub](https://github.com/ROCm/aiter) - AMD GPU operator library
- 📊 Leaderboards:
  - [ISL=1024, OSL=1024](https://daniehua-gptoss-fp4-vllm-isl1024osl1024.hf.space)
  - [ISL=1024, OSL=8192](https://daniehua-gptoss-fp4-vllm-isl1024osl8192.hf.space)
  - [ISL=8192, OSL=1024](https://daniehua-gptoss-fp4-vllm-isl8192osl1024.hf.space)


**Good luck! 🚀**

Remember:
- **Use submit mode directly**: All-in-one, view ranking real-time ⭐
- **Performance matters, accuracy matters more!** All optimizations must pass accuracy validation
- **Rapid iteration**: Submit immediately after each optimization, see effects instantly
- **Focus on MoE kernel optimization**: GPT-OSS is a MoE model, MoE kernel performance is critical!


