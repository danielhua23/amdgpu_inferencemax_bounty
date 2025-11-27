# 🚀 Multi-Concurrency Testing - Quick Guide

## 📑 Table of Contents

- [⚡ Minimalist Usage (Recommended)](#-minimalist-usage-recommended)
- [📊 Leaderboard Auto-Routing](#-leaderboard-auto-routing)
  - [Submission Data Format](#submission-data-format)
- [🔧 Command Details](#-command-details)
  - [Basic Syntax](#basic-syntax)
  - [Parameter Description](#parameter-description)
  - [Example Usage](#example-usage)
- [🎯 Automatic Operations](#-automatic-operations)
- [📁 Result Output](#-result-output)
  - [JSON Result File Contents](#json-result-file-contents)
- [⏱️ Time Estimation](#️-time-estimation)
- [🆚 Comparison: Multi-Concurrency Mode vs Single Config Mode](#-comparison-multi-concurrency-mode-vs-single-config-mode)
- [❓ FAQ](#-faq)
  - [Q: Can I test only partial CONC values?](#q-can-i-test-only-partial-conc-values)
  - [Q: Can I interrupt the test midway?](#q-can-i-interrupt-the-test-midway)
  - [Q: How to retest failed CONC?](#q-how-to-retest-failed-conc)
  - [Q: Can multiple ISL-OSL tests run in parallel?](#q-can-multiple-isl-osl-tests-run-in-parallel)
- [🎉 Quick Start Example](#-quick-start-example)
- [📚 Related Documentation](#-related-documentation)

---

## ⚡ Minimalist Usage (Recommended)

Complete all tests and submit with just **4 commands**!

```bash
cd /workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x

# 1. Launch server
export MODEL="amd/DeepSeek-R1-0528-MXFP4-Preview"
export PORT=8888
export TP=8
bash launch_sglang_server.sh &

# Wait for server ready (see "Uvicorn running..."), then run the following commands

# 2-4. Test 3 ISL-OSL combinations (each auto-tests 5 CONC values)
./dsr1_benchmark submit "YourTeam" -isl 1024 -osl 1024
./dsr1_benchmark submit "YourTeam" -isl 1024 -osl 8192
./dsr1_benchmark submit "YourTeam" -isl 8192 -osl 1024
```

**Done!** All 15 configurations (3 ISL-OSL × 5 CONC) have been tested and submitted to the corresponding Leaderboards.

---

## 📊 Leaderboard Auto-Routing

Each ISL-OSL combination automatically submits to its dedicated Leaderboard:

| ISL | OSL | Leaderboard URL |
|-----|-----|----------------|
| 1024 | 1024 | https://daniehua-dsr1-fp4-sgl-isl1024osl1024.hf.space |
| 1024 | 8192 | https://daniehua-dsr1-fp4-sgl-isl1024osl8192.hf.space |
| 8192 | 1024 | https://daniehua-dsr1-fp4-sgl-isl8192osl1024.hf.space |

### Submission Data Format

Each submission contains the following fields (in order):

1. **Team Name** - Team name
2. **CONC** - Concurrency (4, 8, 16, 32, or 64)
3. **MI355X E2E (median)** (ms) - MI355X end-to-end latency median
4. **MI355X Throughput per GPU** (tokens/s) - MI355X single GPU throughput
5. **B200 E2E (median)** (ms) - B200 Baseline end-to-end latency
6. **B200 Throughput per GPU** (tokens/s) - B200 Baseline single GPU throughput
7. **E2E Ratio** - MI355X/B200 latency ratio (lower is better)
8. **Throughput Ratio** - MI355X/B200 throughput ratio (higher is better)
9. **bits_per_byte** - WikiText accuracy metric
10. **byte_perplexity** - WikiText accuracy metric
11. **word_perplexity** - WikiText accuracy metric

**Important Notes**:
- **E2E uses median** (`median_e2el_ms`): More stable, not affected by outliers
- **Throughput per GPU** (`tput_per_gpu = total_token_throughput / 8`): Normalized to single GPU performance
- **Auto-includes B200 Baseline**: Direct comparison of MI355X vs B200 performance
- **Performance Ratio Interpretation**:
  - `Throughput Ratio > 1.0` = MI355X is faster ✅
  - `E2E Ratio < 1.0` = MI355X has lower latency ✅

**Example**:
```json
{
  "data": [
    "MyTeam",      // Team Name
    16,            // CONC
    15979.59,      // MI355X E2E (median)
    1017.26,       // MI355X Throughput per GPU
    10407.0,       // B200 E2E (median)
    344.564,       // B200 Throughput per GPU
    1.5349,        // E2E Ratio (MI355X/B200)
    2.9523,        // Throughput Ratio (MI355X/B200)
    0.4485,        // bits_per_byte
    1.3646,        // byte_perplexity
    3.2522         // word_perplexity
  ]
}
```

---

## 🔧 Command Details

### Basic Syntax

```bash
cd /workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x
./dsr1_benchmark <mode> [team_name] -isl <value> -osl <value>
```

### Parameter Description

- **`<mode>`**: Test mode
  - `acc` - Accuracy test only (quick validation)
  - `perf` - Accuracy + performance test
  - `submit <team>` - Full test and submit to Leaderboard

- **`-isl <value>`**: Input Sequence Length (1024 or 8192)
- **`-osl <value>`**: Output Sequence Length (1024 or 8192)

### Example Usage

```bash
cd /workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x

# Quick accuracy validation (no submit)
./dsr1_benchmark acc -isl 1024 -osl 1024

# Full test (no submit)
./dsr1_benchmark perf -isl 1024 -osl 1024

# Full test and submit
./dsr1_benchmark submit "MyTeam" -isl 1024 -osl 1024
```

---

## 🎯 Automatic Operations

When you run `-isl -osl` mode, the script automatically:

1. ✅ **Loop test 5 CONC values**: 4, 8, 16, 32, 64
2. ✅ **For each CONC run**:
   - Accuracy test (WikiText)
   - Accuracy validation (baseline ±3%)
   - Performance test (if mode is perf/submit)
3. ✅ **Auto-calculate NUM_PROMPTS**:
   - OSL=8192: `NUM_PROMPTS = CONC × 20`
   - Others: `NUM_PROMPTS = CONC × 50`
4. ✅ **Save all results**: Independent directory `batch_isl{ISL}_osl{OSL}_{timestamp}/`
5. ✅ **Submit to Leaderboard** (submit mode): Auto-route to corresponding ISL-OSL Leaderboard
6. ✅ **Generate summary report**: `summary.txt`

---

## 📁 Result Output

```
batch_isl1024_osl1024_20251125_150000/
├── summary.txt                           # Summary report
├── result_isl1024_osl1024_conc4.json     # CONC=4 result
├── result_isl1024_osl1024_conc8.json     # CONC=8 result
├── result_isl1024_osl1024_conc16.json    # CONC=16 result
├── result_isl1024_osl1024_conc32.json    # CONC=32 result
└── result_isl1024_osl1024_conc64.json    # CONC=64 result
```

### JSON Result File Contents

Each JSON file contains the following information:

**1. Performance Metrics (SGLang Summary Statistics)**
```json
{
  "successful_requests": 3200,
  "benchmark_duration": 805.30,
  "total_token_throughput": 8138.10,
  "mean_ttft_ms": 1450.87,
  "median_ttft_ms": 1683.46,
  "mean_tpot_ms": 14.31,
  "median_e2el_ms": 15979.59,
  "tput_per_gpu": 1017.26,
  ...
}
```

**2. Benchmark Parameters**
```json
{
  "benchmark_args": {
    "model": "amd/DeepSeek-R1-0528-MXFP4-Preview",
    "backend": "vllm",
    "random_input_len": 1024,
    "random_output_len": 1024,
    "max_concurrency": 16,
    "num_prompts": 3200,
    ...
  }
}
```

**3. B200 Baseline Comparison** 🆚
```json
{
  "b200_baseline_nv1126": {
    "b200_median_e2e_1126": 10407,
    "b200_tput_pergpu_1126": 344.564
  },
  "tput_per_gpu_ratio_vs_b200_1126": 2.9523,   // >1.0 = MI355X faster ✅
  "median_e2e_ratio_vs_b200_1126": 1.5349      // <1.0 = MI355X faster
}
```

**4. Accuracy Metrics**
```json
{
  "accuracy": {
    "bits_per_byte": 0.4485,
    "byte_perplexity": 1.3646,
    "word_perplexity": 3.2522
  },
  "accuracy_validation": {
    "status": "PASSED"
  }
}
```

**Important Notes**:
- ✅ **Only saves summary statistics**, does not include per-request detail arrays (reduces file size)
- 📊 **Auto-includes B200 baseline** and performance ratios, easy comparison
- 🎯 **Includes complete parameter configuration**, easy to reproduce experiments

**Summary Report Example**:
```
Multi-Concurrency Test Results
ISL: 1024, OSL: 1024
Mode: submit
Time: Mon Nov 25 15:00:00 2025
============================================

[1/5] ISL=1024 OSL=1024 CONC=4: PASSED (180s)
[2/5] ISL=1024 OSL=1024 CONC=8: PASSED (220s)
[3/5] ISL=1024 OSL=1024 CONC=16: PASSED (280s)
[4/5] ISL=1024 OSL=1024 CONC=32: PASSED (350s)
[5/5] ISL=1024 OSL=1024 CONC=64: PASSED (450s)

============================================
Multi-Concurrency Test Complete!
============================================
Total tests: 5
Passed: 5
Failed: 0

Results saved in: batch_isl1024_osl1024_20251125_150000/
============================================
```

---

## ⏱️ Time Estimation

| Mode | Single ISL-OSL Combination | All 3 Combinations |
|------|------------------|--------------|
| **acc** | ~30-50 mins | ~1.5-2.5 hours |
| **perf** | ~1.5-2.5 hours | ~5-7.5 hours |
| **submit** | ~1.5-2.5 hours | ~5-7.5 hours |


---

## 🆚 Comparison: Multi-Concurrency Mode vs Single Config Mode

| Feature | Multi-Conc Mode `-isl -osl` | Single Config Mode |
|------|----------------------|-----------|
| Commands (15 configs) | **3 commands** | **15 commands** |
| CONC Setup | **Auto-loop** 4,8,16,32,64 | Manual set each |
| NUM_PROMPTS | **Auto-calculate** | Manual set each |
| Result Organization | **Auto-group** (by ISL-OSL) | Scattered |
| Leaderboard | **Auto-route** to corresponding URL | Unified URL |
| Use Case | ⭐ **Final submission** | Quick validate single config |

---

## ❓ FAQ

### Q: Can I test only partial CONC values?

No. Multi-concurrency mode tests 5 fixed CONC values (4,8,16,32,64). If you need customization, please use single config mode.

### Q: Can I interrupt the test midway?

Yes. Press `Ctrl+C` to interrupt. Completed CONC results will be saved, summary report will mark incomplete tests.

### Q: How to retest failed CONC?

Rerun the entire ISL-OSL combination, or use single config mode to test failed CONC individually.

### Q: Can multiple ISL-OSL tests run in parallel?

**Not recommended**. All tests share the same SGLang server. Parallel runs will cause port conflicts. Sequential execution is recommended.

---

## 🎉 Quick Start Example

```bash
# Complete workflow (3 ISL-OSL combinations, 15 configs total)
cd /workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x

# 1. Launch server
export MODEL="amd/DeepSeek-R1-0528-MXFP4-Preview"
export PORT=8888
export TP=8
bash launch_sglang_server.sh &

# Wait for server ready (~20-30 mins, first launch)
# View logs: tail -f /tmp/sglang-server-*.log

# 2. Test and submit
./dsr1_benchmark submit "MyAwesomeTeam" -isl 1024 -osl 1024
./dsr1_benchmark submit "MyAwesomeTeam" -isl 1024 -osl 8192
./dsr1_benchmark submit "MyAwesomeTeam" -isl 8192 -osl 1024

# 3. View results
cat batch_isl*/summary.txt

# 4. Check Leaderboards
# https://daniehua-dsr1-fp4-sgl-isl1024osl1024.hf.space
# https://daniehua-dsr1-fp4-sgl-isl1024osl8192.hf.space
# https://daniehua-dsr1-fp4-sgl-isl8192osl1024.hf.space
```

**Done!** 🚀

---

## 📚 Related Documentation

- [Competition Quick Start Guide](./COMPETITION_QUICKSTART.md)
- [Test Mode Details](./BENCHMARK_MODES_README.md)
- [Project Overview](./PROJECT_SUMMARY.md)


