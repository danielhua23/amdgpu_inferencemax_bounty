# AMD Instinct MI355X Performance Optimization Bounty Program

## 🎯 Overview

Welcome to the **AMD Instinct MI355X Performance Optimization Bounty Program**! Stage 2! This competition challenges developers to optimize LLM inference performance on AMD's latest Instinct MI355X GPUs, targeting to beat baseline performance

## 🏆 Stage 2 Competition Rules

### Tracks Overview

#### Track 1: GPT-OSS-120B FP4 with AMD ATOM or vLLM
- ISL/OSL: 8k/1k
- Concurrency: 4, 32, 128
- TP: no restriction, but less or equal to 8, single node
- EP: no restriction, but less or equal to 8, single node

#### Track 2: DeepSeek-R1-0528 FP4 + MTP with AMD ATOM or SGLang
- ISL/OSL: 8k/1k
- Concurrency: 4, 32, 128
- TP: no restriction, but less or equal to 8, single node
- EP: no restriction, but less or equal to 8, single node

### Required Submission

Participants must submit the following to qualify:

#### a) Code Link

- GitHub repository link (may be your forked **AMD ATOM** / **vLLM** / **SGLang**).
- Pull Request in your GitHub repo that clearly describes b), c), d), and any additional comments.

#### b) Leaderboard Results

- Submit your results to the official target leaderboard.

#### c) Performance Results

- **Throughput per GPU** (tokens/s/GPU, y-axis) vs. **E2E latency** (s, x-axis).
- **Throughput per GPU** (tokens/s/GPU, y-axis) vs. **interactivity** (tokens/s/user, x-axis).
- Must include all required concurrency levels (4, 32, 128) for the corresponding ISL/OSL configuration (8k/1k).

#### d) Technical Documentation

- **Why**: Technical explanation of how your solution achieves superior performance.
- **How**: Complete step-by-step reproduction guide.
- **What**: Specific optimizations, kernel changes, or algorithmic improvements.

#### Metrics (How to Calculate)

| Metric | Formula |
|--------|---------|
| Interactivity (token/s/user) | `1000.0 / median_tpot_ms` |
| E2E latency (s) | `median_e2el_ms / 1000` |
| Total token throughput per GPU (token/s/GPU) | `total_token_throughput / num_GPUs_you_used` (num_GPUs_you_used ≤ 8) |

#### Unallowed Optimization

- Optimization that will benefit all chips (e.g. MTP core algorithms and scheduler).
- Optimization that obviously won’t be accepted by vLLM/SGLang (e.g. new enhanced coupling AMD-specific dependencies with no downgrade options). Please check vLLM/SGLang contribution guides for basic information.

**Exception**: If you are optimizing using **AMD ATOM**, this is fully controlled by AMD; AMD-specific optimization is allowed.

> Let’s keep the conversation going on the **AMD E2E Speedrun Discord** channel if you have any questions.

---

### Grand Prize Eligibility (One per Track)

To win the Grand Prize for a track, the submission must meet or exceed all of the following performance targets at the specified concurrency levels.

#### Track 1: GPT-OSS Grand Prize Targets

No matter what TP and EP size you use, you should meet:

**Interactivity vs. total token throughput per GPU**

| CONC | Interactivity | Throughput per GPU |
|------|---------------|---------------------|
| 128 | ≥ 40 token/s/user | ≥ 50600 token/s/GPU |
| 32 | ≥ 150 token/s/user | ≥ 21500 token/s/GPU |
| 4 | ≥ 270 token/s/user | ≥ 9500 token/s/GPU |

**E2E Latency vs. total token throughput per GPU**

| CONC | E2E latency | Throughput per GPU |
|------|-------------|---------------------|
| 128 | ≤ 21 s | ≥ 50600 token/s/GPU |
| 32 | ≤ 6 s | ≥ 21500 token/s/GPU |
| 4 | ≤ 3.5 s | ≥ 9500 token/s/GPU |

- **Accuracy**: GPT-OSS (AMD ATOM or vLLM): GSM8K accuracy ≥ 0.38

#### Track 2: DeepSeek-R1-MTP Grand Prize Targets

No matter what TP and EP size you use, you should meet:

**Interactivity vs. total token throughput per GPU**

| CONC | Interactivity | Throughput per GPU |
|------|---------------|---------------------|
| 128 | ≥ 48 token/s/user | ≥ 6000 token/s/GPU |
| 32 | ≥ 50 token/s/user | ≥ 3900 token/s/GPU |
| 4 | ≥ 165 token/s/user | ≥ 1500 token/s/GPU |

**E2E Latency vs. total token throughput per GPU**

| CONC | E2E latency | Throughput per GPU |
|------|-------------|---------------------|
| 128 | ≤ 22 s | ≥ 6000 token/s/GPU |
| 32 | ≤ 18 s | ≥ 3900 token/s/GPU |
| 4 | ≤ 5 s | ≥ 1500 token/s/GPU |

- **Accuracy**: DeepSeek-R1-MTP (AMD ATOM or SGLang): GSM8K accuracy ≥ 0.93

#### Additional Mandatory Requirements for Grand Prize

- **Code Mergeability**: Your code must be mergeable into AMD ATOM / vLLM / SGLang. During Stage 2, we will collaborate closely to facilitate merging. After the deadline, you may need to wait 2–4 weeks until we open a pull request and get it merged. If your code is assessed as unlikely to be merged, you will not receive the grand prize.
- **Accuracy**: GPT-OSS (AMD ATOM/vLLM): GSM8K ≥ 0.38; DeepSeek-R1-MTP (AMD ATOM / SGLang): GSM8K ≥ 0.93.
- **Code Legality**: No copyright infringement or license violations.
- **Reproducibility**: Results must be fully reproducible with submitted code and documentation.

---

### Ranking & Scoring Rules

Each track is ranked independently. For each model track, we evaluate performance across **3 concurrency levels** (4, 32, 128).

#### 1. How to Rank Teams

Ranking is determined by:

1. **Dominance in performance curve**: Teams whose curve (throughput vs. latency / throughput vs. interactivity) is higher and more to the left (throughput vs. latency) or right (throughput vs. interactivity) across concurrency points rank higher.
2. **Tie-breaker**: If curves intersect or it’s hard to determine which is better, we weight: **Throughput per GPU (50%) > Interactivity (30%) > E2E Latency (20%)**.
3. **Consistency**: Teams that perform well at all concurrency levels rank above teams strong at only one.

#### 2. Point Allocation

- Each concurrency level contributes up to **1000 points**.
- 1st place: 1000 pts · 2nd: 900 pts · 3rd: 800 pts · and so on in descending order.
- **Total score** = sum of points across all 3 concurrency levels. Higher total score = higher rank; the team with the highest score is awarded 1st place.

---

### Prize Structure

| Prize | Amount |
|-------|--------|
| **Grand Prize** | $1,000,000 total ($500,000 per track) |
| **First Place** | $60,000 total ($30,000 per track) |
| **Second Place** | $40,000 total ($20,000 per track) |

*AMD reserves the right of final interpretation.*


## 📊 Benchmarked Configurations
 
### Models & Backends
**Important Note: Backend is not limited, SGLang and vLLM is just for example, you can choose any framework you familiar with. It will be accepted only if the model performance surpass baseline**

**This bounty benchmarks only one case: ISL=8192, OSL=1024 (8k/1k long-context), with CONC = 4, 32, 128.**

| Model | Backend | Directory | Leaderboard |
|-------|---------|-----------|-------------|
| **DeepSeek-R1 MTP FP4** | SGLang | `dsr1-fp4-sglang-mtp-mi355x/` | https://daniehua-dsr1-fp4-isl8192osl1024.hf.space |
| **DeepSeek-R1 MTP FP4** | AMD ATOM | `dsr1-fp4-atom-mtp-mi355x/` | https://daniehua-dsr1-fp4-isl8192osl1024.hf.space |
| **GPT-OSS FP4 (120B)** | vLLM | `gptoss-fp4-vllm-mi355x/` | https://daniehua-gptoss-fp4-isl8192osl1024.hf.space |
| **GPT-OSS FP4 (120B)** | AMD ATOM | `gptoss-fp4-atom-mi355x/` | https://daniehua-gptoss-fp4-isl8192osl1024.hf.space |

### Test Configurations

**Only the following configuration is benchmarked** for all four tracks above:

| ISL | OSL | Description | CONC |
|-----|-----|-------------|------|
| 8192 | 1024 | Long context (8k/1k) | **4, 32, 128** |

**CONC** = Maximum concurrent requests. You must run and submit results for all three CONC values (4, 32, 128) for the single supported case ISL=8192, OSL=1024.

## 💰 Bounty Structure

Each **Model** represents an independent bounty. AMD ATOM also can be your choice of inference engine besides vLLM/SGLang. **Only ISL=8192, OSL=1024 (8k/1k)** is benchmarked, with **CONC = 4, 32, 128**:

### DeepSeek-R1 MTP FP4 (AMD ATOM/SGLang) - 1 Bounty
- ISL=8192, OSL=1024 (8k/1k), CONC=4, 32, 128

### GPT-OSS FP4 120B (AMD ATOM/vLLM) - 1 Bounty
- ISL=8192, OSL=1024 (8k/1k), CONC=4, 32, 128

**Total: 2 Bounties**

## 🚀 Quick Start Guide

**Benchmark case: ISL=8192, OSL=1024 (8k/1k) only; CONC = 4, 32, 128.**

### GPT-OSS (vLLM)
> Note: it's not mandatory that you must use vLLM to optimize GPT-OSS; AMD ATOM is also the choice and it might be the better choice given that it's taken fully controlled by AMD and might be with better performance.

→ [gptoss-fp4-vllm-mi355x GPTOSS_COMPETITION_QUICKSTART_EN.md](gptoss-fp4-vllm-mi355x/GPTOSS_COMPETITION_QUICKSTART_EN.md)

### GPT-OSS (AMD ATOM)
→ [gptoss-fp4-atom-mi355x GPTOSS_COMPETITION_QUICKSTART_EN.md](gptoss-fp4-atom-mi355x/GPTOSS_COMPETITION_QUICKSTART_EN.md)

### DeepSeek-R1 MTP (SGLang)
> Note: it's not mandatory that you must use SGLang to optimize DeepSeek-R1 MTP; AMD ATOM is also the choice and it might be the better choice given that it's taken fully controlled by AMD and might be with better performance.

→ [dsr1-fp4-sglang-mtp-mi355x COMPETITION_QUICKSTART_EN.md](dsr1-fp4-sglang-mtp-mi355x/COMPETITION_QUICKSTART_EN.md)

### DeepSeek-R1 MTP (AMD ATOM)
→ [dsr1-fp4-atom-mtp-mi355x COMPETITION_QUICKSTART_EN.md](dsr1-fp4-atom-mtp-mi355x/COMPETITION_QUICKSTART_EN.md)

### Hardware
- AMD Instinct MI355X GPU (8 GPUs)

### 🔗 Important Links

- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **SGLang GitHub**: https://github.com/sgl-project/sglang
- **AMD ATOM GitHub**: https://github.com/ROCm/ATOM/
- **AMD ROCm**: https://rocm.docs.amd.com/

### Leaderboards

**This bounty uses only ISL=8192, OSL=1024 (8k/1k), CONC=4, 32, 128.**

No matter what inference engine you use, results are submitted to the leaderboard which depends on model. 

| Track | Leaderboard |
|-------|-------------|
| DeepSeek-R1 MTP | https://daniehua-dsr1-fp4-isl8192osl1024.hf.space |
| GPT-OSS | https://daniehua-gptoss-fp4-isl8192osl1024.hf.space |

## ❓ FAQ

**Q: Can I participate in multiple bounties?**  
A: Yes! Each Model is independent. You can submit solutions for any or all of the 2 bounties (ISL=8192, OSL=1024, CONC=4,32,128).

**Q: Can I use proprietary optimizations?**  
A: No. Your solution must be open-source and mergeable into vLLM/SGLang/AMD ATOM public repositories.

**Q: Do I need to beat baselines on ALL CONC values?**  
A: To win grand prize, it's mandantory. Otherwise, it's not.

## Acknowledgments

- **vLLM Team** for the vLLM inference engine
- **SGLang Team** for the SGLang inference framework
- **AMD ATOM Team** for the ATOM inference framework
- **AMD** for the Instinct MI355X GPUs and ROCm platform

