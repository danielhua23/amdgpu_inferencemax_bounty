# AMD Instinct MI355X InferenceMAX Bounty Program

[中文版](https://github.com/danielhua23/amdgpu_inferencemax_bounty/blob/main/README_ZH.md)
## 🎯 Overview

Welcome to the **AMD Instinct MI355X InferenceMAX Bounty Program**! This competition challenges developers to optimize LLM inference performance on AMD's latest Instinct MI355X GPUs, targeting to beat NVIDIA B200 baseline performance on [InferenceMAX](https://inferencemax.semianalysis.com/) benchmarks and contribute your optimizations back to the open-source community.

## 🏆 Competition Rules

### 1. Performance Benchmark

- **Target**: Your solution must exceed the **best B200 baseline** recorded during the competition period
- **Baseline Updates**: B200 baselines will be **dynamically updated** throughout the competition to align with [InferenceMAX](https://inferencemax.semianalysis.com/)
- **Evaluation Period**: Performance is measured against the best B200 results within the competition timeframe

### 2. Required Submissions

Participants must submit the following to qualify:

#### a) Leaderboard Results
- Submit your results to the targeted leaderboard using our provided scripts
- See benchmark instructions below for automation

#### b) Performance Curve Plot
- Provide a throughput vs. latency curve similar to [InferenceMAX](https://inferencemax.semianalysis.com/)
- Must include all tested concurrency levels for the specific ISL-OSL configuration
- **Qualification Criteria**: At least **50% of data points** must be above the B200 TRT-LLM baseline to enter the evaluation pool

#### c) Technical Documentation
Your submission must include detailed documentation explaining:
1. **Why**: Technical explanation of why your solution achieves superior performance
2. **How**: Complete reproduction guide with step-by-step instructions
3. **What**: Specific optimizations, kernel changes, or algorithmic improvements

### 3. Winning Criteria

- **Selection Process**: We will select the **first participant** who beats the B200 baseline for each configuration during regular competition time
- **Multiple Winners**: Each model-ISL-OSL configuration has its own bounty (see Bounty Structure below)

### 4. Prize Claiming Process

Once selected as a winner:

1. **PR Submission** (Within 2 weeks):
   - Submit a Pull Request to the upstream repository:
     - **GPT-OSS (vLLM)**: [vLLM GitHub](https://github.com/vllm-project/vllm)
     - **DeepSeek-R1 (SGLang)**: [SGLang GitHub](https://github.com/sgl-project/sglang)
   
2. **Code Merge** (Within 2 weeks):
   - Your PR must be **merged** into the main branch within 2 weeks of selection
   - Code must meet the project's quality standards and review requirements
   
3. **Prize Award**:
   - Upon successful merge, you will receive the bounty prize
   
4. **Fallback**:
   - If the selected winner fails to merge their PR within 2 weeks, we will select the **next qualifying participant** who beat B200 for that configuration

## 📊 Benchmarked Configurations

### Models & Backends

| Model | Backend | Directory | Leaderboard |
|-------|---------|-----------|-------------|
| **DeepSeek-R1 FP4** | SGLang | `dsr1-fp4-sglang-mi355x/` | https://daniehua-dsr1-fp4-sgl-isl*-osl*.hf.space |
| **GPT-OSS FP4 (120B)** | vLLM | `gptoss-fp4-vllm-mi355x/` | https://daniehua-gptoss-fp4-vllm-isl*-osl*.hf.space |

### Test Configurations

Each model is tested with multiple **ISL (Input Sequence Length)** and **OSL (Output Sequence Length)** combinations:

| ISL | OSL | Description | DeepSeek-R1 CONC | GPT-OSS CONC |
|-----|-----|-------------|------------------|--------------|
| 1024 | 1024 | Standard short | 4,8,16,32,64 | 4,8 |
| 1024 | 8192 | Long generation | 4,8,16,32,64 | 4,8,16 |
| 8192 | 1024 | Long context | 4,8,16,32,64 | 4,8 |

**CONC** = Maximum concurrent requests

## 💰 Bounty Structure

Each **Model × ISL × OSL** configuration represents an independent bounty:

### DeepSeek-R1 (SGLang) - 3 Bounties
- ISL=1024, OSL=1024: **$X,XXX** *(TBD)*
- ISL=1024, OSL=8192: **$X,XXX** *(TBD)*
- ISL=8192, OSL=1024: **$X,XXX** *(TBD)*

### GPT-OSS (vLLM) - 3 Bounties
- ISL=1024, OSL=1024: **$X,XXX** *(TBD)*
- ISL=1024, OSL=8192: **$X,XXX** *(TBD)*
- ISL=8192, OSL=1024: **$X,XXX** *(TBD)*

**Total: 6 Bounties** (amounts to be announced)

## 🚀 Quick Start Guide

### For vLLM (GPT-OSS) Optimization

Please refer to [GPTOSS_COMPETITION_QUICKSTART.md](https://github.com/danielhua23/amdgpu_inferencemax_bounty/blob/main/gptoss-fp4-vllm-mi355x/GPTOSS_COMPETITION_QUICKSTART_EN.md)

### For SGLang (DeepSeek-R1) Optimization

Please refer to [COMPETITION_QUICKSTART.md](https://github.com/danielhua23/amdgpu_inferencemax_bounty/blob/main/dsr1-fp4-sglang-mi355x/COMPETITION_QUICKSTART_EN.md)

## 📈 Performance Metrics

Your submission will be evaluated on:

1. **Throughput** (tokens/sec/GPU) - Higher is better
2. **Latency** (ms) - Lower is better
3. **Interactivity** (tokens/sec/user) - Higher is better
4. **E2E Latency** (ms) - Lower is better

All metrics are compared against NVIDIA B200 TRT baselines from [InferenceMAX](https://inferencemax.semianalysis.com/).

## 📋 Submission Checklist

Before submitting, ensure you have:

- [ ] Run benchmarks using provided scripts
- [ ] Submitted results to the leaderboard (automated in scripts)
- [ ] Plotted throughput vs. latency curve with all CONC values
- [ ] Verified >50% of curve data points beat B200 baseline
- [ ] Prepared technical documentation explaining:
  - [ ] Why your solution works (technical details)
  - [ ] How to reproduce results (step-by-step)
  - [ ] What optimizations were made (code changes)
- [ ] Ready to submit PR to vLLM/SGLang within 2 weeks

## 🛠️ Technical Requirements

### Hardware
- AMD Instinct MI355X GPU (8 GPUs)

### Models
- **DeepSeek-R1**: `amd/DeepSeek-R1-0528-MXFP4-Preview`
- **GPT-OSS**: `openai/gpt-oss-120b` (FP4 quantized)

## 📚 Repository Structure

```
amdgpu_inferencemax_bounty/
├── dsr1-fp4-sglang-mi355x/             # DeepSeek-R1 + SGLang
│   ├── dsr1_benchmark                  # Benchmark script
│   └── COMPETITION_QUICKSTART.md       # Competition guide
│
├── gptoss-fp4-vllm-mi355x/             # GPT-OSS + vLLM
│   ├── gptoss_benchmark                # Benchmark script
│   └── GPTOSS_COMPETITION_QUICKSTART.md # Competition guide

```

## 🔗 Important Links

- **InferenceMAX Platform**: https://inferencemax.semianalysis.com/
- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **SGLang GitHub**: https://github.com/sgl-project/sglang
- **AMD ROCm**: https://rocm.docs.amd.com/

### Leaderboards

#### DeepSeek-R1 (SGLang)
- ISL=1024, OSL=1024: https://daniehua-dsr1-fp4-sgl-isl1024osl1024.hf.space
- ISL=1024, OSL=8192: https://daniehua-dsr1-fp4-sgl-isl1024osl8192.hf.space
- ISL=8192, OSL=1024: https://daniehua-dsr1-fp4-sgl-isl8192osl1024.hf.space

#### GPT-OSS (vLLM)
- ISL=1024, OSL=1024: https://daniehua-gptoss-fp4-vllm-isl1024osl1024.hf.space
- ISL=1024, OSL=8192: https://daniehua-gptoss-fp4-vllm-isl1024osl8192.hf.space
- ISL=8192, OSL=1024: https://daniehua-gptoss-fp4-vllm-isl8192osl1024.hf.space

## 💡 Tips for Success

1. **Study InferenceMAX**: Review current B200 performance curves on [InferenceMAX](https://inferencemax.semianalysis.com/) to understand the target
2. **Performance Profiling**: Use ROCm profiling tools to identify bottlenecks
3. **Kernel Optimization**: Focus on high-impact kernels (attention, GEMM, quantization)
4. **Memory Management**: Optimize GPU memory usage and data transfers
5. **Documentation**: Good documentation increases your chances of PR acceptance
6. **Community**: Engage with vLLM/SGLang communities for best practices

## ❓ FAQ

**Q: Can I participate in multiple bounties?**  
A: Yes! Each Model×ISL×OSL configuration is independent. You can submit solutions for all 6 bounties.

**Q: What if multiple people beat B200 at the same time?**  
A: We will select the one with more curve data points above B200. If tied, we will select the one with higher sum of throughput ratios.

**Q: Can I use proprietary optimizations?**  
A: No. Your solution must be open-source and mergeable into vLLM/SGLang public repositories.

**Q: What if my PR is rejected?**  
A: You have 2 weeks to address feedback and get it merged. Work closely with maintainers.

**Q: Do I need to beat B200 on ALL CONC values?**  
A: No. You need ≥50% of your curve data points above B200 AND all curve data points must not be lower than AMD's existing performance points on InferenceMAX to qualify, then we compare overall performance.

## 📞 Support & Contact

For questions or issues:
- **Technical Issues**: Open an issue in the vLLM/SGLang repositories
- **Competition Rules**: [Contact information TBD]
- **Leaderboard Problems**: [Contact information TBD]

## 📅 Timeline

- **Competition Start**: [TBD]
- **Competition End**: [TBD]
- **B200 Baseline Updates**: Continuous (aligned with InferenceMAX)
- **PR Submission Deadline**: 2 weeks after winner selection
- **Prize Distribution**: After successful PR merge

## 🎖️ Recognition

Winners will be:
- Listed on the official leaderboards
- Credited in vLLM/SGLang release notes
- Featured in AMD and SemiAnalysis communications

---

**Good luck, and happy optimizing! 🚀**

*Let's push the boundaries of LLM inference performance together!*

---

## License

This benchmark suite and documentation are provided under [LICENSE TBD].

## Acknowledgments

- **SemiAnalysis** for the InferenceMAX platform
- **vLLM Team** for the vLLM inference engine
- **SGLang Team** for the SGLang inference framework
- **AMD** for the Instinct MI355X GPUs and ROCm platform

