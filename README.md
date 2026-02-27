# AMD Instinct MI355X Performance Optimization Bounty Program

[中文版](https://github.com/danielhua23/amdgpu_bounty_optimization/blob/main/README_ZH.md)
## 🎯 Overview

Welcome to the **AMD Instinct MI355X Performance Optimization Bounty Program**! This competition challenges developers to optimize LLM inference performance on AMD's latest Instinct MI355X GPUs, targeting to beat baseline performance

## 🏆 Competition Rules
### 1.Models list
four models for your selection to be optimized, and you can choose optimize all or any of them. One model corresponds to one PR for our review and judgement.
- (amd/DeepSeek-R1-0528-MXFP4)[https://huggingface.co/amd/DeepSeek-R1-0528-MXFP4]
- (amd/DeepSeek-R1-0528-mtp-mxfp4)[https://huggingface.co/amd/DeepSeek-R1-0528-mtp-mxfp4]
- (openai/gpt-oss-120b)[https://huggingface.co/openai/gpt-oss-120b]
- (moonshotai/Kimi-K2.5)[https://huggingface.co/moonshotai/Kimi-K2.5]

### 2. Performance Benchmark

- **Target**: Your solution must exceed the **baseline** during the competition period on the models you selected

### 3. Required Submissions

Participants must submit the following to qualify:

#### a) code link
- github repo link, can be your forked vllm/sglang/...
- PR at your github repo, clearly describe the below b),c),d) and something you want comment

#### b) Leaderboard Results
- Submit your results to the targeted leaderboard using our provided scripts
- See benchmark instructions below for automation

#### c) Performance Curve Plot
- Provide a throughput per gpu(y axis) vs. e2e latency(x axis) curve
- Must include all tested concurrency levels for the specific ISL-OSL configuration

#### d) Technical Documentation
Your submission must include detailed documentation explaining:
1. **Why**: Technical explanation of why your solution achieves superior performance
2. **How**: Complete reproduction guide with step-by-step instructions
3. **What**: Specific optimizations, kernel changes, or algorithmic improvements

### 3. Winning Criteria

- **solution be able to be merged**: you solution must be able to merged into the mainstream framework vllm or sglang or others, otherwise it's illegal.So you should balance the trade-off between PR merging and extreme performance
- **Selection Process**: We will comprehensively consider all the submitted solution and select the one whose PR or soluton is easy to merge first then consider who surpass the baseline most

### 4. Prize Claiming Process

Once selected as a winner:

1. **PR Submission** (Within 2 weeks):
   - Waiting for us to open a Pull Request to the upstream repository like:
     - [vLLM GitHub](https://github.com/vllm-project/vllm)
     - [SGLang GitHub](https://github.com/sgl-project/sglang)
     - [ATOM GitHub](https://github.com/ROCm/ATOM)
     - others
   
2. **Code Merge** (Within 2 weeks):
   - The PR must be **merged** into the main branch within 2 weeks
   - Code must meet the project's quality standards and review requirements
   
3. **Prize Award**:
   - Upon successful merge, you will receive the bounty prize
   
4. **Fallback**:
   - If the selected winner fails to merge their PR within 2 weeks, we will select the **next qualifying participant** who beat baseline for that configuration

## 📊 Benchmarked Configurations
 
### Models & Backends
**Important Note: Backend is not limited, SGLang and vLLM is just for example, you can choose any framework you familiar with. It will be accepted only if the model performance surpass baseline**
| Model | Backend | Directory | Leaderboard |
|-------|---------|-----------|-------------|
| **DeepSeek-R1 FP4** | SGLang | `dsr1-fp4-sglang-mi355x/` | https://daniehua-dsr1-fp4-sgl-isl*-osl*.hf.space |
| **DeepSeek-R1 MTP FP4** | SGLang | `dsr1-fp4-atom-mtp-mi355x/` | https://daniehua-dsr1-atom-mtp-sgl-isl*-osl*.hf.space |
| **GPT-OSS FP4 (120B)** | vLLM | `gptoss-fp4-vllm-mi355x/` | https://daniehua-gptoss-fp4-vllm-isl*-osl*.hf.space |
| **moonshotai/Kimi-K2.5** | vLLM | `kimik25-int4-vllm-mi355x/` | https://daniehua-kimik25-int4-vllm-isl*-osl*.hf.space |

### Test Configurations

Each model is tested with multiple **ISL (Input Sequence Length)** and **OSL (Output Sequence Length)** combinations:

| ISL | OSL | Description | DeepSeek-R1 CONC | GPT-OSS CONC | Kimi-K2.5 CONC | DeepSeek-R1 MTP CONC |
|-----|-----|-------------|------------------|--------------|----------------|----------------------|
| 1024 | 1024 | Standard short | 4,8,32,64,128,256 | 4,8,16,32,64,256 | 4,8,16,32,64,128 | 8,16,32,64,128,256 |
| 1024 | 8192 | Long generation | 4,8,16,32,64,128 | 4,8,16,32,64,256 | 4,8,16,32,64,128 | 8,16,32,64,128,256 |
| 8192 | 1024 | Long context | 4,8,16,32,64,128 | 4,8,16,32,64,256 | 4,8,16,32,64,128 | 8,16,32,64,128,256 |

**CONC** = Maximum concurrent requests

## 💰 Bounty Structure

Each **Model** represents an independent bounty:

### DeepSeek-R1 - 1 Bounty
- ISL=1024, OSL=1024
- ISL=1024, OSL=8192
- ISL=8192, OSL=1024

### GPT-OSS - 1 Bounty
- ISL=1024, OSL=1024
- ISL=1024, OSL=8192
- ISL=8192, OSL=1024

### Kimi-K2.5 - 1 Bounty
- ISL=1024, OSL=1024
- ISL=1024, OSL=8192
- ISL=8192, OSL=1024

### DeepSeek-R1 MTP - 1 Bounty
- ISL=1024, OSL=1024
- ISL=1024, OSL=8192
- ISL=8192, OSL=1024

**Total: 4 Bounties** (amounts to be announced)

## 🚀 Quick Start Guide

### For vLLM Optimization
Note: it's not mandatory that you must use vllm to optimize GPT-OSS, any inference framework you familiar with is accepted, but it's better to pick mainstream framework

Please refer to [GPTOSS_COMPETITION_QUICKSTART.md](https://github.com/danielhua23/amdgpu_bounty_optimization/blob/main/gptoss-fp4-vllm-mi355x/GPTOSS_COMPETITION_QUICKSTART_EN.md)

### For SGLang Optimization
Note: it's not mandatory that you must use sglang to optimize DeepSeek-R1, any inference framework you familiar with is accepted, but it's better to pick mainstream framework

Please refer to [COMPETITION_QUICKSTART.md](https://github.com/danielhua23/amdgpu_bounty_optimization/blob/main/dsr1-fp4-sglang-mi355x/COMPETITION_QUICKSTART_EN.md)

## 📈 Performance Metrics

Your submission will be evaluated on:

1. **Throughput** (tokens/sec/GPU) - Higher is better
2. **Latency** (ms) - Lower is better
3. **Interactivity** (tokens/sec/user) - Higher is better
4. **E2E Latency** (ms) - Lower is better

## 📋 Submission Checklist

Before submitting, ensure you have:

- [ ] Run benchmarks using provided scripts
- [ ] Submitted results to the leaderboard (automated in scripts)
- [ ] Plotted throughput vs. latency curve with all CONC values
- [ ] Verified >50% of curve data points beat baseline, one data point is one CONC value
- [ ] Prepared technical documentation explaining:
  - [ ] Why your solution works (technical details)
  - [ ] How to reproduce results (step-by-step)
  - [ ] What optimizations were made (code changes)
- [ ] Ready to submit PR to vLLM/SGLang within 2 weeks

## 🛠️ Technical Requirements

### Hardware
- AMD Instinct MI355X GPU (8 GPUs)

## 🔗 Important Links

- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **SGLang GitHub**: https://github.com/sgl-project/sglang
- **AMD ROCm**: https://rocm.docs.amd.com/

### Leaderboards

#### DeepSeek-R1
- ISL=1024, OSL=1024: https://daniehua-dsr1-fp4-sgl-isl1024osl1024.hf.space
- ISL=1024, OSL=8192: https://daniehua-dsr1-fp4-sgl-isl1024osl8192.hf.space
- ISL=8192, OSL=1024: https://daniehua-dsr1-fp4-sgl-isl8192osl1024.hf.space

#### GPT-OSS
- ISL=1024, OSL=1024: https://daniehua-gptoss-fp4-vllm-isl1024osl1024.hf.space
- ISL=1024, OSL=8192: https://daniehua-gptoss-fp4-vllm-isl1024osl8192.hf.space
- ISL=8192, OSL=1024: https://daniehua-gptoss-fp4-vllm-isl8192osl1024.hf.space

#### DeepSeek-R1 MTP
- ISL=1024, OSL=1024: https://daniehua-dsr1-fp4-atom-mtp-isl1024-osl1024.hf.space
- ISL=1024, OSL=8192: https://daniehua-dsr1-fp4-atom-mtp-isl1024-osl8192.hf.space
- ISL=8192, OSL=1024: https://daniehua-dsr1-fp4-atom-mtp-isl8192-osl1024.hf.space

#### kimi-k2.5
- ISL=1024, OSL=1024: https://daniehua-kimik25-int4-vllm-isl1024-osl1024.hf.space
- ISL=1024, OSL=8192: https://daniehua-kimik25-int4-vllm-isl1024-osl8192.hf.space
- ISL=8192, OSL=1024: https://daniehua-kimik25-int4-vllm-isl8192-osl1024.hf.space

## 💡 Tips for Success

1. **Performance Profiling**: Use ROCm profiling tools to identify bottlenecks
2. **Kernel Optimization**: Focus on high-impact kernels (attention, GEMM, quantization)
3. **Memory Management**: Optimize GPU memory usage and data transfers
4. **Documentation**: Good documentation increases your chances of PR acceptance
5. **Community**: Engage with vLLM/SGLang communities for best practices

## ❓ FAQ

**Q: Can I participate in multiple bounties?**  
A: Yes! Each Model×ISL×OSL configuration is independent. You can submit solutions for all 6 bounties.

**Q: What if multiple people beat baselines at the same time?**  
A: We will select the one with more curve data points above baseline. If tied, we will select the one with higher sum of throughput ratios in the regular time.

**Q: Can I use proprietary optimizations?**  
A: No. Your solution must be open-source and mergeable into vLLM/SGLang public repositories.

**Q: Do I need to beat baselines on ALL CONC values?**  
A: No. But you need ≥50% of your curve data points above baselines AND all curve data points must not be lower than AMD's existing performance points to qualify, then we compare overall performance.

## 📞 Support & Contact

For questions or issues:
- **Technical Issues**: Open an issue in the vLLM/SGLang repositories
- **Competition Rules**: [Contact information TBD]
- **Leaderboard Problems**: [Contact information TBD]

## 📅 Timeline

- **Competition Start**: [TBD]
- **Competition End**: [TBD]
- **PR Submission Deadline**: 2 weeks after winner selection
- **Prize Distribution**: After successful PR merge

## License

This benchmark suite and documentation are provided under [LICENSE TBD].

## Acknowledgments

- **vLLM Team** for the vLLM inference engine
- **SGLang Team** for the SGLang inference framework
- **AMD** for the Instinct MI355X GPUs and ROCm platform

