# 🏆 GPT-OSS 竞赛快速开始指南

## 📑 目录

- [目标](#目标)
- [📌 重要说明](#-重要说明)
- [核心文件](#核心文件)
- [快速开始（5 步走）](#快速开始5-步走)
  - [1️⃣ 准备工作目录（在宿主机）](#1️⃣-准备工作目录在宿主机)
  - [2️⃣ 启动开发容器](#2️⃣-启动开发容器)
  - [3️⃣ 在容器内安装最新版本的可编辑 vLLM](#3️⃣-在容器内安装最新版本的可编辑-vllm)
  - [4️⃣ 示例: 修改代码后如何recompile](#4️⃣-示例-修改代码后如何recompile)
  - [5️⃣ 测试优化效果](#5️⃣-测试优化效果)
- [测试模式对比](#测试模式对比)
- [两种测试方式对比](#两种测试方式对比)
- [评分标准](#评分标准)
  - [性能指标（主要）](#性能指标主要)
  - [准确性要求（必须满足）](#准确性要求必须满足)
  - [B200 Baseline 对比 📊](#b200-baseline-对比-)
- [优化方向建议](#优化方向建议)
- [开发技巧](#开发技巧)
- [常见问题](#常见问题)
- [推荐的工作流程](#推荐的工作流程)
- [资源链接](#资源链接)

---

## 目标

在 AMD MI355X GPU 上优化 vLLM 推理性能（GPT-OSS 120B FP4 模型），同时保持模型准确性。

## 📌 重要说明

本竞赛的测试基准**对齐 [InferenceMAX](https://github.com/InferenceMAX/InferenceMAX)** 仓库的 AMD MI355X 测试配置，并会随着 InferenceMAX 的更新而同步更新。

**模型差异**：
- **模型**：`openai/gpt-oss-120b` (FP4 量化)
- **框架**：vLLM
- **特性**：使用 AMD AITER 优化的 MoE 和 attention kernels

## 核心文件

| 文件 | 用途 |
|------|------|
| `amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x/launch_vllm_server.sh` | 启动 vLLM 服务器 |
| `amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x/gptoss_benchmark` | 运行测试并提交结果（二进制文件）|
| `amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x/all_conc_var.sh` | 多并发测试环境变量配置 |
| `amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x/specific_conc_var.sh` | 单配置测试环境变量配置 |

## 快速开始（5 步走）

### 1️⃣ 准备工作目录（在宿主机）

```bash
# 在宿主机上创建工作目录
mkdir -p ~/competition
cd ~/competition

# 克隆 vLLM（你将在此基础上优化）
git clone https://github.com/vllm-project/vllm.git

# 克隆 AITER（AMD GPU算子库）
git clone --recursive https://github.com/ROCm/aiter.git

# 克隆脚本文件所在仓库
git clone https://github.com/danielhua23/amdgpu_inferencemax_bounty.git
```

### 2️⃣ 启动开发容器

**注意**：请将 `HF_TOKEN` 替换为你的 Hugging Face Token。

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

**挂载说明**：
- 宿主机的 `~/competition/*` → 容器内 `/workspace/*`
- 在宿主机修改代码，容器内立即生效（反之亦然）
- 测试脚本位于 `/workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x/` 目录

### 3️⃣ 在容器内安装最新版本的可编辑 vLLM

> refer to https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#build-wheel-from-source

```bash
# 卸载容器内部已有的相关库
pip uninstall -y aiter vllm

# 安装 AITER
cd /workspace/aiter
python3 setup.py develop
```

验证 AITER 安装：
```bash
root@mi355:/workspace# pip list | grep aiter
aiter                             0.1.7.post3.dev39+g1f5b378dc        /workspace/aiter
```

安装 vLLM：
```bash
# 进入 vLLM 目录
cd /workspace/vllm

# 安装依赖
pip install --upgrade pip
pip install -r requirements/rocm.txt

# 安装 vLLM（可编辑模式）
pip install --upgrade numba \
    scipy \
    huggingface-hub[cli,hf_transfer] \
    setuptools_scm
pip install "numpy<2"

# Build vLLM for MI GPU
export PYTORCH_ROCM_ARCH="gfx942;gfx950"
python3 setup.py develop

# 验证
python -c "import vllm; print(vllm.__file__)"
# 期望输出: /workspace/vllm/vllm/__init__.py
```

>note: you might meet some error like : ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. transformers 4.56.2 requires huggingface-hub<1.0,>=0.34.0, but you have huggingface-hub 1.1.5 which is incompatible. vllm 0.11.2.dev346+g18523b87f.rocm702 requires numba==0.61.2, but you have numba 0.62.1 which is incompatible. **Just try pip install tranformers -U**

### 4️⃣ 示例: 修改代码后如何recompile

```bash
# 在容器内或宿主机（VS Code）都可以编辑
# 示例：优化调度器
cd /workspace/vllm
vim vllm/engine/llm_engine.py

# 如果修改了 Python 代码，无需重新编译（可编辑模式自动生效）

# 如果修改了 C++/CUDA/HIP 扩展，需要重新编译：
cd /workspace/vllm
rm -r vllm/*.so
rm -r ./build
pip uninstall -y vllm
python3 setup.py develop
```

### 5️⃣ 测试优化效果

#### 推荐工作流程 ⭐

```
开发阶段（快速迭代）
  ↓
1. 单配置测试并提交（方式 1）
   - 用 submit 模式测试单个配置（~15-20分钟）
   - 自动提交到 Leaderboard，实时查看排名
  ↓
2. 多并发批量测试并提交（方式 2）
   - 用 submit 模式测试所有 CONC（~1-2小时/ISL-OSL）
   - 自动提交所有结果
  ↓
完成！实时查看 Leaderboard 排名 🎉
```

**为什么推荐直接用 submit mode？**
- ✅ **一步到位**：submit = 准确性测试 + 性能测试 + 自动提交
- ✅ **实时反馈**：立即看到 Leaderboard 排名，快速迭代
- ✅ **节省时间**：无需先 perf 再 submit，直接提交即可

---

#### 方式 1: 单配置测试（快速验证）⚡

**适用场景**：开发阶段快速验证单个配置的性能

```bash
cd /workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x

# 1. 加载环境变量（无需手动 export）
source specific_conc_var.sh

# 2. 启动 vLLM 服务器（首次启动需要 20+ 分钟 JIT 编译）
bash launch_vllm_server.sh

# 等待服务器就绪后（看到 "Uvicorn running..."），运行测试

# 3. 推荐：直接测试并提交（~15-20分钟）⭐
./gptoss_benchmark submit "YourTeam"

# 可选：如果只想快速验证准确性（~5-10分钟）
./gptoss_benchmark acc

# 可选：如果只想测试性能但不提交（~15-20分钟）
./gptoss_benchmark perf
```

**环境变量说明**：`specific_conc_var.sh` 会设置：
- `MODEL`, `PORT`, `TP`（服务器配置）
- `ISL`, `OSL`, `CONC`（测试配置）
- `MAX_MODEL_LEN`, `RANDOM_RANGE_RATIO`, `NUM_PROMPTS`, `RESULT_FILENAME`（测试参数）

**提示**：所有 `.sh` 脚本都位于 `/workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x/` 目录

---

#### 方式 2: 多并发批量测试（一键测试所有 CONC）🚀

**适用场景**：批量测试所有 CONC 值并提交到 Leaderboard

**只需 3 条命令，自动测试所有配置并提交！⭐**

```bash
cd /workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x

# 1. 加载环境变量（无需手动 export）
source all_conc_var.sh

# 2. 启动 vLLM 服务器（首次启动需要 20+ 分钟 JIT 编译）
bash launch_vllm_server.sh

# 等待服务器就绪后（看到 "Uvicorn running..."），运行以下命令

# ========== 推荐：直接测试并提交（一步到位）========== 

# 提交 ISL=1024, OSL=1024 的所有结果（自动跑 CONC=4,8,16，~1小时）
./gptoss_benchmark submit "YourTeam" -isl 1024 -osl 1024

# 提交 ISL=1024, OSL=8192 的所有结果（自动跑 CONC=4,8,16，~1小时）
./gptoss_benchmark submit "YourTeam" -isl 1024 -osl 8192

# 提交 ISL=8192, OSL=1024 的所有结果（自动跑 CONC=4,8，~40分钟）
./gptoss_benchmark submit "YourTeam" -isl 8192 -osl 1024

# ========== 可选：如果只想测试不提交，用 perf 模式 ========== 

# 测试 ISL=1024, OSL=1024（不提交，~1小时）
./gptoss_benchmark perf -isl 1024 -osl 1024

# 测试 ISL=1024, OSL=8192（不提交，~1小时）
./gptoss_benchmark perf -isl 1024 -osl 8192

# 测试 ISL=8192, OSL=1024（不提交，~40分钟）
./gptoss_benchmark perf -isl 8192 -osl 1024
```

**结果会自动提交到对应的 Leaderboard**：
- ISL=1024, OSL=1024 → https://daniehua-gptoss-fp4-vllm-isl1024osl1024.hf.space
- ISL=1024, OSL=8192 → https://daniehua-gptoss-fp4-vllm-isl1024osl8192.hf.space
- ISL=8192, OSL=1024 → https://daniehua-gptoss-fp4-vllm-isl8192osl1024.hf.space

**提交内容**：每个 CONC 配置会独立提交，包含：
- 队伍名称 + CONC 值
- **MI355X vs B200 直接对比**：E2E、吞吐量、性能比率
- 准确性指标：bits_per_byte, byte_perplexity, word_perplexity

**CONC 范围说明**：
- ISL=1024, OSL=1024: CONC=4,8,16（3个配置）
- ISL=1024, OSL=8192: CONC=4,8,16（3个配置）
- ISL=8192, OSL=1024: CONC=4,8（2个配置）⚠️ 注意：8192-1024 只测试 CONC=4,8

## 测试模式对比

| 模式 | 命令示例 | 运行内容 | 耗时（单配置）| 使用场景 |
|------|---------|---------|-------------|---------|
| **submit** ⭐ | `./gptoss_benchmark submit "Team"` | 准确性 + 性能 + 提交 | ~15-20分钟 | **推荐：一步到位，实时查看排名** |
| **submit -isl -osl** ⭐ | `./gptoss_benchmark submit "Team" -isl 1024 -osl 1024` | 自动测试 3 个 CONC + 提交 | ~1小时 | **推荐：批量测试并提交** |
| **acc** | `./gptoss_benchmark acc` | 仅准确性测试 | ~5-10分钟 | 可选：快速验证准确性 |
| **perf** | `./gptoss_benchmark perf` | 准确性 + 性能（不提交）| ~15-20分钟 | 可选：测试性能但不提交 |
| **perf -isl -osl** | `./gptoss_benchmark perf -isl 1024 -osl 1024` | 自动测试 3 个 CONC（不提交）| ~1小时 | 可选：批量测试但不提交 |

## 两种测试方式对比

| 方式 | 推荐命令 | 配置数 | 耗时估算 | 推荐场景 |
|------|---------|-------|---------|---------|
| **方式 1: 单配置** ⭐ | `./gptoss_benchmark submit "Team"` | 1个 | ~15-20分钟 | **开发阶段快速迭代** |
| **方式 2: 多并发** ⭐ | `./gptoss_benchmark submit "Team" -isl 1024 -osl 1024` | 3个 | ~1小时 | **批量测试所有CONC** |

**推荐工作流程** 🎯：
1. **开发阶段**：使用**方式 1**（单配置 + submit）快速迭代，实时查看 Leaderboard
2. **批量提交**：使用**方式 2**（多并发 + submit）一次性测试并提交所有配置

**为什么直接用 submit？**
- ✅ submit = 准确性测试 + 性能测试 + 自动提交（一步到位）
- ✅ 实时查看 Leaderboard 排名，立即知道优化效果
- ✅ 节省时间，无需先 perf 再 submit

## 评分标准

### 性能指标（主要）

- **Throughput per GPU** (`tput_per_gpu`) - 权重最高 🏅
  - 单GPU归一化吞吐量 = `total_token_throughput / 8`
  - 与 B200 baseline 直接对比
- **E2E (median)** (ms) - 端到端延迟中位数
  - 与 B200 baseline 直接对比

### 准确性要求（必须满足）

所有指标必须在基线 ± 3% 范围内：
- bits_per_byte ≤ 2.0558 × 1.03 = **2.1175**
- byte_perplexity ≤ 4.1577 × 1.03 = **4.2824**
- word_perplexity ≤ 222.7893 × 1.03 = **229.4730**

❌ 超出范围会立即终止测试，不运行性能基准

### B200 Baseline 对比 📊

**自动对比功能**：每个结果 JSON 自动包含 NVIDIA B200 (会随着inferenceMax上记录的B200性能数据周期性同步更新) 的 baseline 数据和性能比率！

**性能比率解读**：
- `tput_per_gpu_ratio_vs_b200_1126 > 1.0` = MI355X 吞吐量更高 ✅
- `median_e2e_ratio_vs_b200_1126 < 1.0` = MI355X 延迟更低 ✅

详见结果 JSON 中的 `b200_baseline_nv1126` 字段。

## 优化方向建议

### 1. Kernel 优化 ⚡
- Attention kernel（Flash Attention、PagedAttention）
- MoE (Mixture of Experts) kernel - GPT-OSS 的关键！
- 量化 kernel (FP4/MXFP4)

### 2. 调度优化 📊
- Continuous batching
- Prefill/decode 切换策略
- KV cache 管理

### 3. 内存优化 💾
- 显存分配策略
- Paged attention
- CUDA graph / compilation config 优化

### 4. ROCm 特定优化 🔧
- AMD GPU 特性利用
- HIP/ROCm API 优化
- AITER 异步迭代器

### 5. vLLM 特定优化 🚀
- Async scheduling
- Block manager
- Compilation config 调优

## 开发技巧

### 查看日志

```bash
# 实时查看服务器日志
tail -f /tmp/vllm-server-*.log

# 过滤错误
tail -f /tmp/vllm-server-*.log | grep -i error
```

### 多并发批量测试（推荐）⭐

```bash
# 1. 加载环境变量
cd /workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x
source all_conc_var.sh

# 2. 启动 vLLM 服务器
bash launch_vllm_server.sh

# 等待服务器就绪后（查看日志 "Uvicorn running..."），运行以下命令

# ========== 推荐：直接测试并提交（一步到位）========== 

# 提交 ISL=1024, OSL=1024 的所有结果（自动测试 CONC=4,8,16，~1小时）
./gptoss_benchmark submit "YourTeam" -isl 1024 -osl 1024

# 提交 ISL=1024, OSL=8192 的所有结果（自动测试 CONC=4,8,16，~1小时）
./gptoss_benchmark submit "YourTeam" -isl 1024 -osl 8192

# 提交 ISL=8192, OSL=1024 的所有结果（自动测试 CONC=4,8，~40分钟）
./gptoss_benchmark submit "YourTeam" -isl 8192 -osl 1024
```

**每条命令会自动**：
- ✅ 测试对应的 CONC 值（1024-1024和1024-8192: 3个，8192-1024: 2个）
- ✅ 运行准确性 + 性能测试
- ✅ 自动提交到对应的 ISL-OSL Leaderboard
- ✅ 保存所有结果到独立目录
- ✅ 生成汇总报告

**Leaderboard 自动路由**：
- `ISL=1024, OSL=1024` → https://daniehua-gptoss-fp4-vllm-isl1024osl1024.hf.space
- `ISL=1024, OSL=8192` → https://daniehua-gptoss-fp4-vllm-isl1024osl8192.hf.space
- `ISL=8192, OSL=1024` → https://daniehua-gptoss-fp4-vllm-isl8192osl1024.hf.space

**结果输出示例**：
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
... (运行测试) ...
✓ CONC=4: PASSED (180s)

============================================
Testing CONC=8
============================================
... (继续测试其他 CONC 值) ...

============================================
Multi-Concurrency Test Complete!
============================================
Total tests: 3
Passed: 3
Failed: 0

Results saved in: batch_isl1024_osl1024_20251127_150000/
============================================
```

**开发阶段快速验证**：
```bash
# 推荐：直接测试并提交（一步到位）⭐
./gptoss_benchmark submit "YourTeam" -isl 1024 -osl 1024

# 可选：只测试准确性（快速验证）
./gptoss_benchmark acc -isl 1024 -osl 1024

# 可选：完整测试但不提交
./gptoss_benchmark perf -isl 1024 -osl 1024
```

## 常见问题

### Q: 准确性验证失败怎么办？

```
ERROR: Accuracy validation FAILED!
bits_per_byte: 6.5000 > 5.1500
```

**解决**：你的优化影响了模型质量，需要调整算法或参数

### Q: 如何只启动服务器不运行测试？

```bash
cd /workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x

# 加载环境变量
source all_conc_var.sh

# 启动服务器
bash launch_vllm_server.sh
```

服务器会在前台运行，日志直接输出到终端。

### Q: 修改了 C++ 代码但没生效？

需要重新编译：

```bash
cd /workspace/vllm
rm -rf build/
pip uninstall -y vllm
VLLM_TARGET_DEVICE=rocm python3 setup.py develop
```

### Q: 多并发测试中途失败了怎么办？

测试会继续运行剩余 CONC 配置，最后生成完整报告。失败的配置会标记为 "FAILED"。

查看失败原因：
```bash
# 查看汇总
cat batch_isl*_osl*/summary.txt

# 查看服务器日志
tail -f /tmp/vllm-server-*.log
```

### Q: 如何只测试特定的 CONC 值？

使用单配置模式：

```bash
cd /workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x

# 1. 编辑 specific_conc_var.sh，修改 CONC 值
vim specific_conc_var.sh  # 修改 CONC=16

# 2. 加载环境变量
source specific_conc_var.sh

# 3. 推荐：直接测试并提交 ⭐
./gptoss_benchmark submit "YourTeam"
```

或者直接手动设置：
```bash
cd /workspace/amdgpu_inferencemax_bounty/gptoss-fp4-vllm-mi355x
source specific_conc_var.sh
export CONC=16  # 覆盖默认值，只测试 CONC=16
export NUM_PROMPTS=160  # GPT-OSS: CONC * 10

# 推荐：直接提交
./gptoss_benchmark submit "YourTeam"

# 可选：只测试不提交
./gptoss_benchmark perf
```

### Q: 测试需要多长时间？

**单配置测试**：
- **submit 模式**: ~15-20分钟 ⭐ **推荐：一步到位**
- **acc 模式**: ~5-10分钟（可选：仅验证准确性）
- **perf 模式**: ~15-20分钟（可选：测试但不提交）

**多并发测试（每个 ISL-OSL 组合）**：
- **ISL=1024, OSL=1024 (3个CONC)**: ~15-20分钟/CONC × 3 = **~1小时** ⭐
- **ISL=1024, OSL=8192 (3个CONC)**: ~15-20分钟/CONC × 3 = **~1小时** ⭐
- **ISL=8192, OSL=1024 (2个CONC)**: ~15-20分钟/CONC × 2 = **~40分钟** ⭐

**全部 3 个 ISL-OSL 组合**（8 个配置）：
- **submit 模式**: ~1小时 + ~1小时 + ~40分钟 = **~2.5-3小时** ⭐

**推荐工作流** 🎯：
1. **开发阶段**：单配置 `submit "YourTeam"` 快速迭代（~15-20分钟/次）
   - 立即看到 Leaderboard 排名，快速验证优化效果
2. **批量提交**：多并发 `submit "YourTeam" -isl -osl` 提交所有配置（~1小时/组合）
   - 一次性完成测试和提交，可在夜间运行

💡 **为什么直接用 submit？**
- ✅ 一步到位，无需先 perf 再 submit
- ✅ 实时查看排名，立即知道优化效果
- ✅ 节省时间，避免重复运行

### Q: GPT-OSS 和 DeepSeek-R1 有什么区别？

**主要区别**：

| 特性 | GPT-OSS | DeepSeek-R1 |
|------|---------|------------|
| 模型大小 | 120B | ~670B |
| 架构 | MoE (Mixture of Experts) | Dense |
| 框架 | vLLM | SGLang |
| CONC 范围 | 4-16 (8192-1024: 4-8) | 4-64 |
| NUM_PROMPTS | CONC × 10 | CONC × 50 (1024-1024/8192-1024) / CONC × 20 (1024-8192) |
| 优化重点 | MoE kernel, vLLM 调度 | 长上下文, chunked prefill |

**优化建议**：
- GPT-OSS：重点优化 MoE kernel（AITER 提供的 A16W4 fused MoE）
- 调整 compilation config 中的 compile_sizes 和 cudagraph_capture_sizes


## 推荐的工作流程

```
第1轮：熟悉基线
  ├─ 运行基线测试：./gptoss_benchmark submit "YourTeam"
  ├─ 了解 vLLM 架构
  └─ 查看 Leaderboard 基线性能

第2轮：低风险优化
  ├─ 调整 compilation config
  ├─ 优化 GPU memory utilization
  └─ 快速验证：./gptoss_benchmark submit "YourTeam"（~15-20分钟）

第3轮：AMD GPU Kernel 优化
  ├─ Profile 找瓶颈
  ├─ 优化 MoE kernel (关键！)
  └─ 实时对比：./gptoss_benchmark submit "YourTeam"，查看 Leaderboard

第4轮：系统优化
  ├─ Async scheduling
  ├─ Block manager
  └─ 端到端调优，每次优化后立即提交验证

第5轮：批量提交
  ├─ 测试所有 ISL-OSL 组合
  ├─ ./gptoss_benchmark submit "YourTeam" -isl 1024 -osl 1024
  ├─ ./gptoss_benchmark submit "YourTeam" -isl 1024 -osl 8192
  └─ ./gptoss_benchmark submit "YourTeam" -isl 8192 -osl 1024
```

**关键优势**：每次优化后直接 submit，实时查看 Leaderboard 排名，快速迭代！

## 资源链接

- 📖 [InferenceMAX 官方仓库](https://github.com/semianalysis/InferenceMAX) - 测试基准参考
- 🔧 [vLLM GitHub](https://github.com/vllm-project/vllm) - 推理框架
- 🔧 [AITER GitHub](https://github.com/ROCm/aiter) - AMD GPU 算子库
- 📊 Leaderboards:
  - [ISL=1024, OSL=1024](https://daniehua-gptoss-fp4-vllm-isl1024osl1024.hf.space)
  - [ISL=1024, OSL=8192](https://daniehua-gptoss-fp4-vllm-isl1024osl8192.hf.space)
  - [ISL=8192, OSL=1024](https://daniehua-gptoss-fp4-vllm-isl8192osl1024.hf.space)


**祝参赛顺利！🚀**

记住：
- **直接用 submit mode**：一步到位，实时查看排名 ⭐
- **性能重要，准确性更重要！** 所有优化必须通过准确性验证
- **快速迭代**：每次优化后立即 submit，立即看到效果
- **重点优化 MoE kernel**：GPT-OSS 是 MoE 模型，MoE kernel 性能至关重要！

