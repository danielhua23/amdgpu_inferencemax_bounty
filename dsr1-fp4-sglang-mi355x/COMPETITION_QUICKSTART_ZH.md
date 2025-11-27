# 🏆 竞赛快速开始指南

## 📑 目录

- [目标](#目标)
- [📌 重要说明](#-重要说明)
- [核心文件](#核心文件)
- [快速开始（5 步走）](#快速开始5-步走)
  - [1️⃣ 准备工作目录（在宿主机）](#1️⃣-准备工作目录在宿主机)
  - [2️⃣ 启动开发容器](#2️⃣-启动开发容器)
  - [3️⃣ 在容器内安装最新版本的可编辑 SGLang](#3️⃣-在容器内安装最新版本的可编辑-sglang)
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

在 AMD MI355X GPU 上优化 SGLang 推理性能，同时保持模型准确性。

## 📌 重要说明

本竞赛的测试基准**对齐 [InferenceMAX](https://github.com/semianalysis/InferenceMAX)** 仓库的 AMD MI355X 测试配置，并会随着 InferenceMAX 的更新而同步更新。

## 核心文件

| 文件 | 用途 |
|------|------|
| `amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x/launch_sglang_server.sh` | 启动 SGLang 服务器 |
| `amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x/dsr1_benchmark` | 运行测试并提交结果（二进制文件）|
| `amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x/all_conc_var.sh` | 多并发测试环境变量配置 |
| `amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x/specific_conc_var.sh` | 单配置测试环境变量配置 |

## 快速开始（5 步走）

### 1️⃣ 准备工作目录（在宿主机）

```bash
# 在宿主机上创建工作目录
mkdir -p ~/competition
cd ~/competition

# 克隆 SGLang（你将在此基础上优化）
git clone https://github.com/sgl-project/sglang.git

# 克隆 AITER（AMD GPU算子库）
git clone --recursive https://github.com/ROCm/aiter.git

# 克隆脚本文件所在仓库
git clone https://github.com/danielhua23/amdgpu_inferencemax_bounty.git
```

### 2️⃣ 启动开发容器

**注意**：请将 `HF_TOKEN` 替换为你的 Hugging Face Token。

```bash
docker run -it \
  --name sglang-dev \
  --ipc=host --shm-size=16g --network=host \
  --privileged --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /nfsdata/hf_hub_cache-1/:/root/.cache/huggingface \
  -v ~/competition:/workspace \
  -v ~/competition/aiter:/workspace/aiter \
  -v ~/competition/sglang:/workspace/sglang \
  -e HF_TOKEN=your_huggingface_token_here \
  rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi35x-20250915 \
  /bin/bash
```

**挂载说明**：
- 宿主机的 `~/competition/*` → 容器内 `/workspace/*`
- 在宿主机修改代码，容器内立即生效（反之亦然）
- 测试脚本位于 `/workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x/` 目录

### 3️⃣ 在容器内安装最新版本的可编辑 SGLang

> refer to https://docs.sglang.io/platforms/amd_gpu.html

```bash
# 卸载容器内部已有的sglang相关库
pip uninstall aiter
pip uninstall sglang
pip uninstall sgl-kernel
# 进入aiter目录
cd /workspace/aiter
python3 setup.py develop
```
verify if newest aiter is installed
```bash
root@mi355:/workspace# pip list | grep aiter
aiter                             0.1.7.post3.dev39+g1f5b378dc        /workspace/aiter
```
lets continue install sgl-kernel
```bash
# 进入 SGLang 目录
cd /workspace/sglang

# Compile sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python setup_rocm.py install
```
verify if newest sgl-kernel is installed
```bash
root@mi355:/workspace# pip list | grep sgl-kernel
sgl-kernel                        0.3.18
```
lets continue install sglang python pkg
```bash
# Install sglang python package
cd ..
rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_hip]"

# verify
python -c "import sglang; print(sglang.__file__)"
# expect ouput: /workspace/sglang/python/sglang/__init__.py
```

### 4️⃣ 示例: 修改代码后如何recompile

```bash
# 在容器内或宿主机（VS Code）都可以编辑
# 示例：优化调度器
cd /workspace/sglang
vim python/sglang/srt/managers/scheduler.py

# 如果修改了 C++/CUDA/HIP 代码，需要重新编译：
cd sgl-kernel
rm -rf build/
pip uninstall sgl-kernel
python setup_rocm.py install
```

### 5️⃣ 测试优化效果

#### 推荐工作流程 ⭐

```
开发阶段（快速迭代）
  ↓
1. 单配置测试并提交（方式 1）
   - 用 submit 模式测试单个配置（~20分钟）
   - 自动提交到 Leaderboard，实时查看排名
  ↓
2. 多并发批量测试并提交（方式 2）
   - 用 submit 模式测试所有 CONC（~2小时/ISL-OSL）
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
cd /workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x

# 1. 加载环境变量（无需手动 export）
source specific_conc_var.sh

# 2. 启动 SGLang 服务器（首次启动需要 20+ 分钟 JIT 编译）
bash launch_sglang_server.sh

# 等待服务器就绪后（看到 "Uvicorn running..."），运行测试

# 3. 推荐：直接测试并提交（~20-30分钟）⭐
./dsr1_benchmark submit "YourTeam"

# 可选：如果只想快速验证准确性（~5-10分钟）
./dsr1_benchmark acc

# 可选：如果只想测试性能但不提交（~20-30分钟）
./dsr1_benchmark perf
```

**环境变量说明**：`specific_conc_var.sh` 会设置：
- `MODEL`, `PORT`, `TP`（服务器配置）
- `ISL`, `OSL`, `CONC`（测试配置）
- `RANDOM_RANGE_RATIO`, `NUM_PROMPTS`, `RESULT_FILENAME`（测试参数）

**提示**：所有 `.sh` 脚本都位于 `/workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x/` 目录

---

#### 方式 2: 多并发批量测试（一键测试所有 CONC）🚀

**适用场景**：批量测试所有 CONC 值并提交到 Leaderboard

**只需 3 条命令，自动测试所有 15 个配置并提交！⭐**

```bash
cd /workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x

# 1. 加载环境变量（无需手动 export）
source all_conc_var.sh

# 2. 启动 SGLang 服务器（首次启动需要 20+ 分钟 JIT 编译）
bash launch_sglang_server.sh

# 等待服务器就绪后（看到 "Uvicorn running..."），运行以下命令

# ========== 推荐：直接测试并提交（一步到位）========== 

# 提交 ISL=1024, OSL=1024 的所有结果（自动跑 CONC=4,8,16,32,64，~2小时）
./dsr1_benchmark submit "YourTeam" -isl 1024 -osl 1024

# 提交 ISL=1024, OSL=8192 的所有结果（自动跑 CONC=4,8,16,32,64，~2小时）
./dsr1_benchmark submit "YourTeam" -isl 1024 -osl 8192

# 提交 ISL=8192, OSL=1024 的所有结果（自动跑 CONC=4,8,16,32,64，~2小时）
./dsr1_benchmark submit "YourTeam" -isl 8192 -osl 1024

# ========== 可选：如果只想测试不提交，用 perf 模式 ========== 

# 测试 ISL=1024, OSL=1024（不提交，~2小时）
./dsr1_benchmark perf -isl 1024 -osl 1024

# 测试 ISL=1024, OSL=8192（不提交，~2小时）
./dsr1_benchmark perf -isl 1024 -osl 8192

# 测试 ISL=8192, OSL=1024（不提交，~2小时）
./dsr1_benchmark perf -isl 8192 -osl 1024
```

**结果会自动提交到对应的 Leaderboard**：
- ISL=1024, OSL=1024 → https://daniehua-dsr1-fp4-sgl-isl1024osl1024.hf.space
- ISL=1024, OSL=8192 → https://daniehua-dsr1-fp4-sgl-isl1024osl8192.hf.space
- ISL=8192, OSL=1024 → https://daniehua-dsr1-fp4-sgl-isl8192osl1024.hf.space

**提交内容**：每个 CONC 配置会独立提交，包含：
- 队伍名称 + CONC 值
- **MI355X vs B200 直接对比**：E2E、吞吐量、性能比率
- 准确性指标：bits_per_byte, byte_perplexity, word_perplexity


## 测试模式对比

| 模式 | 命令示例 | 运行内容 | 耗时（单配置）| 使用场景 |
|------|---------|---------|-------------|---------|
| **submit** ⭐ | `./dsr1_benchmark submit "Team"` | 准确性 + 性能 + 提交 | ~20-30分钟 | **推荐：一步到位，实时查看排名** |
| **submit -isl -osl** ⭐ | `./dsr1_benchmark submit "Team" -isl 1024 -osl 1024` | 自动测试 5 个 CONC + 提交 | ~2小时 | **推荐：批量测试并提交** |
| **acc** | `./dsr1_benchmark acc` | 仅准确性测试 | ~5-10分钟 | 可选：快速验证准确性 |
| **perf** | `./dsr1_benchmark perf` | 准确性 + 性能（不提交）| ~20-30分钟 | 可选：测试性能但不提交 |
| **perf -isl -osl** | `./dsr1_benchmark perf -isl 1024 -osl 1024` | 自动测试 5 个 CONC（不提交）| ~2小时 | 可选：批量测试但不提交 |

## 两种测试方式对比

| 方式 | 推荐命令 | 配置数 | 耗时估算 | 推荐场景 |
|------|---------|-------|---------|---------|
| **方式 1: 单配置** ⭐ | `./dsr1_benchmark submit "Team"` | 1个 | ~20分钟 | **开发阶段快速迭代** |
| **方式 2: 多并发** ⭐ | `./dsr1_benchmark submit "Team" -isl 1024 -osl 1024` | 5个 | ~2小时 | **批量测试所有CONC** |

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
- bits_per_byte ≤ 0.4485 × 1.03 = **0.4620**
- byte_perplexity ≤ 1.3646 × 1.03 = **1.4055**
- word_perplexity ≤ 3.2522 × 1.03 = **3.3498**

❌ 超出范围会立即终止测试，不运行性能基准

### B200 Baseline 对比 📊

**自动对比功能**：每个结果 JSON 自动包含 NVIDIA B200 (会随着inferenceMax上记录的B200性能数据周期性同步更新) 的 baseline 数据和性能比率！

**性能比率解读**：
- `tput_per_gpu_ratio_vs_b200_1126 > 1.0` = MI355X 吞吐量更高 ✅
- `median_e2e_ratio_vs_b200_1126 < 1.0` = MI355X 延迟更低 ✅

详见结果 JSON 中的 `b200_baseline_nv1126` 字段。

## 优化方向建议

### 1. Kernel 优化 ⚡
- Attention kernel
- MoE (Mixture of Experts) kernel  
- 量化 kernel (FP4/FP8)

### 2. 调度优化 📊
- Batch scheduler
- Prefill/decode 切换策略
- KV cache 管理

### 3. 内存优化 💾
- 显存分配策略
- 减少内存碎片
- Paged attention

### 4. ROCm 特定优化 🔧
- AMD GPU 特性利用
- HIP/ROCm API 优化
- AITER 异步迭代器

## 开发技巧

### 查看日志

```bash
# 实时查看服务器日志
tail -f /tmp/sglang-server-*.log

# 过滤错误
tail -f /tmp/sglang-server-*.log | grep -i error
```

### 多并发批量测试（推荐）⭐

```bash
# 1. 加载环境变量
cd /workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x
source all_conc_var.sh

# 2. 启动 SGLang 服务器
bash launch_sglang_server.sh

# 等待服务器就绪后（查看日志 "Uvicorn running..."），运行以下命令

# ========== 推荐：直接测试并提交（一步到位）========== 

# 提交 ISL=1024, OSL=1024 的所有结果（自动测试 CONC=4,8,16,32,64，~2小时）
./dsr1_benchmark submit "YourTeam" -isl 1024 -osl 1024

# 提交 ISL=1024, OSL=8192 的所有结果（自动测试 CONC=4,8,16,32,64，~2小时）
./dsr1_benchmark submit "YourTeam" -isl 1024 -osl 8192

# 提交 ISL=8192, OSL=1024 的所有结果（自动测试 CONC=4,8,16,32,64，~2小时）
./dsr1_benchmark submit "YourTeam" -isl 8192 -osl 1024
```

**每条命令会自动**：
- ✅ 测试 5 个 CONC 值（4, 8, 16, 32, 64）
- ✅ 运行准确性 + 性能测试
- ✅ 自动提交到对应的 ISL-OSL Leaderboard
- ✅ 保存所有结果到独立目录
- ✅ 生成汇总报告

**Leaderboard 自动路由**：
- `ISL=1024, OSL=1024` → https://daniehua-dsr1-fp4-sgl-isl1024osl1024.hf.space
- `ISL=1024, OSL=8192` → https://daniehua-dsr1-fp4-sgl-isl1024osl8192.hf.space
- `ISL=8192, OSL=1024` → https://daniehua-dsr1-fp4-sgl-isl8192osl1024.hf.space

**结果输出示例**：
```
============================================
Multi-Concurrency Testing Mode
============================================
ISL: 1024
OSL: 1024
Mode: submit
CONC values: 4, 8, 16, 32, 64
Team: YourTeam
Leaderboard: https://daniehua-dsr1-fp4-sgl-isl1024osl1024.hf.space
============================================

Results directory: batch_isl1024_osl1024_20251125_150000

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
Total tests: 5
Passed: 5
Failed: 0

Results saved in: batch_isl1024_osl1024_20251125_150000/
============================================
```

**开发阶段快速验证**：
```bash
# 推荐：直接测试并提交（一步到位）⭐
./dsr1_benchmark submit "YourTeam" -isl 1024 -osl 1024

# 可选：只测试准确性（快速验证）
./dsr1_benchmark acc -isl 1024 -osl 1024

# 可选：完整测试但不提交
./dsr1_benchmark perf -isl 1024 -osl 1024
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
cd /workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x

# 加载环境变量
source all_conc_var.sh

# 启动服务器
bash launch_sglang_server.sh
```

服务器会在前台运行，日志直接输出到终端。

### Q: 修改了 C++ 代码但没生效？

需要重新编译：

```bash
cd /workspace/sglang/sgl-kernel
rm -rf build/
pip uninstall sgl-kernel
python setup_rocm.py install
```

### Q: 多并发测试中途失败了怎么办？

测试会继续运行剩余 CONC 配置，最后生成完整报告。失败的配置会标记为 "FAILED"。

查看失败原因：
```bash
# 查看汇总
cat batch_isl*_osl*/summary.txt

# 查看服务器日志
tail -f /tmp/sglang-server-*.log
```

### Q: 如何只测试特定的 CONC 值？

使用单配置模式：

```bash
cd /workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x

# 1. 编辑 specific_conc_var.sh，修改 CONC 值
vim specific_conc_var.sh  # 修改 CONC=64

# 2. 加载环境变量
source specific_conc_var.sh

# 3. 推荐：直接测试并提交 ⭐
./dsr1_benchmark submit "YourTeam"
```

或者直接手动设置：
```bash
cd /workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x
source specific_conc_var.sh
export CONC=64  # 覆盖默认值，只测试 CONC=64
export NUM_PROMPTS=3200

# 推荐：直接提交
./dsr1_benchmark submit "YourTeam"

# 可选：只测试不提交
./dsr1_benchmark perf
```

### Q: 测试需要多长时间？

**单配置测试**：
- **submit 模式**: ~20-30分钟 ⭐ **推荐：一步到位**
- **acc 模式**: ~5-10分钟（可选：仅验证准确性）
- **perf 模式**: ~20-30分钟（可选：测试但不提交）

**多并发测试（每个 ISL-OSL 组合，5 个 CONC 值）**：
- **submit -isl -osl**: ~20-30分钟/CONC × 5 = **~2小时** ⭐ **推荐**
- **acc -isl -osl**: ~5-10分钟/CONC × 5 = **25-50分钟**（可选）
- **perf -isl -osl**: ~20-30分钟/CONC × 5 = **~2小时**（可选）

**全部 3 个 ISL-OSL 组合**（15 个配置）：
- **submit 模式**: ~2小时 × 3 = **~6小时** ⭐

**推荐工作流** 🎯：
1. **开发阶段**：单配置 `submit "YourTeam"` 快速迭代（~20分钟/次）
   - 立即看到 Leaderboard 排名，快速验证优化效果
2. **批量提交**：多并发 `submit "YourTeam" -isl -osl` 提交所有配置（~2小时/组合）
   - 一次性完成测试和提交，可在夜间运行

💡 **为什么直接用 submit？**
- ✅ 一步到位，无需先 perf 再 submit
- ✅ 实时查看排名，立即知道优化效果
- ✅ 节省时间，避免重复运行



## 推荐的工作流程

```
第1轮：熟悉基线
  ├─ 运行基线测试：./dsr1_benchmark submit "YourTeam"
  ├─ 了解 SGLang 架构
  └─ 查看 Leaderboard 基线性能

第2轮：低风险优化
  ├─ 调整超参数
  ├─ 优化配置
  └─ 快速验证：./dsr1_benchmark submit "YourTeam"（~20分钟）

第3轮：AMD GPU Kernel 优化
  ├─ Profile 找瓶颈
  ├─ 优化关键 kernel
  └─ 实时对比：./dsr1_benchmark submit "YourTeam"，查看 Leaderboard

第4轮：系统优化
  ├─ 内存管理
  ├─ 通信优化
  └─ 端到端调优，每次优化后立即提交验证

第5轮：批量提交
  ├─ 测试所有 ISL-OSL 组合
  ├─ ./dsr1_benchmark submit "YourTeam" -isl 1024 -osl 1024
  ├─ ./dsr1_benchmark submit "YourTeam" -isl 1024 -osl 8192
  └─ ./dsr1_benchmark submit "YourTeam" -isl 8192 -osl 1024
```

**关键优势**：每次优化后直接 submit，实时查看 Leaderboard 排名，快速迭代！

## 资源链接

- 📖 [InferenceMAX 官方仓库](https://github.com/semianalysis/InferenceMAX) - 测试基准参考
- 🔧 [SGLang GitHub](https://github.com/sgl-project/sglang) - 推理框架
- 📊 Leaderboards:
  - [ISL=1024, OSL=1024](https://daniehua-dsr1-fp4-sgl-isl1024osl1024.hf.space)
  - [ISL=1024, OSL=8192](https://daniehua-dsr1-fp4-sgl-isl1024osl8192.hf.space)
  - [ISL=8192, OSL=1024](https://daniehua-dsr1-fp4-sgl-isl8192osl1024.hf.space)


**祝参赛顺利！🚀**

记住：
- **直接用 submit mode**：一步到位，实时查看排名 ⭐
- **性能重要，准确性更重要！** 所有优化必须通过准确性验证
- **快速迭代**：每次优化后立即 submit，立即看到效果

