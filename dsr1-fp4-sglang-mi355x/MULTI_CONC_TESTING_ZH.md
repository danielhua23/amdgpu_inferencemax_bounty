# 🚀 多并发测试 - 快速指南

## 📑 目录

- [⚡ 极简用法（推荐）](#-极简用法推荐)
- [📊 Leaderboard 自动路由](#-leaderboard-自动路由)
  - [提交数据格式](#提交数据格式)
- [🔧 命令详解](#-命令详解)
  - [基本语法](#基本语法)
  - [参数说明](#参数说明)
  - [示例用法](#示例用法)
- [🎯 自动执行的操作](#-自动执行的操作)
- [📁 结果输出](#-结果输出)
  - [JSON 结果文件内容](#json-结果文件内容)
- [⏱️ 耗时估算](#️-耗时估算)
- [🆚 对比：多并发模式 vs 单配置模式](#-对比多并发模式-vs-单配置模式)
- [❓ 常见问题](#-常见问题)
  - [Q: 可以只测试部分 CONC 值吗？](#q-可以只测试部分-conc-值吗)
  - [Q: 测试中途可以中断吗？](#q-测试中途可以中断吗)
  - [Q: 如何重新测试失败的 CONC？](#q-如何重新测试失败的-conc)
  - [Q: 多个 ISL-OSL 测试可以并行运行吗？](#q-多个-isl-osl-测试可以并行运行吗)
- [🎉 快速开始示例](#-快速开始示例)
- [📚 相关文档](#-相关文档)

---

## ⚡ 极简用法（推荐）

只需 **4 条命令** 完成所有测试并提交！

```bash
cd /workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x

# 1. 启动服务器
export MODEL="amd/DeepSeek-R1-0528-MXFP4-Preview"
export PORT=8888
export TP=8
bash launch_sglang_server.sh &

# 等待服务器就绪后（看到 "Uvicorn running..."），运行以下命令

# 2-4. 测试 3 个 ISL-OSL 组合（每个自动测试 5 个 CONC 值）
./dsr1_benchmark submit "YourTeam" -isl 1024 -osl 1024
./dsr1_benchmark submit "YourTeam" -isl 1024 -osl 8192
./dsr1_benchmark submit "YourTeam" -isl 8192 -osl 1024
```

**完成！** 所有 15 个配置（3 ISL-OSL × 5 CONC）已测试并提交到对应的 Leaderboard。

---

## 📊 Leaderboard 自动路由

每个 ISL-OSL 组合自动提交到专属 Leaderboard：

| ISL | OSL | Leaderboard URL |
|-----|-----|----------------|
| 1024 | 1024 | https://daniehua-dsr1-fp4-sgl-isl1024osl1024.hf.space |
| 1024 | 8192 | https://daniehua-dsr1-fp4-sgl-isl1024osl8192.hf.space |
| 8192 | 1024 | https://daniehua-dsr1-fp4-sgl-isl8192osl1024.hf.space |

### 提交数据格式

每次提交包含以下字段（按顺序）：

1. **Team Name** - 队伍名称
2. **CONC** - 并发数 (4, 8, 16, 32, 或 64)
3. **MI355X E2E (median)** (ms) - MI355X 端到端延迟中位数
4. **MI355X Throughput per GPU** (tokens/s) - MI355X 单GPU吞吐量
5. **B200 E2E (median)** (ms) - B200 Baseline 端到端延迟
6. **B200 Throughput per GPU** (tokens/s) - B200 Baseline 单GPU吞吐量
7. **E2E Ratio** - MI355X/B200 延迟比率（越小越好）
8. **Throughput Ratio** - MI355X/B200 吞吐量比率（越大越好）
9. **bits_per_byte** - WikiText 准确性指标
10. **byte_perplexity** - WikiText 准确性指标
11. **word_perplexity** - WikiText 准确性指标

**重要说明**：
- **E2E 使用中位数 (median)** (`median_e2el_ms`)：更稳定，不受极端值影响
- **Throughput per GPU** (`tput_per_gpu = total_token_throughput / 8`)：归一化到单GPU性能
- **自动包含 B200 Baseline**：直接对比 MI355X vs B200 性能
- **性能比率解读**：
  - `Throughput Ratio > 1.0` = MI355X 更快 ✅
  - `E2E Ratio < 1.0` = MI355X 延迟更低 ✅

**示例**：
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

## 🔧 命令详解

### 基本语法

```bash
cd /workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x
./dsr1_benchmark <mode> [team_name] -isl <value> -osl <value>
```

### 参数说明

- **`<mode>`**: 测试模式
  - `acc` - 仅测试准确性（快速验证）
  - `perf` - 准确性 + 性能测试
  - `submit <team>` - 完整测试并提交到 Leaderboard

- **`-isl <value>`**: Input Sequence Length（1024 或 8192）
- **`-osl <value>`**: Output Sequence Length（1024 或 8192）

### 示例用法

```bash
cd /workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x

# 快速验证准确性（不提交）
./dsr1_benchmark acc -isl 1024 -osl 1024

# 完整测试（不提交）
./dsr1_benchmark perf -isl 1024 -osl 1024

# 完整测试并提交
./dsr1_benchmark submit "MyTeam" -isl 1024 -osl 1024
```

---

## 🎯 自动执行的操作

当你运行 `-isl -osl` 模式时，脚本会自动：

1. ✅ **循环测试 5 个 CONC 值**：4, 8, 16, 32, 64
2. ✅ **每个 CONC 运行**：
   - 准确性测试（WikiText）
   - 准确性验证（基线 ±3%）
   - 性能测试（如果模式是 perf/submit）
3. ✅ **自动计算 NUM_PROMPTS**：
   - OSL=8192: `NUM_PROMPTS = CONC × 20`
   - 其他: `NUM_PROMPTS = CONC × 50`
4. ✅ **保存所有结果**：独立目录 `batch_isl{ISL}_osl{OSL}_{timestamp}/`
5. ✅ **提交到 Leaderboard**（submit 模式）：自动路由到对应的 ISL-OSL Leaderboard
6. ✅ **生成汇总报告**：`summary.txt`

---

## 📁 结果输出

```
batch_isl1024_osl1024_20251125_150000/
├── summary.txt                           # 汇总报告
├── result_isl1024_osl1024_conc4.json     # CONC=4 结果
├── result_isl1024_osl1024_conc8.json     # CONC=8 结果
├── result_isl1024_osl1024_conc16.json    # CONC=16 结果
├── result_isl1024_osl1024_conc32.json    # CONC=32 结果
└── result_isl1024_osl1024_conc64.json    # CONC=64 结果
```

### JSON 结果文件内容

每个 JSON 文件包含以下信息：

**1. 性能指标（SGLang 汇总统计）**
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

**2. Benchmark 参数配置**
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

**3. B200 Baseline 对比** 🆚
```json
{
  "b200_baseline_nv1126": {
    "b200_median_e2e_1126": 10407,
    "b200_tput_pergpu_1126": 344.564
  },
  "tput_per_gpu_ratio_vs_b200_1126": 2.9523,   // >1.0 = MI355X更快 ✅
  "median_e2e_ratio_vs_b200_1126": 1.5349      // <1.0 = MI355X更快
}
```

**4. 准确性指标**
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

**重要说明**：
- ✅ **只保存汇总统计**，不包含每个请求的详细数组（减少文件大小）
- 📊 **自动包含 B200 baseline** 和性能比率，方便对比
- 🎯 **包含完整参数配置**，便于复现实验

**汇总报告示例**：
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

## ⏱️ 耗时估算

| 模式 | 单个 ISL-OSL 组合 | 全部 3 个组合 |
|------|------------------|--------------|
| **acc** | ~30-50 分钟 | ~1.5-2.5 小时 |
| **perf** | ~1.5-2.5 小时 | ~5-7.5 小时 |
| **submit** | ~1.5-2.5 小时 | ~5-7.5 小时 |


---

## 🆚 对比：多并发模式 vs 单配置模式

| 特性 | 多并发模式 `-isl -osl` | 单配置模式 |
|------|----------------------|-----------|
| 命令数（15个配置） | **3 条** | **15 条** |
| CONC 设置 | **自动循环** 4,8,16,32,64 | 手动设置每个 |
| NUM_PROMPTS | **自动计算** | 手动设置每个 |
| 结果组织 | **自动分组**（按 ISL-OSL） | 分散 |
| Leaderboard | **自动路由**到对应 URL | 统一 URL |
| 适用场景 | ⭐ **最终提交** | 快速验证单个配置 |

---

## ❓ 常见问题

### Q: 可以只测试部分 CONC 值吗？

不行。多并发模式固定测试 5 个 CONC 值（4,8,16,32,64）。如果需要自定义，请使用单配置模式。

### Q: 测试中途可以中断吗？

可以。按 `Ctrl+C` 中断。已完成的 CONC 结果会保存，汇总报告会标记未完成的测试。

### Q: 如何重新测试失败的 CONC？

重新运行整个 ISL-OSL 组合，或使用单配置模式单独测试失败的 CONC。

### Q: 多个 ISL-OSL 测试可以并行运行吗？

**不推荐**。所有测试共享同一个 SGLang 服务器。并行运行会导致端口冲突。建议顺序执行。

---

## 🎉 快速开始示例

```bash
# 完整流程（3个ISL-OSL组合，共15个配置）
cd /workspace/amdgpu_inferencemax_bounty/dsr1-fp4-sglang-mi355x

# 1. 启动服务器
export MODEL="amd/DeepSeek-R1-0528-MXFP4-Preview"
export PORT=8888
export TP=8
bash launch_sglang_server.sh &

# 等待服务器就绪（~20-30分钟，首次启动）
# 查看日志: tail -f /tmp/sglang-server-*.log

# 2. 测试并提交
./dsr1_benchmark submit "MyAwesomeTeam" -isl 1024 -osl 1024
./dsr1_benchmark submit "MyAwesomeTeam" -isl 1024 -osl 8192
./dsr1_benchmark submit "MyAwesomeTeam" -isl 8192 -osl 1024

# 3. 查看结果
cat batch_isl*/summary.txt

# 4. 检查 Leaderboard
# https://daniehua-dsr1-fp4-sgl-isl1024osl1024.hf.space
# https://daniehua-dsr1-fp4-sgl-isl1024osl8192.hf.space
# https://daniehua-dsr1-fp4-sgl-isl8192osl1024.hf.space
```

**搞定！** 🚀

---

## 📚 相关文档

- [竞赛快速开始指南](./COMPETITION_QUICKSTART.md)
- [测试模式详解](./BENCHMARK_MODES_README.md)
- [项目总览](./PROJECT_SUMMARY.md)

