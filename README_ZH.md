# AMD Instinct MI355X 性能优化赏金计划

[English](https://github.com/danielhua23/amdgpu_bounty_optimization/blob/main/README.md)
## 🎯 概述

欢迎参加 **AMD Instinct MI355X 性能优化赏金计划**！本竞赛挑战开发者在 AMD 最新的 Instinct MI355X GPU 上优化 LLM 推理性能，目标是超越基线性能。

## 🏆 竞赛规则
### 1. 模型列表
可从以下四个模型中选择进行优化，可全部或任选优化。每个模型对应一个 PR，供我们审核与评定。
- [amd/DeepSeek-R1-0528-MXFP4](https://huggingface.co/amd/DeepSeek-R1-0528-MXFP4)
- [amd/DeepSeek-R1-0528-mtp-mxfp4](https://huggingface.co/amd/DeepSeek-R1-0528-mtp-mxfp4)
- [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
- [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5)

### 2. 性能基准

- **目标**：您选择的模型在竞赛期间的表现必须超过**基线**

### 3. 提交要求

参与者须提交以下内容方可获得资格：

#### a) 代码链接
- GitHub 仓库链接（可为您 fork 的 vllm/sglang/...）
- 在您仓库的 PR，清晰说明以下 b)、c)、d) 以及您希望我们关注的内容

#### b) 排行榜结果
- 使用我们提供的脚本将结果提交到目标排行榜
- 基准与自动化说明见下文

#### c) 性能曲线图
- 提供每 GPU 吞吐量（y 轴）与端到端延迟（x 轴）的曲线
- 须包含该 ISL-OSL 配置下所有测试过的并发档位

#### d) 技术文档
提交须包含详细技术文档，说明：
1. **为什么**：从技术角度说明方案为何能取得更优性能
2. **如何**：完整复现步骤与操作说明
3. **做了什么**：具体优化、内核改动或算法改进

### 3. 获胜标准

- **方案须具备可合并性**：方案必须能够合并进主流框架 vllm、sglang 或其他指定仓库，否则视为无效。因此需要在 PR 可合并性与极致性能之间权衡。
- **评选流程**：我们将综合评估所有提交方案，优先考虑 PR 或方案更易被合并的参与者，再比较谁超越基线最多。

### 4. 领奖流程

一旦被选为获胜者：

1. **PR 提交**（2 周内）：
   - 等待我们向上游仓库发起 Pull Request，例如：
     - [vLLM GitHub](https://github.com/vllm-project/vllm)
     - [SGLang GitHub](https://github.com/sgl-project/sglang)
     - 其他仓库
   
2. **代码合并**（2 周内）：
   - 该 PR 须在 2 周内**合并**到主分支
   - 代码须满足项目质量与审查要求
   
3. **奖金发放**：
   - 合并成功后发放赏金
   
4. **备选**：
   - 若被选中的获胜者在 2 周内未能完成 PR 合并，我们将选择该配置下**下一个**超越基线的合格参与者

## 📊 基准测试配置
 
### 模型与后端
**重要说明：后端不限于 SGLang 与 vLLM，二者仅为示例，您可使用任何熟悉的框架。只要模型性能超越基线即可被接受。**
| 模型 | 后端 | 目录 | 排行榜 |
|-------|---------|-----------|-------------|
| **DeepSeek-R1 FP4** | SGLang | `dsr1-fp4-sglang-mi355x/` | https://daniehua-dsr1-fp4-sgl-isl*-osl*.hf.space |
| **DeepSeek-R1 MTP FP4** | SGLang | `dsr1-fp4mtp-sglang-mi355x/` | https://daniehua-dsr1-fp4mtp-sgl-isl*-osl*.hf.space |
| **GPT-OSS FP4 (120B)** | vLLM | `gptoss-fp4-vllm-mi355x/` | https://daniehua-gptoss-fp4-vllm-isl*-osl*.hf.space |
| **moonshotai/Kimi-K2.5** | vLLM | `kimik25-int4-vllm-mi355x/` | https://daniehua-kimik25-int4-vllm-isl*-osl*.hf.space |

### 测试配置

每个模型使用多组 **ISL（输入序列长度）** 与 **OSL（输出序列长度）** 组合进行测试：

| ISL | OSL | 描述 | DeepSeek-R1 并发 | GPT-OSS 并发 | Kimi-K2.5 并发 | DeepSeek-R1 MTP 并发 |
|-----|-----|-------------|------------------|--------------|----------------|----------------------|
| 1024 | 1024 | 标准短序列 | 4,8,32,64,128,256 | 4,8,16,32,64,256 | 4,8,16,32,64,128 | 8,16,32,64,128,256 |
| 1024 | 8192 | 长生成 | 4,8,16,32,64,128 | 4,8,16,32,64,256 | 4,8,16,32,64,128 | 8,16,32,64,128,256 |
| 8192 | 1024 | 长上下文 | 4,8,16,32,64,128 | 4,8,16,32,64,256 | 4,8,16,32,64,128 | 8,16,32,64,128,256 |

**CONC** = 最大并发请求数

## 💰 赏金结构

每个**模型**对应一个独立赏金：

### DeepSeek-R1 - 1 个赏金
- ISL=1024, OSL=1024
- ISL=1024, OSL=8192
- ISL=8192, OSL=1024

### GPT-OSS - 1 个赏金
- ISL=1024, OSL=1024
- ISL=1024, OSL=8192
- ISL=8192, OSL=1024

### Kimi-K2.5 - 1 个赏金
- ISL=1024, OSL=1024
- ISL=1024, OSL=8192
- ISL=8192, OSL=1024

### DeepSeek-R1 MTP - 1 个赏金
- ISL=1024, OSL=1024
- ISL=1024, OSL=8192
- ISL=8192, OSL=1024

**合计：4 个赏金**（金额待公布）

## 🚀 快速开始指南

### 针对 vLLM 优化 GPT-OSS
注意，并不是一定要用vLLM优化GPT-OSS模型，可选任一你熟悉的框架，但最好为主流框架

请参阅 [GPTOSS_COMPETITION_QUICKSTART.md](https://github.com/danielhua23/amdgpu_bounty_optimization/blob/main/gptoss-fp4-vllm-mi355x/GPTOSS_COMPETITION_QUICKSTART_EN.md)

### 针对 SGLang 优化 DeepSeek-R1
注意，并不是一定要用SGLang优化DeepSeek模型，可选任一你熟悉的框架，但最好为主流框架

请参阅 [COMPETITION_QUICKSTART.md](https://github.com/danielhua23/amdgpu_bounty_optimization/blob/main/dsr1-fp4-sglang-mi355x/COMPETITION_QUICKSTART_EN.md)

## 📈 性能指标

提交将按以下指标评估：

1. **吞吐量**（tokens/秒/GPU）— 越高越好
2. **延迟**（毫秒）— 越低越好
3. **交互性**（tokens/秒/用户）— 越高越好
4. **端到端延迟**（毫秒）— 越低越好


## 📋 提交清单

提交前请确认：

- [ ] 已使用提供的脚本完成基准测试
- [ ] 已将结果提交到排行榜（脚本中可自动化）
- [ ] 已绘制吞吐量 vs 延迟曲线，并包含所有 CONC 档位
- [ ] 已确认超过 50% 的曲线数据点高于基线（每个数据点对应一个 CONC 值）
- [ ] 已准备技术文档，说明：
  - [ ] 方案为何有效（技术细节）
  - [ ] 如何复现（步骤说明）
  - [ ] 做了哪些优化（代码与改动）
- [ ] 可在 2 周内向 vLLM/SGLang 等提交 PR

## 🛠️ 技术要求

### 硬件
- AMD Instinct MI355X GPU（8 卡）

## 🔗 重要链接

- **vLLM GitHub**：https://github.com/vllm-project/vllm
- **SGLang GitHub**：https://github.com/sgl-project/sglang
- **AMD ROCm**：https://rocm.docs.amd.com/

### 排行榜

#### DeepSeek-R1
- ISL=1024, OSL=1024：https://daniehua-dsr1-fp4-sgl-isl1024osl1024.hf.space
- ISL=1024, OSL=8192：https://daniehua-dsr1-fp4-sgl-isl1024osl8192.hf.space
- ISL=8192, OSL=1024：https://daniehua-dsr1-fp4-sgl-isl8192osl1024.hf.space

#### GPT-OSS
- ISL=1024, OSL=1024：https://daniehua-gptoss-fp4-vllm-isl1024osl1024.hf.space
- ISL=1024, OSL=8192：https://daniehua-gptoss-fp4-vllm-isl1024osl8192.hf.space
- ISL=8192, OSL=1024：https://daniehua-gptoss-fp4-vllm-isl8192osl1024.hf.space

#### DeepSeek-R1 MTP
- ISL=1024, OSL=1024：https://daniehua-dsr1-fp4mtp-sgl-isl1024-osl1024.hf.space
- ISL=1024, OSL=8192：https://daniehua-dsr1-fp4mtp-sgl-isl1024-osl8192.hf.space
- ISL=8192, OSL=1024：https://daniehua-dsr1-fp4mtp-sgl-isl8192-osl1024.hf.space

#### kimi-k2.5
- ISL=1024, OSL=1024：https://daniehua-kimik25-int4-vllm-isl1024-osl1024.hf.space
- ISL=1024, OSL=8192：https://daniehua-kimik25-int4-vllm-isl1024-osl8192.hf.space
- ISL=8192, OSL=1024：https://daniehua-kimik25-int4-vllm-isl8192-osl1024.hf.space

## 💡 成功提示

1. **性能分析**：用 ROCm 分析工具定位瓶颈
2. **内核优化**：重点优化高影响内核（attention、GEMM、量化）
3. **内存管理**：优化 GPU 内存与数据传输
4. **文档**：清晰的文档有助于 PR 被接纳
5. **社区**：参与 vLLM/SGLang 社区，借鉴最佳实践

## ❓ 常见问题

**问：可以同时参加多个赏金吗？**  
答：可以。每个模型×ISL×OSL 配置独立，您可为全部 4 个赏金提交方案。

**问：若多人同时超越基线如何决定？**  
答：优先选择曲线数据点中高于基线的数量更多的一方；若相同，则比较常规时间内吞吐比（throughput ratio）总和更高的一方。

**问：可以使用闭源或专有优化吗？**  
答：不可以。方案须开源且能合并到 vLLM/SGLang 等公共仓库。

**问：是否必须在所有 CONC 上都超越基线？**  
答：不需要。但须满足：曲线数据点中 ≥50% 高于基线，且所有数据点均不低于 AMD 现有的最佳性能，方可进入评选，再比较整体表现。

## 📞 支持与联系

如有疑问或问题：
- **技术问题**：请在 vLLM/SGLang 等仓库提交 issue
- **竞赛规则**：[联系方式待定]
- **排行榜问题**：[联系方式待定]

## 📅 时间表

- **竞赛开始**：[待定]
- **竞赛结束**：[待定]
- **PR 提交截止**：获胜者公布后 2 周内
- **奖金发放**：PR 成功合并后

## License

本基准与文档按 [LICENSE TBD] 提供。

## 致谢

- **vLLM 团队** 提供 vLLM 推理引擎
- **SGLang 团队** 提供 SGLang 推理框架
- **AMD** 提供 Instinct MI355X GPU 与 ROCm 平台
