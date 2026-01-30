# AMD Instinct MI355X InferenceMAX 赏金计划

## 🎯 项目概述

欢迎参加 **AMD Instinct MI355X InferenceMAX 赏金计划**！本竞赛挑战开发者在 AMD 最新的 Instinct MI355X GPU 上优化 LLM 推理性能，目标是使用 vLLM 和 SGLang 后端在 [InferenceMAX](https://inferencemax.semianalysis.com/) 基准测试中击败 NVIDIA B200 基线性能，并将您的优化贡献回开源社区。

## 🏆 竞赛规则

### 1. 性能基准

- **目标**：您的解决方案必须超过竞赛期间记录的**最佳 B200 基线**
- **基线更新**：B200 基线将在整个竞赛期间**动态更新**，以与 [InferenceMAX](https://inferencemax.semianalysis.com/) 保持一致
- **评估周期**：性能根据竞赛时间范围内的最佳 B200 结果进行衡量

### 2. 提交要求

参与者必须提交以下内容才有资格：

#### a) 排行榜结果
- 使用我们提供的脚本将您的结果提交到目标排行榜
- 请参阅下面的基准测试说明以了解自动化流程

#### b) 性能曲线图
- 提供类似于 [InferenceMAX](https://inferencemax.semianalysis.com/) 的吞吐量与延迟曲线图
- 必须包含特定 ISL-OSL 配置的所有测试并发级别
- **资格标准**：至少 **50% 的数据点**必须高于 B200 TRT-LLM 基线才能进入评选池

#### c) 技术文档
您的提交必须包含详细的技术文档，解释：
1. **为什么**：技术解释为什么您的解决方案能实现卓越性能
2. **如何**：完整的复现指南，包含逐步说明
3. **做了什么**：具体的优化、内核修改或算法改进

### 3. 获胜标准

- **选择流程**：我们将选择在常规竞赛时间内**第一个**击败 B200 基线的参与者（针对每个配置）
- **多个获胜者**：每个模型-ISL-OSL 配置都有自己的赏金（见下方赏金结构）

### 4. 领奖流程

一旦被选为获胜者：

1. **PR 提交**（2 周内）：
   - 向上游仓库提交 Pull Request：
     - **GPT-OSS (vLLM)**：[vLLM GitHub](https://github.com/vllm-project/vllm)
     - **DeepSeek-R1 (SGLang)**：[SGLang GitHub](https://github.com/sgl-project/sglang)
   
2. **代码合并**（2 周内）：
   - 您的 PR 必须在被选中后的 2 周内**合并**到主分支
   - 代码必须符合项目的质量标准和审查要求
   
3. **奖金发放**：
   - 成功合并后，您将收到赏金奖励
   
4. **备选方案**：
   - 如果被选中的获胜者在 2 周内未能合并其 PR，我们将选择该配置的**下一个**符合条件的参与者

## 📊 基准测试配置

### 模型与后端

| 模型 | 后端 | 目录 | 排行榜 |
|-------|---------|-----------|-------------|
| **DeepSeek-R1 FP4** | SGLang | `dsr1-fp4-sglang-mi355x/` | https://daniehua-dsr1-fp4-sgl-isl*-osl*.hf.space |
| **GPT-OSS FP4 (120B)** | vLLM | `gptoss-fp4-vllm-mi355x/` | https://daniehua-gptoss-fp4-vllm-isl*-osl*.hf.space |

### 测试配置

每个模型使用多个 **ISL（输入序列长度）** 和 **OSL（输出序列长度）** 组合进行测试：

| ISL | OSL | 描述 | DeepSeek-R1 并发值 | GPT-OSS 并发值 |
|-----|-----|-------------|------------------|--------------|
| 1024 | 1024 | 标准短序列 | 4,8,32,64,128,256 | 4,8,16,32,64,256 |
| 1024 | 8192 | 长生成 | 4,8,16,32,64,128 | 4,8,16,32,64,256 |
| 8192 | 1024 | 长上下文 | 4,8,16,32,64,128 | 4,8,16,32,64,256 |

**CONC** = 最大并发请求数

## 💰 赏金结构

每个 **模型 × ISL × OSL** 配置代表一个独立的赏金：

### DeepSeek-R1 (SGLang) - 3 个赏金
- ISL=1024, OSL=1024：**$X,XXX** *(待定)*
- ISL=1024, OSL=8192：**$X,XXX** *(待定)*
- ISL=8192, OSL=1024：**$X,XXX** *(待定)*

### GPT-OSS (vLLM) - 3 个赏金
- ISL=1024, OSL=1024：**$X,XXX** *(待定)*
- ISL=1024, OSL=8192：**$X,XXX** *(待定)*
- ISL=8192, OSL=1024：**$X,XXX** *(待定)*

**总计：6 个赏金**（金额待公布）

## 🚀 快速开始指南

### 针对 vLLM（GPT-OSS）优化

请跳转至 [GPTOSS_COMPETITION_QUICKSTART.md](https://github.com/danielhua23/amdgpu_inferencemax_bounty/blob/main/gptoss-fp4-vllm-mi355x/GPTOSS_COMPETITION_QUICKSTART_ZH.md)

### 针对 SGLang（DeepSeek-R1）优化

请跳转至 [COMPETITION_QUICKSTART.md](https://github.com/danielhua23/amdgpu_inferencemax_bounty/blob/main/dsr1-fp4-sglang-mi355x/COMPETITION_QUICKSTART_ZH.md)

## 📈 性能指标

您的提交将根据以下指标进行评估：

1. **吞吐量**（tokens/秒/GPU）- 越高越好
2. **延迟**（毫秒）- 越低越好
3. **交互性**（tokens/秒/用户）- 越高越好
4. **端到端延迟**（毫秒）- 越低越好

所有指标都与 [InferenceMAX](https://inferencemax.semianalysis.com/) 上的 NVIDIA B200 TRT 基线进行比较。

## 📋 提交清单

提交前，请确保您已完成：

- [ ] 使用提供的脚本运行基准测试
- [ ] 将结果提交到排行榜（脚本中自动完成）
- [ ] 绘制包含所有并发值的吞吐量与延迟曲线图
- [ ] 验证 >50% 的折线数据点超过 B200 基线
- [ ] 准备技术文档，解释：
  - [ ] 为什么您的解决方案有效（技术细节）
  - [ ] 如何复现结果（逐步说明）
  - [ ] 做了哪些优化（代码更改）
- [ ] 准备在 2 周内向 vLLM/SGLang 提交 PR

## 🛠️ 技术要求

### 硬件
- AMD Instinct MI355X GPU（8 个 GPU）

### 模型
- **DeepSeek-R1**：`amd/DeepSeek-R1-0528-MXFP4-Preview`
- **GPT-OSS**：`openai/gpt-oss-120b`（FP4 量化）

## 📚 仓库结构

```
amdgpu_inferencemax_bounty/
├── dsr1-fp4-sglang-mi355x/             # DeepSeek-R1 + SGLang
│   ├── dsr1_benchmark                  # 基准测试
│   └── COMPETITION_QUICKSTART.md       # 竞赛指南
│
├── gptoss-fp4-vllm-mi355x/              # GPT-OSS + vLLM
│   ├── gptoss_benchmark                 # 基准测试
│   └── GPTOSS_COMPETITION_QUICKSTART.md # 竞赛指南

```

## 🔗 重要链接

- **InferenceMAX 平台**：https://inferencemax.semianalysis.com/
- **vLLM GitHub**：https://github.com/vllm-project/vllm
- **SGLang GitHub**：https://github.com/sgl-project/sglang
- **AMD ROCm**：https://rocm.docs.amd.com/

### 排行榜

#### DeepSeek-R1 (SGLang)
- ISL=1024, OSL=1024：https://daniehua-dsr1-fp4-sgl-isl1024osl1024.hf.space
- ISL=1024, OSL=8192：https://daniehua-dsr1-fp4-sgl-isl1024osl8192.hf.space
- ISL=8192, OSL=1024：https://daniehua-dsr1-fp4-sgl-isl8192osl1024.hf.space

#### GPT-OSS (vLLM)
- ISL=1024, OSL=1024：https://daniehua-gptoss-fp4-vllm-isl1024osl1024.hf.space
- ISL=1024, OSL=8192：https://daniehua-gptoss-fp4-vllm-isl1024osl8192.hf.space
- ISL=8192, OSL=1024：https://daniehua-gptoss-fp4-vllm-isl8192osl1024.hf.space

## 💡 成功提示

1. **研究 InferenceMAX**：查看 [InferenceMAX](https://inferencemax.semianalysis.com/) 上当前的 B200 性能曲线以了解目标
2. **性能分析**：使用 ROCm 性能分析工具识别瓶颈
3. **内核优化**：专注于高影响力的内核（attention、GEMM、量化）
4. **内存管理**：优化 GPU 内存使用和数据传输
5. **文档编写**：良好的文档会提高您的 PR 被接受的机会
6. **社区参与**：与 vLLM/SGLang 社区互动以了解最佳实践

## ❓ 常见问题

**问：我可以参加多个赏金吗？**  
答：可以！每个模型×ISL×OSL 配置都是独立的。您可以为全部 6 个赏金提交解决方案。

**问：如果多个人同时击败 B200 怎么办？**  
答：选择折线点数量比B200高的更多的那个，如果打平，则选择throughput ratio总和更大的那个

**问：我可以使用专有优化吗？**  
答：不可以。您的解决方案必须是开源的，并且可以合并到 vLLM/SGLang 公共仓库中。

**问：如果我的 PR 被拒绝怎么办？**  
答：您有 2 周时间解决反馈并使其合并。请与维护者密切合作。

**问：我需要在所有并发值上都击败 B200 吗？**  
答：不需要。您需要 ≥50% 的折线点高于 B200且所有折线点不能低于AMD在InferenceMax上现有性能点 才有资格，然后我们比较整体性能。

## 📞 支持与联系

如有问题或疑问：
- **技术问题**：在 vLLM/SGLang 仓库中开启 issue
- **竞赛规则**：[联系方式待定]
- **排行榜问题**：[联系方式待定]

## 📅 时间表

- **竞赛开始**：[待定]
- **竞赛结束**：[待定]
- **B200 基线更新**：持续进行（与 InferenceMAX 保持一致）
- **PR 提交截止日期**：获胜者选定后 2 周
- **奖金分发**：PR 成功合并后

## 🎖️ 认可

获胜者将：
- 列在官方排行榜上
- 在 vLLM/SGLang 发布说明中获得致谢
- 在 AMD 和 SemiAnalysis 的传播中得到展示

---

**祝您好运，优化愉快！🚀**

*让我们一起推动 LLM 推理性能的边界！*

---

## 许可证

本基准测试套件和文档在 [许可证待定] 下提供。

## 致谢

- **SemiAnalysis** 提供 InferenceMAX 平台
- **vLLM 团队** 提供 vLLM 推理引擎
- **SGLang 团队** 提供 SGLang 推理框架
- **AMD** 提供 Instinct MI355X GPU 和 ROCm 平台

