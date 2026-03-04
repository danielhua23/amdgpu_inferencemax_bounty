# GPT-OSS FP4 on AMD Atom – Competition Quick Start Guide

## Table of Contents

- [Objective](#objective)
- [Important Notice](#important-notice)
- [Core Files](#core-files)
- [Quick Start (5 Steps)](#quick-start-5-steps)
- [Test Mode Comparison](#test-mode-comparison)
- [Evaluation Criteria](#evaluation-criteria)
- [Resource Links](#resource-links)

---

## Objective

Run and benchmark **GPT-OSS 120B FP4** on [AMD Atom](https://github.com/ROCm/ATOM)  on AMD MI355X GPUs while meeting accuracy. Supported cases: **ISL=8192, OSL=1024 (8k/1k)** with CONC = 4, 32, 128;

## Important Notice

- **Image**: this guide doc is an example to get you quickly start using `rocm/atom:rocm7.1.1-ubuntu24.04-pytorch2.9-atom0.1.1-MI350x`
- **Model**: `openai/gpt-oss-120b`
- **Framework**: AMD Atom
- **Benchmark case**: uses **ISL=8192, OSL=1024**; CONC = 4, 32, 128.
- **Something you cannot change**:
  - model: openai/gpt-oss-120b
  - ISL: 8192
  - OSL: 1024
  - CONC: 4, 32, 128
  - NUM_PROMPTS: CONC * 10
  - basline data in gptoss_benchmark.cpp
  - others, if you're not sure, let's actively talk on discord

## Core Files

| File | Purpose |
|------|--------|
| `launch_atom_server.sh` | Start Atom OpenAI-compatible server (model, TP, EP, block-size, max-model-len) |
| `gptoss_benchmark` | Run tests and submit (compile from `gptoss_benchmark.cpp` if needed) |
| `all_conc_var.sh` | Multi-concurrency test env vars |
| `specific_conc_var.sh` | Single-config test env vars |

## Quick Start (5 Steps)

### 1. Prepare working directory (host)

```bash
mkdir -p ~/competition
cd ~/competition
git clone https://github.com/danielhua23/amdgpu_bounty_optimization.git
# Clone AMD ATOM
git clone https://github.com/ROCm/ATOM.git
# Clone AITER (AMD GPU operator library)
git clone --recursive https://github.com/ROCm/aiter.git
```

### 2. Launch container (Atom image)

Use the Atom image to quickstart:

```bash
docker run -it \
  --name atom-gptoss \
  --ipc=host --shm-size=16g --network=host \
  --privileged --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  -v /nfsdata/hf_hub_cache-1/:/root/.cache/huggingface \
  -v ~/competition:/workspace \
  -v ~/competition/aiter:/workspace/aiter \
  -v ~/competition/ATOM:/workspace/ATOM \
  -e HF_TOKEN=your_huggingface_token_here \
  rocm/atom:rocm7.1.1-ubuntu24.04-pytorch2.9-atom0.1.1-MI350x \
  /bin/bash
```
You can develop at `/app/ATOM` and `/app/aiter-test`, after launch container, we jump to **4. Compile benchmark binary**.

Adjust `/nfsdata/hf_hub_cache-1/` and `HF_TOKEN` as needed.

### 3. (Optional) Install Latest Editable ATOM in Container
**Note:** if you are not using above ATOM image but referring the installing steps from https://github.com/ROCm/ATOM using rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0 for installing, you can refer this step to install latest editable ATOM in container

```bash
# Install AITER
cd /workspace/aiter
python3 setup.py develop
# Verify
python -c "import aiter; print(aiter.__file__)"
```

Verify AITER installation:
```bash
root@mi355:/workspace# pip list | grep aiter
aiter                             0.1.7.post3.dev39+g1f5b378dc        /workspace/aiter
```

Install ATOM:
```bash
# Enter ATOM directory
cd /workspace/ATOM
# may need ninja
pip install ninja
# Upgrade pip and install atom
pip install .

# Verify
python -c "import atom; print(atom.__file__)"
# Expected output: /workspace/ATOM/atom/__init__.py
```
> Note: I won't say you 100% get atom successfully compiled since aiter is the dependency of atom, and both atom and aiter are fast-iterated projects, they are changing in the fly. So you may encounter some issues. But I would say they work for me at 33e0aac6d7f5f2e505bcfe18a22d68110bfb3331 for atom and cbbdc5066d3ab0f25f022bb5d79cccc8465b7d1c for aiter

### 4. Compile benchmark binary

If `gptoss_benchmark` is not provided as a binary:

```bash
cd /workspace/amdgpu_bounty_optimization/gptoss-fp4-atom-mi355x
g++ -std=c++17 -o gptoss_benchmark gptoss_benchmark.cpp -lcurl -pthread -O2
```

### 5. Start Atom server (terminal 1)

```bash
cd /workspace/amdgpu_bounty_optimization/gptoss-fp4-atom-mi355x
source specific_conc_var.sh   # or all_conc_var.sh
bash launch_atom_server.sh
```

Wait until the server is ready (e.g. health at `http://0.0.0.0:8888/health`).

### 6. Run benchmark / submit (terminal 2)

**Single config (e.g. CONC=4):**

```bash
docker exec -it atom-gptoss bash
cd /workspace/amdgpu_bounty_optimization/gptoss-fp4-atom-mi355x
source specific_conc_var.sh
./gptoss_benchmark submit "YourTeam"
```

**Results are submitted to Leaderboard**:
- ISL=8192, OSL=1024 → https://daniehua-gptoss-fp4-isl8192osl1024.hf.space

**Multi-concurrency (CONC=4,32,128) for 8k/1k:**

```bash
cd /workspace/amdgpu_bounty_optimization/gptoss-fp4-atom-mi355x
source all_conc_var.sh
./gptoss_benchmark submit "YourTeam" -isl 8192 -osl 1024
```

## Test Mode Comparison

| Mode | Command | What runs |
|------|---------|-----------|
| **submit** | `./gptoss_benchmark submit "Team"` | Accuracy + performance + submit |
| **submit -isl -osl** | `./gptoss_benchmark submit "Team" -isl 8192 -osl 1024` | Batch CONC=4,32,128 + submit |
| **acc** | `./gptoss_benchmark acc` | Accuracy only |
| **perf** | `./gptoss_benchmark perf` | Accuracy + performance, no submit |

## Evaluation Criteria

- **Performance**: Throughput per GPU, E2E latency (median); compare to baseline as in result JSON.
- **Accuracy**: `gsm8k_metric > 0.38`.

Good luck.
