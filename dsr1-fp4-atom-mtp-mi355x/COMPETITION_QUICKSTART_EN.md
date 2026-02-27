# 🏆 Competition Quick Start Guide – DSR1 FP4 ATOM MTP (MI355X)

## 📑 Table of Contents

- [Objective](#objective)
- [Core Files](#core-files)
- [Quick Start (5 Steps)](#quick-start-5-steps)
- [Test Mode Comparison](#test-mode-comparison)
- [Evaluation Criteria](#evaluation-criteria)
- [Resource Links](#resource-links)

---

## Objective

Optimize inference performance of **DeepSeek-R1 MXFP4** using **ATOM with MTP (Medusa-style speculative decoding)** on AMD MI355X GPUs, surpass baseline while maintaining model accuracy.

**Image**: `rocm/atom:rocm7.2.0-ubuntu24.04-pytorch2.9-atom0.1.1`  
**Model**: `amd/DeepSeek-R1-0528-MXFP4`

---

## Core Files

| File | Purpose |
|------|--------|
| `dsr1-fp4-atom-mtp-mi355x/launch_atom_server.sh` | Launch ATOM OpenAI-compatible server (MTP) |
| `dsr1-fp4-atom-mtp-mi355x/dsr1_benchmark` | Run tests and submit results (binary) |
| `dsr1-fp4-atom-mtp-mi355x/all_conc_var.sh` | Multi-concurrency test environment variables |
| `dsr1-fp4-atom-mtp-mi355x/specific_conc_var.sh` | Single configuration test environment variables |
| `dsr1-fp4-atom-mtp-mi355x/bench_atom.py` | Optional: GSM8K evaluation via OpenAI API |

---

## Quick Start (5 Steps)

### 1️⃣ Prepare Working Directory (on Host)

```bash
mkdir -p ~/competition
cd ~/competition
git clone https://github.com/danielhua23/amdgpu_bounty_optimization.git
```

### 2️⃣ Launch Container (ATOM image)

```bash
docker run -it \
  --name atom-mtp-dev \
  --ipc=host --shm-size=16g --network=host \
  --privileged --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /nfsdata/hf_hub_cache-1/:/root/.cache/huggingface \
  -v ~/competition:/workspace \
  -e HF_TOKEN=your_huggingface_token_here \
  rocm/atom:rocm7.2.0-ubuntu24.04-pytorch2.9-atom0.1.1 \
  /bin/bash
```

Mount your workspace so scripts are available at `/workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x/` (or your repo path).

### 3️⃣ Compile Benchmark Binary (optional, for submit/perf/acc)

```bash
cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
g++ -std=c++17 -o dsr1_benchmark dsr1_benchmark.cpp -lcurl -pthread -O2
```

### 4️⃣ Launch ATOM Server

```bash
cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
source specific_conc_var.sh
bash launch_atom_server.sh
```

Server runs in foreground. Wait until health check passes: `curl http://0.0.0.0:8888/health`.

### 5️⃣ Run Tests (in another terminal)

```bash
docker exec -it atom-mtp-dev bash
cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
source specific_conc_var.sh

# Accuracy + performance + submit
./dsr1_benchmark submit "YourTeam"

# Accuracy only
./dsr1_benchmark acc

# Performance only (no submit)
./dsr1_benchmark perf
```

**Environment Variables** (from `specific_conc_var.sh`):

- Server: `MODEL`, `PORT`, `TP`, `EP_SIZE`, `DP_ATTENTION`
- Test: `ISL`, `OSL`, `CONC`, `RANDOM_RANGE_RATIO`, `NUM_PROMPTS`, `RESULT_FILENAME`

---

## Multi-Concurrency Batch Testing

```bash
cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
source all_conc_var.sh
bash launch_atom_server.sh
# In another terminal:
source all_conc_var.sh
./dsr1_benchmark submit "YourTeam" -isl 1024 -osl 1024
./dsr1_benchmark submit "YourTeam" -isl 1024 -osl 8192
./dsr1_benchmark submit "YourTeam" -isl 8192 -osl 1024
```

Results are submitted to the corresponding ATOM MTP leaderboards (URLs are in `dsr1_benchmark.cpp`).

---

## Test Mode Comparison

| Mode | Command | What Runs | Time (single config) |
|------|---------|-----------|----------------------|
| **submit** | `./dsr1_benchmark submit "Team"` | Accuracy + performance + submit | ~20–30 min |
| **submit -isl -osl** | `./dsr1_benchmark submit "Team" -isl 1024 -osl 1024` | Batch CONC + submit | ~2–3 h |
| **acc** | `./dsr1_benchmark acc` | Accuracy only | ~5–10 min |
| **perf** | `./dsr1_benchmark perf` | Accuracy + performance, no submit | ~20–30 min |

---

## Server Launch Details

The ATOM server is started with:

- `python3 -m atom.entrypoints.openai_server`
- Args: `--model`, `--server-port`, `-tp`, `--kv_cache_dtype fp8`, `--method mtp`
- Optional: `--max-model-len 10240` when ISL/OSL ≠ 1024/1024;
- Env: `AMDGCN_USE_BUFFER_OPS=1`, `OMP_NUM_THREADS=1`

Benchmark uses: `run_benchmark_serving` with `--backend vllm` and `--use-chat-template` (OpenAI-compatible API).

---

## Evaluation Criteria

- **Performance**: Throughput per GPU, E2E latency (median); compare to baseline.
- **Accuracy**: Must meet required GSM8K (or specified) threshold; otherwise performance run is not counted.

---

## Resource Links

- ATOM / ROCm Atom stack (refer to AMD/ROCm docs for image and usage).
- Leaderboards (replace with actual URLs if different):
  - ISL=1024, OSL=1024: `https://daniehua-dsr1-fp4-atom-mtp-isl1024osl1024.hf.space`
  - ISL=1024, OSL=8192: `https://daniehua-dsr1-fp4-atom-mtp-isl1024osl8192.hf.space`
  - ISL=8192, OSL=1024: `https://daniehua-dsr1-fp4-atom-mtp-isl8192osl1024.hf.space`

Good luck! 🚀
