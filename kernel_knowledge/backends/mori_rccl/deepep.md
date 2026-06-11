---
title: DeepEP on AMD — ROCm port and UCCL-EP portable all-to-all
kind: backend
backend: mori
gens: [gfx942]
dtypes: [bf16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/deepseek-ai/DeepEP
  - https://github.com/ROCm/DeepEP
  - https://github.com/uccl-project/uccl
  - https://arxiv.org/html/2512.19849v1
---

# DeepEP on AMD (ROCm port) + UCCL-EP

## TL;DR
**DeepEP** (`deepseek-ai/DeepEP`) is DeepSeek's NVIDIA-native expert-parallel all-to-all library
(NVSHMEM + IBGDA). Two routes bring it to Instinct: **`ROCm/DeepEP`** (AMD's official port on rocSHMEM,
MI300X/MI308X, CX-7) and **UCCL-EP** (`uccl-project/uccl`, a *portable* drop-in DeepEP replacement that
runs on AMD GPUs + non-NVIDIA NICs). On AMD, MoRI-EP is the first-party SOTA ([mori_ep.md](mori_ep.md));
DeepEP-on-ROCm / UCCL-EP are the **portable / framework-compat** options — reach for them when you need
the upstream DeepEP API unchanged or heterogeneous GPU/NIC support. `operator: moe_dispatch_combine`.

## Concepts — what DeepEP provides
High-throughput **and** low-latency all-to-all GPU kernels for MoE **dispatch** (tokens→experts) and
**combine** (experts→tokens), with FP8 dispatch / BF16 combine and near-zero SM occupation. Two kernel
families:
- **normal (high-throughput)**: bulk dispatch/combine for prefill / high-concurrency decode
  (`test_intranode.py`, `test_internode.py`).
- **low-latency**: small-batch decode path that overlaps comm with compute (`test_low_latency.py`).

This **normal vs low-latency** split is the same distinction sglang exposes via `--enable-deepep-moe`
(DeepEP dispatch mode picked by batch regime) and that MoRI-EP mirrors with its kernel-type enum.

## The ROCm port (`ROCm/DeepEP`)
| requirement | value |
|---|---|
| GPU | MI308X / MI300X (gfx942); more archs later |
| ROCm | 6.3.4 |
| dependency | **rocSHMEM** (all internode RDMA dispatch/combine) |
| interconnect | xGMI intranode, RDMA internode; one node = 8 GPUs |
| NIC (`--nic`) | `cx7` (Mellanox CX-7, default), `thor2` (Broadcom), `io` (AMD Pensando AI NIC) |
| API | unified Python API **identical to upstream** deepseek-ai/DeepEP |

Build (no MPI): build rocSHMEM with `-DUSE_EXTERNAL_MPI=OFF`, then
`python3 setup.py --variant rocm --nic <NIC_TYPE> build develop --user`. Low-latency tests need
`ROCSHMEM_MAX_NUM_CONTEXTS=144` and `ROCSHMEM_HEAP_SIZE=2147483648` (DeepSeek-size models);
internode normal uses `ROCSHMEM_MAX_NUM_CONTEXTS=64`.

**Differences vs upstream**: NVIDIA-only optimizations (PTX loads) are **not mirrored** — the AMD path
focuses on **correctness + baseline overlap** and may lag NVIDIA in advanced tuning; implementation may
differ from the DeepSeek-V3 paper. The README ships no measured bandwidth/latency numbers.

## UCCL-EP (portable drop-in)
`uccl-project/uccl` (UCCL-EP) has the **same interface and functionality as DeepEP** and enables
GPU-initiated token-level comm across **heterogeneous** GPUs (NVIDIA, AMD) and NICs (EFA, Broadcom, CX-7) —
a **drop-in replacement with no code changes**. Motivation: DeepEP / ROCm-DeepEP assume specific GPU↔NIC
pairings (the ROCm port is effectively *AMD GPU + NVIDIA NIC*); UCCL-EP breaks that coupling while keeping
IBGDA-level performance.
- Build for MI300X: `bash build.sh roc7 ep --install` (ROCm 7 wheel), or `python setup.py install`.
- Arch foot-gun: a `CUDA error: invalid device function` means arch auto-detect failed → set
  `TORCH_CUDA_ARCH_LIST=gfx942` (MI300X/MI325X) or `gfx950` (MI355X) at compile time.
- Ecosystem: used by **AMD Primus** training and pulled into **AMD TheRock** (UCCL-Tran/EP/P2P).

## Measured perf (vendor / paper, version-tagged)
- UCCL-EP paper (arXiv 2512.19849): on **AMD + Broadcom**, UCCL-EP reaches **comparable performance to
  original DeepEP on NVIDIA-only**; **+up to 40%** SGLang serving throughput (NVIDIA+EFA); **+up to 45%**
  DeepSeek-V3 training throughput on a **16-node AMD+Broadcom** cluster (Primus/Megatron-LM). First work to
  run GPU-initiated token-level comm on **non-NVIDIA NICs** (paper-reported).
- No first-party measured GB/s in `ROCm/DeepEP`; compare against MoRI-EP's published table when choosing
  ([mori_ep.md](mori_ep.md): MI300X+CX7 307/330 GB/s dispatch/combine).

## Engage in frameworks
- **sglang**: `--enable-deepep-moe` (DP attention + EP MoE); pairs with RCCL tuning ([rccl_tuning.md](rccl_tuning.md)).
- **vLLM**: all2all backend selection (`--all2all-backend ...` / `VLLM_ALL2ALL_BACKEND`); on ROCm AMD
  steers production toward `mori_low_latency` (MoRI-EP) — DeepEP/UCCL-EP are the portability path.

## Pitfalls
- ROCm-DeepEP needs a working **rocSHMEM** build first — the #1 install blocker; follow the rocSHMEM guide.
- Default `--nic cx7` assumes a Mellanox NIC; on Broadcom/Pollara set `thor2`/`io` (or use UCCL-EP).
- Exactly **8 GPUs/node** (`NUM_MAX_NVL_PEERS=8`) for the intranode path.
- Correctness-first port → don't expect NVIDIA-tier tuning; benchmark against MoRI-EP before committing.

## Verify
- Run `test_internode.py` / `test_low_latency.py` from the port for a fabric sanity check; trace
  dispatch/combine with rocprofv3 and confirm overlap with grouped-GEMM.
- Numeric parity vs torch reference MoE (greedy) after switching all2all backend.

## Alternatives / cross-links
[mori_ep.md](mori_ep.md) (first-party AMD SOTA) · [rccl_tuning.md](rccl_tuning.md) (RDMA/xGMI knobs) ·
[overview.md](overview.md) · FlashDMoE single-kernel design ref: arXiv 2506.04667.

## Sources
- DeepEP (upstream, normal vs low-latency, FP8): https://github.com/deepseek-ai/DeepEP
- ROCm/DeepEP (rocSHMEM, MI300X/MI308X, NIC types, ROCSHMEM_* env, correctness-first): https://github.com/ROCm/DeepEP + https://github.com/ROCm/DeepEP/blob/main/README.md
- UCCL-EP (drop-in, heterogeneous, build, Primus/TheRock): https://github.com/uccl-project/uccl + https://github.com/uccl-project/uccl/tree/main/ep
- UCCL-EP paper (40%/45% numbers, AMD+Broadcom): https://arxiv.org/html/2512.19849v1
