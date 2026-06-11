---
title: MoRI + RCCL — communication backend overview (collectives, EP all-to-all, KV transfer)
kind: backend
backend: mori
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/ROCm/mori
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
  - https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
  - https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
---

# MoRI + RCCL — the AMD communication backend

## TL;DR
On Instinct, **RCCL** is the NCCL-compatible workhorse for **TP** collectives (all-reduce / all-gather /
reduce-scatter), and **MoRI** (Modular RDMA Interface, `ROCm/mori`) is AMD's **next-gen** comm stack for
the *hard* distributed-inference cases: **EP all-to-all** (MoRI-EP), **KV-cache transfer** for
prefill/decode disaggregation (MoRI-IO), and latency-sensitive collectives (MoRI-CCL). The decision rule:
TP-dominated dense models → tune RCCL; large **MoE + expert parallelism / PD-disaggregation** → MoRI-EP
(or DeepEP, see [deepep.md](deepep.md)). LLM inference is usually compute/memory-bound, so comm tuning is
single-digit % *until* comm becomes the bottleneck (sub-island TP, EP all-to-all, multi-node) — know
**when** to spend here.

## Concepts — which library owns which collective

| Collective / op | Where it fires | Parallelism | Backend |
|---|---|---|---|
| **all-reduce** | after attn + MLP (sum partials), 2×/layer | **TP** (dominant inference collective) | RCCL / framework custom-AR |
| **all-gather / reduce-scatter** | sequence-parallel, sharded weights | TP/SP | RCCL / MoRI-CCL |
| **all-to-all dispatch/combine** | MoE token routing (tokens→experts→tokens) | **EP** | **MoRI-EP** / DeepEP |
| **KV-cache P2P transfer** | prefill→decode disaggregation | PD | **MoRI-IO** |
| **broadcast / gather** | sampling, DP coordination | DP | RCCL |

MoRI is a **suite** built on shared MoRI-Core primitives (IBGDA, GPUDirect, symmetric memory):
- **MoRI-EP** — MoE dispatch & combine kernels (intra-node xGMI + inter-node RDMA). See [mori_ep.md](mori_ep.md).
- **MoRI-IO** — ultra-low-overhead P2P for KV-cache transfer (layer-wise PUSH, zero-copy, GDR).
- **MoRI-CCL** — lightweight collective library for latency-sensitive / resource-constrained cases.
- **MoRI-SHMEM** — symmetric-memory / GPU-initiated comm APIs (backs ByteDance Triton-distributed EP).
- **MoRI-UMBP** — unified memory & bandwidth pool with tiered storage.

Sharing one foundation lets MoRI-IO (KV) and MoRI-EP (tokens) do **unified network-priority management**
— joint scheduling of KV-transfer and dispatch traffic on the same NIC.

## The levers
- **Pick the backend by parallelism**: TP → RCCL (+ framework custom-AR); EP MoE → MoRI-EP / DeepEP; PD
  KV → MoRI-IO. Don't reach for MoRI on a dense TP-only model.
- **RCCL env tuning** (single biggest knob for sub-8-GPU TP): `NCCL_MIN_NCHANNELS=112`,
  `RCCL_MSCCLPP_THRESHOLD`. Full table in [rccl_tuning.md](rccl_tuning.md).
- **MoRI-EP kernel mode**: high-throughput vs low-latency, auto-switched (`MORI_EP_LAUNCH_CONFIG_MODE=AUTO`).
- **MoRI arch pin**: `MORI_GPU_ARCHS=gfx950` (MI355X) / `gfx942` (MI300X) to stop wrong-arch JIT.
- **Wide-EP topology**: small-radius EP + PD disaggregation (e.g. 2P2D) for fault isolation at scale.

## Hardware reality (8× MI300X node)
xGMI is **fully connected** (direct link to each of the other 7 GPUs), 64 GB/s/link theoretical,
~45–48 GB/s realized → ~316–330 GB/s all-reduce busbw on a healthy node. No central switch (unlike
NVSwitch): collective algos and channel counts behave differently than on H100. **Use all 8 GPUs** to
get full bandwidth; stay **within one xGMI island (≤8 GPU) for TP**, use PP across nodes.

## Measured (vendor / community, version-tagged)
- AMD Wide-EP DeepSeek-R1 on **32× MI300X** (2P2D, AITER + MoRI): **32.3k input tok/s** and **12.4k
  output tok/s** per node @ 2000-token inputs, ROCm 6.3.1, 2025-11-12 (AMD-reported).
- EP16 vs EP8 **1.3×** higher output throughput under SLO; **2×** offline vs TP8 (AMD-reported, same blog).
- Moreh MoAI on **8× MI300X**, DeepSeek-R1 decode: **21,224 tok/s per decode node** (~85% of peak decode),
  using MoRI-EP dispatch/combine inside a HIP graph + expert load-balancing, 2025-11 (vendor-reported).
- MoRI-EP all-to-all bandwidth (4096 tok, hidden 7168, top-8, FP8 dispatch + BF16 combine, EP8): MI300X+CX7
  **307 GB/s dispatch / 330 GB/s combine** XGMI; MI355X+AINIC **345 / 420 GB/s** (mori repo, v1.2.0, 2026-06).

## Pitfalls
- MoRI kernels are **JIT-compiled on first use** (cached `~/.mori/jit/`) → first iteration is slow; precompile
  with `MORI_PRECOMPILE=1`. Wrong-arch auto-select is a common foot-gun (pin `MORI_GPU_ARCHS`).
- MoRI-EP DeepEP-compatible (3D-layout) API needs the **`ENABLE_STANDARD_MOE_ADAPT=ON`** build flag, else
  those methods `RuntimeError`. See [mori_ep.md](mori_ep.md).
- `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` is **incompatible with MoRI** (vllm envs.py note).
- Setting `NCCL_MIN_NCHANNELS` **bypasses RCCL's MI300X tuning model** — A/B it, don't assume it wins.
- Don't stretch TP across nodes (off xGMI). Multi-node → PP + RDMA tuning ([rccl_tuning.md](rccl_tuning.md)).

## Verify
- Baseline the fabric first: `rccl-tests all_reduce_perf -b 8 -e 16G -f 2 -g 1 -G 1` → expect ~316–330 GB/s
  busbw; lower means a slow xGMI link (fix hardware before software).
- EP: confirm the all2all backend banner in the server log (`mori_low_latency` for vLLM); trace
  dispatch/combine kernels with rocprofv3 and check they overlap grouped-GEMM.
- Any quantized custom-AR (Quick Reduce INT/FP) is an **accuracy gate** — re-run a small eval.

## Sources
- ROCm/mori (suite, hardware support, JIT, milestones, EP bandwidth table): https://github.com/ROCm/mori @ v1.2.0 (2026-06-08)
- Wide-EP DeepSeek on MI300X (32 GPU, 2P2D, 32.3k/12.4k tok/s, MoRI+AITER): https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html (2025-11-12, ROCm 6.3.1)
- Moreh 21,224 tok/s/decode-node: https://moreh.io/technical-report/21k-output-tokens-per-second-deepseek-inference-on-amd-instinct-mi300x-gpus-with-expert-parallelism-251113/
- RCCL usage tips / xGMI bandwidth: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html · https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
- vLLM distributed inference with MoRI (launch flags): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/vllm-mori-distributed.html
