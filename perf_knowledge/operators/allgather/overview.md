---
title: allgather — overview
kind: operator_overview
operator: allgather
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/all_gather.py
  - https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
---

# allgather  (gather each rank's shard to all ranks)

## TL;DR
All-gather collects each rank's shard `[N/P, …]` into the full tensor `[N, …]` on every rank — it appears
in **sequence/tensor parallelism** (gather sharded activations/weights) and as the **second half of a
2-shot all-reduce** (reduce-scatter + all-gather). On 8× MI300X it rides the same fully-connected xGMI mesh
as [[allreduce]] (~316–330 GB/s class). The most important fact: all-gather is **pure data movement** (no
reduction → no accuracy concern), so the lever is **overlap with the producing/consuming GEMM** (AsyncTP)
and using xGMI P2P / GPU-initiated (Iris) copies, often offloaded to **SDMA** copy engines to free the CUs.

## Math contract
`out[r·S : (r+1)·S] = in_r` for all ranks `r` (S = shard size), result replicated. No reduction → bitwise
identical across backends (modulo layout). dtype unchanged in→out.

## Shape regimes
- **prefill / SP**: gather sequence-sharded activations → large messages, bandwidth-bound, ring algo.
- **decode**: small gathers → latency-bound, tree / 1-shot / GPU-initiated.
- As the **2-shot AR tail**: message size = the reduce-scatter shard.

## Where it matters (Amdahl)
Standalone, all-gather is small except in sequence-parallel layouts and as the AR tail. The win is
**overlapping** it with the adjacent GEMM (AsyncTP fuses RS/AG with the GEMMs) and **offloading to SDMA** so
it doesn't steal CU cycles. As a lone op it rarely moves e2e; as part of the SP-rewrite of all-reduce it does.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| mori_rccl (RCCL all-gather + MoRI-CCL) | 🟢 sota (default) | [backends/rccl.md](backends/rccl.md) |
| hip | 🟢 (xGMI P2P / Iris GPU-initiated; aiter Triton all_gather) | [backends/hip.md](backends/hip.md) |

## Fusion neighbors
- **reduce-scatter + RMSNorm + all-gather** (SP rewrite of all-reduce) — aiter ships a fused Triton kernel
  for exactly this ([[reduce_scatter]], [[fused_allreduce_rmsnorm]]).
- **all-gather ↔ GEMM overlap** (AsyncTP). See [fusion.md](fusion.md).

## Numerics
Pure copy → parity-safe (no reduction). See [numerics.md](numerics.md).

## How to bench
`rccl-tests all_gather_perf -b 8 -e 16G -f 2 -g 1 -G 1`; e2e SP serving tok/s; trace AG↔GEMM overlap.

## Sources
- RCCL env / algos: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
- aiter Triton all_gather (Iris GPU-initiated, SDMA): `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/all_gather.py`.
- xGMI mesh: https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
