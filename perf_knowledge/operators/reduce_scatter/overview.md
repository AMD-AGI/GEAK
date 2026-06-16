---
title: reduce_scatter — overview
kind: operator_overview
operator: reduce_scatter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/reduce_scatter.py
  - https://docs.vllm.ai/en/latest/design/fusions/
---

# reduce_scatter  (reduce across ranks, scatter shards)

## TL;DR
Reduce-scatter sums each rank's partial and gives each rank only **its shard** of the result `[N/P, …]` —
it is the **first half of a 2-shot all-reduce** and the key op in the **sequence-parallel rewrite** of
all-reduce (`AR → reduce_scatter + local RMSNorm + all_gather`). On 8× MI300X it rides the same xGMI mesh
as [[allreduce]]. The most important fact: RS lets the subsequent **RMSNorm run on the smaller sharded
tensor** (less work) and lets the AsyncTP pass **overlap RS with the producing GEMM** — and aiter ships a
**single fused RS+RMSNorm+quant+AG** kernel that realizes the whole SP collective.

## Math contract
`out_r[i] = Σ_s in_s[r·S + i]` — rank `r` receives the reduced shard `[r·S : (r+1)·S]`. dtype bf16/fp16
in/out, fp32 accumulate (RCCL). It is a reduction (unlike all-gather) → reduction-order numerics apply.

## Shape regimes
- **prefill / SP**: reduce sequence-sharded activations → large messages, ring algo, bandwidth-bound.
- **decode**: small → latency-bound.
- As the **2-shot AR head**: feeds the all-gather tail.

## Where it matters (Amdahl)
Standalone RS is the AR head; its e2e value is the **SP rewrite** — running the norm on the sharded tensor
+ overlapping RS/AG with the GEMMs (AsyncTP). On TP-heavy layers with sequence parallelism this is a real
win; as a lone op it's single-digit %.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| mori_rccl (RCCL reduce-scatter + MoRI-CCL) | 🟢 sota (default) | [backends/rccl.md](backends/rccl.md) |
| hip | 🟢 (xGMI P2P / Iris; aiter Triton reduce_scatter + fused SP kernel) | [backends/hip.md](backends/hip.md) |

## Fusion neighbors
- **reduce-scatter + RMSNorm + fp8 quant + all-gather** in one fused Triton kernel (the SP collective) →
  [[allgather]], [[fused_allreduce_rmsnorm]].
- **GEMM → reduce-scatter (AsyncTP)** overlap. See [fusion.md](fusion.md).

## Numerics
fp32 accumulate; reduction-order deltas benign (like all-reduce). fp8 quant in the fused SP kernel is a
gate. See [numerics.md](numerics.md).

## How to bench
`rccl-tests reduce_scatter_perf -b 8 -e 16G -f 2 -g 1 -G 1`; e2e SP serving tok/s; trace RS↔GEMM overlap.

## Sources
- RCCL env / algos: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
- aiter Triton reduce_scatter + fused SP kernel: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/reduce_scatter.py`, `.../fused/reduce_scatter_rmsnorm_quant_all_gather.py`.
- SP rewrite (AR → RS + norm + AG): https://docs.vllm.ai/en/latest/design/fusions/
