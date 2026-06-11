---
title: allreduce — overview
kind: operator_overview
operator: allreduce
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, int6, int4]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
  - https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
---

# allreduce  (sum partials across TP ranks)

## TL;DR
All-reduce sums each rank's partial activation and broadcasts the result — it fires **twice per layer**
(after attention, after MLP) and is **the dominant collective of tensor-parallel inference**. On an 8×
MI300X node it runs over a **fully-connected xGMI mesh** (no NVSwitch), sustaining ~316–330 GB/s busbw out
of the box. The single most important fact: RCCL is the default, but **framework custom all-reduce**
(AITER custom AR, vLLM Quick Reduce) often **beats RCCL for small/decode messages** by using xGMI P2P
directly (1-shot/2-shot) and optionally **quantizing the reduction** to cut wire bytes (an accuracy gate).

## Math contract
`out[i] = Σ_r in_r[i]` for every rank `r`, result replicated to all ranks. dtype bf16/fp16 in/out, fp32
accumulate (RCCL); quantized custom AR reduces in int8/int6/int4/fp8 on the wire then dequantizes.
Algorithms: **ring** (bandwidth-optimal, large msgs), **tree** (latency, small msgs), **1-shot**
(every rank reads all peers, small msgs), **2-shot** (reduce-scatter + all-gather, mid msgs).

## Shape regimes
- **prefill (large M)**: large all-reduce messages → ring / 2-shot, bandwidth-bound; comm amortized under
  compute.
- **decode (M=batch)**: tiny messages, latency-bound → tree / 1-shot custom AR / Quick Reduce; this is
  where custom AR beats RCCL.

## Where it matters (Amdahl)
On TP-dominated dense inference, all-reduce is single-digit % **except** sub-island TP (TP=2/4), where
fewer xGMI links are used, and decode (latency-bound small messages). The decision: TP=8 single island,
dense → leave RCCL's tuning model on (small gains); TP=2/4 or decode → custom AR + `NCCL_MIN_NCHANNELS`.
First **baseline the fabric** (rccl-tests) — a slow xGMI link, not software, is the usual culprit.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| mori_rccl (RCCL + MSCCL++) | 🟢 sota (default TP collective) | [backends/rccl.md](backends/rccl.md) |
| hip | 🟢 (1-shot/2-shot custom AR over xGMI P2P; the kernels) | [backends/hip.md](backends/hip.md) |
| vllm_kernels | 🟢 (Quick Reduce quantized custom AR; AITER custom AR) | [backends/vllm_kernels.md](backends/vllm_kernels.md) |

## Fusion neighbors
- **all-reduce + RMSNorm** (the next layer's norm folded in) → [[fused_allreduce_rmsnorm]].
- **all-reduce → reduce-scatter + RMSNorm + all-gather** (sequence parallelism rewrite) →
  [[reduce_scatter]] / [[allgather]].
- overlap with the next GEMM (high-priority streams). See [fusion.md](fusion.md).

## Numerics
RCCL fp32 accumulate = parity-safe. **Quick Reduce quantizes the reduction** (int8/int6/int4/fp8) → an
accuracy gate (changes reduced values). See [numerics.md](numerics.md).

## How to bench
`rccl-tests all_reduce_perf -b 8 -e 16G -f 2 -g 1 -G 1` → expect ~316–330 GB/s busbw; lower = a slow link.
e2e: TP serving tok/s with the custom-AR flag on/off; trace that AR overlaps the next GEMM.

## Sources
- xGMI mesh / busbw / algorithms: https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
- RCCL env / tuning model / MSCCL++: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
- Quick Reduce quantized custom AR: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- Full RCCL knob table: [`backends/mori_rccl/rccl_tuning.md`](../../backends/mori_rccl/rccl_tuning.md).
