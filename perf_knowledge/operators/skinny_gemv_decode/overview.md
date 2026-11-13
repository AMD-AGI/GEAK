---
title: skinny_gemv_decode — overview
kind: operator_overview
operator: skinny_gemv_decode
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
updated: 2026-11-13
sources:
  - https://github.com/ROCm/aiter
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
---

# skinny_gemv_decode

## TL;DR
> The tall-skinny GEMM / GEMV of LLM decode — `M=1..8` tokens × large weight matrix — which is **memory
> bandwidth bound**, so the key fact is: you optimize for **weight read bandwidth and CU utilization**
> (split-K across CUs), not MFMA throughput.

## Math contract
- `Y[M,N] = X[M,K] @ W[K,N] (+ bias)` with `M` tiny (1..8). For `M=1` it is a matrix-vector product (GEMV).
- Output dtype bf16/fp16; weights bf16/fp16 or fp8 (→ [../scaled_quant_gemm/overview.md](../scaled_quant_gemm/overview.md));
  accumulate fp32.
- Dominant decode GEMMs: QKV proj, O proj, MLP up/gate/down at batch≈1.

## Shape regimes
- **Decode only**: M = batch, N,K = model hidden dims (large). Arithmetic intensity is low →
  bandwidth bound. Large-batch/prefill uses dense GEMM instead
  ([../dense_gemm/overview.md](../dense_gemm/overview.md)).
- **M is driven by concurrency and capped by `max_num_seqs`**, whichever binds first — concurrency
  counts client sessions, not in-flight requests, and sub-agents fan out without taking their own slot,
  so M runs above the configured concurrency by a factor that is itself not constant. M = 1..8 is the
  low-concurrency corner only: decode M p50 runs **8 / 17 / 36 / 71 / 145** at concurrency
  8 / 16 / 32 / 64 / 128, with p90 at **11 / 22 / 44 / 82 / 162**. Above ~M=64 the skinny path stops
  being the right kernel; check the crossover to dense GEMM rather than assuming.
- **Do not assume powers of two.** When `max_num_seqs` is not binding, M is spread, not concentrated:
  at concurrency 64 it covers p10–p90 = 66..82 with 71/68/70 the most common values. When
  `max_num_seqs` *is* binding it pins M exactly, and that is the only case where a single tuned M is
  safe. Under speculative decoding multiply by `draft_len+1`. Sweep construction and the
  time-weighting rule: [[optimization/workload_shape_selection.md]].

## Where it matters (Amdahl)
- At batch≈1 decode, these projections are a large fraction of per-token latency and are bandwidth-limited;
  the GPU's MFMA peak is irrelevant — only how fast you stream W and how well you fill the 304 CUs
  (MI300X) / 256 CUs (MI350X) matters. This is why aiter ships a dedicated skinny / wvSplitK-style path.

## Backend landscape (link table → SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota | [backends/aiter.md](backends/aiter.md) |
| triton | 🟡 competitive | [backends/triton.md](backends/triton.md) |
| asm | 🟡 competitive | [backends/asm.md](backends/asm.md) |
| hip | 🟡 competitive | [backends/hip.md](backends/hip.md) |
| flydsl | 🟡 competitive (gfx950 small-M family) | [backends/flydsl.md](backends/flydsl.md) |

## Fusion neighbors
- Bias/activation epilogue, fp8 dequant; for decode MoE the per-expert M is also tiny →
  [../grouped_gemm_moe/overview.md](../grouped_gemm_moe/overview.md). Split-K is the core technique →
  [../splitk_streamk_gemm/overview.md](../splitk_streamk_gemm/overview.md).

## Numerics
- fp32 accumulate; split-K reduction order; fp8 weight scales → [numerics.md](numerics.md).

## How to bench
- Bench at M=1 and M=2/4/8 separately (regimes differ); median ≥3 warm reps; oracle = dense fp32
  reference. Report achieved GB/s vs HBM peak (bandwidth bound), not TFLOPS.

## Sources
- AITER (skinny GEMM path): https://github.com/ROCm/aiter
- vLLM decode/GEMV custom kernels on ROCm: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- Decode-M vs concurrency (8/17/36/71/145) and the non-power-of-two spread: AgentX v1.0 1M corpus,
  https://inferencex.semianalysis.com/agentx/methodology ; derivation and sweep construction in
  [[optimization/workload_shape_selection.md]].
