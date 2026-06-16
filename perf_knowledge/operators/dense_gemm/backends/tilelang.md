---
title: dense_gemm on tilelang — SOTA card
kind: sota_card
operator: dense_gemm
backend: tilelang
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/tile-ai/tilelang
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# dense_gemm × tilelang

## TL;DR
> TileLang is a Python tile-DSL that compiles to MFMA kernels — choose it when you want
> Triton-level authoring ergonomics with finer tile/pipeline control, e.g. for a fused or odd-shape
> GEMM you'd otherwise hand-write. On plain dense bf16 GEMM it lands roughly **~0.94–1.05× of Triton**
> on MI300X, i.e. competitive but not a clear win over the tuned hipBLASLt that aiter dispatches.
> Prefer [aiter.md](aiter.md)/[hipblaslt.md](hipblaslt.md) for the live path.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| TileLang GEMM kernel | `tile-ai/tilelang` | gfx942/950; bf16/fp16/fp8 | **~0.94–1.05× of Triton GEMM** on MI300X (no public absolute TFLOP figure that beats tuned hipBLASLt) → use [hipblaslt.md](hipblaslt.md) for plain GEMM | custom/fused GEMM authoring with tile control |

## Config space / knobs
- `T.Kernel` grid + `block_M/block_N/block_K`; keep tiles 8-multiples, prefer 16×16 MFMA mapping.
- Pipeline: `T.Pipelined(..., num_stages)` for K-deep prefill; software-pipelined copy → shared → MFMA.
- `T.alloc_shared` / `T.alloc_fragment` layouts; `T.copy` with vectorized loads; swizzle for LDS-bank
  conflict avoidance. `num_threads` per block ~256.
- Tune via TileLang's autotuner over (block tiles × stages × threads).

## Numerics / parity
fp32 accumulate, bf16 out; parity-safe for bf16. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
Compiles to a callable kernel/extension — call directly from your op or register as an aiter candidate.
No env-overlay into sglang's live GEMM; engagement requires a code seam.

## Pitfalls & anti-patterns
- Expecting it to beat tuned hipBLASLt on vanilla GEMM — it usually won't; its value is authoring
  fused/irregular kernels.
- Under-tuned tiles/stages → far below Triton; always run the autotuner per shape.

## How to verify
Bench the generated kernel vs Triton and `hipblaslt-bench` on the same (M,N,K); adopt only with a code
seam and a win on the target shape.

## Alternatives / cross-links
[triton.md](triton.md) · [aiter.md](aiter.md) · [hipblaslt.md](hipblaslt.md) · [ck.md](ck.md) ·
[../overview.md](../overview.md) · language ref [[languages/tilelang/...]].

## Sources
- TileLang project: https://github.com/tile-ai/tilelang.
- ~0.94–1.05× of Triton on MI300X: perf_knowledge dense_gemm overview landscape (see ../overview.md).
- MFMA/tile levers: ROCm workload guide.
