---
title: gemm_epilogue_fused on hipblaslt — SOTA card
kind: sota_card
operator: gemm_epilogue_fused
backend: hipblaslt
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/ROCm/hipBLASLt
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# gemm_epilogue_fused × hipblaslt

## TL;DR
> hipBLASLt provides a **fixed menu** of fused epilogues (bias, GELU/RELU, scale, fp8 output) via its
> `epilogue` enum — fast and the executed kernel under aiter, but limited (no arbitrary act, no gated
> silu·mul, residual support varies). Use it for the standard bias+gelu / scaled-fp8 cases; for richer
> fusion (gated act, residual chains, mxfp4) use [ck.md](ck.md). On serving it's reached via aiter.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| hipBLASLt matmul + epilogue (`HIPBLASLT_EPILOGUE_{BIAS,GELU,GELU_BIAS,RELU,...}`, fp8 scale) | `ROCm/hipBLASLt` | gfx942/950; bf16/fp16/fp8 | fp8 GEMM core: ~2750 TFLOP/s @4096, ~3130 @8192 on MI355X (epilogue ≈ free); bf16 ceiling ~890 TFLOP/s on MI300X | standard bias+gelu / scaled-fp8 fused GEMM |

## Config space / knobs
- `hipblasLtMatmulDescSetAttribute`: `EPILOGUE` enum, `BIAS_POINTER`, `A/B/C/D_SCALE_POINTER` (fp8),
  `AMAX` for output scale.
- Solution selection (the perf knob) → done by aiter's race, not by hand.
- `--algo_method all` in `hipblaslt-bench` to enumerate fused solutions offline.

## Numerics / parity
fp32 accumulate, epilogue in fp32 before down-cast; bias/gelu parity-safe; fp8 output task-gated. See
[../numerics.md](../numerics.md).

## Integration (rebind seam)
In serving reached **through aiter** (no direct rebind in sglang). Direct API for non-aiter stacks.
`HIPBLASLT_TUNING_FILE` does **not** engage the aiter path.

## Pitfalls & anti-patterns
- Needing an epilogue outside the enum (gated silu·mul, residual+act chain, mxfp4) → not supported; use CK.
- `HIPBLASLT_TUNING_FILE`/TunableOp assumed to tune serving — bypassed by aiter.

## How to verify
`hipblaslt-bench` with the epilogue flag on the exact shape; for serving verify via aiter engagement marker.

## Alternatives / cross-links
[ck.md](ck.md) (richer epilogues) · [aiter.md](aiter.md) (live/tune lever) · [triton.md](triton.md) ·
[../overview.md](../overview.md) · dense: [[operators/dense_gemm/backends/hipblaslt.md]].

## Sources
- hipBLASLt epilogue API: https://github.com/ROCm/hipBLASLt.
- fp8 GEMM TFLOP/s (MI355X): ROCm CDNA4 GEMM blog.
- TunableOp/tuning-file bypass: perf_knowledge dense_gemm/backends/aiter.md.
