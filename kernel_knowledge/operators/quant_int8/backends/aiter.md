---
title: quant_int8 on aiter — SOTA card
kind: sota_card
operator: quant_int8
backend: aiter
gens: [gfx942, gfx950]
dtypes: [int8]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/quant.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/include/smoothquant.h
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/quant_kernels.cu
---

# quant_int8 × aiter

## TL;DR
aiter is the live INT8 path on sglang/vLLM, and its strength is the **fused SmoothQuant**:
`smoothquant_fwd` applies the per-channel smooth scale and emits the per-token scale in one kernel, and
`moe_smooth_per_token_scaled_quant` does the same per-expert for MoE. The INT8 GEMM is `gemm_a8w8` (shared
with FP8, INT32 accumulate, dequant epilogue). Use aiter int8 when the model is INT8-quantized and feeds
an aiter GEMM/MoE; on CDNA3/4 prefer FP8 for accuracy unless INT8 is required.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `smoothquant_fwd(out,input,x_scale,y_scale)` | `csrc/include/smoothquant.h` → `csrc/py_itfs_ck/smoothquant_kernels.cu` | gfx942/950, int8 | one-pass smooth + per-token | SmoothQuant W8A8 activations |
| `pertoken_quant(..., quant_dtype=i8, x_scale=smooth)` | `aiter/ops/quant.py:42` | all | per-token + smooth ref | per-token int8 |
| `moe_smooth_per_token_scaled_quant_v1/v2` | `csrc/kernels/quant_kernels.cu:1334,1562` | gfx942/950 | per-expert smooth + route | MoE int8 |

## Config space / knobs
- per-token activation × per-channel weight (the SmoothQuant default).
- `x_scale` = per-channel smooth factor (offline-calibrated); `y_scale` = per-token (emitted).
- MoE: per-expert smooth via map/hash (`has_smscale_map`).
- HIP `thread_data_size`/block size → [[tuning.md]].

## Numerics / parity
INT32 accumulate; symmetric `s=amax/127`; SmoothQuant α offline. Gate on task accuracy / err-ratio →
[[numerics.md]].

## Integration (rebind seam)
`aiter.ops.quant.pertoken_quant` / `smoothquant_fwd`; INT8 GEMM via `gemm_a8w8`. In vLLM gated by
`VLLM_ROCM_USE_AITER_LINEAR=1`.

## Pitfalls & anti-patterns
- Per-tensor int8 activations → outliers; use smooth + per-token.
- MoE single global smooth scale instead of per-expert.
- INT8 over FP8 on CDNA3/4 without reason.

## How to verify
`AITER_LOG_MORE=1` (dispatch); round-trip error + gsm8k parity; confirm INT32 accumulate in `gemm_a8w8`.

## Alternatives / cross-links
[vllm_kernels.md](vllm_kernels.md) · [hip.md](hip.md) · [triton.md](triton.md) · [overview.md](../overview.md) ·
[[operators/fused_moe_grouped_gemm]] · [[operators/scaled_quant_gemm]].

## Sources
- aiter smoothquant + pertoken + MoE int8: `ROCm/aiter@a6bb49937:aiter/ops/quant.py`, `csrc/include/smoothquant.h`, `csrc/kernels/quant_kernels.cu`.
