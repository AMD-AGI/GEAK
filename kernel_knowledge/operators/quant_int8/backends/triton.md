---
title: quant_int8 on triton — SOTA card
kind: sota_card
operator: quant_int8
backend: triton
gens: [gfx942, gfx950]
dtypes: [int8]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/quant/quant.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/moe/moe_op_gemm_int8_smoothquant.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# quant_int8 × triton

## TL;DR
Triton's INT8 niche on AMD is the **fused SmoothQuant MoE GEMM** (`moe_op_gemm_int8_smoothquant.py`):
per-expert smoothquant + INT8 grouped GEMM in one kernel, with a gfx942 gluon variant. For standalone
per-token INT8 quant, `dynamic_per_token_quant_fp8_i8` handles both i8 and fp8 (the `_i8` suffix). Triton
is the portable/fusion path; the standalone quant is memory-bound so it doesn't lose to asm.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `dynamic_per_token_quant_fp8_i8` (i8 mode) | `aiter/ops/triton/quant/quant.py:94` | gfx942/950, int8 | grid=(rows,) row reduce | standalone per-token int8 |
| `static_per_tensor_quant_fp8_i8` | `:27` | gfx942/950 | scalar scale | static/calibrated |
| `moe_gemm_int8_smoothquant` | `aiter/ops/triton/moe/moe_op_gemm_int8_smoothquant.py` | gfx942/950 | fused per-expert smooth + grouped GEMM | MoE W8A8 |

## Config space / knobs
- `BLOCK_SIZE`≈padded hidden; `num_warps` 4–8 wide; `num_stages` 1–2 (AMD pipeliner).
- MoE: `get_kernel_config(m,n,k,routing_data)` picks `BLOCK_M/N/K`; `can_overflow_int32` guard.
- `DTYPE_MAX` from `torch.iinfo` (int) vs `torch.finfo` (fp) — the shared kernel branches on dtype.

## Numerics / parity
INT32 accumulate; symmetric `amax/127`; RNE. Gate on task accuracy → [[numerics.md]].

## Integration (rebind seam)
`aiter.ops.triton.quant.*` and `aiter.ops.triton.moe.moe_gemm_int8_smoothquant`; Python kernel → overlay a
tuned config via autotune without editing site-packages.

## Pitfalls & anti-patterns
- INT32 accumulator overflow on very long K (`can_overflow_int32`) → upcast indices.
- `num_warps=8` carried from NVIDIA → spill.
- Standalone quant in decode → launch-bound; fuse.

## How to verify
`AMDGCN_ENABLE_DUMP=1`; round-trip error; e2e gsm8k parity; rocprof confirm the int8 path ran.

## Alternatives / cross-links
[aiter.md](aiter.md) · [hip.md](hip.md) · [vllm_kernels.md](vllm_kernels.md) · [[languages/triton_amd]] ·
[[operators/fused_moe_grouped_gemm]] · [overview.md](../overview.md).

## Sources
- Triton int8 quant + fused MoE smoothquant: `ROCm/aiter@a6bb49937:aiter/ops/triton/quant/quant.py`, `.../moe/moe_op_gemm_int8_smoothquant.py`.
- Triton AMD knobs: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
