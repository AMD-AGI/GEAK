---
title: skinny_gemv_decode on triton — SOTA card
kind: sota_card
operator: skinny_gemv_decode
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
status: competitive
updated: 2026-06-05
sources:
  - https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
  - https://github.com/ROCm/aiter
---

# skinny_gemv_decode × triton

## TL;DR
> A Triton split-K matmul with small BLOCK_M is the **authorable** decode-GEMV path — easy to set
> `SPLIT_K` high to fill CUs and to sweep tiling. Competitive and good for prototyping/coverage; aiter's
> skinny asm usually wins the live path.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton split-K GEMV/skinny matmul (small BLOCK_M, high SPLIT_K) | https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html | gfx942/950; bf16, fp8 | no first-party number reproduced; bandwidth-bound, sweep SPLIT_K | authorable decode GEMM, coverage |

## Config space / knobs
- `BLOCK_M` (16/32), `SPLIT_K` (high, to fill CUs), `BLOCK_N/K`, `matrix_instr_nonkdim=16` (mfma_16x16),
  num_warps, waves_per_eu, reduction mode. See [../tuning.md](../tuning.md).

## Numerics / parity
- fp32 accumulate; split-K reduction (atomic vs workspace) → [../numerics.md](../numerics.md).

## Integration (rebind seam)
- Overlay the triton GEMM module the framework calls (or aiter's triton path); verify kernel + autotune key
  in trace.

## Pitfalls & anti-patterns
- Default dense BLOCK_M=128 at M=1 wastes the tile — force small BLOCK_M.
- Over-splitting K → reduction cost > BW gain; sweep.

## How to verify
- HBM GB/s vs peak + A/B vs aiter skinny + dense fp32 oracle ([../numerics.md](../numerics.md)).

## Alternatives / cross-links
[aiter.md](aiter.md) · [asm.md](asm.md) · [hip.md](hip.md) · [../overview.md](../overview.md)

## Sources
- Triton split-K/persistent matmul: https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
- AITER triton GEMM paths: https://github.com/ROCm/aiter
