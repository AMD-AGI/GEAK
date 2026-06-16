---
title: <operator> — overview
kind: operator_overview
operator: <operator_id>
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: YYYY-MM-DD
sources: []
---

# <operator>

## TL;DR
> What it is in one sentence + the single most important optimization fact.

## Math contract
- exact definition / shapes / layouts (transpose, bias, epilogue), dtype in/out/accumulate.

## Shape regimes
- prefill vs decode (M/batch ranges), the shapes that dominate GPU time for typical LLMs.

## Where it matters (Amdahl)
- typical %GPU-time on real models; what speedup is needed to move e2e.

## Backend landscape (link table → SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢/🟡/⚪ | [backends/triton.md](backends/triton.md) |
| flydsl | … | [backends/flydsl.md](backends/flydsl.md) |
| hip | … | [backends/hip.md](backends/hip.md) |
| ck | … | [backends/ck.md](backends/ck.md) |
| asm | … | [backends/asm.md](backends/asm.md) |
| tilelang | … | [backends/tilelang.md](backends/tilelang.md) |
| aiter | … | [backends/aiter.md](backends/aiter.md) |
| hipblaslt | … | [backends/hipblaslt.md](backends/hipblaslt.md) |
| … | | |

## Fusion neighbors
- ops it can fuse with (bias/act/residual/quant epilogue; norm+quant; etc.) → see `fusion.md`.

## Numerics
- accuracy considerations → see `numerics.md`.

## How to bench
- canonical isolated benchmark + oracle (so every backend card compares apples-to-apples).

## Sources
- <primary sources>
