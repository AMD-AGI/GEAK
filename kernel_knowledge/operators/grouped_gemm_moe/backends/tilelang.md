---
title: grouped_gemm_moe on tilelang — SOTA card
kind: sota_card
operator: grouped_gemm_moe
backend: tilelang
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: experimental
updated: 2026-06-05
sources:
  - https://github.com/tile-ai/tilelang
  - https://github.com/ROCm/aiter
---

# grouped_gemm_moe × tilelang

## TL;DR
> TileLang can express a grouped/variable-M GEMM at the tile level with explicit pipelining, attractive
> for authoring custom MoE fusions with less boilerplate than HIP. Experimental on AMD/MoE here — no
> reproduced perf; use [aiter.md](aiter.md) for production, TileLang for research/codegen.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| TileLang grouped GEMM (tile program over expert offsets) | https://github.com/tile-ai/tilelang | gfx942/950; bf16, fp8 | no AMD MoE number reproduced — experimental | research / custom fusion codegen |

## Config space / knobs
- Tile-program block sizes (`block_M/N/K`), pipeline stages, MFMA mapping, explicit LDS staging; expert
  offset table drives the grouped loop.

## Numerics / parity
- fp32 accumulate; mask padding; per-expert scales → [../numerics.md](../numerics.md).

## Integration (rebind seam)
- Compile to a callable op and wire into the MoE layer; verify kernel name in trace.

## Pitfalls & anti-patterns
- Toolchain/ROCm-version maturity varies; validate correctness and perf before relying on it in serving.

## How to verify
- Per-expert dense oracle ([../numerics.md](../numerics.md)) + A/B vs aiter/triton.

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [ck.md](ck.md) · [hip.md](hip.md) · [../overview.md](../overview.md)

## Sources
- TileLang: https://github.com/tile-ai/tilelang
- AITER (production reference): https://github.com/ROCm/aiter
