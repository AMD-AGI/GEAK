---
title: splitk_streamk_gemm on hipblaslt — SOTA card
kind: sota_card
operator: splitk_streamk_gemm
backend: hipblaslt
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-05
sources:
  - https://github.com/ROCm/hipBLASLt
  - https://arxiv.org/abs/2301.03598
---

# splitk_streamk_gemm × hipblaslt

## TL;DR
> hipBLASLt's Tensile-generated solutions **include split-K and stream-K (GSU / global-split-U) variants**;
> the library's solution selection picks one for CU-underutilized shapes automatically. You don't author it
> — you let the solution search/heuristic choose. Good for covered shapes; no Python rebind seam to inject
> a custom split here (tune via aiter's per-shape DB instead).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| hipBLASLt Tensile solution w/ Global-Split-U (split-K) / stream-K | `ROCm/hipBLASLt@HEAD` (Tensile-generated `Cijk_*` kernels) | gfx942/950; bf16, fp16, fp8 | no first-party number reproduced; selected per shape by the library | small-M / large-K covered shapes |

## Config space / knobs
- Not directly user-authored: GSU (global split-U = split-K), stream-K, and workgroup mapping are baked
  into Tensile solutions. Influence selection via solution index / the aiter tuned DB
  ([../../dense_gemm/backends/aiter.md](../../dense_gemm/backends/aiter.md)) which races hipBLASLt solutions
  per shape.

## Numerics / parity
- Same-math algorithm swap; GSU uses fp32 reduction → parity-safe within fp tolerance → [../numerics.md](../numerics.md).

## Integration (rebind seam)
- ⚠ No clean Python rebind to force a *custom* split-K kernel into hipBLASLt. The practical lever is
  **aiter's per-shape DB**, which selects the best hipBLASLt solution (incl. GSU/stream-K) for the shape —
  use that to get the split-K benefit on the live path.

## Pitfalls & anti-patterns
- `HIPBLASLT_TUNING_FILE` / TunableOp hook PyTorch dispatch which aiter bypasses → 0 engagement on the
  serving path (see dense aiter card). Tune through aiter.

## How to verify
- rocprof shows a `Cijk_*` GSU/stream-K kernel; A/B via aiter DB swap, dense fp32 oracle.

## Alternatives / cross-links
[triton.md](triton.md) · [ck.md](ck.md) · [asm.md](asm.md) · [hip.md](hip.md) · [../overview.md](../overview.md)

## Sources
- hipBLASLt (Tensile GSU/stream-K solutions): https://github.com/ROCm/hipBLASLt
- Stream-K: https://arxiv.org/abs/2301.03598
