---
title: linear_attention_gated_delta on TileLang — SOTA card
kind: sota_card
operator: linear_attention_gated_delta
backend: tilelang
gens: [gfx90a, gfx942]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
status: experimental
updated: 2026-06-08
sources:
  - https://github.com/tile-ai/tilelang
  - https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
  - https://arxiv.org/abs/2511.08083
---

# linear_attention_gated_delta × TileLang

## TL;DR
TileLang **can express** chunked linear-attention (the tile model handles the chunk matmul + state carry,
and the upstream tile-ai project has linear-attention / chunk-scan examples), but there is **no published,
AMD-tuned TileLang Gated-DeltaNet kernel** as of 2026-06. On AMD, GDN is a solved problem in Triton (FLA /
aiter), so TileLang here is **research only**: prototype a chunk-scan fast, but ship the Triton path
([triton.md](triton.md)).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| TileLang chunk-scan (extend from FA/linear examples) | `tile-ai/tilelang` | gfx90a/942; fp16/bf16 | no AMD GDN number published | fast iteration on a chunked-scan variant |

> **Primarily Triton-portable on AMD; TileLang GDN is experimental.** No hand-tuned TileLang Gated-DeltaNet
> kernel known as of 2026-06; the portable + production path is FLA/aiter Triton.

## Config space / knobs ([[tilelang]])
- `block_M`/`block_N` for the chunk matmul, `threads`, `num_stages`, `GemmWarpPolicy`; the triangular
  solve / cumsum must be hand-expressed (TileLang has no built-in WY/solve_tril primitive).
- Autotune sweeps per shape (~1 s) — won't generalize across d_k/d_v.

## Numerics / parity
fp32 state accumulate; chunk-boundary state parity vs the FLA fp32 reference is the gate. TileLang adds no
special handling. See [numerics.md](../numerics.md).

## Integration (rebind seam)
Python-importable callable; sglang's `tilelang` backend is the dense-attention path, **not** a GDN path —
wiring a TileLang GDN kernel into the hybrid KV-cache manager is custom work.

## Pitfalls & anti-patterns
- CDNA3-only maturity; MI350/CDNA4 unproven (HipKittens: missing AMD-constraint abstractions; CUTLASS/CK
  deps on some paths).
- The serial chunk dependency (solve_tril + cumsum) is the hard part — TileLang gives you tiles, not the
  algorithm; you must encode the recurrence correctly.
- Don't present dense-FA TileLang numbers as GDN performance.

## How to verify
Bench vs the Triton GDN path at the served shape; chunk-boundary state parity vs FLA fp32; confirm the
autotuned config; re-measure on your stack.

## Alternatives / cross-links
[overview.md](../overview.md) · [triton.md](triton.md) · [hip.md](hip.md) · languages: [[tilelang]] ·
[[triton_amd]] · ops: [[causal_conv1d]] · [[cumsum_scan]].

## Sources
- tile-ai/tilelang (linear-attention / chunk-scan examples): https://github.com/tile-ai/tilelang
- TileLang AMD dense FA (context, not GDN): https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
- HipKittens TileLang AMD critique: https://arxiv.org/abs/2511.08083
