---
title: sparse_attention_nsa on TileLang — SOTA card
kind: sota_card
operator: sparse_attention_nsa
backend: tilelang
gens: [gfx90a, gfx942]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
status: experimental
updated: 2026-06-08
sources:
  - https://github.com/tile-ai/tilelang
  - https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
  - ROCm/composable_kernel:example/ck_tile/50_sparse_attn
  - https://arxiv.org/abs/2511.08083
---

# sparse_attention_nsa × TileLang

## TL;DR
TileLang is a **promising but unproven** NSA path on AMD. It is CDNA3-validated for *dense* FlashAttention
(~1.53× Triton, vendor-reported) and its tile model can express the selected/compressed branches, but
there is **no published, AMD-tuned TileLang NSA kernel** as of 2026-06. The closest AMD sparse-attention
artifact is the **CK-Tile `50_sparse_attn` example** (jenga / VSA sparse attention) — a CK path, not
TileLang. Treat TileLang NSA on MI300X as research: prototype fast, but the production sparse path is
Triton ([triton.md](triton.md)).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| TileLang dense FA (basis to extend to NSA branches) | `tile-ai/tilelang` | gfx90a/942; fp16/bf16 | dense FA ~1.53× Triton (vendor, MI300X, ROCm 7.0.1) — **not** an NSA number | fast iteration on a branch kernel |
| CK-Tile `50_sparse_attn` (jenga/VSA) | `ROCm/composable_kernel:example/ck_tile/50_sparse_attn` (on-box) | gfx942 | no published number | CK reference for block-sparse attention |

> **Primarily Triton-portable on AMD; TileLang/CK NSA is experimental.** No hand-tuned TileLang NSA kernel
> known as of 2026-06; the portable path is Triton + the CK `50_sparse_attn` example as a CK reference.

## Config space / knobs (TileLang — [[tilelang]])
- `block_M`/`block_N` (align `block_N` to NSA `block_size=64`), `threads`, `num_stages`, `num_split_q`,
  `qk_coalesced_width`/`v_coalesced_width`, `GemmWarpPolicy`, `enable_rasterization`.
- Autotune sweeps the product (~1 s) — but per-shape; an NSA selected-branch tune won't generalize.

## Numerics / parity
Same three-branch + top-k risks as the operator (see [numerics.md](../numerics.md)); TileLang has no
special handling — the indexer/selection must be validated independently.

## Integration (rebind seam)
TileLang kernels are Python-importable callables; on sglang the `tilelang` attention backend is the
generic dense path, **not** an NSA path — wiring an NSA TileLang kernel is custom integration work.

## Pitfalls & anti-patterns
- **CDNA3-only maturity**; treat MI350/CDNA4 as unproven (HipKittens: TileLang lacks AMD-constraint
  abstractions, uses 32×32 MFMA / buffer_load / XCD swizzle less than peak).
- Some TileLang paths call into CUTLASS/CK backends — verify on AMD.
- Do not present dense-FA TileLang numbers as NSA performance.

## How to verify
Bench the TileLang branch kernel vs the Triton NSA path at the served shape; selection-overlap + greedy
parity; confirm the autotuned config; re-measure on your stack (vendor numbers are dense-FA only).

## Alternatives / cross-links
[overview.md](../overview.md) · [triton.md](triton.md) · [hip.md](hip.md) · languages: [[tilelang]] ·
[[triton_amd]] · [[composable_kernel]] · core: [[mla_attention]].

## Sources
- TileLang AMD dense FA (1.53× Triton, ROCm 7.0.1): https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
- tile-ai/tilelang: https://github.com/tile-ai/tilelang
- CK-Tile sparse-attn example: `ROCm/composable_kernel:example/ck_tile/50_sparse_attn` (on-box via aiter 3rdparty).
- HipKittens TileLang AMD critique: https://arxiv.org/abs/2511.08083
