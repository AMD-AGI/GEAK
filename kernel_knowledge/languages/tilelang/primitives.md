---
title: TileLang primitives — the tile-level API
kind: language
gens: [gfx90a, gfx942]
dtypes: [fp16, bf16]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
  - https://github.com/tile-ai/tilelang
  - https://arxiv.org/abs/2504.17577
---

# TileLang primitives

## TL;DR
TileLang programs are written with a small set of `T.*` primitives inside a `@tilelang.jit` /
`@tilelang.autotune` kernel. The vocabulary covers memory scopes, data movement, MFMA-backed compute,
reductions, and parallel/pipelined loops. The named primitives below are the ones exercised in AMD's
MI300X FlashAttention blog (verified against the source). See [overview.md](overview.md) for the 3-level
model and [autotune.md](autotune.md) for the tuning decorators.

## Core concepts — the primitive vocabulary
| Primitive | role |
|---|---|
| `T.Kernel(...)` | declares the kernel grid / block context |
| `T.alloc_shared(shape, dtype)` | allocate an **LDS** (shared-memory) tile |
| `T.alloc_fragment(shape, dtype)` | allocate a **register/fragment** tile (per-lane MFMA storage) |
| `T.alloc_var(dtype)` | a scalar accumulator (e.g. running softmax stat) |
| `T.copy(src, dst, coalesced_width=...)` | move global↔shared↔fragment; vectorized/coalesced |
| `T.gemm(A, B, C, transpose_B=..., k_pack=..., policy=GemmWarpPolicy.FullRow)` | MFMA-backed tile GEMM |
| `T.reduce_max(...)` / `T.reduce_sum(...)` | row reductions (the FA softmax stats) |
| `T.Parallel(...)` | parallel loop over a tile dim (maps to lanes/threads) |
| `T.Pipelined(range, num_stages=...)` | software-pipelined loop (prefetch depth = `num_stages`) |
| `T.use_swizzle(...)` | block-scheduling swizzle for cache reuse (a.k.a. rasterization) |

A FlashAttention forward is ~80 lines: allocate Q in fragments + K/V in shared, loop KV tiles with
`T.Pipelined`, `S = T.gemm(Q, K, transpose_B=True)`, `m = T.reduce_max(S)`,
`P = exp(S - m)` (via `T.Parallel`), `l = T.reduce_sum(P)`, `O = O*scale + T.gemm(P, V)`,
`T.copy(O, out)`.

## The levers
- **`coalesced_width`** on `T.copy` — set so each lane's global load is ≥128 bit (vectorized). The FA
  kernel exposes `qk_coalesced_width` / `v_coalesced_width` as separate autotune params.
- **`k_pack`** on `T.gemm` — K elements packed per MFMA operand.
- **`policy=GemmWarpPolicy.FullRow`** — warp→tile mapping; `FullRow` assigns full rows of the output tile
  to a warp (good for the MI300X warp-scheduling strategy in the FA blog).
- **`transpose_B=True`** — for `S = Q·Kᵀ` without a physical transpose.
- **`num_stages`** on `T.Pipelined` — prefetch depth; larger overlaps loads with MFMA at the cost of LDS
  / registers (bounded by MI300X's 64 KB LDS).
- **`T.use_swizzle` / `enable_rasterization=True`** — reorder block execution for L2/LLC reuse.

## Pitfalls
- `T.annotate_layout` and other low-level layout hints exist in TileLang but are **not** used in the
  MI300X FA blog — don't assume a primitive is AMD-validated just because it exists; check the example.
- Over-deep `num_stages` overruns the 64 KB LDS budget on gfx942 → the autotuner should reject it, but a
  hand-set value can spill.
- `GemmWarpPolicy` choice interacts with MFMA tile shape; the autotuner sweeps it — don't hard-code
  blindly.
- Tensor-Core acceleration / some intrinsics are flagged "future" in the AMD blog — feature coverage on
  AMD lags the NVIDIA path.

## Verify
- Inspect the emitted config (`block_M`, `block_N`, `threads`, `num_stages`, coalesced widths) from the
  autotuner output and confirm it matches expectations (FA optimal: `128/32/512`, rasterization on).
- Numerics: greedy temp=0 parity vs a reference; isolated kernel bench vs Triton at the same shape.

## Sources
- TileLang FlashAttention on MI300X (primitive names `T.alloc_shared`/`T.alloc_fragment`/`T.copy`/`T.gemm`/`T.reduce_*`/`T.Pipelined`/`T.Parallel`/`T.use_swizzle`, `GemmWarpPolicy.FullRow`, coalesced widths): https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
- tile-ai/tilelang (API surface, examples): https://github.com/tile-ai/tilelang
- TileLang paper (arXiv 2504.17577 — language design / 3-level abstraction): https://arxiv.org/abs/2504.17577
