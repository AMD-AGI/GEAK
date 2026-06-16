---
title: TileLang — tile DSL on AMD Instinct (CDNA3)
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
  - https://arxiv.org/abs/2511.08083
---

# TileLang overview

## TL;DR
TileLang (`tile-ai/tilelang`) is a Python tile DSL (built on a TVM/TIR backend) that compiles concise
tile programs to AMD via LLVM IR. On MI300X (CDNA3/gfx942) it is **competitive**: FlashAttention fwd
~**1.53× Triton** and ~**2.7× PyTorch**, and FlashMLA reaches **~parity with hand-tuned AITER asm** while
beating Triton ~1.98× (vendor/project-reported). Its strength is editability-with-near-asm-performance in
~80 lines of Python. Its weakness (per HipKittens, arXiv 2511.08083): it lacks abstractions for several
AMD constraints and leans on CUTLASS/CK backend calls — so it is **CDNA3-validated**, not yet a CDNA4
peak path. Choose TileLang for fast iteration on attention; for the absolute ceiling use AITER asm or
[../asm_mfma/](../asm_mfma/overview.md). See [primitives.md](primitives.md), [autotune.md](autotune.md),
[vs_triton.md](vs_triton.md), [pitfalls.md](pitfalls.md).

## Core concepts — the 3-level abstraction
TileLang layers the API so beginners use built-in primitives and experts drop to low-level control:
1. **High level** — declarative tile ops (`T.gemm`, `T.copy`, `T.reduce_*`) with the compiler choosing
   layouts, swizzles, and pipelining.
2. **Mid level** — explicit memory scopes (`T.alloc_shared`, `T.alloc_fragment`), parallel/pipelined
   loops (`T.Parallel`, `T.Pipelined`), and hints (`coalesced_width`, `k_pack`, `GemmWarpPolicy`).
3. **Low level** — layout/swizzle annotations and (on supported targets) direct intrinsic access; the
   compiler still handles AMD-specific bank-conflict swizzling automatically.

The compiler maps tiles onto MFMA ("Auto MatrixCore" support; validated on MI250 and MI300X with Async
Copy support) and emits the LDS swizzle for AMD's bank-conflict rules without code changes.

## The levers
- **Tile sizes** `block_M`, `block_N` (FA optimal config: `block_M=128, block_N=32`).
- **`threads`** per block (512 in the FA optimal config) and **`num_stages`** (software pipeline depth).
- **`num_split_q`** (split the Q dimension across blocks).
- **Coalescing widths** `qk_coalesced_width` / `v_coalesced_width` (vectorized global loads).
- **`GemmWarpPolicy`** (e.g. `FullRow`) — warp→tile mapping for the MFMA.
- **`T.use_swizzle` / rasterization** (`enable_rasterization=True`) — block-scheduling reorder for cache
  reuse.
- **Autotune** over `itertools.product` of the above (108 FA candidates, ~1 s search — see
  [autotune.md](autotune.md)).

## MI300X vs Hopper (what TileLang handles for you on CDNA3)
- **No TMA, no WGMMA** on MI300X → the compiler does not need warp specialization / TMA; tile sizes are
  more flexible (`block_m` need not be a multiple of 64).
- **64 KB shared memory** (vs Hopper 228 KB) → tighter LDS budgeting; the autotuner respects it.
- **Different bank-conflict rules** → a different swizzle strategy, applied automatically (no code diff
  vs the Hopper version of the same kernel).

## Pitfalls (summary — full list in [pitfalls.md](pitfalls.md))
- **CDNA3-only** today: HipKittens notes TileLang lacks abstractions for flexible tile sizing under
  register pressure, thread-block scheduling, and cache-aware grid ordering, and uses **32×32×16 MFMA /
  `buffer_load_dwordx4` / XCD swizzle** less than peak AMD asm. Treat MI350/CDNA4 perf as unproven.
- Reported single-attention-kernel **257 TFLOPs on MI300X** (HipKittens) — well below peak; good for
  iteration, not the ceiling.
- Depends on **CUTLASS/CK** backend calls for some paths.

## Verify
```bash
pip install tilelang        # validated: ROCm 7.0.1, PyTorch 2.9.0, Triton 3.0.0, TileLang 0.1.7
# run the example, compare latency vs Triton/PyTorch at the same shape; check the autotuned config
```
Greedy temp=0 parity vs a reference attention; isolated kernel bench vs Triton at identical b/h/s/d.

## Sources
- Quickly Developing Powerful Flash Attention Using TileLang on AMD MI300X (ROCm Blog, 2026-01-20 — 1.53× Triton, 2.7× PyTorch, 3-level API, MI300X-vs-Hopper notes, ROCm 7.0.1/TileLang 0.1.7): https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
- tile-ai/tilelang (AMD MI250/MI300X support, FlashMLA AMD example ~AITER asm parity, FA2 fwd on MI300X #1406): https://github.com/tile-ai/tilelang
- TileLang paper (arXiv 2504.17577): https://arxiv.org/abs/2504.17577
- HipKittens critique (arXiv 2511.08083 — 257 TFLOPs, lacks AMD-constraint abstractions, CUTLASS/CK deps): https://arxiv.org/abs/2511.08083
