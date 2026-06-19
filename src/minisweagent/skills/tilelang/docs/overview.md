---
title: TileLang — tile DSL on AMD Instinct (CDNA3)
kind: language
gens: [gfx90a, gfx942]
dtypes: [fp16, bf16, fp8]
regimes: [both]
status: competitive
updated: 2026-06-19
sources:
  - https://github.com/tile-ai/tilelang
  - upstream tilelang 0.1.11 (tilelang/language/, examples/amd/, examples/gemm/)
  - https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
  - https://arxiv.org/abs/2504.17577
  - https://arxiv.org/abs/2511.08083
---

# TileLang overview

## TL;DR
TileLang (`tile-ai/tilelang`) is a Python tile DSL on a TVM/TIR backend that compiles concise tile
programs to AMD (HIP) and NVIDIA (CUDA). A kernel is a `@T.prim_func` written with `T.*` primitives,
wrapped by `@tilelang.jit` (compile) and optionally `@tilelang.autotune` (search configs). On MI300X
(CDNA3/gfx942) it is **competitive**: vendor/project-reported FlashAttention fwd ~**1.53× Triton** /
~**2.7× PyTorch**, FlashMLA ~**parity with hand-tuned AITER asm**. Its strength is near-asm performance
with an editable ~80-line tile program. Its weakness (HipKittens, arXiv 2511.08083): it lacks
abstractions for some AMD constraints and leans on CUTLASS/CK backend calls — **CDNA3-validated**, not a
proven CDNA4 peak path. Use TileLang to author/rewrite a hot kernel fast; use AITER asm for the last few
percent. See [primitives.md](primitives.md), [autotune.md](autotune.md), [vs_triton.md](vs_triton.md),
[pitfalls.md](pitfalls.md).

## Core concepts — the 3-level abstraction
1. **High level** — declarative tile ops (`T.gemm`, `T.copy`, `T.reduce_max/sum`) where the compiler
   chooses layouts, LDS swizzle, and pipelining.
2. **Mid level** — explicit memory scopes (`T.alloc_shared`, `T.alloc_fragment`, `T.alloc_var`),
   parallel/pipelined loops (`T.Parallel`, `T.Pipelined`), and hints (`coalesced_width`, `k_pack`,
   `policy=GemmWarpPolicy.*`, `T.use_swizzle`).
3. **Low level** — layout/swizzle annotations (`T.annotate_layout`, `Fragment`/`Layout`) and intrinsics
   (`T.ds_read_tr16_b64`, `T.ldg128`, etc.); the compiler still applies AMD bank-conflict swizzling.

The compiler maps `T.gemm` tiles onto MFMA (MatrixCore) and emits the LDS swizzle for AMD's
bank-conflict rules without code changes.

## Anatomy of a kernel
```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])           # mark which arg(s) are outputs
def matmul(M, N, K, block_M, block_N, block_K, num_stages, threads,
           dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype),
             B: T.Tensor((N, K), dtype),
             C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return main
```
`T.Kernel(grid_x, grid_y[, grid_z], threads=...)` returns the block indices; allocations and loops live
inside the `with` body. The grid order in the AMD GEMM example is `(ceildiv(N,block_N), ceildiv(M,block_M))`
→ `(bx, by)`. Compile with `matmul(...).compile(...)` or call the jitted fn directly.

## The levers (full set in [autotune.md](autotune.md))
- **Tile sizes** `block_M`, `block_N`, `block_K` (GEMM; FA uses `block_M`/`block_N`).
- **`threads`** per block and **`num_stages`** (`T.Pipelined` software-pipeline depth, 0 = none).
- **`num_split_q`** (FA: split/persist the Q dimension across blocks for occupancy).
- **`k_pack`** on `T.gemm` (K elements per MFMA operand; 2 on CDNA3, 1 on RDNA/WMMA).
- **`coalesced_width`** on `T.copy` (`qk_coalesced_width` / `v_coalesced_width` in FA).
- **`policy=GemmWarpPolicy.{Square,FullRow,FullCol}`** — warp→tile mapping.
- **`T.use_swizzle(panel_size, enable=...)`** — threadblock swizzle (a.k.a. rasterization) for L2/LLC
  reuse; `enable_rasterization`/`panel_size` are common autotune params.

## MI300X / gfx942 specifics (what TileLang handles, what to budget)
- **No TMA, no WGMMA** on MI300X → no warp specialization needed; tile sizes are flexible (`block_M` need
  not be a multiple of 64). `T.gemm` dispatches to HIP/MFMA, not WGMMA.
- **64 KB LDS per CU** (vs Hopper 228 KB) → tight LDS budget; this bounds `num_stages` and tile sizes.
  Each shared tile costs `rows*cols*dtype_bytes`; a pipeline of depth `s` roughly multiplies staged
  shared buffers by `s`. The AMD FA example uses `num_stages ∈ {0,1}` and the GEMM example
  `{0,1,2,3}` precisely because of this. See [pitfalls.md](pitfalls.md).
- **Different bank-conflict rules** → a different LDS swizzle, applied automatically (no code diff vs the
  Hopper version of the same kernel).
- **MFMA / k_pack**: CDNA3 uses `k_pack=2`; RDNA WMMA needs `k_pack=1` and `block_M/N ≤ 32`.

## Parity gating (rewrite workflow)
1. Get it **correct** first: `profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)` (FA/GEMM
   examples use 1e-2). Greedy/temp=0 parity vs a PyTorch reference before any perf claim.
2. Then **bench**: `profiler.do_bench()` vs the reference at the exact shape (≥3 warm repeats, median).
3. Only ship a config that passes parity **and** beats the baseline at the serving shape.

## Verify
```bash
pip install tilelang          # AMD: needs a ROCm/HIP build
python examples/amd/example_amd_flash_attn_fwd.py --seq_len 4096 --dim 128
python examples/gemm/example_gemm_autotune.py --use_autotune
```

## Sources
- tile-ai/tilelang upstream (0.1.11): `tilelang/language/` API, `examples/amd/example_amd_flash_attn_fwd.py`, `examples/gemm/example_gemm_autotune.py`, `examples/quickstart.py`: https://github.com/tile-ai/tilelang
- TileLang FlashAttention on MI300X (ROCm Blog — 1.53× Triton, 2.7× PyTorch, 3-level API, MI300X-vs-Hopper notes): https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
- TileLang paper (arXiv 2504.17577): https://arxiv.org/abs/2504.17577
- HipKittens critique (arXiv 2511.08083 — lacks some AMD-constraint abstractions, CUTLASS/CK deps): https://arxiv.org/abs/2511.08083
