---
title: TileLang primitives — the tile-level API
kind: language
gens: [gfx90a, gfx942]
dtypes: [fp16, bf16, fp8]
regimes: [both]
status: competitive
updated: 2026-06-19
sources:
  - https://github.com/tile-ai/tilelang
  - upstream tilelang 0.1.11 (tilelang/language/__init__.py, allocate.py, copy_op.py, gemm_op.py, reduce_op.py, loop.py, annotations.py)
  - https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
---

# TileLang primitives

## TL;DR
A TileLang kernel is a `@T.prim_func` written with `T.*` primitives, wrapped by `@tilelang.jit`
(+ optional `@tilelang.autotune`). The vocabulary below is **verified against upstream
`tilelang/language/` (0.1.11)** — these symbols are all exported from `tilelang.language`. The covered
areas: kernel/grid context, memory scopes, data movement, MFMA-backed compute, reductions, parallel and
pipelined loops, and scalar/elementwise math. See [overview.md](overview.md) for the 3-level model and
[autotune.md](autotune.md) for the tuning decorators.

## Verified primitive vocabulary
| Primitive | role |
|---|---|
| `T.prim_func` | decorator marking the device function inside the jitted wrapper |
| `T.Tensor(shape, dtype)` | typed kernel argument (global tensor) |
| `T.Kernel(gx, gy[, gz], threads=...)` | grid/block context manager; yields block index tuple |
| `T.ceildiv(a, b)` | grid-sizing helper |
| `T.alloc_shared(shape, dtype)` | allocate an **LDS** (shared-memory) tile |
| `T.alloc_fragment(shape, dtype)` | allocate a **register/fragment** tile (per-lane MFMA storage) |
| `T.alloc_local(shape, dtype)` | allocate thread-local registers |
| `T.alloc_var(dtype)` | a scalar (e.g. a running index/stat) |
| `T.fill(buf, value)` / `T.clear(buf)` | initialize a tile (`clear` = fill 0) |
| `T.copy(src, dst, coalesced_width=...)` | move global↔shared↔fragment; vectorized/coalesced |
| `T.gemm(A, B, C, transpose_A=, transpose_B=, k_pack=, policy=)` | MFMA-backed tile GEMM, accumulates into C |
| `T.reduce_max(buf, out, dim=, clear=)` / `T.reduce_sum(buf, out, dim=, clear=)` | row/col reductions (softmax stats) |
| `T.reduce_abssum`, `T.cumsum`, `T.cummax` | other reductions / scans |
| `T.Parallel(*extents)` | parallel loop over tile dims (maps to lanes/threads) |
| `T.Pipelined(stop, num_stages=...)` | software-pipelined serial loop (prefetch depth = `num_stages`) |
| `T.serial`, `T.unroll`, `T.vectorized` | other loop forms |
| `T.use_swizzle(panel_size, order="row", enable=...)` | threadblock swizzle (rasterization) for cache reuse |
| `T.annotate_layout(...)` | low-level layout hint (advanced) |
| `T.exp`, `T.max`, `T.cast`, `T.if_then_else`, `T.infinity(dtype)`, `T.abs`, `T.log` | scalar/elementwise math |

`GemmWarpPolicy` is imported from `tilelang.language` (`from tilelang.tileop.base import GemmWarpPolicy`);
values: **`Square = 0`, `FullRow = 1`, `FullCol = 2`** (verified in `tilelang/tileop/base.py`).

## FlashAttention-fwd skeleton (canonical, MI300X — upstream `examples/amd/example_amd_flash_attn_fwd.py`)
This is the **real** upstream AMD structure (FA-V2, online softmax, Q persisted across blocks). Key
points: K/V tiles in shared, accumulators in fragments, `transpose_B=True` for `S = Q·Kᵀ`,
`policy=GemmWarpPolicy.FullRow`, and the f32→f16 `acc_s → acc_s_cast` recast between the two GEMMs.
```python
with T.Kernel(num_split_q, batch * heads, threads=threads) as (b_split, byz):
    T.use_swizzle(panel_size, enable=enable_rasterization)
    bz, by = byz // heads, byz % heads
    num_q_blocks = T.ceildiv(seq_len, block_M)
    bx = T.alloc_var(T.int32); bx = b_split
    while bx < num_q_blocks:                                   # persistent Q loop
        acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
        m_i   = T.alloc_fragment([block_M], accum_dtype)
        l_i   = T.alloc_fragment([block_M], accum_dtype)
        T.fill(acc_o, 0); T.fill(m_i, -T.infinity(accum_dtype)); T.fill(l_i, 0)
        q0 = bx * block_M
        Q_shared = T.alloc_shared([block_M, dim], dtype)
        K_shared = T.alloc_shared([block_N, dim], dtype)
        V_shared = T.alloc_shared([block_N, dim], dtype)
        acc_s      = T.alloc_fragment([block_M, block_N], accum_dtype)
        acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
        T.copy(Q[bz, q0:q0+block_M, by, :], Q_shared, coalesced_width=qk_coalesced_width)
        loop_end = T.ceildiv(q0 + block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)
        for k in T.Pipelined(loop_end, num_stages=num_stages):
            kv = k * block_N
            T.copy(K[bz, kv:kv+block_N, by//groups, :], K_shared, coalesced_width=qk_coalesced_width)
            T.copy(V[bz, kv:kv+block_N, by//groups, :], V_shared, coalesced_width=v_coalesced_width)
            T.clear(acc_s)                                     # or causal mask via T.if_then_else
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                   k_pack=k_pack, policy=GemmWarpPolicy.FullRow)
            # --- online softmax: update m_i, rescale acc_o/l_i, exp(acc_s - m_i) ---
            m_prev = T.alloc_fragment([block_M], accum_dtype)
            T.copy(m_i, m_prev); T.reduce_max(acc_s, m_i, dim=1, clear=False)
            for i in T.Parallel(block_M): m_i[i] = T.max(m_i[i], m_prev[i])
            for i in T.Parallel(block_M):
                sf = T.exp(m_prev[i]*scale - m_i[i]*scale); l_i[i] *= sf
                for j in T.Parallel(dim): acc_o[i, j] *= sf      # rescale running output
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp(acc_s[i, j]*scale - m_i[i]*scale)
            row_sum = T.alloc_fragment([block_M], accum_dtype)
            T.reduce_sum(acc_s, row_sum, dim=1)
            for i in T.Parallel(block_M): l_i[i] += row_sum[i]
            T.copy(acc_s, acc_s_cast)                          # f32 D-layout -> f16 A-layout
            T.gemm(acc_s_cast, V_shared, acc_o, policy=GemmWarpPolicy.FullRow)
        for i, j in T.Parallel(block_M, dim):
            Output[bz, q0+i, by, j] = acc_o[i, j] / l_i[i]     # final normalize
        bx = bx + num_split_q                                  # advance persistent Q index
```
Notes for a rewrite: `scale = (1/dim)**0.5`; on **RDNA/WMMA** the `acc_s → acc_s_cast` recast must route
through a `T.alloc_shared([block_M, block_N])` tile (D and A register layouts differ) and `block_M/N ≤ 32`,
`k_pack=1`. On **CDNA3 (MI300X)** the direct fragment `T.copy(acc_s, acc_s_cast)` is correct, `k_pack=2`.

## GEMM skeleton (canonical — upstream `examples/gemm/example_gemm_autotune.py`)
```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
    A_shared = T.alloc_shared((block_M, block_K), dtype)
    B_shared = T.alloc_shared((block_N, block_K), dtype)
    C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
    C_shared = T.alloc_shared((block_M, block_N), dtype)
    T.use_swizzle(panel_size=10, enable=enable_rasterization)
    T.clear(C_local)
    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
        T.copy(A[by * block_M, k * block_K], A_shared)
        T.copy(B[bx * block_N, k * block_K], B_shared)
        T.gemm(A_shared, B_shared, C_local, transpose_B=True)  # B is (N, K): A @ Bᵀ
    T.copy(C_local, C_shared)
    T.copy(C_shared, C[by * block_M, bx * block_N])            # stage through LDS before global store
```

## The levers
- **`coalesced_width`** on `T.copy` — vectorize each lane's global load (FA uses 8 for QK, 4 for V).
- **`k_pack`** on `T.gemm` — K per MFMA operand (CDNA3: 2; RDNA: 1).
- **`policy=GemmWarpPolicy.{Square,FullRow,FullCol}`** — warp→output-tile mapping; FA uses `FullRow`.
- **`transpose_A` / `transpose_B`** — for `Q·Kᵀ` or `A·Bᵀ` without a physical transpose.
- **`num_stages`** on `T.Pipelined` — prefetch depth; larger overlaps loads with MFMA at the cost of LDS
  (bounded by MI300X's 64 KB LDS — see [pitfalls.md](pitfalls.md)).
- **`T.use_swizzle(panel_size, enable=...)`** — threadblock-schedule swizzle for L2/LLC reuse.

## Pitfalls
- `T.annotate_layout` / `Fragment`/`Layout` exist but are advanced; the AMD FA/GEMM examples do **not**
  need them. Don't hand-write layouts unless you have a measured reason.
- Over-deep `num_stages` overruns the 64 KB LDS budget on gfx942 → compile failure or VGPR spill; the
  autotuner should reject it, a hand-pin can break.
- `GemmWarpPolicy` interacts with MFMA tile shape and requires M (FullRow) / N (FullCol) divisibility by
  the warp factor — let the autotuner sweep it rather than hard-coding.
- The bridging recast between FA's two GEMMs is layout-sensitive: direct on CDNA3, via shared on RDNA.

## Verify
- Inspect the winning config from the autotuner and confirm it is physically valid (LDS, VGPR).
- Numerics: `profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)`; then bench vs Triton/PyTorch
  at the exact shape.

## Sources
- upstream `tilelang/language/__init__.py` + `allocate.py`/`copy_op.py`/`gemm_op.py`/`reduce_op.py`/`loop.py`/`annotations.py` (0.1.11) — exact exported `T.*` symbols and signatures.
- `examples/amd/example_amd_flash_attn_fwd.py` — canonical MI300X FA-fwd structure (`GemmWarpPolicy.FullRow`, `k_pack`, coalesced widths, persistent-Q, online softmax).
- `examples/gemm/example_gemm_autotune.py` — canonical GEMM tile program.
- TileLang FlashAttention on MI300X (ROCm Blog): https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
