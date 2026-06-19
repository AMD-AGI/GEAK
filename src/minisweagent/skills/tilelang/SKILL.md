---
name: tilelang
description: >
  Use when rewriting or authoring TileLang kernels (`@tilelang.jit` /
  `@tilelang.autotune` over a `@T.prim_func` written with `T.*` primitives) on
  AMD Instinct GPUs (MI250/MI300X, gfx90a/gfx942). Covers the full lifecycle:
  writing a correct tile program (GEMM, FlashAttention), optimizing it with the
  autotune levers and MI300X LDS/MFMA budgets, and debugging correctness vs a
  reference before quoting perf.
---

# TileLang Kernel Skills

This skill covers rewriting a hot GPU kernel into an optimized TileLang tile
program on AMD (CDNA3/gfx942): **write** (tile programming), **optimize**
(autotune + memory/compute levers), and **debug** (parity-gated correctness).

A TileLang kernel is a `@T.prim_func` device function wrapped by `@tilelang.jit`
(compile) and optionally `@tilelang.autotune` (config search). The compiler maps
`T.gemm` tiles onto MFMA and emits the AMD LDS bank-conflict swizzle for you.

Choose your entry point based on the task:

| Task | Start with |
|------|-----------|
| Write a new TileLang kernel or port from Triton/PyTorch | Tile Programming (below) |
| Improve performance of an existing TileLang kernel | Optimization (below) |
| Fix wrong results, NaN, compile errors, or LDS spills | Debugging (below) |

---

## Tile Programming

Use this workflow to get the first **correct** kernel structure.

1. **Classify** the kernel: GEMM-like (`T.gemm` + LDS staging), attention/softmax
   (online-softmax loop, two GEMMs), reduction (`T.reduce_max/sum` + `T.Parallel`),
   or elementwise (`T.copy` + `T.Parallel`).
2. **Start from a canonical skeleton.** The verified upstream GEMM and
   FlashAttention-fwd tile programs are in `docs/primitives.md` — copy the
   structure (`T.Kernel` grid, `T.alloc_shared`/`T.alloc_fragment`, the
   `T.Pipelined` K/KV loop, `T.gemm(..., transpose_B=True, policy=...)`).
3. **Fill in** compute with `T.*` math (`T.exp`, `T.max`, `T.cast`,
   `T.if_then_else`, `T.infinity`).
4. **Wire up** `@tilelang.jit(out_idx=[...])` and a `ref_program` for parity.

Full primitive vocabulary (verified vs upstream 0.1.11), `GemmWarpPolicy` values,
and both skeletons: `docs/primitives.md`.

---

## Optimization

Use this once the kernel is correct. Structural choices dominate; parameter
tuning is the last lever.

1. **Structural** (highest impact): fusion, the right tile shapes, persistent-Q
   / split-K, removing redundant LDS round-trips.
2. **Memory hierarchy**: `coalesced_width` on `T.copy` (≥128-bit lanes),
   `T.use_swizzle` rasterization for L2/LLC reuse, LDS staging through
   `T.alloc_shared`.
3. **Compute**: `k_pack` (2 on CDNA3), `policy=GemmWarpPolicy.{Square,FullRow,FullCol}`,
   `T.Pipelined(num_stages=...)` to overlap loads with MFMA.
4. **Autotune**: sweep the levers with `@tilelang.autotune` /
   `AutoTuner.from_kernel`. The real upstream FA and GEMM config spaces, both
   autotune styles, and the 64 KB-LDS bound on `num_stages` are in
   `docs/autotune.md`.

How TileLang positions vs Triton and AITER asm (where it wins, where it doesn't):
`docs/vs_triton.md`.

---

## Debugging

Parity-gated triage on a runnable kernel.

1. **Parity first**: `profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)`
   before any perf claim. A fast kernel that fails allclose is not a valid rewrite.
2. **Common failures**: wrong `k_pack` on RDNA, the FA `acc_s → acc_s_cast`
   layout recast (direct on CDNA3, via shared on RDNA), `GemmWarpPolicy`
   M/N-divisibility, and `num_stages` overrunning the 64 KB LDS budget (compile
   error or silent VGPR spill).
3. **Re-tune**: autotuned configs are shape- and build-specific — re-tune per
   serving shape and after ROCm/TileLang upgrades.

Full pitfall list, LDS-budget arithmetic, and MI300X/gfx942-specific traps:
`docs/pitfalls.md`.

---

## Reference Documentation

The `docs/` subdirectory contains the detailed, upstream-verified guides:

- `docs/overview.md` — what TileLang is, the 3-level abstraction, kernel anatomy, MI300X/gfx942 specifics, parity-gating workflow
- `docs/primitives.md` — verified `T.*` API, `GemmWarpPolicy`, canonical GEMM and FlashAttention-fwd skeletons
- `docs/autotune.md` — `@tilelang.autotune` + `@tilelang.jit` and `AutoTuner.from_kernel`, the real FA/GEMM config spaces, LDS-bounded `num_stages`
- `docs/vs_triton.md` — TileLang vs Triton vs AITER asm on MI300X (where each wins)
- `docs/pitfalls.md` — AMD-specific traps: LDS budget, k_pack, warp policy, non-portable configs, parity-before-perf
