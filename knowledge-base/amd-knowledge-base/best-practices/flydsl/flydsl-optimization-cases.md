---
layer: "best-practices"
category: "flydsl"
subcategory: "optimization"
tags: ["flydsl", "kernel-optimization", "gpu", "mfma", "fusion"]
rocm_version: "7.0+"
last_updated: 2026-03-31
---

# FlyDSL Kernel Optimization — Proven Patterns

General-purpose optimization patterns from successful GEAK runs on FlyDSL kernels. Ordered by observed impact.

---

## High-Impact Patterns

### 1. Kernel Fusion

**When**: The `@flyc.jit` host wrapper launches 2+ `@flyc.kernel` functions that share input data, especially when individual kernels are memory-bound or small (launch-overhead-dominated).

**How**: Merge multiple kernels into one. Assign thread subgroups to cover different original workloads within a single warp. Load shared data once.

**Why it works**: Eliminates launch overhead and redundant HBM reads. For memory-bound micro-kernels, removing one launch + one redundant global read often dominates any compute-level optimization.

**Measured**: Significant speedup on a dual-kernel RoPE+KV-cache operation where both originals loaded the same cos/sin caches.

### 2. Fast-Path Relaxation

**When**: The kernel has optimized code paths guarded by overly-restrictive conditions (e.g. `if False and ...`, `N % tile_cols == 0` when `N % vec_width == 0` would suffice, `causal`-only paths that also work for non-causal).

**How**: Relax the guard condition so more shapes/modes use the fast vectorized path. Ensure the relaxed condition still guarantees alignment/correctness.

**Why it works**: The fast path exists because it's faster — the restrictive guard was likely added during development for safety and never relaxed.

**Measured**: Significant speedup on RMSNorm by changing the fast-path condition from `N % (BLOCK_THREADS * VEC_WIDTH) == 0` to `N % vec_width == 0`, letting all benchmark shapes use vectorized loads. Also effective on Flash Attention by removing an unnecessary `causal`-only restriction on the N128 path.

### 3. Loop Restructuring (Constexpr → SCF)

**When**: The kernel uses `range_constexpr` to unroll a loop with a large iteration count, causing code bloat and register pressure.

**How**: Convert to `scf.for` with loop-carried state (flatten/unflatten helpers for complex state). Keep `range_constexpr` for small loops where unrolling is beneficial.

**Why it works**: Constexpr unrolling duplicates the loop body N times in the binary. For large N this bloats instruction cache and increases register pressure. `scf.for` emits a single loop body with explicit state passing.

**Measured**: Major speedup on MoE GEMM 2-stage by converting the ping-pong main loop from constexpr unrolling to scf.for when pair iterations exceeded a threshold.

---

## Medium-Impact Patterns

### 4. Software Pipelining / Load-Compute Overlap

**When**: The kernel has a main loop that loads data, then computes on it, with no overlap between the two phases.

**How**: Move global loads earlier so they complete while ALU/MFMA work is in progress. Use `sched_barrier`, `sched_mfma`, `sched_dswr`/`sched_dsrd` to control instruction interleaving.

**Measured**: Notable speedup on Flash Attention by moving V global-to-LDS store into the exp2 computation window, overlapping memory with compute.

### 5. Pre-loading Across Passes

**When**: The kernel makes multiple passes over data (e.g. pass 1: reduction, pass 2: normalize+output) and loads data in pass 2 that could have been loaded in pass 1.

**How**: Load data needed by pass 2 (e.g. gamma weights) during pass 1, cache in registers or local arrays, and reuse in pass 2.

**Measured**: Contributed to RMSNorm speedup — pre-loading gamma during the sumsq reduction pass eliminated redundant HBM reads in the normalization pass.

### 6. Scheduler Tuning

**When**: The kernel uses `sched_mfma`, `sched_dswr`, `sched_dsrd` for instruction scheduling.

**How**: Verify that group counts match actual instruction counts. Off-by-one errors in `sched_dswr` timing can cause pipeline bubbles.

**Measured**: Single-line scheduler fix on a preshuffle GEMM kernel yielded notable improvement.

### 7. MFMA Instruction Selection

**When**: The kernel uses MFMA intrinsics that may not be available or optimal on the target arch.

**How**: Check `get_hip_arch()` and select the correct variant. When a wider MFMA isn't available, split into multiple narrower calls and adjust scheduler accordingly.

**Measured**: Correct MFMA selection + NaN sanitization on MoE GEMM when switching from unavailable scaled-MFMA to the correct fp8 variant with doubled call count.

---

## Low-Impact Patterns

### 8. Block and Tile Size Tuning

Adjust `BLOCK_THREADS`, tile dimensions, and `known_block_size` hints. Low risk, small gains.

### 9. Reduction Mode Selection

Try `ds_bpermute` vs `xor` for wavefront-level reductions. Architecture-dependent.

### 10. Vectorization Width Adjustment

Match vector width to element size for 128-bit loads: `vec(8, T.f16)` or `vec(4, T.f32)`.

---

## Correctness Patterns (No speedup, prevents silent corruption)

- **FP8 NaN Sanitization**: E4M3FNUZ uses `0x80` as NaN. AND each byte with `0x7F` at load time.
- **f32→f16 Range Clamping**: Clamp to ±65504 before `arith.trunc_f` to avoid Inf.
- **LDS Budget Verification**: Always verify total LDS bytes against arch limit. Overlap non-concurrent allocations when near the limit.
