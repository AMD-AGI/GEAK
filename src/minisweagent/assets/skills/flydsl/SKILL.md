---
name: flydsl
description: >
  Use when working with FlyDSL kernels (`@flyc.kernel` / `flydsl.compiler`) on
  AMD GPUs. Covers three complementary workflows: writing new tile-programmed
  kernels, optimizing existing kernels for performance, and debugging
  correctness issues (NaN, wrong results, compilation errors, hangs).
---

# FlyDSL Kernel Skills

This skill covers the full lifecycle of FlyDSL GPU kernels on AMD GPUs:
**write** (tile programming), **optimize** (performance tuning), and
**debug** (correctness triage).

Choose your entry point based on the task:

| Task | Start with |
|------|-----------|
| Write a new `@flyc.kernel` from scratch or port from Triton | Tile Programming (below) |
| Improve performance of an existing kernel | Optimization (below) |
| Fix NaN, wrong results, compilation errors, or hangs | Debugging (below) |

---

## Tile Programming

Use this workflow to design the first correct kernel structure with FlyDSL's
tile programming model (CuTe-style layout algebra).

**Scope**: Start here for a new kernel structure. Switch to Debugging once you
have runnable code, then to Optimization once correctness is established.

### Kernel Type Classification

| Pattern | Examples | Key Primitives |
|---------|----------|---------------|
| **Elementwise** | vecadd, scale, relu, abs | `logical_divide` + `copy_atom_call` |
| **Reduction** | sum, max, softmax, layernorm | `buffer_load` + warp shuffle + LDS |
| **Tiled Copy** | transpose, permute, gather | `zipped_divide` + `TiledCopy` |
| **GEMM** | matmul, batched gemm | `TiledMma` + `TiledCopy` + LDS |
| **Fused** | fused attention, GEMM+epilogue | Combine GEMM + elementwise |

### Design Steps

1. **Classify** the kernel type using the table above
2. **Generate** the appropriate skeleton from pattern templates
3. **Fill in** compute logic using FlyDSL arith ops on vectors
4. **Add** synchronization and shared memory if needed
5. **Test** and debug using the common error table

Full tile programming guide with kernel skeletons, compute recipes, control flow,
LDS usage, and MFMA reference: `docs/flydsl_tile_programming.md`

---

## Optimization

Use this workflow when optimizing an existing FlyDSL kernel for performance.
Parameter tuning alone yields marginal gains. **Prioritize structural
optimizations in early patches**, then fall back to tuning in later patches.

### Optimization Priority

1. **Structural** (highest impact): kernel fusion, fast-path relaxation, loop restructuring, redundant work elimination, algorithm replacement
2. **Memory hierarchy** (medium): LDS utilization, vectorized access, load/compute overlap, data layout
3. **Compute** (medium): MFMA instruction selection, software pipelining, scheduler tuning
4. **Parameter tuning** (low): block size, tile dimensions, unroll factors

### Bottleneck Classification

- **Memory-bound** → reduce data movement (fusion, LDS caching, vectorization)
- **Compute-bound** → improve instruction throughput (MFMA selection, software pipelining)
- **Latency-bound** (small shapes) → reduce kernel launch count (fusion)

Full optimization workflow and detailed strategies: `docs/flydsl_optimization.md`

---

## Debugging

Use this workflow for correctness, stability, and hang triage on runnable
FlyDSL kernels.

### Debug Strategy (classify → isolate → fix)

1. **Cache check**: If a fix looks ineffective, rerun with `FLYDSL_RUNTIME_ENABLE_CACHE=0`
2. **Classify the error** using the table below
3. **Isolate** with diagnostic workflow (all-1s test, single-partition, host-side prints)
4. **Fix** using the pattern-specific guidance in the debug doc

### Error Classification

| Symptom | Likely Cause |
|---|---|
| All NaN output | Softmax -inf/-inf, division by zero |
| All zeros output | Wrong output address, uninitialized buffer |
| >50% mismatch | Wrong partition count, layout mismatch |
| 1-5% mismatch | FP8 quantization, scale factor |
| Compilation error | Type mismatch, range vs range_constexpr |
| GPU hang | Infinite loop, barrier deadlock |

### Common Pitfalls Checklist

- [ ] `range_constexpr()` for compile-time loops (not `range()`)
- [ ] No Python `if` on runtime GPU values
- [ ] `buffer_load` offset units match dtype
- [ ] `vector.store` uses vector type (not scalar)
- [ ] `scf.for` state packed with raw SSA values
- [ ] MFMA operand order: `mfma(LHS, RHS, acc)`

Full debugging guide with detailed fixes and diagnostic workflow: `docs/flydsl_debug_kernel.md`

---

## Reference Documentation

The `docs/` subdirectory contains detailed guides:

- `flydsl_tile_programming.md` — Kernel skeletons, compute recipes, control flow, LDS, MFMA reference
- `flydsl_optimization.md` — Optimization workflow, tier-by-tier strategies, correctness constraints, key APIs
- `flydsl_debug_kernel.md` — NaN/zeros debugging, mismatch triage, compilation errors, GPU hangs
