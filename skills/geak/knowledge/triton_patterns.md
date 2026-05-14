# Triton Kernel Optimization Patterns for AMD GPUs

## Backend Detection
A Triton kernel is identified by:
- File extension: `.py`
- Contains `@triton.jit` or `tl.` patterns
- Kernel type: "triton"

## Planning Policy

When optimizing Triton kernels:
- Fill most task slots with "Prefer First" strategies
- Only add autotune/launch/wrapper tasks after at least 3 kernel-body tasks
- Leave GPUs idle rather than spending them on low-value wrapper work

## Prefer First (High Priority)

1. **Algorithmic kernel-body rewrites** that change the reduction tree, tiling scheme, decomposition, or math formulation.
2. **Operation fusion or launch-count reduction** when adjacent work can be merged into the Triton kernel body.

### Memory-Bound Additions
- Memory-access rewrites inside the kernel body: better blocking, fewer redundant loads/stores, and higher SRAM/L2 reuse
- Masking, pointer-arithmetic, or load/store simplifications that reduce HBM traffic on the hottest path

### Compute-Bound Additions
- Instruction-count reduction and control-flow simplification inside hot loops
- MFMA / `tl.dot`-friendly reformulations, cheaper math primitives, or algorithmic approximations when correct

### Latency-Bound Additions
- Fuse adjacent short kernels so each launch performs materially more work
- Increase work per program or use persistent/multi-tile kernel patterns that amortize launch overhead

### LDS-Bound Additions
- LDS-bank-conflict reduction and staged-access restructuring inside the kernel body
- Move transient data from LDS to registers when it reduces LDS pressure without hurting occupancy

## Consider Next (Medium Priority)

- Shape-specialized kernel variants when different input regimes clearly want different algorithms or tile structures
- Kernel-body memory-layout and live-range cleanup that directly supports the hottest profiled path
- Vectorized or blocked load/store patterns (memory-bound kernels)
- Register-pressure and live-range reductions (compute-bound kernels)
- Shape-specialized kernel variants for small vs large shapes (latency-bound kernels)

## Deprioritize Until Later (Low Priority)

- `@triton.autotune`-only config sweeps
- Pure `num_warps` / `num_stages` / `BLOCK_*` parameter search without kernel-body change
- Python dispatch, import-routing, or wrapper-only edits unless profiling clearly shows the wrapper dominates

## Triton-Specific Optimization Techniques

### @triton.autotune
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        # ... more configs
    ],
    key=['M', 'N'],  # Re-tune when these change
)
@triton.jit
def kernel(...)
```

### Tiling & Blocking
- Use `tl.constexpr` for tile sizes to enable compile-time optimization
- Choose tile sizes that divide evenly into common input shapes
- Consider L2 cache size (256 MB on MI300X) when choosing tile sizes

### Memory Access
- Use `tl.load` with `mask` and `other` for boundary handling
- Prefer contiguous loads (stride-1 in the innermost dimension)
- Use `tl.store` with `mask` for boundary-safe writes
- Consider `eviction_policy` hints for streaming vs. reuse patterns

### Reduction Patterns
- `tl.sum`, `tl.max`, `tl.min` for intra-block reductions
- For cross-block reductions, consider atomic operations or multi-pass approaches
- Split-K patterns for GEMMs: split K dimension across blocks, atomically accumulate

### Compiler Hints
- `tl.constexpr` for values known at compile time
- Avoid `tl.where` in hot loops (creates predication overhead)
- Minimize live variables across `tl.dot` calls (register pressure)
- Use `tl.math.rsqrt` instead of `1.0 / tl.sqrt(x)`

### AMD-Specific Triton
- `waves_per_eu` controls occupancy on AMD
- `tl.dot` maps to MFMA instructions on gfx942
- LDS (shared memory) usage visible via `num_stages` pipeline depth
- AMD wavefronts are 64-wide (not 32 like NVIDIA warps)

### Constexpr Parameterization (High Impact)
Use `tl.constexpr` for values known at JIT time to enable compile-time optimization:
```python
@triton.jit
def kernel(K: tl.constexpr, BLOCK_SIZE: tl.constexpr, ...):
    # K and BLOCK_SIZE are compile-time constants
    # Compiler can eliminate dead branches, unroll perfectly, and optimize register allocation
    local_data = tl.zeros([K], dtype=tl.float32)
```

### Multi-Program Cooperative Patterns
For compute-bound kernels, consider splitting work across multiple programs that cooperate:
```python
@triton.jit
def kernel_cooperative(pid, num_programs: tl.constexpr, N, ...):
    # Each program processes N/num_programs elements
    start = pid * (N // num_programs)
    end = min(start + N // num_programs, N)
    # Process subset, write partial results to output
    # Host-side: launch with grid=(num_programs,) then reduce
```
