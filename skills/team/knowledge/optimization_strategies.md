# Optimization Strategy Catalog

## Priority Scheme

| Priority | Category | Description |
|----------|----------|-------------|
| 0 | Algorithmic | Novel kernel rewrites: different data structures, different algorithms, work decomposition |
| 1 | Data Reuse | LDS tiling, multi-query workgroups, register blocking, loop tiling |
| 2 | Fusion | Fuse multiple kernels or operations into one |
| 3 | Memory | Coalescing, vectorized loads, layout transformation, prefetching |
| 4 | Compute | Strength reduction, FMA, branchless, unrolling, MFMA |
| 5 | Parallelism | Block sizing, occupancy, persistent kernels, grid-stride loops |
| 6 | Shape-adaptive | Size-specialized variants, template dispatch |
| 8 | Autotuning | Parameter sweeps, compiler flag search |
| 15 | Wrapper/dispatch | Python-side changes, launch config, dtype routing |

**CRITICAL DIRECTIVE:** Kernel algorithmic improvement is the PRIMARY goal. Wrapper/dispatch changes are LOW priority. Even if the kernel appears well-optimized, always attempt algorithmic improvements first. Generate at least 3 genuinely different algorithmic approaches per kernel.

## Per-Bottleneck Strategy Selection

### memory-bound
1. **LDS tiling** (P1): Cache repeatedly-accessed data in LDS. Share across threads/queries.
2. **Vectorized loads** (P3): float4/float2 loads to maximize bandwidth utilization.
3. **Coalesced access** (P3): Ensure consecutive threads access consecutive addresses.
4. **Operation fusion** (P2): Eliminate intermediate memory traffic between stages.
5. **Algorithmic reduction** (P0): Change algorithm to need less data (e.g., spatial indexing, sampling).
6. **Layout transformation** (P3): AoS → SoA or vice versa for access pattern.

### compute-bound
1. **Algorithmic rewrite** (P0): Reduce instruction count via better algorithm.
2. **Template parameterization** (P0/P6): Compile-time constants for sizes → unrolling, register optimization.
3. **Warp-cooperative** (P0): Distribute work across wavefront threads.
4. **Strength reduction** (P4): Replace expensive operations (div → mul, pow → exp+log).
5. **MFMA utilization** (P4): Convert to matrix operations where applicable.
6. **Loop unrolling** (P4): `#pragma unroll` for tight inner loops.

### latency-bound
1. **Kernel fusion** (P2): Combine multiple small kernels.
2. **Increase work per thread** (P5): Process multiple elements per thread.
3. **Persistent kernels** (P5): Single launch, loop over all work items.
4. **Batch operations** (P0): Restructure to process batches in one kernel call.

### lds-bound
1. **Reduce LDS usage** (P1): Use registers instead where possible.
2. **Padding** (P1): Add padding bytes to avoid bank conflicts.
3. **Layout reorganization** (P1): Change LDS access pattern to minimize conflicts.
4. **Split workgroups** (P5): Smaller workgroups that each use less LDS.

### balanced (no single bottleneck)
1. **Algorithmic changes** (P0): Fundamental restructuring that improves both compute and memory.
2. **Data reuse + compute reduction** (P0+P1): Combined approaches.
3. **Better instruction scheduling** (P4): Interleave memory and compute.

## Compound Optimization Strategies

These combine multiple techniques for larger gains. Use in later optimization rounds:

1. **Template + Warp-cooperative**: Compile-time K enables optimal register allocation for warp-cooperative merge. (P0+P0)

2. **Tiled + Multi-query**: LDS tile reference data AND process multiple queries per workgroup. Multiplied bandwidth savings. (P1+P1)

3. **Template + Tiled + Vectorized**: Compile-time sizes enable perfect unrolling of tiled loops with vectorized loads. (P0+P1+P3)

4. **Fusion + Split-K**: Fuse operations AND split work across more workgroups. (P2+P0)

5. **Algorithmic + Layout**: New algorithm designed around the optimal memory layout. (P0+P3)

## Wrapper and Interface Optimizations (Priority 15 — but HIGH IMPACT)

Despite low priority number, these can yield 2-3x improvement by reducing per-call overhead:

1. **Handle data layout variants in kernel**: If the Python wrapper transposes inputs before calling the kernel, move the transposed addressing into the kernel itself. GPU transpose kernels (`.transpose().contiguous()`) cost 10-20μs each. For small problem sizes this dominates total time.

2. **Direct output layout**: If the consumer expects output in format (B,K,M) but the kernel writes (B,M,K) and Python transposes after, change the kernel to write in the consumer's format directly. Saves a GPU memcpy kernel per call.

3. **Eliminate unnecessary outputs**: If a kernel writes both `idx` and `dist2` but only `idx` is used downstream, remove the `dist2` parameter and writes. Saves tensor allocation + global memory bandwidth.

4. **Use `torch.empty` over `new_zeros`**: When the kernel writes every output element, zero-initialization is wasted work. `torch.empty()` skips the GPU memset.

5. **Minimize tensor allocations in hot path**: Each `torch.zeros/empty` call has fixed overhead (~5μs). Reuse pre-allocated buffers when possible.

**When to apply**: These are HIGH VALUE in Round 1 when test shapes include small/medium problems where per-call overhead is significant relative to kernel compute time. For large problems (>100ms kernel time), kernel algorithmic improvements dominate.

## Task Generation Guidelines

When generating tasks for engineers:
- Each task should target a **different priority level** or **different bottleneck aspect**
- At least 2 tasks must be Priority 0-1 (algorithmic/data-reuse)
- Maximum 1 task can be Priority 6+ (tuning/dispatch)
- Each task must include:
  - Specific optimization focus and expected impact
  - Key code locations to modify
  - How to verify correctness
  - Expected metric changes (e.g., "should reduce Dependency Wait Cycles by >50%")
- Tasks must NOT include:
  - Instructions to modify test harness or benchmarks
  - GPU device selection (handled by framework)
  - Build cache clearing (handled by engineer workflow)
