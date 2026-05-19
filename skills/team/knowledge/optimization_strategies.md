# Optimization Strategy Catalog

## Priority Hierarchy

Priority determines which strategies to try first. Lower number = higher priority = try first.

| Priority | Category | Description |
|----------|----------|-------------|
| P0 | Algorithm Restructuring | Novel algorithmic rewrites: template params, warp-cooperative, complexity reduction |
| P1 | Data Reuse | Shared memory tiling, register blocking, cache optimization |
| P2 | Memory Access | Coalescing, vectorized loads, SoA layouts, non-temporal hints |
| P3 | Compute | Branchless patterns, ILP, FMA, loop unrolling |
| P4 | Launch Config | Block size, grid size, occupancy tuning, launch bounds |
| P5 | Autotuning | Parameter search, multi-config dispatch |

| PW | Wrapper/Binding | Python wrapper overhead: autograd bypass, output format, allocation reduction |

**Rule**: At least 2/3 of engineer tasks per round MUST be P0-P2 (kernel algorithmic work). At most 1 task can be P4-P5 (tuning) or PW (wrapper). Exception: when overhead detection triggers (all cases at similar latency), PW becomes mandatory.

## Critical Pattern Detection

Before bottleneck-driven selection, check for these high-impact patterns:

**Search/Scan Pattern**: If the kernel has 1 thread iterating over N elements (brute-force search, argmin/argmax, top-K selection, reduction over large arrays):
→ **ALWAYS assign warp-cooperative as the #1 priority task in Round 1.** This pattern gives 5-30x speedup. See `hip_optimization.md` → "Warp-Cooperative Algorithms" for the pattern with shared-memory merge.

**Oversized Arrays**: If the kernel declares arrays with hardcoded large sizes (e.g., `float arr[100]`) but actual sizes are much smaller at runtime:
→ **ALWAYS assign template parameterization as a top priority task.** This eliminates register spill.

These two patterns often compose well together (template + warp-cooperative = excellent). Assign them to different engineers in the same round for potential merge.

## Bottleneck-Driven Strategy Selection

### Memory-Bound (HBM bandwidth > 60% utilized, compute < 40%)
1. P1: LDS/shared memory tiling to reduce global memory traffic
2. P2: Coalesced access patterns (SoA layout)
3. P2: Vectorized loads (float4)
4. P0: Algorithmic data reuse (e.g., tiled matrix multiply)
5. P2: Non-temporal hints for streaming data
6. P3: Mixed precision (fp16 loads → fp32 compute)

### Compute-Bound (ALU utilization > 60%, memory < 40%)
1. P0: Algorithmic complexity reduction (O(N²) → O(N log N))
2. P3: Instruction-level parallelism (interleave independent ops)
3. P3: FMA usage (fmaf instead of mul+add)
4. P0: Warp-cooperative work distribution
5. P3: Branchless computation (eliminate divergence)
6. P5: Mixed precision compute (fp16 ALU has 2x throughput)

### Latency-Bound (Low utilization on both memory and compute)
1. P0: Increase parallelism (more work per thread, more threads)
2. P4: Launch configuration tuning (more blocks, better occupancy)
3. P1: Prefetching / software pipelining
4. P0: Warp-cooperative to increase work per wavefront
5. P3: Reduce instruction dependencies (break dependency chains)
6. P4: Persistent threads for small workloads

### LDS-Bound (High LDS utilization, bank conflicts)
1. P1: Padding to avoid bank conflicts (+1 padding)
2. P1: Restructure LDS access patterns
3. P1: Reduce LDS usage (split into multiple passes)
4. P2: Use registers instead of LDS where possible
5. P4: Reduce block size to increase LDS per thread

### Balanced (No single dominant bottleneck)
1. P0: Template parameterization (reduce register spill, improve everything)
2. P0: Warp-cooperative algorithms (improve both compute and memory efficiency)
3. P1: Tiled data loading (improve memory, free up compute)
4. P2: Vectorized + coalesced access
5. P3: Loop unrolling + register blocking
6. P4: Launch configuration tuning

## Task Generation Guidelines

When creating engineer tasks for a round:

1. **Diversity**: Each task must target a DIFFERENT strategy category. No two tasks in the same round should use the same P-level approach.

2. **Independence**: Tasks should modify different parts of the kernel or use completely different approaches so patches can potentially be merged.

3. **Specificity**: Each task prompt must include:
   - The specific optimization technique to apply
   - Which part of the kernel to modify
   - Why this approach is expected to help (based on profiling data)
   - Quantitative target (e.g., "target 2x reduction in global memory loads")

4. **Adaptation**: After each round, the bottleneck likely shifts. Re-profile and generate new tasks targeting the NEW bottleneck, not the old one.

### Overhead-Bound (All test cases at similar latency regardless of problem size)

This indicates the bottleneck is Python/C++ wrapper overhead, NOT kernel compute. The kernel GPU time is negligible compared to framework overhead.

1. PW: Replace `torch.autograd.Function.apply()` with `@torch.no_grad()` direct function (3-5us saved)
2. PW: Use `torch.empty()` instead of `torch.zeros()` / `new_zeros()` (1-3us per allocation)
3. PW: Modify kernel to output in expected format — avoid `.transpose().contiguous()` (3-20us)
4. PW: Remove unnecessary output buffer allocations (e.g., scratch buffers unused by callers) (2-5us)
5. PW: Add specialized dispatch paths for template-supported parameters
6. PW: Skip `CHECK_CONTIGUOUS` in C++ binding (caller ensures contiguous)

See `wrapper_optimization.md` for complete patterns. **This is the ONLY category that requires modifying Python wrapper and C++ binding files** — all other categories modify only the kernel source.

## Compound Strategies

Some optimizations compose multiplicatively. When planning multiple rounds, consider these powerful combinations:

**Round 1 → Round 2 compounding:**
- Template parameterization → then LDS tiling (fewer registers = more LDS available)
- Warp-cooperative → then vectorized loads (fewer threads = wider loads per thread)
- Coalesced access → then prefetching (coalesced loads benefit more from prefetch)

**Within-round compatibility (for merge engineer):**
- Template + Launch bounds → compatible (both reduce register usage)
- LDS tiling + Coalesced access → compatible (tiling enables coalescing)
- Branchless + Loop unrolling → compatible (independent transforms)
- Two different tiling schemes → INCOMPATIBLE (LDS conflict)
- Two different warp-cooperative approaches → INCOMPATIBLE
