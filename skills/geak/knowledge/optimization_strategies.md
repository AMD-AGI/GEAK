# Per-Bottleneck Optimization Strategies

## Task Priority Scheme (lower = higher priority = runs first)

- **0**: Novel algorithmic kernel rewrites (different algorithm, different reduction/scan tree, split kernel variants, eliminate expensive ops)
- **2**: Operation fusion (fuse adjacent kernels, fuse elementwise ops, fuse normalization + quantization)
- **4**: Cross-language kernel rewrite (HIP to Triton or vice versa for launch-overhead-bound kernels)
- **5**: Kernel-body memory access restructuring, computation reordering, LDS optimization, register pressure optimization
- **6**: Shape-adaptive optimization (@triton.autotune, shape-specialized kernel variants)
- **8**: Autotune configs, parameter search (BLOCK_S, num_warps, num_stages)
- **15**: Wrapper/launch-config/dispatch-only changes (lowest priority)

## Bottleneck: Balanced

"Balanced" means no single resource is saturated. Actionable kernel-body approaches:

1. **INCREASE ARITHMETIC INTENSITY**: Fuse adjacent operations into the kernel loop so more compute happens per memory access.
2. **REDUCE MEMORY TRAFFIC**: Cache intermediate results in registers or LDS instead of reading/writing global memory.
3. **IMPROVE PARALLELISM**: Restructure loops to expose more independent work per wavefront; consider split-K or multi-pass approaches.
4. **ALTERNATIVE ALGORITHMS**: Try a fundamentally different algorithm (different reduction tree, different scan, tiled vs non-tiled, etc.).
5. **COMPILER GUIDANCE**: Restructure Triton/HIP code to help the compiler generate better ISA -- avoid `tl.where` in hot loops, use `tl.constexpr` aggressively, minimize live variables across `tl.dot` calls.

## Bottleneck: Memory-Bound

The kernel is limited by memory bandwidth. Focus on kernel-body changes:

1. **VECTORIZED LOADS**: Use float4/float2 vector loads to maximize HBM throughput.
2. **COALESCED ACCESS**: Ensure adjacent threads access adjacent memory addresses.
3. **LDS STAGING**: Stage global memory reads through LDS to improve access patterns.
4. **REDUCE DATA MOVEMENT**: Recompute values instead of storing and reloading them.
5. **OPERATION FUSION**: Fuse the memory-bound kernel with adjacent elementwise ops to amortize memory access cost over more computation.
6. **TILING / BLOCKING**: Increase tile sizes to improve data reuse from L2 cache.

## Bottleneck: Compute-Bound

The kernel is limited by arithmetic throughput. Focus on kernel-body changes:

1. **REDUCE INSTRUCTION COUNT**: Simplify expressions, use hardware intrinsics (`tl.math.rsqrt`, `fma`), eliminate redundant computations.
2. **USE MFMA INSTRUCTIONS**: On AMD GPUs, restructure computation to use Matrix Fused Multiply-Add for dense linear algebra.
3. **STRENGTH REDUCTION**: Replace expensive ops (div, mod, pow) with cheaper equivalents (shifts, masks, lookup tables).
4. **LOOP UNROLLING**: Manually unroll inner loops to help the compiler schedule instructions more aggressively.
5. **ALGORITHM CHANGE**: Switch to an algorithm with lower computational complexity (e.g., O(n log n) vs O(n^2), approximate methods).
6. **TEMPLATE PARAMETERIZATION**: Replace fixed-size per-thread arrays with template parameters to eliminate register waste and boost occupancy. This is critical when arrays are sized for worst-case but runtime values are much smaller.
7. **WARP-COOPERATIVE PARALLELISM**: Change from 1-thread-per-work-item to 1-wavefront-per-work-item. Divide the work across 64 threads, each processing a subset, then merge via LDS reduction. Often yields 5-10x speedup for compute-bound kernels.
8. **SIMPLER DATA STRUCTURES**: Replace complex data structures (heaps, trees) with simpler ones (insertion sort, linear scan) when problem size is small. Simpler structures have less control flow overhead and are more amenable to unrolling.

## Bottleneck: Latency-Bound

The kernel is too short to saturate any resource. Focus on kernel-body changes:

1. **INCREASE WORK PER KERNEL**: Process more elements per thread or per block to amortize kernel launch overhead.
2. **FUSE KERNELS**: Merge this kernel with adjacent ones to eliminate launch gaps.
3. **PERSISTENT KERNEL**: Convert to a persistent kernel pattern that stays resident and processes multiple tiles without relaunching.
4. **INCREASE BLOCK SIZE**: Use larger thread blocks to improve GPU occupancy for this short-running kernel.

## Bottleneck: LDS-Bound

The kernel is limited by LDS (Local Data Share) bandwidth or capacity.

1. **REDUCE LDS BANK CONFLICTS**: Pad shared memory arrays to avoid stride-32 access patterns (on AMD: 32 banks, 4 bytes each).
2. **REDUCE LDS USAGE**: Move data from LDS to registers where possible to free LDS capacity and improve occupancy.
3. **OPTIMIZE LDS ACCESS PATTERN**: Restructure loops so that LDS reads/writes are coalesced within each wavefront.
4. **SPLIT COMPUTATION**: Break the kernel into phases that use LDS at different times to reduce peak LDS pressure.

## Priority Directive

**KERNEL ALGORITHMIC IMPROVEMENT IS THE PRIMARY GOAL.**

The primary goal is **algorithmic improvement of the GPU kernel body** -- the `@triton.jit` functions, HIP `__global__` / `__device__` kernels, CK template bodies, or ASM routines. This means changing *how the computation is performed*: different tiling strategies, different reduction algorithms, fused operations, restructured memory access patterns, alternative scan/sort/attention algorithms -- all **inside** the kernel body itself.

**Wrapper changes are LOW priority**: Launch config tuning (`num_warps`, `BLOCK_SIZE`), Python dispatch changes, import routing changes, and wrapper fixes are acceptable ONLY after exhausting kernel-body approaches.

**Do NOT give up**: Even if the kernel looks well-optimized by human experts, you MUST attempt novel algorithmic improvements. The entire purpose is to discover improvements that humans missed. Generate at least 3-5 genuinely different *algorithmic* approaches per kernel -- not 3-5 variations of launch config parameters.

## Search/Pointer-Chasing Workloads

For latency-bound HIP search workloads (binary_search, lower_bound, KNN, etc.):
- Prioritize branchless search logic
- Operation-specific specialization (remove generic functionality the hot path doesn't need)
- Size-specialized kernel variants for small/medium/huge data paths
- Wavefront-cooperative upper-level search or coarse-index narrowing
- Deprioritize generic vectorization or bandwidth-maximization
