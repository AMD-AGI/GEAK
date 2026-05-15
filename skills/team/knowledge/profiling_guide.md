# Profiling Interpretation Guide (rocprof-compute / omniperf)

## Key Sections to Analyze

### Section 2: System Speed-of-Light (SoL)
The most important section. Shows utilization as percentage of peak for each pipeline.

| Metric | What It Tells You |
|--------|-------------------|
| VALU Utilization | Vector ALU usage (FP/INT compute) |
| VMEM Utilization | Vector memory pipeline (global loads/stores) |
| MFMA Utilization | Matrix instructions (GEMMs, matmuls) |
| LDS Utilization | Local data share usage |
| HBM Bandwidth | Memory bandwidth achieved vs peak (~5.3 TB/s) |

**How to read:** >60% = saturated (bottleneck). 20-60% = moderate. <20% = underutilized.

### Section 7.2: Wavefront Runtime Statistics
Breaks down what wavefronts spend time doing.

| Metric | Meaning |
|--------|---------|
| Active Cycles | Useful compute work |
| Dependency Wait Cycles | Stalled waiting for memory → **memory-bound signal** |
| Issue Wait Cycles | Stalled waiting for execution slots → **compute-bound signal** |
| Barrier Wait Cycles | Waiting at `__syncthreads()` → **load imbalance signal** |

**Key ratio:** Dependency Wait / Active Cycles
- \> 3x → strongly memory-bound
- 1-3x → moderately memory-bound
- < 1x → compute-bound or latency-bound

### Section 11: Compute Pipeline
CU utilization, instruction throughput, and pipeline breakdown.

### Sections 13-16: Cache Hierarchy

| Level | Good Hit Rate | Poor Hit Rate |
|-------|--------------|---------------|
| L1 | >60% | <40% (random/strided access) |
| L2 | >50% | <30% (streaming/working set too large) |

**HBM bandwidth vs peak:** <20% means most data comes from cache (compute-bound) or kernel is too small (latency-bound).

## Bottleneck Classification

### memory-bound
**Evidence:** Dependency Wait >> Active Cycles (>3x), high HBM bandwidth utilization, low cache hit rates.
**Root causes:** Strided access, large working set, no data reuse.
**Optimization directions:**
- Coalesced access patterns
- LDS tiling for data reuse
- Vectorized loads (float4)
- Reduce data movement (algorithmic changes)
- Operation fusion to avoid intermediate buffers

### compute-bound
**Evidence:** High VALU/MFMA utilization (>60%), Issue Wait Cycles significant, low VMEM utilization.
**Root causes:** Too many instructions per element, suboptimal algorithm.
**Optimization directions:**
- Algorithmic rewrites (reduce instruction count)
- Strength reduction (replace expensive ops)
- MFMA utilization (matrix ops)
- Loop unrolling and instruction-level parallelism
- Template parameterization (compile-time constants)

### latency-bound
**Evidence:** Very short kernel duration (<100μs), low utilization everywhere (<20%), few wavefronts.
**Root causes:** Not enough work to fill GPU, kernel launch overhead dominates.
**Optimization directions:**
- Increase work per kernel (fuse operations)
- Persistent kernels
- Batch multiple calls into one kernel
- Increase block size

### lds-bound
**Evidence:** High LDS contention, bank conflict metrics elevated, high Barrier Wait Cycles.
**Root causes:** Bank conflicts from stride patterns, LDS capacity exceeded.
**Optimization directions:**
- Padding to avoid bank conflicts (+1 per row)
- Reorganize LDS layout
- Reduce LDS usage per workgroup
- Split computation to use less LDS

### balanced
**Evidence:** All metrics 20-60% of peak, no single bottleneck dominates.
**Optimization directions:**
- Algorithmic changes that reduce both compute and memory
- Better instruction scheduling
- Compound optimizations

## Profiling Summary Format

After analyzing profiling data, produce a summary with:

1. **PRIMARY BOTTLENECK**: Classification + 2-3 sentences with quantitative evidence.
   - Severity: CRITICAL (dominant, >3x gap) / MODERATE (significant) / MILD (addressable)
2. **KEY METRICS**: List of CAUSE → EFFECT → IMPACT chains.
   - Example: "Strided global reads (stride-3) → 52% L1 hit rate → 40% of peak HBM bandwidth wasted on cache misses"
3. **OPTIMIZATION OPPORTUNITIES**: Ranked by expected impact.
   - HIGH (>20% improvement expected)
   - MEDIUM (5-20%)
   - LOW (<5%)

## Metrics to Ignore

- Values showing 0.0, nan, or empty — missing counter data
- "Parallel execution" info — not relevant for single-kernel optimization
- Device assignment details
- Profiler overhead warnings
