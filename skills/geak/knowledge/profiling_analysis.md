# rocprof-compute Profiling Analysis Guide

## How to Profile

```bash
# Profile a kernel
rocprof-compute profile --no-roof -- python3 /path/to/harness.py --profile

# Analyze the results
rocprof-compute analyze -p /path/to/workloads/<timestamp>/ --report-diff -o /path/to/output/
```

## Analysis Guidelines

When analyzing rocprof-compute output, focus on these high-signal sections:

### Section 2: System Speed-of-Light
- Focus on "Pct of Peak" columns
- This tells you how close you are to hardware limits
- Key metrics: VALU utilization, VMEM utilization, MFMA utilization, HBM bandwidth

### Section 7.2: Wavefront Runtime Stats
- **Dependency Wait Cycles**: Time waiting for data (memory-bound signal)
- **Issue Wait Cycles**: Time waiting for execution slots (compute-bound signal)
- **Active Cycles**: Time doing useful work
- Ratio of Dependency Wait / Active = memory pressure indicator

### Section 11: Compute Pipeline
- Utilization and throughput of compute units
- Low utilization = kernel not compute-bound or poor occupancy

### Cache Sections (13-16)
- **L1 Hit Rate**: Should be >60% for non-streaming access
- **L2 Hit Rate**: <50% usually indicates random or streaming access
- **HBM Bandwidth**: Compare against peak (~5.3 TB/s for MI300X)

## Key Diagnostic Ratios

1. **Kernel Efficiency**: Active Cycles / Total Wave Cycles
   - <20% = severe bottleneck, CRITICAL
   - 20-60% = moderate, room for improvement
   - >60% = acceptable

2. **Memory vs Compute Bound**:
   - Dependency Wait >> Active Cycles = memory-bound
   - Issue Wait >> Active Cycles = compute-bound
   - Both similar = balanced

3. **Cache Effectiveness**:
   - Hit rates >80% = good cache utilization
   - Hit rates 60-80% = some optimization possible
   - Hit rates <60% = likely memory-bound

## Bottleneck Classification

Based on the analysis, classify the kernel as one of:

### memory-bound
- **Evidence**: High Dependency Wait Cycles, low cache hit rates, high HBM traffic
- **Key indicator**: Dependency Wait / Active > 3x
- **Focus**: Reduce memory traffic, improve cache locality, use vector loads

### compute-bound
- **Evidence**: High VALU/MFMA utilization, Issue Wait cycles
- **Key indicator**: Compute utilization >60% of peak
- **Focus**: Reduce instruction count, use MFMA, strength reduction

### latency-bound
- **Evidence**: Very short kernel duration, low resource utilization across the board
- **Key indicator**: Kernel duration <100us with <20% utilization
- **Focus**: Increase work per kernel, fuse kernels, persistent patterns

### lds-bound
- **Evidence**: High LDS bank conflicts, high LDS utilization
- **Key indicator**: LDS contention metrics elevated
- **Focus**: Reduce bank conflicts, reduce LDS footprint, split phases

### balanced
- **Evidence**: No single resource is saturated, moderate utilization everywhere
- **Key indicator**: All metrics 20-60% of peak
- **Focus**: Algorithmic rewrites, fusion, increase arithmetic intensity

## Required Output Format

When reporting profiling analysis, structure as:

### 1. PRIMARY BOTTLENECK (2-3 sentences)
- State the specific resource/subsystem that is the bottleneck
- Provide quantitative evidence (e.g., "X% of peak", "Y cycles wasted")
- Explain WHY this is occurring (root cause)
- Severity: CRITICAL (<20% of peak), MODERATE (20-60%), ACCEPTABLE (>60%)

### 2. KEY METRICS (3-4 observations)
For each: CAUSE -> EFFECT -> IMPACT chain

### 3. OPTIMIZATION DIRECTIONS (2-3, ranked by impact)
HIGH IMPACT (>20% improvement potential):
- Code-level approach
- Target metrics
- Rationale

MEDIUM IMPACT (5-20%):
- Secondary optimization

## Metrics to Ignore
- Metrics showing 0.0, nan, or empty values (not active for this kernel)
- Parallel execution info (handled by framework)
- Device assignment info (handled by framework)
