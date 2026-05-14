# Phase 3: Kernel Profiling

## Objective
Profile the kernel using rocprof-compute (formerly omniperf), classify the bottleneck,
and extract baseline performance metrics.

## Steps

### 3.1 Run Baseline Benchmark

First, establish the baseline performance:
```bash
python3 "$HARNESS_PATH" --benchmark
```

Extract `GEAK_RESULT_LATENCY_MS` from the output. This is the baseline latency.

Run the full benchmark for authoritative measurement:
```bash
python3 "$HARNESS_PATH" --full-benchmark
```

Save raw benchmark output to `$EVAL_DIR/logs/baseline_benchmark.txt`.

### 3.2 Profile with rocprof-compute

Use the profiling wrapper script:
```bash
bash ${SKILL_DIR}/scripts/profile_kernel.sh \
  "$HARNESS_PATH" \
  "$EVAL_DIR/logs/profiling_raw"
```

Or manually:
```bash
# Profile (generates raw data)
rocprof-compute profile --no-roof -- python3 "$HARNESS_PATH" --profile

# Find the workload directory
WORKLOAD_DIR=$(ls -td /tmp/workloads/*/ 2>/dev/null | head -1)

# Analyze (generates readable report)
rocprof-compute analyze -p "$WORKLOAD_DIR" -o "$EVAL_DIR/logs/profiling_raw/"
```

### 3.3 Analyze Profiling Output

Read the profiling report and analyze using the guide in `knowledge/profiling_analysis.md`.

Focus on:
1. **System Speed-of-Light** (Section 2): Check "Pct of Peak" for each resource
2. **Wavefront Runtime Stats** (Section 7.2): Compare Dependency Wait vs Active Cycles
3. **Compute Pipeline** (Section 11): Check utilization
4. **Cache Hierarchy** (Sections 13-16): Check hit rates and bandwidth

### 3.4 Classify Bottleneck

Based on the analysis, classify as ONE of:
- **memory-bound**: Dependency Wait >> Active Cycles, low cache hit rates
- **compute-bound**: High compute utilization, Issue Wait cycles
- **latency-bound**: Very short kernel, low utilization everywhere
- **lds-bound**: High LDS contention, bank conflicts
- **balanced**: No single resource saturated

### 3.5 Generate Baseline Metrics

Create `$EVAL_DIR/baseline/baseline_metrics.json`:
```json
{
  "duration_us": <total_kernel_duration_microseconds>,
  "benchmark_duration_us": <benchmark_measured_duration>,
  "bottleneck": "<bottleneck_type>",
  "gpu_arch": "gfx942",
  "gpu_name": "MI300X",
  "metrics": {
    "memory.hbm_bandwidth_utilization": <float_pct>,
    "memory.l2_hit_rate": <float_pct>,
    "compute.valu_utilization": <float_pct>,
    "compute.mfma_utilization": <float_pct>,
    "wavefront.occupancy": <float_pct>,
    "wavefront.active_cycles_pct": <float_pct>
  },
  "top_kernels": [
    {
      "name": "<kernel_name>",
      "duration_us": <float>,
      "pct_of_total": <float>,
      "bottleneck": "<bottleneck_type>"
    }
  ]
}
```

### 3.6 Generate Profiling Summary

Create a human-readable profiling summary at `$EVAL_DIR/logs/profiling_summary.md`:

```markdown
# Profiling Summary

## Primary Bottleneck: <type> (CRITICAL/MODERATE/ACCEPTABLE)
<2-3 sentences with quantitative evidence>

## Key Metrics
- <Metric>: <value> (<pct> of peak)
  ROOT CAUSE: <why>
  IMPACT: <how it limits performance>

## Optimization Directions
HIGH IMPACT (>20%):
- <specific optimization>

MEDIUM IMPACT (5-20%):
- <secondary optimization>
```

### 3.7 Extract GPU Architecture Context

If available from the profiling data, extract GPU specs and append to profiling summary:
```markdown
## GPU Architecture: MI300X (gfx942)
- Compute Units: 304
- Peak HBM bandwidth: 5300 GB/s
- LDS per CU: 64 KB (32 banks)
- VGPRs per CU: 512
- Wavefront size: 64
```
