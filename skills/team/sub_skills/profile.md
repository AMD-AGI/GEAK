# Phase: Profiling

Profile the kernel to establish baseline metrics and classify the performance bottleneck.

## Steps

### 1. Establish Baseline Performance

Run the benchmark command with GPU locking to get accurate timing:

```bash
bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID python3 scripts/task_runner.py performance
```

Or for Makefile-based projects:
```bash
bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID ./<binary> --mode bench
```

Parse the output. Look for:
- `GEAK_RESULT_LATENCY_MS=<float>` — canonical format
- `performance_report.json` — structured per-test-case results
- `Perf: <float> ms` — fallback format

Record per-test-case baseline latencies.

### 2. Profile with rocprof-compute

Use the profiling script with GPU locking:

```bash
bash $SKILL_DIR/scripts/profile_kernel.sh \
    "python3 scripts/task_runner.py performance" \
    $EVAL_DIR/logs/profiling \
    $GPU_ID \
    3
```

If `profile_kernel.sh` cannot find a profiling tool, proceed with benchmark-only data plus source code analysis.

### 3. Analyze Profiling Output

Read `$SKILL_DIR/knowledge/profiling_guide.md` for interpretation guidance.

Read the profiling report at `$EVAL_DIR/logs/profiling/profile_report.txt`. Focus on:
- **Section 2 (System Speed-of-Light)**: Utilization of each pipeline as % of peak
- **Section 7.2 (Wavefront Runtime Stats)**: Time breakdown (Active, Dependency Wait, Issue Wait, Barrier Wait)
- **Section 11 (Compute Pipeline)**: CU utilization and throughput
- **Sections 13-16 (Cache)**: L1/L2 hit rates, HBM bandwidth achieved

### 4. Classify Bottleneck

Based on profiling data, classify as ONE of:
- **memory-bound**: Dependency Wait >> Active Cycles, high HBM bandwidth
- **compute-bound**: High VALU/MFMA utilization, Issue Wait significant
- **latency-bound**: Very short kernel (<100μs), low utilization everywhere
- **lds-bound**: High LDS contention, bank conflicts
- **balanced**: No single bottleneck dominates

### 5. Identify Specific Optimization Opportunities

Go beyond bottleneck classification. Identify concrete opportunities:
- What specific memory access patterns are inefficient? (strided reads, uncoalesced, etc.)
- What data could be cached in LDS? How much bandwidth would that save?
- What loops could benefit from unrolling or algorithmic changes?
- Are there oversized data structures wasting registers?
- Is the kernel under-utilizing the GPU (low occupancy, few wavefronts)?

Frame each opportunity as: **CAUSE** (what's happening) → **EFFECT** (why it's slow) → **OPPORTUNITY** (what to change) → **EXPECTED IMPACT** (HIGH/MEDIUM/LOW)

### 6. Output

Write `$EVAL_DIR/logs/baseline_metrics.json`:

```json
{
  "baseline_latency_ms": 0.5,
  "bottleneck": "compute-bound",
  "per_test_case": [
    {"test_case_id": "shape_0", "execution_time_ms": 0.05},
    {"test_case_id": "shape_1", "execution_time_ms": 0.20}
  ],
  "key_metrics": {
    "valu_utilization_pct": 45.0,
    "vmem_utilization_pct": 12.0,
    "hbm_bandwidth_pct": 8.0,
    "l1_hit_rate_pct": 52.0,
    "l2_hit_rate_pct": 99.0,
    "dependency_wait_to_active_ratio": 0.5
  },
  "top_kernel": "knn_kernel",
  "top_kernel_pct": 97.8
}
```

Write `$EVAL_DIR/logs/profiling_summary.md` — human-readable summary with:
1. PRIMARY BOTTLENECK with quantitative evidence
2. KEY METRICS with CAUSE → EFFECT → IMPACT chains
3. OPTIMIZATION OPPORTUNITIES ranked by expected impact
