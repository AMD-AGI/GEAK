# Phase C/H: Kernel Profiling & Bottleneck Analysis

## Objective
Profile the kernel to identify performance bottlenecks. Used both for initial baseline (Phase C) and re-profiling after optimization rounds (Phase H).

## Steps

### Step 1: Run Baseline Benchmark

Establish the performance baseline using COMMANDMENT commands:

```bash
# Setup (clear cache)
cd $KERNEL_PATH && rm -rf build/ __pycache__/ *.so

# Run benchmark with GPU lock
bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID <benchmark_command>
```

Parse output and record per-test-case latencies.

### Step 2: Profile the Kernel

Run the profiling script:

```bash
bash $SKILL_DIR/scripts/profile_kernel.sh $GPU_ID "<profile_command>" $EVAL_DIR/profile_output
```

This will:
1. Run 3 warmup iterations
2. Profile with best available profiler (rocprof-compute → omniperf → rocprof → benchmark-only)
3. Save the profile report

### Step 3: Analyze Profile Data

Read the profile report at `$EVAL_DIR/profile_output/profile_report.txt`.

Following the profiling_guide.md knowledge file, extract these key metrics:

**System Speed-of-Light:**
- VALU Utilization %
- VMEM Utilization %
- LDS Utilization %
- Effective HBM Bandwidth (GB/s vs 5300 GB/s peak)

**Wavefront Runtime:**
- Active Cycles / Total Cycles ratio
- Dependency Wait / Total Cycles ratio
- Issue Wait / Total Cycles ratio

**Cache Hierarchy:**
- L1 Hit Rate %
- L2 Hit Rate %
- Memory Coalescing Efficiency %

**Compute:**
- Branch Divergence %
- Active Threads per instruction (vs 64 ideal)

### Step 4: Classify Bottleneck

Apply the decision tree from profiling_guide.md:

```
1. SoL: VALU vs VMEM utilization
   ├─ VALU > 60%, VMEM < 40% → COMPUTE-BOUND
   ├─ VMEM > 60%, VALU < 40% → MEMORY-BOUND
   ├─ Both > 50% → BALANCED
   ├─ Both < 40% → check wavefront stats → LATENCY-BOUND
   └─ LDS > 50% → LDS-BOUND

2. Secondary indicators:
   - Low L1 hit rate → memory locality issue
   - Low coalescing → access pattern issue
   - High divergence → branching issue
```

### Step 5: Output

Write `$EVAL_DIR/baseline_metrics.json` (or `$EVAL_DIR/round_N_metrics.json` for re-profiling):
```json
{
  "bottleneck": "<memory-bound|compute-bound|latency-bound|lds-bound|balanced>",
  "metrics": {
    "valu_utilization_pct": 0.0,
    "vmem_utilization_pct": 0.0,
    "lds_utilization_pct": 0.0,
    "hbm_bandwidth_gbps": 0.0,
    "active_cycle_ratio": 0.0,
    "dependency_wait_ratio": 0.0,
    "l1_hit_rate_pct": 0.0,
    "l2_hit_rate_pct": 0.0,
    "coalescing_efficiency_pct": 0.0,
    "branch_divergence_pct": 0.0
  },
  "profiler_used": "<rocprof-compute|omniperf|rocprof|benchmark-only>",
  "baseline_latency_ms": {
    "per_case": [{"name": "...", "latency_ms": 0.0}],
    "geomean_ms": 0.0
  }
}
```

Write `$EVAL_DIR/profiling_summary.md` — human-readable analysis including:
1. Raw metric values with interpretation
2. Bottleneck classification with evidence
3. Top 3 optimization opportunities identified

## Phase H: Re-Profiling (After Optimization Round)

When called after an optimization round, ALSO produce a bottleneck shift analysis:

### Shift Analysis

Compare the new metrics against the previous round's metrics:

```markdown
## Bottleneck Shift Analysis — Round N

BEFORE (Round N-1): [bottleneck] — [key metric value]
AFTER  (Round N):   [bottleneck] — [key metric value]

### What Changed
- VALU util: X% → Y% ([+/-]Z%)
- VMEM util: X% → Y%
- L1 hit rate: X% → Y%
- Coalescing: X% → Y%

### Shift Explanation
[Why the bottleneck shifted. E.g., "Template parameterization reduced register spill, 
freeing compute resources. Now memory bandwidth is the limiting factor."]

### Recommended Next Strategies
1. [Strategy targeting the new bottleneck]
2. [Alternative strategy]
3. [Complementary strategy]
```

Save to `$EVAL_DIR/round_N_shift_analysis.md`.
