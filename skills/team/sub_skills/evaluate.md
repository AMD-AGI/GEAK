# Phase: Evaluation

Collect engineer results, verify the best candidate, and produce a structured report.

## Steps

### 1. Collect Results

Read `worker_result.json` from each engineer's output directory:
```bash
find $EVAL_DIR/logs/workers/ -name "worker_result.json" -exec cat {} \;
```

Each result contains:
```json
{
  "worker_id": 0,
  "status": "success|failed",
  "best_speedup_geo": 4.5,
  "best_speedup_arith": 6.2,
  "best_latency_ms": 0.12,
  "baseline_latency_ms": 0.54,
  "strategy": "Description of what was done",
  "patch_file": "best_patch.diff",
  "per_test_case": [
    {"test_case_id": "shape_0", "baseline_ms": 0.05, "optimized_ms": 0.04, "speedup": 1.25}
  ],
  "iterations_tried": 5
}
```

### 2. Rank Candidates

Sort successful results by geometric mean speedup (highest first). If tied, use arithmetic mean as tiebreaker.

Skip results where:
- `status` is `failed`
- `best_speedup_geo` ≤ 1.0
- No patch file exists

### 3. Verify Top Candidates

For the top 2 candidates (or all if fewer than 2):

```bash
# Start from clean state
cd $REPO_ROOT
git checkout -- .

# Apply the candidate's patch
git apply $EVAL_DIR/logs/workers/worker_N/best_patch.diff

# Clear build cache
rm -rf $REPO_ROOT/build
rm -rf ~/.cache/torch_extensions/*/$(basename $REPO_ROOT)/

# Run correctness
python3 scripts/task_runner.py correctness

# Run full benchmark with GPU lock
bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID python3 scripts/task_runner.py performance
```

Parse benchmark results. The VERIFIED speedup is authoritative — worker-reported speedups are provisional.

**Rejection criteria:**
- Correctness test fails
- Patch doesn't apply cleanly
- Verified speedup < 1.0x (regression)
- Patch modifies test harness, benchmark scripts, or evaluation infrastructure

### 4. Select Best

Choose the verified candidate with the highest geometric mean speedup.

### 5. Save Results

Copy the winning kernel:
```bash
cp $TASK_PATH $EVAL_DIR/optimized/$(basename $TASK_PATH)
cp $EVAL_DIR/logs/workers/worker_N/best_patch.diff $EVAL_DIR/optimized/best_patch.diff
```

### 6. Generate Round Report

Write per-round results to `$EVAL_DIR/logs/round_N_results.json`:

```json
{
  "round": 1,
  "num_engineers": 3,
  "best_speedup_geo": 4.5,
  "best_speedup_arith": 6.2,
  "winning_worker": 0,
  "winning_strategy": "Template parameterization + warp-cooperative search",
  "workers": [
    {"worker_id": 0, "speedup_geo": 4.5, "status": "success"},
    {"worker_id": 1, "speedup_geo": 2.1, "status": "success"},
    {"worker_id": 2, "speedup_geo": 0.0, "status": "failed"}
  ],
  "verified_speedup_geo": 4.3,
  "verified_speedup_arith": 6.0,
  "per_test_case": [...]
}
```

### 7. Generate Final Report (after all rounds)

Write `$EVAL_DIR/report/final_report.json`:

```json
{
  "status": "success",
  "kernel_name": "knn_kernel",
  "kernel_type": "hip",
  "total_rounds": 3,
  "test_cases": [
    {
      "test_case_id": "shape_0_standard",
      "baseline_ms": 0.0491,
      "optimized_ms": 0.0050,
      "speedup": 9.82
    }
  ],
  "speedup_summary": {
    "geometric_mean": 8.5,
    "arithmetic_mean": 10.2,
    "best_case": 19.0,
    "worst_case": 1.02,
    "num_test_cases": 15
  },
  "round_progression": [
    {"round": 1, "speedup_geo": 4.5, "strategy": "..."},
    {"round": 2, "speedup_geo": 7.2, "strategy": "..."},
    {"round": 3, "speedup_geo": 8.5, "strategy": "..."}
  ],
  "optimizations_applied": "Description of all optimizations in the final kernel"
}
```

Write `$EVAL_DIR/report/summary.md` with:
- Per-test-case speedup table
- Round progression table
- Strategy descriptions
- Before/after comparison
