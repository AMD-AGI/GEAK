# Phase 6: Evaluation & Result Selection

## Objective
Collect results from all optimization workers, verify the best candidates using the
authoritative FULL_BENCHMARK, and generate structured output under `$EVAL_DIR`.

## Output Directory Recap

```
$EVAL_DIR/
├── baseline/                        # Already populated in Phase 1 & 3
│   ├── <kernel_source_file>         # Original kernel
│   └── baseline_metrics.json        # Baseline performance
├── optimized/                       # Populated in this phase
│   ├── <kernel_source_file>         # Best optimized kernel
│   └── best_patch.diff              # Patch from baseline to optimized
├── logs/                            # Already populated in Phase 1-5
│   └── workers/worker_*/            # Per-worker results
└── report/                          # Populated in this phase
    ├── final_report.json            # Machine-readable report
    └── summary.md                   # Human-readable report
```

## Steps

### 6.1 Collect Worker Results

Gather results from all workers:
```bash
find "$EVAL_DIR/logs/workers" -name "worker_result.json" | sort
```

For each worker, extract:
- Best speedup achieved
- Best latency (ms)
- Path to best patch
- Strategies tried and which succeeded
- Summary of optimizations

### 6.2 Rank Results

Sort workers by speedup (highest first):
```
Worker 0: 1.23x (patch: .../worker_0/best_patch.diff)
Worker 1: 1.15x (patch: .../worker_1/best_patch.diff)
Worker 2: 0.98x (no improvement)
```

### 6.3 Validate Best Candidates

For the top 2-3 candidates, verify in a clean environment:

```bash
# For each candidate patch:

# 1. Reset to clean state
cd "$REPO_ROOT" && git checkout -- .

# 2. Apply the patch
git apply "$PATCH_PATH"

# 3. Clear build cache
rm -rf "$REPO_ROOT/build"

# 4. Verify correctness
python3 "$HARNESS_PATH" --correctness
if [ $? -ne 0 ]; then
    echo "REJECTED: Correctness test failed"
    git checkout -- .
    continue
fi

# 5. Run FULL_BENCHMARK (authoritative)
python3 "$HARNESS_PATH" --full-benchmark
# Parse per-test-case results from output
# Each test case produces: "Performance: X.XXXX ms (test_case_id)"
# OR: "GEAK_RESULT_LATENCY_MS=X.XXXX" for single-test-case kernels

# 6. Compute per-test-case verified speedup
# For each test case i:
#   speedup_i = baseline_ms_i / optimized_ms_i
#
# Aggregate across all test cases:
#   geometric_mean  = exp(mean(log(speedup_i)))
#   arithmetic_mean = mean(speedup_i)

# 7. Reset for next candidate
git checkout -- .
```

### 6.4 Rejection Criteria

Reject a result if:
1. **Correctness test fails** after applying the patch
2. **Patch doesn't apply cleanly** (conflicts)
3. **FULL_BENCHMARK shows regression** (speedup < 1.0x)
4. **The optimization modified evaluation infrastructure** (test harness, benchmark, COMMANDMENT)
5. **The optimization only changed wrapper/dispatch** without kernel-body modifications (unless no kernel-body approaches worked)

### 6.5 Select Best Result

The result with the highest **verified geometric mean speedup** from FULL_BENCHMARK wins.
When geometric means are tied (within 1%), prefer the candidate with higher arithmetic mean.
Note: Worker-reported speedups are provisional; FULL_BENCHMARK results are authoritative.

### 6.6 Populate `optimized/` Directory

Apply the winning patch and copy the optimized kernel:

```bash
# Apply best patch
cd "$REPO_ROOT" && git checkout -- .
git apply "$BEST_PATCH_PATH"

# Get kernel filename
KERNEL_FILENAME=$(basename "$KERNEL_PATH")

# Copy optimized kernel source
cp "$KERNEL_PATH" "$EVAL_DIR/optimized/$KERNEL_FILENAME"

# Save the patch
cp "$BEST_PATCH_PATH" "$EVAL_DIR/optimized/best_patch.diff"
```

### 6.7 Generate `report/final_report.json`

The final report must include **per-test-case results** and **both geometric mean and
arithmetic mean** speedups. This enables accurate comparison across kernels with
different numbers and sizes of test cases.

```json
{
  "status": "success|no_improvement",
  "kernel_name": "<kernel_name>",
  "kernel_path": "<absolute_path>",
  "kernel_type": "triton|hip",
  "bottleneck_type": "<type>",
  "correctness": "PASS|FAIL",
  "test_cases": [
    {
      "test_case_id": "<id>",
      "params": { "<param_name>": "<value>", "..." : "..." },
      "baseline_ms": <float>,
      "optimized_ms": <float>,
      "speedup": <float>
    }
  ],
  "speedup_summary": {
    "geometric_mean": <float>,
    "arithmetic_mean": <float>,
    "best": <float>,
    "worst": <float>,
    "best_test_case": "<test_case_id>",
    "worst_test_case": "<test_case_id>",
    "num_test_cases": <int>
  },
  "optimizations_applied": ["<description_1>", "<description_2>"],
  "optimization_summary": "Description of the winning optimization",
  "strategies_applied": ["strategy1", "strategy2"],
  "eval_dir": "<absolute_path_to_eval_dir>",
  "files": {
    "baseline_kernel": "baseline/<kernel_file>",
    "optimized_kernel": "optimized/<kernel_file>",
    "patch": "optimized/best_patch.diff",
    "baseline_metrics": "baseline/baseline_metrics.json",
    "profiling_summary": "logs/profiling_summary.md"
  },
  "workers": {
    "total": <N>,
    "with_improvement": <count>,
    "results": [
      {
        "worker_id": 0,
        "speedup_geo_mean": <float>,
        "speedup_arith_mean": <float>,
        "verified_speedup_geo_mean": <float_or_null>,
        "verified_speedup_arith_mean": <float_or_null>,
        "strategies": ["list"],
        "patch_path": "<relative_path_or_null>"
      }
    ]
  }
}
```

**Speedup calculation:**
```
Per test case:    speedup_i = baseline_ms_i / optimized_ms_i
Geometric mean:   exp( (1/N) * sum(log(speedup_i)) )
Arithmetic mean:  (1/N) * sum(speedup_i)
```

Write to `$EVAL_DIR/report/final_report.json`.

### 6.8 Generate `report/summary.md`

Write a human-readable summary to `$EVAL_DIR/report/summary.md`:

```markdown
# GEAK Optimization Report

## Result
| Metric | Value |
|--------|-------|
| Kernel | <kernel_name> (<kernel_type>) |
| Bottleneck | <bottleneck_type> |
| Correctness | PASS |
| Test Cases | <N> |

## Speedup Summary
| Metric | Value |
|--------|-------|
| **Geometric Mean** | **<X.XX>x** |
| **Arithmetic Mean** | **<X.XX>x** |
| Best | <X.XX>x (<test_case_id>) |
| Worst | <X.XX>x (<test_case_id>) |

## Per-Test-Case Results
| Test Case | Params | Baseline (ms) | Optimized (ms) | Speedup |
|-----------|--------|---------------|----------------|---------|
| shape_0 | B=2,N=256 | 0.1234 | 0.0567 | 2.18x |
| shape_1 | B=4,N=1024 | 0.4567 | 0.0890 | 5.13x |
| ... | ... | ... | ... | ... |

## Winning Strategy
<Description of what was changed and why it helped>

## Optimizations Applied
1. <optimization_description_1>
2. <optimization_description_2>

## Files
- Baseline kernel: `baseline/<kernel_file>`
- Optimized kernel: `optimized/<kernel_file>`
- Patch: `optimized/best_patch.diff`

## Workers Summary
| Worker | Geo Mean | Arith Mean | Strategy | Status |
|--------|----------|------------|----------|--------|
| 0 | <X.XX>x | <X.XX>x | <strategy> | <success/failed> |
| 1 | <X.XX>x | <X.XX>x | <strategy> | <success/failed> |

## Strategies Attempted
1. <strategy_name>: <result>
2. <strategy_name>: <result>
```

### 6.9 Report to User

Print a clear summary to the console, including:
- The speedup achieved
- The `$EVAL_DIR` path for all artifacts
- Key file locations within the eval directory

### 6.10 Multi-Round Decision (if applicable)

If running multiple optimization rounds:
- If the best result shows improvement: apply the patch as the new baseline for the next round
- If no improvement: adjust strategy for next round (try different approach categories)
- After final round: select the overall best result across all rounds
