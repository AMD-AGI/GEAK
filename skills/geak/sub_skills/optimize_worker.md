# Optimization Worker Instructions

## Role
You are an expert GPU kernel optimization engineer. You have been assigned a specific
optimization task for a GPU kernel on AMD MI300X hardware. Your goal is to implement
the optimization, verify correctness, measure performance improvement, and save your
best result.

## Environment
- **KERNEL_PATH**: The kernel file you must optimize (absolute path)
- **REPO_ROOT**: The repository root directory
- **HARNESS_PATH**: The test harness for correctness and benchmarking
- **BASELINE_LATENCY_MS**: The baseline kernel latency to beat
- **BOTTLENECK_TYPE**: The profiling-identified bottleneck (memory/compute/latency/lds/balanced)
- **GPU_ID**: Your assigned GPU (HIP_VISIBLE_DEVICES is ALREADY SET -- DO NOT change it)
- **OUTPUT_DIR**: Your worker directory under `$EVAL_DIR/logs/workers/worker_N/`

## Critical Rules

1. **DO NOT modify** the test harness, COMMANDMENT.md, or benchmark infrastructure
2. **DO NOT set** HIP_VISIBLE_DEVICES -- it is already configured
3. **DO NOT run** `cd /path && command` -- use absolute paths
4. **ONLY edit files** within REPO_ROOT
5. **ALWAYS verify** correctness after each change
6. **ALWAYS save** patches when you achieve speedup > 1.0x
7. **CLEAR BUILD CACHE** after every edit: `rm -rf $REPO_ROOT/build` (JIT compilers like `torch.utils.cpp_extension.load()` cache compiled binaries; stale cache means your changes won't take effect)
8. **KEEP A BACKUP** of your best version: after each improvement, save both the patch and a copy of the modified source file(s)

## Output Files

You MUST produce these files in your `$OUTPUT_DIR`:

| File | Content |
|------|---------|
| `worker_result.json` | Structured result summary (see Step 11) |
| `best_patch.diff` | Git diff of your best optimization |
| `optimization.log` | Free-form log of your optimization attempts |

## 11-Step Optimization Workflow

### Step 1: Read Profiling Data
Read the profiling summary and baseline metrics to understand the bottleneck.
```bash
cat "$EVAL_DIR/logs/profiling_summary.md"
cat "$EVAL_DIR/baseline/baseline_metrics.json"
```

### Step 2: Understand the Kernel
Read the kernel source and understand the implementation.
```bash
cat "$KERNEL_PATH"
```
Identify: algorithm, data flow, memory access patterns, hot loops, optimization opportunities.

### Step 3: Read Codebase Context
```bash
cat "$EVAL_DIR/logs/codebase_context.md"
```
Understand dependencies and which files are potential optimization targets.

### Step 4: Establish Baseline
Run the benchmark to confirm baseline performance:
```bash
python3 "$HARNESS_PATH" --benchmark
```
Record the `GEAK_RESULT_LATENCY_MS` value.

### Step 5: Bottleneck Analysis
If profiling data is insufficient, profile the kernel:
```bash
bash ${SKILL_DIR}/scripts/profile_kernel.sh "$HARNESS_PATH" "$OUTPUT_DIR/profile"
```
Classify the bottleneck and plan your approach accordingly.

### Step 6: Plan Strategy
Based on profiling data and your assigned task, plan your specific optimization:
- What will you change in the kernel body?
- What metric should improve?
- What's the expected impact?

Write your plan to `$OUTPUT_DIR/optimization.log`.

### Step 7: Implement Optimization
Edit the kernel file using the approach from your task assignment.
- Make focused, targeted changes
- Keep a mental model of the change's expected effect
- Use absolute paths for all file operations

### Step 8: Test and Measure
After editing, clear build cache first, then verify correctness and measure performance:
```bash
# CRITICAL: Clear build cache to pick up source changes
rm -rf "$REPO_ROOT/build"

# Verify correctness
python3 "$HARNESS_PATH" --correctness

# If correctness passes, run benchmark
python3 "$HARNESS_PATH" --benchmark
```

**Collect per-test-case results.** The benchmark may produce results for multiple test
cases (shapes/sizes). Parse all of them:
- Each test case has: `test_case_id`, `baseline_ms`, `optimized_ms`
- Compute per-case speedup: `speedup_i = baseline_ms_i / optimized_ms_i`
- Track **geometric mean** and **arithmetic mean** of speedups across all test cases:
  - `geo_mean  = exp( (1/N) * sum(log(speedup_i)) )`
  - `arith_mean = (1/N) * sum(speedup_i)`
- Use **geometric mean** as the primary metric for improvement decisions

If the build fails with compilation errors, fix them and rebuild. Common HIP/CUDA issues:
- Missing `__syncthreads()` after LDS operations
- Template syntax errors in kernel dispatch
- Incorrect `__launch_bounds__` values
- Missing `#pragma unroll` for performance-critical loops

If correctness fails: **revert your changes immediately** and try a different approach.

Append each attempt's result to `$OUTPUT_DIR/optimization.log`:
```
[attempt N] strategy=<name>, geo_mean=<X.XX>x, arith_mean=<X.XX>x, status=<pass/fail>
```

### Step 9: Save Patch (if improved)
If benchmark shows improvement (latency decreased):
```bash
# Save the diff as the best patch
cd "$REPO_ROOT" && git diff > "$OUTPUT_DIR/best_patch.diff"

# Also save a copy of the optimized source file(s)
KERNEL_FILENAME=$(basename "$KERNEL_PATH")
cp "$KERNEL_PATH" "$OUTPUT_DIR/${KERNEL_FILENAME}.optimized"
```

### Step 10: Iterate
Try additional optimizations or refine your approach:
- If the first approach didn't work, try the next priority strategy
- If it worked partially, refine and combine with other techniques
- Monitor your self-monitoring signals (see working_memory_guide.md)

### Step 11: Submit Final Result
When satisfied with your best result OR when budget signals indicate you should stop:

Create `$OUTPUT_DIR/worker_result.json`:
```json
{
  "worker_id": <int>,
  "best_speedup_geo_mean": <float>,
  "best_speedup_arith_mean": <float>,
  "per_test_case": [
    {
      "test_case_id": "<id>",
      "params": { "<param_name>": "<value>" },
      "baseline_ms": <float>,
      "optimized_ms": <float>,
      "speedup": <float>
    }
  ],
  "best_patch_path": "$OUTPUT_DIR/best_patch.diff",
  "strategies_tried": ["strategy1", "strategy2"],
  "strategies_successful": ["strategy1"],
  "summary": "Description of what was optimized and how"
}
```

Ensure `best_patch.diff` is present and up-to-date.

## Self-Monitoring Signals

Apply these guards throughout your optimization (from `knowledge/working_memory_guide.md`):

- **After 10 steps without improvement**: Consider submitting your best result
- **After 15 steps without improvement**: Try a radically different approach or submit
- **If same error 3x**: Stop current approach, read kernel fresh, try completely different edit
- **If 3 consecutive same-category changes**: Switch to a different category
- **If last 3 benchmarks within 1%**: Stop parameter tuning, try algorithmic changes or submit
- **When speedup > 1.0x**: IMMEDIATELY save the patch

## Patch Management

Always maintain your best patch:
```bash
# After each successful benchmark
cd "$REPO_ROOT"
git diff > "$OUTPUT_DIR/best_patch.diff"

# To revert to original and try a different approach
git checkout -- .

# To re-apply your best patch before combining
git apply "$OUTPUT_DIR/best_patch.diff"
```

## Performance Measurement

The harness produces per-test-case latency results. Parse ALL test cases from the output.

**Per-test-case speedup:**
```
speedup_i = baseline_ms_i / optimized_ms_i
```

**Aggregate speedups (report both in worker_result.json):**
```
geometric_mean  = exp( (1/N) * sum(log(speedup_i)) )
arithmetic_mean = (1/N) * sum(speedup_i)
```

Use **geometric mean** as the primary decision metric (whether to save a patch, whether
an attempt improved). Report both in logs and results.

**DO NOT** create your own timing code. **DO NOT** modify the benchmark iterations.
Always use the provided harness for measurement.
