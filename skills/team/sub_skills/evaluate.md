# Phase G: Evaluation & Ranking

## Objective
Collect engineer results, rank by performance, verify top candidates in a clean environment, and select the round winner.

## Steps

### G1: Collect Results

Read `worker_result.json` from each engineer's output directory:
```
$EVAL_DIR/round_N/engineer_0/worker_result.json
$EVAL_DIR/round_N/engineer_1/worker_result.json
...
```

Expected format:
```json
{
  "engineer_id": 0,
  "task": "description of assigned task",
  "strategy": "what was actually implemented",
  "speedup_geomean": 1.5,
  "speedup_arithmetic": 1.6,
  "per_case": [
    {"name": "case_0", "baseline_ms": 0.5, "optimized_ms": 0.3, "speedup": 1.67}
  ],
  "status": "success|partial|failed",
  "patch_file": "best_patch.diff",
  "strategies_tried": ["P0-ALG: template", "P2-MEM: vectorized loads"],
  "notes": "optional notes"
}
```

### G2: Rank Engineers

Sort by geometric mean speedup (descending). Use arithmetic mean as tiebreaker.

```
Ranking:
1. Engineer 2: 3.5x geomean (3.8x arithmetic) — template parameterization
2. Engineer 0: 2.1x geomean (2.3x arithmetic) — LDS tiling
3. Engineer 1: 1.0x geomean (1.0x arithmetic) — FAILED (correctness issue)
```

Filter out:
- `status == "failed"` — no valid result
- `speedup_geomean < 1.0` — regression
- Missing `patch_file`

### G3: Verify Top Candidates

For the top 2-3 candidates (or all candidates with speedup > 1.0x), independently verify:

For each candidate:
```bash
# 1. Reset to clean state (current best, or baseline for round 1)
cd $KERNEL_PATH
git checkout -- .  # Reset all kernel files
# If not round 1, apply the current-best patch first
git apply $EVAL_DIR/current_best.diff 2>/dev/null || true

# 2. Apply this candidate's patch
git checkout -- .  # Clean slate
git apply $EVAL_DIR/round_N/engineer_X/best_patch.diff

# 3. Clear build cache
rm -rf build/ __pycache__/ *.so

# 4. Run correctness test
<correctness_command>

# 5. Run FULL benchmark (authoritative) with gpu_lock
bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID <full_benchmark_command>
```

**Rejection criteria:**
- Correctness test fails → REJECT
- Patch fails to apply → REJECT
- Verified speedup < 1.0x → REJECT (regression)
- Patch modifies test harness or COMMANDMENT → REJECT

Record verified speedups. These override engineer-reported speedups.

### G4: Select Round Winner

The candidate with the highest **verified** geometric mean speedup wins the round.

Save the winning patch as `$EVAL_DIR/current_best.diff`:
```bash
# Apply winning patch
cd $KERNEL_PATH
git checkout -- .
git apply $EVAL_DIR/round_N/engineer_X/best_patch.diff

# Save as current best
git diff > $EVAL_DIR/current_best.diff
```

### G5: Output

Write `$EVAL_DIR/round_N/round_result.json`:
```json
{
  "round": 1,
  "num_engineers": 3,
  "rankings": [
    {
      "rank": 1,
      "engineer_id": 2,
      "strategy": "template parameterization",
      "reported_speedup": 3.5,
      "verified_speedup": 3.4,
      "status": "verified"
    }
  ],
  "round_winner": {
    "engineer_id": 2,
    "verified_speedup_geomean": 3.4,
    "verified_speedup_arithmetic": 3.7,
    "patch_file": "current_best.diff",
    "per_case": [...]
  },
  "merge_result": {
    "attempted": true,
    "merged_speedup": 4.1,
    "patches_merged": [2, 0],
    "status": "improved|no_improvement|skipped"
  },
  "cumulative_speedup_geomean": 3.4,
  "budget_used": 3,
  "budget_remaining": 3
}
```

Write a human-readable round summary to `$EVAL_DIR/round_N/summary.md`.
