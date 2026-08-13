# Engineer Self-Monitoring Protocol

## State Tracking

Maintain these variables throughout your optimization session:

```
best_speedup = 1.0           # Best geomean speedup achieved so far
best_patch_saved = false      # Whether you saved a patch for the best result
strategies_tried = []         # List of strategies attempted
steps_since_improvement = 0   # Steps since last speedup improvement
consecutive_same_category = 0 # Consecutive attempts in same strategy category
error_count = 0               # Consecutive identical errors
last_3_speedups = []          # Last 3 benchmark results
benchmarks_since_best = 0     # Full measurements since best_speedup last improved (see Guard 3)
```

## Guard Signals

### 1. Stall Detection
- **steps_since_improvement >= 8**: WARNING — You are stalling. Try a radically different approach. Switch to a different P-level category.
- **steps_since_improvement >= 12**: CRITICAL — Force submit your best result. Stop iterating. Write worker_result.json with what you have.
- **steps_since_improvement >= 15**: EMERGENCY — Submit immediately regardless of state.

### 2. Dead-End Detection
- **3 consecutive failures of the same strategy type**: Switch to a completely different category. If you were doing P2 memory optimizations and they keep failing, try P0 algorithmic or P3 compute instead.
- Never try the exact same optimization twice. If it didn't work, move on.

### 3. Ceiling Detection
- **Last 3 benchmarks within 1% of each other**: You've hit a ceiling for this approach. Stop tuning parameters and either:
  - Switch to a different strategy category, OR
  - Submit your best result if speedup is already good
- **3 consecutive full measurements that fail to BEAT your best**: submit your best and stop. This
  threshold is measured, not guessed: across 22 engineer sessions in two instrumented Qwen3-14B runs,
  the longest run of non-improving measurements that was ever *later* broken by a new best was **2**.
  A 3-strike stop would not have cost a single win in either run, and two sessions that ignored it
  spent $17 re-measuring a plateau they had already topped out on.

### 3a. Do NOT stop early just because you have taken many steps
Counting steps is not evidence of a ceiling; only measurements are. In those same 22 sessions the
best result arrived at **76–98% of the session's length in 19 of them** — engineers win late, after a
long grind of edit/compile/debug punctuated by only 1–12 actual benchmarks. Truncating on step count
would have thrown away most of the wins. Grind as long as you are still making the code work; stop
when the *measurements* stop improving (§3), not when the work feels long.

### 4. Crash Loop Recovery
- **3 identical compile/runtime errors in a row**: STOP modifying incrementally. Instead:
  1. Re-read the ORIGINAL kernel source file
  2. Start fresh from the baseline
  3. Re-apply only your best-performing changes
  4. Try a completely different approach

### 5. Diversity Enforcement
- **3 consecutive changes in the same strategy category**: Must switch to a different category for the next attempt.
- Track which categories you've tried. Prefer unexplored categories.

## Priority Ordering by Bottleneck

When choosing what to try next, follow this order based on the profiling data:

**Memory-bound kernel**: P1 (tiling) → P2 (coalescing) → P0 (algorithmic) → P3 (compute)
**Compute-bound kernel**: P0 (algorithmic) → P3 (branchless/ILP) → P1 (register blocking) → P4 (occupancy)
**Latency-bound kernel**: P0 (parallelism) → P4 (launch config) → P1 (prefetch) → P3 (break deps)
**LDS-bound kernel**: P1 (reduce LDS/pad banks) → P2 (use registers instead) → P4 (smaller blocks)
**Balanced kernel**: P0 (template/warp-coop) → P1 (tiling) → P2 (vectorize) → P4 (tune)

## Patch Save Rules

1. **ALWAYS save a patch when speedup > 1.0x**. Run `git diff` and save to `best_patch.diff`.
2. **Update patch when you beat your previous best**. Overwrite `best_patch.diff`.
3. **Never submit without a patch file** if you achieved any speedup.
4. Before saving: verify correctness passes. A fast-but-wrong kernel is worth 0x.

## Benchmark Discipline

1. You usually do NOT need to clear the build cache — ninja rebuilds what changed. **Never use `rm`**
   (it triggers an approval prompt that blocks the run, and your workspace is already artifact-free).
   If you genuinely suspect a stale build after editing headers, MOVE the cache aside instead:
   `mv .torch_ext .torch_ext.stale_$(date +%s)_$$ 2>/dev/null || true`
2. Always run correctness test BEFORE benchmarking. Don't waste time benchmarking broken code.
3. Use the COMMANDMENT commands exactly. Don't invent your own benchmark.
4. Run benchmark at least 2 times if the result seems surprisingly good or bad.
5. Track BOTH geometric mean AND per-test-case speedups.

## Change Classification

When implementing changes, classify each as:

| Category | Description | Example |
|----------|-------------|---------|
| P0-ALG | Algorithmic restructuring | Template params, warp-cooperative |
| P1-REUSE | Data reuse optimization | LDS tiling, register blocking |
| P2-MEM | Memory access pattern | Coalescing, vectorized loads |
| P3-COMP | Compute optimization | Branchless, FMA, unrolling |
| P4-LAUNCH | Launch configuration | Block size, occupancy |
| P5-TUNE | Autotuning | Parameter search |

Track: `strategies_tried.append("P0-ALG: template parameterization for K")`

## Submission Checklist

Before writing worker_result.json:
- [ ] Best patch saved to best_patch.diff
- [ ] Correctness verified with best patch applied
- [ ] Final benchmark run with best patch (record exact numbers)
- [ ] Geometric mean calculated across ALL test cases
- [ ] worker_result.json contains: speedup, strategy, per-test-case results
- [ ] Mini-report written: what you tried, what worked, what didn't
