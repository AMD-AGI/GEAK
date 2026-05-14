# Working Memory & Self-Monitoring Guide for Optimization Workers

## Purpose

During iterative kernel optimization, you must self-monitor to avoid common failure modes:
spinning on the same approach, ignoring benchmark noise, running out of budget, or
forgetting to save successful patches.

## State Tracking

Track these variables throughout your optimization session:

- **best_speedup**: Best speedup achieved vs baseline (e.g., 1.15x)
- **best_latency_ms**: Best kernel latency achieved
- **baseline_latency_ms**: Original kernel latency (DO NOT modify)
- **strategies_tried**: List of strategies attempted
- **strategies_failed**: List of strategies that produced no improvement
- **steps_since_improvement**: Steps since last speedup improvement
- **bottleneck_type**: Current bottleneck classification

## Priority Ordering by Bottleneck

### Memory-Bound
1. Memory coalescing (vectorized loads, reduce bandwidth, improve locality)
2. Algorithmic kernel rewrites
3. Operation fusion
4. Parameter tuning (tile sizes, warps)
5. Dispatch-path optimization (last resort)

### Compute-Bound
1. Algorithmic kernel rewrites (reduce FLOPs, better math)
2. Parameter tuning (tile sizes, warps, split-K)
3. Operation fusion
4. Memory coalescing
5. Dispatch-path optimization (last resort)

### Latency/Balanced/Unknown
1. Algorithmic kernel rewrites
2. Operation fusion
3. Memory coalescing (vectorized loads, reduce global memory traffic)
4. Parameter tuning (tile sizes, warps)
5. Dispatch-path optimization (last resort)

## Guard Signals

### Stall Detection
- **After 10 steps without improvement**: "STALLED: No improvement for N steps. Consider submitting."
- **After 15+ steps without improvement**: "WARNING: Try a RADICALLY different approach: bypass the current kernel entirely, restructure the call graph, eliminate unnecessary operations, or save your best patch and submit."
- **After 20+ steps without improvement**: "EARLY_STOP: Submit now."

### Dead-End Detection
- When 3+ attempts of the same strategy category fail, that category is a dead end.
- Switch to a completely different approach category.

### Crash Loop Recovery
- If the same error repeats 3+ times:
  1. Read the kernel file fresh
  2. Use a completely different edit strategy
  3. Run the test command directly to verify the environment

### Diminishing Returns / Ceiling Detection
- If last 3 benchmark results are within 1% of each other AND you've done 3+ tuning steps:
  "CEILING REACHED. STOP parameter tuning. OPTIONS: (1) SUBMIT best result, (2) Try fundamentally different algorithm, (3) Try @triton.autotune with shape-specific configs."

### Diversity Enforcement
- If last 3+ changes were all the same category (e.g., all TUNING):
  "DIVERSITY REQUIRED. You MUST try a different category."
  - If stuck on TUNING: try ALGORITHMIC or FUSION
  - If stuck on ALGORITHMIC: try TUNING or MEMORY LAYOUT
  - If stuck on FUSION: try ALGORITHMIC or TUNING

### Step 5 Checkpoint
At step 5, if best_speedup <= 1.01x:
"Have you identified the DOMINANT bottleneck? Re-read profiling data. Look for: algorithmic shortcuts, fusion opportunities, unnecessary memory copies, redundant ops."

## Patch Save Enforcement

**CRITICAL**: When you achieve speedup > 1.0x, you MUST immediately save the patch.
Unsaved improvements are LOST when the session ends.

After editing the kernel, ALWAYS run the test to:
1. Capture the speedup measurement
2. Save the patch (git diff) to disk
3. Verify correctness hasn't regressed

## Budget Management

Track your step count against the maximum allowed steps:
- At 70%: "BUDGET_WARN: ~N steps remaining"
- At 85%: "BUDGET_CRITICAL: Wrap up and submit best result"
- At 95%: "BUDGET_FORCE: MUST submit immediately"

## Change Classification

Classify each code change to enable diversity tracking:
- **algorithmic**: New kernel function, restructured algorithm, different reduction tree
- **fusion**: Fused operations, merged kernels, combined ops
- **tuning**: BLOCK_SIZE, num_warps, num_stages, @triton.autotune configs
- **wrapper**: Python dispatch, import routing, launch config changes

## Insight Buffer

Keep a rolling log of the last 15 observations:
```
[WIN] step 3: TUNE(BLOCK_M=128) -> latency: 0.0234ms
[FAIL] step 5: ALGO(algorithm rewrite) -> Exit 1: shape mismatch
[OK]  step 7: Benchmark latency: 0.0228ms
```

Use tags: WIN (improvement), FAIL (regression/error), OK (neutral), WARN (warning).
