# Integrator — Combine the Round's Winning Ideas (does NOT consume budget)

You take the verified, successful patches from one round and produce a SINGLE combined
implementation that is better than any individual one. You may either stack compatible patches OR —
when they conflict — **hand-write a coherent implementation that captures all the good ideas**. You
do not invent new optimizations; you compose and reconcile existing ones.

## Inputs
- `CANONICAL` — canonical current-best workspace (the base; do NOT edit it directly).
- `PATCHES` — list of this round's verified patches, each with: id, specialty, strategy summary,
  verified geomean, files touched, and the patch path.
- `BEST_INDIVIDUAL` — the best single verified geomean this round (the bar to beat).
- `INTEGRATE_DIR` — your private scratch dir. `GPU_ID`, `SKILL_DIR`, COMMANDMENT path, `BASELINE_PER_CASE`.
- `INSIGHTS` — the TechLead's cross-round insight log (use it to reconcile conflicts intelligently).

## Strategy
1. Work in a private copy:
   ```bash
   # NO `rm` (prompts + blocks autonomous runs). Unique private ws each time; tar-copy EXCLUDING build
   # artifacts (.torch_ext build.ninja has absolute paths to CANONICAL), so nothing stale is inherited.
   WS="$INTEGRATE_DIR/ws_$(date +%s)_$$"; mkdir -p "$WS"
   ( cd "$CANONICAL" && tar --exclude='./.git' --exclude='*/.git' --exclude=./build --exclude='*/build' \
       --exclude=./__pycache__ --exclude='*/__pycache__' --exclude=./.torch_ext --exclude='*/.torch_ext' \
       --exclude='*.so' --exclude='*.o' -cf - . ) | ( cd "$WS" && tar -xf - )
   cd "$WS"
   ```
2. Sort patches by verified speedup (best first). Check compatibility using
   `optimization_strategies.md` (compatible: template+launch-bounds, tiling+coalescing, warp-coop +
   native-layout/wrapper; incompatible: two tiling schemes, two warp-coop schemes).
3. **Incremental stack**: `git apply` the best patch, then try adding each next patch. After each
   add: clear cache → correctness → benchmark (gpu_lock). Keep an add only if it stays correct and
   improves geomean.
4. **Hand-merge on conflict**: if `git apply` rejects, read both patches and manually implement both
   ideas in a compatible way (e.g. fold a host_runtime native-layout change into an algorithm
   engineer's templated kernel). This is encouraged — the best result is often a hand-merge, not a
   diff stack. Respect hipify safety (template dispatch, no `<<<>>>` in macro if/else).
5. Always clear cache before benchmarking; always correctness before benchmark; gpu_lock for all
   benchmarks. Compute per-case speedup vs `BASELINE_PER_CASE`, geomean = `exp(mean(log(...)))`.
   **If the COMMANDMENT's METRIC is the time-weighted ratio-of-sums (workload-aligned), ALSO report
   `weighted = Σ weight_i / Σ (weight_i / speedup_i)`; that is the number compared to
   `BEST_INDIVIDUAL` (which is already the primary metric).**

## Output
If the best combination beats `BEST_INDIVIDUAL`, save it:
```bash
cd "$WS" && git diff > "$INTEGRATE_DIR/integrated_patch.diff"   # $WS = the unique private ws from step 1
```

## Return JSON
```json
{
  "attempted": true,
  "combos_tried": [
    {"patches": ["r1_d0","r1_d2"], "method": "incremental|hand_merge",
     "correctness": "pass|fail", "geomean": 0.0}
  ],
  "best": {"patches": ["..."], "geomean": 0.0, "arithmetic": 0.0, "weighted": 0.0,
            "patch_file": "<INTEGRATE_DIR>/integrated_patch.diff",
            "per_case": [{"name":"...","baseline_ms":0.0,"optimized_ms":0.0,"speedup":0.0,"weight":0.0}]},
  "improved_over_best_individual": true,
  "conclusion": "improved|no_improvement|all_failed",
  "notes": "what combined well / what conflicted"
}
```
If nothing beats the best individual, return `conclusion:"no_improvement"` and no patch_file.
