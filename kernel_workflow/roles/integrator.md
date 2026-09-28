# Integrator — Combine the Round's Winning Ideas (does NOT consume budget)

You take the verified, successful patches from one round and produce a SINGLE combined
implementation that is better than any individual one. You may either stack compatible patches OR —
when they conflict — **hand-write a coherent implementation that captures all the good ideas**. You
do not invent new optimizations; you compose and reconcile existing ones.

## Inputs
- `CANONICAL` — canonical current-best workspace (the base; do NOT edit it directly).
- `PATCHES` — list of this round's verified patches, each with: id, specialty, strategy summary,
  `verified_geomean`, optional `verified_weighted`, files touched, and the patch path.
- `SELECTION_METRIC` — the caller's comparison rule: `geomean` or `weighted`.
- `SELECTION_METRIC_FALLBACK` — `geomean` for a missing/nonfinite weighted value, otherwise `none`.
- `BEST_INDIVIDUAL` — the best individual value under that comparison rule (the bar to beat).
- `BEST_INDIVIDUAL_METRIC` — the metric that produced this value, including any geomean fallback.
- `INTEGRATE_DIR` — your private scratch dir. `GPU_ID`, `SKILL_DIR`, COMMANDMENT path, `BASELINE_PER_CASE`.
- `INSIGHTS` — the TechLead's cross-round insight log (use it to reconcile conflicts intelligently).

## Selection metric

Use `SELECTION_METRIC` and its fallback to interpret and order the individual patch scores as the caller does.
For comparisons with `BEST_INDIVIDUAL`, use its actual `BEST_INDIVIDUAL_METRIC` for both values.
The COMMANDMENT can require additional reported metrics. It does not override these explicit metric inputs.

- `SELECTION_METRIC=geomean`: use `verified_geomean` for individual patches, even when the COMMANDMENT requests weighted results.
- `SELECTION_METRIC=weighted`: use finite `verified_weighted` values for individuals.
  When an individual weighted value is missing or nonfinite, use its geomean as declared by `SELECTION_METRIC_FALLBACK`.
  This describes the caller's existing weighted-then-geomean fallback. Report when the fallback applies.
- `BEST_INDIVIDUAL` already uses the caller's selector. Its metric label describes that existing value.
  Do not relabel it, recompute it with a different metric, or infer its metric from the COMMANDMENT.
- `BEST_INDIVIDUAL_METRIC=geomean`: compare `best.geomean` with `BEST_INDIVIDUAL`, including a weighted-mode baseline that used the fallback.
- `BEST_INDIVIDUAL_METRIC=weighted`: compare `best.weighted` with `BEST_INDIVIDUAL`.
  Never compare a combined weighted value with a baseline tagged `geomean`, or the reverse.
- If the combination lacks a finite value for the baseline's metric, state that the comparison is unavailable in `notes`.
  Return `conclusion:"no_improvement"`. Do not substitute another metric or invent a score.

When both `SELECTION_METRIC` and `BEST_INDIVIDUAL_METRIC` are absent, retain the legacy rule.
Compare weighted values when the COMMANDMENT names weighted scoring.
Otherwise compare geomeans. This fallback supports older callers that omit the metric inputs.

## Strategy
1. Work in a private copy:
   ```bash
   # Issue #429: ALWAYS use materialize_workspace.sh — do NOT inline tar/cp. Nested aiter/jit/*.so
   # must never land in integrate clones. Script excludes recursive *.so/*.o and aiter/jit, preserves
   # symlinks (never -h), and may share immutable aiter via --link-aiter.
   WS="$INTEGRATE_DIR/ws_$(date +%s)_$$"
   bash "${WORKFLOW_DIR:-$SKILL_DIR}/scripts/materialize_workspace.sh" \
     --src "$CANONICAL" --dst "$WS" \
     --shared-root "${EVAL_DIR:-$(dirname "$INTEGRATE_DIR")}/_shared" --link-aiter
   cd "$WS"
   ```
2. Sort patches by the selection metric (best first). Check compatibility using
   `optimization_strategies.md` (compatible: template+launch-bounds, tiling+coalescing, warp-coop +
   native-layout/wrapper; incompatible: two tiling schemes, two warp-coop schemes).
3. **Incremental stack**: `git apply` the best patch, then try adding each next patch. After each
   add: clear cache → correctness → benchmark (gpu_lock). Keep an add only if it stays correct and
   improves the comparison metric described above.
4. **Hand-merge on conflict**: if `git apply` rejects, read both patches and manually implement both
   ideas in a compatible way (e.g. fold a host_runtime native-layout change into an algorithm
   engineer's templated kernel). This is encouraged — the best result is often a hand-merge, not a
   diff stack. Respect hipify safety (template dispatch, no `<<<>>>` in macro if/else).
5. Always clear cache before benchmarking; always correctness before benchmark; gpu_lock for all
   benchmarks. Compute per-case speedup vs `BASELINE_PER_CASE`, geomean = `exp(mean(log(...)))`.
   Also report `weighted = Σ weight_i / Σ (weight_i / speedup_i)` when the selection rule or COMMANDMENT requests it.
   Keep `geomean` and `weighted` in their distinct output fields. Use the selection rule above to compare with `BEST_INDIVIDUAL`.

## Output
If the best combination beats `BEST_INDIVIDUAL` under the selection rule, save it:
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
