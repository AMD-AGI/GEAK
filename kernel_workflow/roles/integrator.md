# Integrator

Combine this round's verified PATCHES in a private workspace. Compose existing
ideas by applying compatible patches or rewriting their interaction; do not assume
a combination improves on BEST_INDIVIDUAL. CANONICAL is read-only. Use INSIGHTS,
COMMANDMENT and BASELINE_PER_CASE, preserving the full case and numerical/state
contract. BEST_INDIVIDUAL is the primary metric, which may be weighted.

Create the private copy with materialize_workspace.sh; do not use inline tar/cp
or copy compiled artifacts. The helper preserves symlinks and isolates builds:
```bash
WS="$INTEGRATE_DIR/ws_$(date +%s)_$$"
bash "${WORKFLOW_DIR:-$SKILL_DIR}/scripts/materialize_workspace.sh" \
  --src "$CANONICAL" --dst "$WS" \
  --shared-root "${EVAL_DIR:-$(dirname "$INTEGRATE_DIR")}/_shared" --link-aiter
cd "$WS"
git init -q
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "integration base"
```

Inspect exact patches for semantic/layout/precision/synchronization compatibility.
Start with a verified patch and add others incrementally. After each addition,
run the COMMANDMENT's build/cache preparation, complete correctness checks and
benchmark via gpu_lock.sh. Keep correct improvements in the primary metric.
If git apply conflicts, read and reconcile the ideas manually; record contributions
retained, changed or rejected. A text merge alone is not evidence of compatibility.
Do not alter the harness, COMMANDMENT or frozen baseline. Exit 86 or
GEAK_SOURCE_INVALID invalidates that invocation, including earlier PASS/timing lines.

Compute per-case speedup = frozen baseline_ms / optimized_ms, geomean =
exp(mean(log(speedups))), and arithmetic mean. For workload-aligned evaluation,
weighted = sum(weight) / sum(weight / speedup) is primary; retain geomean as a
diagnostic. Use complete matching cases and supplied weights, never count instead.

Preserve the oracle's GEAK_WEIGHTED_SPEEDUP or audited aggregation when supplied;
recompute from static weights only when that is the declared metric. Before saving,
mark any new editable source files with git add -N so the diff includes them; do
not include generated artifacts. If returning `best`, its `patch_file` must
reconstruct the exact measured source from the supplied CANONICAL. Save the full
diff against that base, or reuse a retained individual patch only if it has the
same base and reproduces that source:
```bash
cd "$WS" && git diff > "$INTEGRATE_DIR/integrated_patch.diff"
```
Return measured combinations and the following JSON. If none beats BEST_INDIVIDUAL,
use conclusion:"no_improvement"; a retained `best` still requires its patch_file.
If no measured source can be reconstructed, omit `best` and keep diagnostics in
combos_tried/notes; use all_failed when appropriate. Missing/invalid measurements
do not establish a gain.

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
