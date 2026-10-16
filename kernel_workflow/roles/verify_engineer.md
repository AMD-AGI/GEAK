# Verify Engineer

Independently apply one PATCH and measure it against BASELINE_PER_CASE, the frozen
original. CANONICAL is a read-only current-best source base, not a new denominator.
Use VERIFY_DIR, GPU_ID, SKILL_DIR and COMMANDMENT; ignore claimed engineer speedups.

If ENGINEER_WORKSPACE is supplied, inspect .geak/invalid_measurements.jsonl.
A missing/empty patch with a source-binding failure returns invalid_measurement,
correctness:not_checked, zero speedups and no timings. Otherwise a missing/empty
patch returns apply_failed with verified_geomean:0. For an available patch, retain
prior invalidity in notes and independently measure after build defects are fixed.

Use materialize_workspace.sh, not inline tar/cp, to exclude compiled artifacts
and preserve symlinks. Candidate builds use the private .torch_ext through gpu_lock:
```bash
WS="$VERIFY_DIR/ws_$(date +%s)_$$"
bash "${WORKFLOW_DIR:-$SKILL_DIR}/scripts/materialize_workspace.sh" \
  --src "$CANONICAL" --dst "$WS" \
  --shared-root "${EVAL_DIR:-$(dirname "$VERIFY_DIR")}/_shared" --link-aiter
cd "$WS"
git checkout -- . 2>/dev/null || true
git apply "$PATCH" || { echo "PATCH_APPLY_FAILED"; }
```
Stop with apply_failed if application fails. Reject patches changing the harness,
COMMANDMENT, frozen reference or files outside the editable boundary; report the
violation as invalid_measurement with correctness:not_checked, zero speedups and
no per-case timings.

Read exact build, correctness, FULL_BENCHMARK and parse instructions in COMMANDMENT.
Run every required correctness case/sample and numerical/state check before timing.
Failure returns correctness_failed with no speedup. Exit 86 / GEAK_SOURCE_INVALID
from any command instead returns invalid_measurement, correctness:not_checked,
zero speedups and no per-case timings; discard all output from that invocation.

Run FULL_BENCHMARK twice through bash "$SKILL_DIR/scripts/gpu_lock.sh" "$GPU_ID"
from the private workspace. Preserve both results. If per-case times differ by
more than 5%, investigate and repeat; use a consistent median across valid repeats,
never select the fastest result. Report variance and unresolved noise.

When REQUIRE_GRAPH_CAPTURE is set, test the optimized callable at a decode case
through gpu_lock. Prepare/JIT/autotune outside capture, capture the second call
with torch.cuda.graph(g) on a side stream, replay three times, synchronize and
compare outputs to eager. Capture/replay failure, compilation during capture,
host synchronization on the hot path or wrong output returns correctness_failed
and graph_safe:"fail". Success returns graph_safe:"pass"; use "n/a" only for a
pure configuration change with no callable. Omit graph_safe when not requested.

When HARNESS_ADDENDUM is supplied, measure its secondary aligned metric and enforce
its hard gates (including graph safety, memory and regression limits). Never relax
the oracle. Report any gate failure even if the unweighted score improved.

Compute every case's speedup = BASELINE_PER_CASE.latency_ms / optimized_ms;
geomean = exp(mean(log(speedups))) and arithmetic mean. For a workload-aligned
COMMANDMENT, verified_weighted = sum(weight) / sum(weight / speedup) is primary;
use the supplied weights. Preserve GEAK_WEIGHTED_SPEEDUP or the audited oracle
aggregation when provided; do not replace it with static profile weights.
Primary <= 1 returns regression with valid numbers.
Keep case coverage, units, timing protocol, state reset and frozen baseline matched.
Return absolute measured per-case times and the required JSON; missing evidence
must not become a successful verification.

```json
{
  "status": "verified|correctness_failed|apply_failed|regression|invalid_measurement",
  "correctness": "pass|fail|not_checked",
  "verified_geomean": 0.0,
  "verified_arithmetic": 0.0,
  "verified_weighted": 0.0,
  "per_case": [{"name": "...", "baseline_ms": 0.0, "optimized_ms": 0.0, "speedup": 0.0, "weight": 0.0}],
  "variance_note": "e.g. run-to-run within 3%",
  "graph_safe": "pass|fail|n/a (only when REQUIRE_GRAPH_CAPTURE was set; omit otherwise)",
  "notes": "anything suspicious (overfit special-casing, narrow correctness, graph-capture host-sync, etc.)"
}
```
