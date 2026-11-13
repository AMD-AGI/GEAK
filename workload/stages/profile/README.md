# Profile

Diagnose the serving bottleneck under the
[invocation contract](../CONTRACT.md#direct-module-calls).

**Input:** workload/runtime identities, the bottleneck question and compatible run evidence.
**Output:** observations, attributed costs, coverage and uncertainty in `REPORT.md`.
**Acceptance:** evidence matches the declared workload, phase and resource scope;
missing coverage stays visible. A diagnostic observation need not produce UTs.

Implementation: the observation and attribution parts of
[kernel-extract](kernel-extract/SKILL.md), or a profiler explicitly named in the plan.
Use traces, native telemetry and CPU/device, queue, cache or communication evidence
as the question requires. Hand computation boundaries needing qualification to
[UT](../ut/README.md); use Roofline when a bound would help the next decision.

## Candidate scope

The scenario's selection parameters override the extractor's defaults.
For the bundled extractor, `select_candidates.py` forwards
`selection_top_k`, `selection_overall_pct_gt` and `selection_phases` to its native
selector. Set `SKILL_ROOT` to `kernel-extract/`, `SHARED_TOOLS` to its `shared/`,
and `EVAL_DIR=OUT`; see the vendored
[pipeline handoff](kernel-extract/references/pipeline-handoff.md).

```bash
python3 "$WORKLOAD_ROOT/stages/profile/select_candidates.py" \
    --source-map "$OUT/evidence/source_map.json"
```

Supply the actual reviewed operator map; omitting it can drop fused work whose member
kernels individually fall below the cutoff. The adapter retains an invocation hash of
the ranking, map, selector and arguments; changed inputs need a fresh output path.
When using the native capture orchestrator, pass the same configured scope through its
`--select-arg` interface.

`selection_candidates.json` is the broad inventory, not one UT per row. Reconcile
complete nonoverlapping operator boundaries with source availability and existing
deliveries. Preserve every selected identity's task or blocker, whole-workload shares,
phase leaders and deferred work. GPU capacity limits concurrency, not extraction scope.
Keep `selection_scope`, `candidate_inventory` and `coverage_summary`; inventory,
task completion and tested-case coverage are separate claims.
Reuse compatible captures when expanding scope and qualify only the added tasks.

## Driver delivery

`run.py run profile` still requests the combined configured Profile/UT delivery.
Its [finalize.py](finalize.py) remains the mapping and acceptance owner; a scoped
diagnostic is not submitted as though this whole delivery were complete.
Use the [UT implementation](../ut/kernel-extract/SKILL.md) for task qualification.

| Artifact | Purpose |
| --- | --- |
| `kernel_ranking.json` | Kernel inventory and selection/coverage evidence |
| `selection_targets.json` | Explicit selected scope, policy receipt and whole-workload shares for Roofline |
| `kernel_workload.json` or ranking `denominators` | Nameable census denominator |
| `task_plan.json` | Boundaries, ownership and blockers |
| `task_gpu_requirements.json` | Optional device widths when frozen metadata lacks them |
| `tasks/<name>/` | Qualified tasks with `unittest.py`, `meta.json`, `kernel_src/` and `cases.py` |
| `tasks/<name>/run/ut_result.json` | Native full-test verdict |
| `tasks/<name>/run/RETURN.json` | Native delivery-gate receipt |
| `results.json`, `REPORT.md` | Identity dispositions and declared/achieved coverage |

The finalizer checks configured scope and delivery consistency using native receipts.
A CPU preflight and the full GPU test are separate evidence; unavailable GPUs leave
qualification incomplete. The vendored skill stays byte-identical.

## Measurement meaning

Measure weights in the actual workload phase, excluding warmup. Keep whole-workload,
capture-window and within-call shares distinct; do not renormalize missing mass.
Map observed requests to executed tensor geometry, including graph padding and routing
occupancy/scope, before freezing cases. Source evidence must support the mapping.

Keep fixture staging and checks outside the scored call while preserving the target's
actual dispatch, allocations and copies. Timing and correctness must exercise the same
computation. Qualify unchanged-source repeatability and timing sensitivity under the
serving execution mode; a timing repair requires fresh performance qualification.
