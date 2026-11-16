# Profile

Diagnose the serving bottleneck under the
[invocation contract](../CONTRACT.md#direct-module-calls).

**Input:** workload/runtime identities, the bottleneck question and compatible run evidence.
**Output:** observations, attributed costs, coverage and uncertainty in `REPORT.md`;
for kernel work, `kernels.json` -- one config's selected kernels with their source path,
served shapes and workload weights ([kernels.py](kernels.py)).
**Acceptance:** evidence matches the declared workload, phase and resource scope;
missing coverage stays visible. Profile decides no task boundary and writes no UT.

Implementation: [kernel-profile](kernel-profile/SKILL.md) for one config's kernels, or a
profiler explicitly named in the plan. Several configs are several calls; an identical
deployment's captures can be reused (`plan_points.py --reuse-capture`). Use traces,
native telemetry and CPU/device, queue, cache or communication evidence as the question
requires. Computation boundaries needing qualification become a `ut_request.json`
([request tools](../../brain/tools/README.md)) for [UT](../ut/README.md); use Roofline
when a bound would help the next decision.

For an unqualified capture path or a collection defect, establish capture, export and
readable evidence within a bounded call before expanding the census; reuse a qualified
path without another admission stage. Keep traffic, its capture trigger, stop/export and
verification in the same owned invocation; a prepared command is not an issued request.
Observe scheduler progress as well as frontend health. A timed-out control request leaves
capture state unknown: preserve logs and inspect owned workers before rearming; return
necessary runtime recovery to Setup. Choose a bounded follow-up from the installed
backend's capabilities and the observed failure, rather than repeating an unchanged call.
Traffic, profiler acknowledgement and process exit do not prove capture success.
Keep changed-mode diagnostic evidence distinct from deployed timing, and report the
remaining attribution/coverage gap through the contract.

## Candidate scope

The scenario's selection parameters override the extractor's defaults.
For the bundled profiler, `select_candidates.py` forwards
`selection_top_k`, `selection_overall_pct_gt` and `selection_phases` to its native
selector. Set `SKILL_ROOT` to `kernel-profile/`, `SHARED_TOOLS` to its `shared/`,
and `EVAL_DIR=OUT`; see its
[pipeline handoff](kernel-profile/references/pipeline-handoff.md).

```bash
python3 "$WORKLOAD_ROOT/stages/profile/select_candidates.py" \
    --source-map "$OUT/evidence/source_map.json"
```

Supply the actual reviewed operator map; omitting it can drop fused work whose member
kernels individually fall below the cutoff. The adapter retains an invocation hash of
the ranking, map, selector and arguments; changed inputs need a fresh output path.
When using the native capture orchestrator, pass the same configured scope through its
`--select-arg` interface.

`selection_candidates.json` is the broad inventory, not one UT per row. Preserve every
selected identity, whole-workload shares, phase leaders and deferred work in
`kernels.json`; task boundaries are drawn downstream. GPU capacity limits concurrency,
not profile scope. Keep `selection_scope`, `candidate_inventory` and `coverage_summary`;
inventory and tested-case coverage are separate claims. Reuse compatible captures when
expanding scope.

## Driver delivery

`run.py run profile` requests one config's kernel profile. [finalize.py](finalize.py)
checks the configured selection scope, the denominator and the inventory, then exports
`kernels.json` ([kernels.py](kernels.py)); `kernels.py verify` re-derives it from the
unchanged evidence.

| Artifact | Purpose |
| --- | --- |
| `kernels.json` | Selected kernels: path (entry, files, frames), points/shapes, weights, gaps; SHA-256 of its evidence |
| `kernel_ranking.json` | Kernel inventory and selection/coverage evidence |
| `selection_targets.json` | Explicit selected scope, policy receipt and whole-workload shares |
| `kernel_workload.json` or ranking `denominators` | Nameable census denominator |
| `sampling_points_final.json`, `evidence/` | Points, censuses, shapes, source map, adjacency, compile variants |
| `REPORT.md` | Sampling plan, achieved coverage, gaps |

## Measurement meaning

Measure weights in the actual workload phase, excluding warmup. Keep whole-workload,
capture-window and within-call shares distinct; do not renormalize missing mass.
Map observed requests to executed tensor geometry, including graph padding and routing
occupancy/scope; source evidence must support the mapping.
