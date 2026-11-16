# Pipeline handoff (`workload/` profile stage)

This reference maps the skill onto the pipeline's `profile` slot. SKILL.md and its
references own observation, sampling, capture and selection; the pipeline supplies
inputs and consumes `$OUT`.

## Inputs

| Pipeline variable | Use in the skill |
|---|---|
| `SKILL_ROOT` | this directory; `SHARED_TOOLS="$SKILL_ROOT/shared"` |
| `OUT` | `EVAL_DIR`. Everything this skill writes goes here, never under `SKILL_ROOT` |
| `IN` / `$RUN_ROOT/baseline/out/baseline.json` | the completed measured baseline: `throughput`, `noise_band`, `config` (TP/EP/GPUs/concurrency) and `provenance` (receipt, logs, node). Use its logs and exports; do not re-measure |
| `PARAMS` | numeric `selection_top_k`, `selection_overall_pct_gt`, `selection_phases`; explanatory `selection`; `max_targets`, `trace_dir`, `serving_env`; `sampling`; optional `reuse_capture`; any `agent_brief` facts |
| `CONFIG` | `target` (model, image, framework commits) the stage must agree with |

`baseline.json` with `measured=false` is not a baseline; report `blocked`. When
`trace_dir` is null and no compatible trace exists under the baseline's provenance,
design a bounded probe per [capture](capture.md) rather than a new full replay.

`params.max_targets` narrows the default selection after it is computed (keep the
first N by overall share); it never widens it or backfills rank 6.

Pass `params.sampling` (`num_points`, `context_tolerance`, `target_pct`) to
`plan_points.py` ([sampling](sampling.md)), and `params.reuse_capture` (another profile's
`EVAL_DIR` of the identical deployment) as `--reuse-capture`.

## Outputs

The slot's `finalize.py` checks `$OUT`, exports `kernels.json` and writes `result.json`:

| File | Skill step |
|---|---|
| `kernel_workload.json`, `kernel_ranking.json` | §2 census; the denominator for every share, with `selection_scope`, `candidate_inventory`, `coverage_summary` |
| `selection_targets.json` | §2, `scripts/select_targets.py`, with the policy it applied |
| `sampling_points_final.json`, `evidence/` | §2: points, censuses, shapes, source map, adjacency, compile variants |
| `kernels.json` | §3, exported by the finalizer: path, shapes and weight of every selected kernel |
| `REPORT.md` | §3 |

```bash
python3 "$WORKLOAD_ROOT/stages/profile/finalize.py"
```

It reports `fail` when the selection ran at a scope other than the configured one, and
`blocked` when the denominator or inventory is missing or `kernels.json` cannot be
derived from the evidence.

## What the pipeline does not ask for

No task boundary, no UT, no optimization and no E2E claim. Task boundaries become a
`ut_request.json` (`brain/tools/README.md`) that the UT stage consumes.
