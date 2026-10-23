# Pipeline handoff (`workload/` profile stage)

This reference maps the five skill stages onto the pipeline's `profile` slot. The
skill's SKILL.md and references still own discovery, sampling, source extraction, UT
authoring and validation; the pipeline supplies inputs and consumes `$OUT`.

## Inputs

| Pipeline variable | Use in the skill |
|---|---|
| `SKILL_ROOT` | this directory; `SHARED_TOOLS="$SKILL_ROOT/shared"` |
| `OUT` | `EVAL_DIR`. Everything this skill writes goes here, never under `SKILL_ROOT` |
| `IN` / `$RUN_ROOT/baseline/out/baseline.json` | the completed measured baseline: `throughput`, `noise_band`, `config` (TP/EP/GPUs/concurrency) and `provenance` (receipt, logs, node). Use its logs and exports; do not re-measure |
| `PARAMS` | `selection` (default rule, stated for audit), `max_targets`, `trace_dir`, `serving_env`; any `agent_brief` facts about this campaign |
| `CONFIG` | `target` (model, image, framework commits) the stage must agree with |

`baseline.json` with `measured=false` is not a baseline; report `blocked`. When
`trace_dir` is null and no compatible trace exists under the baseline's provenance,
design a bounded probe per [capture](capture.md) rather than a new full replay.

`params.max_targets` narrows the default selection after it is computed (keep the
first N by overall share); it never widens it or backfills rank 6.

## Outputs

The slot's `finalize.py` maps `$OUT` to `result.json`. It requires these names:

| File | Skill step |
|---|---|
| `kernel_workload.json`, `kernel_ranking.json` | §2 census; the denominator for every share |
| `selection_targets.json` | §2, `scripts/select_targets.py`; the roofline stage ranks on `targets[].overall_pct` |
| `task_plan.json` | §3, `build_task_plan.py` from the reviewed `task_decls.json`; every selected identity owned by a task or listed in `blocked_identities` with a blocker |
| `tasks/<name>/` | §4; each holds `unittest.py`, `meta.json`, `cases.py`, `kernel_src/` and is run by the optimize stage one GPU per task |
| `REPORT.md`, `results.json` | §5 |

Finish with:

```bash
python3 "$WORKLOAD_ROOT/stages/profile/finalize.py"
```

It reports `blocked` while a selected identity has neither a task nor a stated
blocker, a task directory lacks one of the four required entries, or a task has no
passing `run/ut_result.json` (`exit_code` 0). A CPU-only pass is `blocked`, not `ok`: say
which GPU gates did not run. The optimize stage reads the same receipt and the
`{"exit": .., "weighted": ..}` line `ut_driver.py` prints.

## What the pipeline does not ask for

No optimization, no E2E claim (only `applyback` measures one), and no GEAK workflow
return: `kernel_extraction_delivery.mjs` is still the per-task preflight from SKILL.md
§5, but nothing outside `$OUT` consumes `RETURN.json`.
