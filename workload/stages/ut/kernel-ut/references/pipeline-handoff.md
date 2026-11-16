# Pipeline handoff (`workload/` UT stage)

## Inputs

| Pipeline variable | Use in the skill |
|---|---|
| `SKILL_ROOT` | this directory; `SHARED_TOOLS="$SKILL_ROOT/shared"` |
| `OUT` | `EVAL_DIR`: `task_plan.json`, `tasks/`, `ut_wave/`, `REPORT.md`, `results.json` |
| `PARAMS` | `request` (path of a `ut_request.json`), `minimum_qualified_tasks`, any `agent_brief` facts |
| `CONFIG` | `target.image` is the `--image` of the UT wave |
| `RUNNER` | the runner script (`runners/CONTRACT.md`) the UT wave starts one lane session with per task: `ut_wave.py start --runner "$RUNNER"` |

The request is `params.request`, else `$OUT/ut_request.json`, else `$IN/ut_request.json`.
Without one the stage is `blocked`: it does not draft task boundaries.

UT generation uses `ut_wave.py start`, with no timing measurements: CPU authors, GPUs
leased only around each `ut_wave.py gpu -- ...` command; start it on the allocated host
without an outer campaign GPU lock. Performance validation runs through the optimizer's
task entrypoint.

## Outputs

| File | Step |
|---|---|
| `task_plan.json` | §1, `scripts/scaffold_tasks.py` from the request; every requested identity owned by a task or listed in `blocked_identities` with a blocker |
| `selection_targets.json` | §1, carried from the request; the roofline stage ranks on `targets[].overall_pct` |
| `tasks/<name>/` | §1-§3; each holds `unittest.py`, `meta.json`, `cases.py`, `kernel_src/` and is run by the optimize stage |
| `REPORT.md`, `results.json` | §4 |

```bash
python3 "$WORKLOAD_ROOT/stages/ut/finalize.py"
```

It reports `blocked` while a requested identity has neither a task directory nor a
stated blocker, a task directory lacks a required entry, or a task has no passing
`run/ut_result.json` or `run/RETURN.json`. A CPU-only pass is `blocked`, not `ok`.
