# Sampling points for one config

Read the config's scheduler observations first (`observed_workload.py`); they decide
which shapes and contexts the deployment actually executed and how often. Observations
can come from existing baseline/CI logs. Collect a short observation window only where
the required raw rows are missing; do not repeat a full workload just to plan points.

## Inputs

`$EVAL_DIR/serving.json` and `$EVAL_DIR/evidence/observed_workload.json`, using the
ordinary probe/observed-workload schemas; optionally `evidence/server_info.json` from the
deployed server (prefill chunk, graph capture sizes). Keep them immutable during the run.
`runtime_identity.json` is needed only to reuse another profile's captures:

```json
{
  "image": "the deployed image digest",
  "model_revision": "the deployed model snapshot or content manifest hash",
  "source_revision": "installed source/overlay revision or content manifest hash",
  "gpu_arch": "the actual GPU architecture"
}
```

Additional identity fields participate in equality. Record actual values from the
environment receipt, not placeholders from this example.

## Plan

```bash
python3 "$SKILL_ROOT/scripts/plan_points.py" --eval-dir "$EVAL_DIR" \
  --num-points 24 --context-tolerance 1.5 --target-pct 90
```

Planning is CPU-only. Observed rows are grouped by phase, exact shape, sequence count,
observed graph mode and a context band; adjacent M values are never assumed to share a
kernel. The representative of a cell is an actual observed row. Context banding is an
explicit approximation; `--context-tolerance 1` requires exact context equality.

`--num-points` is the capture budget, chosen from the number of distinct phase/shape
domains and the available capture time. The budget goes greedily to uncovered proxy
mass (prefill new tokens, decode iterations), each phase normalized separately.
`sampling_plan.json` records planned proxy coverage per phase, `budget_limited` when a
phase misses its target, and every unselected cell. The proxy is not GPU-time coverage.

## Reuse another profile's captures

```bash
python3 "$SKILL_ROOT/scripts/plan_points.py" --eval-dir "$EVAL_DIR" --num-points 24 \
  --reuse-capture <another profile's EVAL_DIR>
```

Allowed only when both deployments are identical: every serving field and argument
except transport addresses/ports, physical GPU allocation, client concurrency and `_`
notes, and the runtime identity. Scheduler settings, graph limits, prefill chunk, KV
policy and TP/EP stay significant; `--max-running-requests` is not a client label. A
different deployment is refused rather than reused.

A cell the other profile captured (signed census present) is planned verbatim, costs no
budget, and its census and probe result are linked into this `EVAL_DIR`; `run_points.py`
resumes those points and captures only the rest. Costs are always reweighted with this
config's own observations by `relabel_points.py`; a reused capture is never presented
as an independent serving run of this config (`profile_run.json` keeps `reused_from`).

## Resume and fill gaps

With unchanged inputs, increase `--num-points` and rerun the planner: point IDs and the
greedy prefix are stable, and `run_points.py` reuses signed censuses. A failed or
shape-shifted probe needs the ordinary probe diagnosis/recapture as well; a larger
budget is not proof that a shape was achieved. Changed inputs or deployment identity
require replanning; a changed capture signature requires explicit `--redo`.
