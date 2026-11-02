---
name: overlay-e2e-ab
description: Install cold-re-verified kernels back into the serving stack through a reversible overlay, re-measure AgentX end to end, and A/B against the baseline — reporting operator speedup and end-to-end delta as two separate numbers.
---

# overlay-e2e-ab

The default implementation of the applyback stage, and the **only** place in the pipeline
permitted to claim an end-to-end gain.

## Why this stage cannot be skipped

The real failure mode of this scenario is not "no speedup was found" — it is **an
operator win with no system movement**. From this very machine's history:

| Change | Operator / microbenchmark | Whole-machine e2e |
| --- | --- | --- |
| FlyDSL MoE sorting | 1.80× | **+0.2356%** |
| MoE split4 | 1.342× (local replay) | **−0.0872%** |

Both were genuine speedups, both were correct, and neither moved the machine. The
optimize stage therefore produces **candidates**, not results; results can only come from
here.

## How the kernels are installed

`site-packages` is never modified. GEAK's `e2e_workflow/scripts/overlay_setup.py` builds
a sitecustomize-driven reversible overlay that takes effect solely through
`OVERLAY_PYTHONPATH` — remove that one variable and the stack is back to original.

How each kernel is wired back is **not guessed here**: the extraction skill already
recorded it in the task's `meta.json["candidate_bind"]` as `{kind, target|module,
impl_module, impl_attr, file}` — the same descriptor it uses for its own candidate leg.
This skill does only two things:

1. Layer every cold-re-verified kernel into **one** overlay, in order, per its
   `candidate_bind` (chain-seeded with `--from` — mounting two overlay directories on
   PYTHONPATH at once does not work, because only the first sitecustomize is imported and
   the second silently has no effect).
2. **Verify the overlay actually takes effect**: `overlay_setup.py check` confirms that
   each target resolves into the overlay rather than its original location. Measuring
   without that check means a reported "no gain" may simply be an overlay that was never
   loaded.

## A/B conventions

- Reference: the baseline stage's `baseline.json` (by default an adopted receipt,
  446.87889 tok/s).
- Candidate: the same AgentX workload at the same TP8/conc10, re-measured `REPEATS` times
  with the overlay, taking the median.
- Criterion: a difference inside the noise band (default ±4.5%) is **no movement** —
  reported honestly as `no_change`, not rounded upward.

The end-to-end serving stack occupies all 8 GPUs, so this stage takes the **node
exclusively** and cannot run alongside optimize.

## Several configs

With a multi-config baseline (`adopt-ci-artifacts`: `multi_config: true`), the overlay is
built and smoke-tested once, then **each representative point** (`baseline.json`
`representative`, by default each group's lowest and highest concurrency) is measured on a
fresh server in `$OUT/points/<pid>/` against **its own CI number**
(`baseline/out/points/<pid>/baseline.json`). CONC and TP come from the point's CI config;
the launch settings that differ between points (KV offload, speculative decoding) come from
`params.point_serving_env.<pid>`, else `serving_env`. `aggregate_points.py` folds the
per-point A/Bs: any regression makes the verdict `regression`, a gain needs a gain somewhere
and no regression, and a point without an A/B makes it `incomplete`. The top-level
`e2e_delta_pct` is the geometric mean over points and is labelled so; the per-point deltas
are the result.

The reference is the CI runner, not this node, so node and session differences are inside
every delta; the noise band has to cover them.

## Preconditions

Run `e2e_workflow/scripts/agentx_smoke.sh` first. `aiperf` and InferenceX are on neither
`$PATH` nor `/opt`; when pieces are missing this stage returns `blocked` and names them.
It must never produce a throughput number that in fact measured something else.

## Output

`$OUT/overlay/` (the reversible overlay itself), `$OUT/e2e_candidate.json`,
`$OUT/ab.json`, `$OUT/result.json`. The mapping is performed by the slot adapter
`stages/applyback/finalize.py`.
