# The `applyback` slot

**What it does**: installs the kernels that **passed cold re-verification** in the
optimize stage back into the serving stack, re-measures end to end, and A/Bs against the
baseline. This is the only stage permitted to claim an end-to-end gain.

**Input**: `$RUN_ROOT/optimize/out/optimized.json` (the kept list, with each kernel's
`kernel_src`, `kernel_src_hashes` and frozen `task_meta_sha256`),
`$TASKS_DIR/<k>/meta.json` (`candidate_bind` — how to wire it back), and
`$RUN_ROOT/baseline/out/baseline.json` (the reference).

Split scenarios export `TASKS_DIR` from `task_source_stage: "ut"`; legacy invocations
fall back to `$RUN_ROOT/profile/out/tasks` in the entry scripts.

**Output**: `$OUT/overlay/`, `$OUT/applied.json`, `$OUT/shadow_check.json`,
`$OUT/e2e/bench_summary.json`, `$OUT/ab.json`, `$OUT/result.json`.

Task metadata supplies the binding, not a replacement working source.

## Implementations

| Skill | Notes |
| --- | --- |
| `overlay-ab/` | the only one. A reversible sitecustomize overlay, proof that the serving ranks loaded it, a fresh stock control, and the A/B. Which benchmark runs is the `harness_cmd` parameter: GEAK's `bench_e2e.sh` when it is absent, the reference harness that produced the baseline when it is present. |

This is an **agent-shape** stage (contract §2B): there is no `run.sh`. Layering an
overlay, starting a clean server and replaying a harness are a handful of commands; what
makes the stage hard is knowing which mistakes silently produce a believable number, and
that is prose. It used to be 1,886 lines of code across two directories whose only real
difference was which benchmark command ran.

To target a different serving stack — patching a fork directly, say — add a directory with
a `SKILL.md` and point `stages.applyback.skill` at it. The slot's `finalize.py` does not
change: the criterion belongs to the slot.

## Artifacts → `result.json`

| result.json | Source |
| --- | --- |
| `artifacts.ab` | `ab.json` |
| `artifacts.applied` | `applied.json` (what was installed, what was skipped, the overlay's hashes) |
| `artifacts.shadow` | `shadow_check.json` (`overlay_setup.py check`) |
| `artifacts.overlay` | `overlay/` (reversible; drop it from `OVERLAY_PYTHONPATH` to return to the original stack) |
| `metrics.e2e_delta_pct` | **whole-machine** change |
| `metrics.op_speedup_max` | **operator** peak speedup |
| `metrics.verdict` | `gain` / `no_change` / `regression` |

## What is judged, and where

`finalize.py` is the verdict (`stages/CONTRACT.md` 3.1) and the rules are written only
there: that the overlay shadowed its targets, that the installed bytes are the
cold-verified bytes, that the configured native-validity fields hold, that both legs share
a metric basis, and the delta against the noise band. Read that file rather than looking
for a second copy here — this slot once carried the same hash check in five places, and
earlier A/B artifacts are retained for audit precisely because a failed current run must
not be able to publish them.

Scenario config supplies whatever a particular harness calls its fields —
`native_validation`, `overlay_receipt`, `serving_source_proof`,
`require_native_interactivity_p90`, `cache_root`, `binding_options`, `harness_cmd` —
because those are one harness's JSON key names, which are data, not skill.

## Two environment constraints

- **Exclusive node**: end-to-end measurement needs all 8 GPUs at TP8/conc10 and cannot run
  alongside optimize.
- **`aiperf` / InferenceX are on neither `$PATH` nor `/opt`**, so `agentx_smoke.sh` runs
  first. When pieces are missing the stage is `blocked` and names them, rather than
  producing a throughput number that actually measured something else.
