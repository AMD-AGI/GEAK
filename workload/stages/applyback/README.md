# The `applyback` slot

**What it does**: installs the kernels that **passed cold re-verification** in the
optimize stage back into the serving stack, re-measures AgentX end to end, and runs an
A/B against the baseline. This is the only stage in the pipeline permitted to claim an
end-to-end gain.

**Input**: `$RUN_ROOT/optimize/out/optimized.json` (the kept list),
`$RUN_ROOT/profile/out/tasks/<k>/meta.json` (`candidate_bind` — how to wire it back), and
`$RUN_ROOT/baseline/out/baseline.json` (the reference).

**Output**: `$OUT/overlay/`, `$OUT/applied.json`, `$OUT/e2e/`, `$OUT/ab.json`,
`$OUT/result.json`.

## Implementations

| Skill | Notes |
| --- | --- |
| `overlay-e2e-ab/` | Reversible sitecustomize overlay, fresh server, A/B — measured with GEAK's own `bench_e2e.sh` and the `agentx` client. |
| `reference-replay-ab/` | For a baseline that was **not** measured by `bench_e2e.sh`: re-runs the reference harness itself, with the overlay handed to it through its own `python_overlay` hook. Both legs then come from the same command. |

Building the overlay is slot-level machinery, not a skill's: `build_overlay.py` sits
beside `finalize.py` and both implementations call it, so the chain-seeding rule and the
shadow check cannot drift apart between them.

To replace them — patching a fork directly, or targeting a different serving stack — create
a directory with a `SKILL.md` and a `run.sh`, and rename `stages.applyback.skill`. The
slot's `finalize.py` (the A/B conventions and the rule that the two numbers stay
separate) does not need rewriting.

## Artifacts → `result.json`

| result.json | Source |
| --- | --- |
| `artifacts.ab` | `ab.json` |
| `artifacts.applied` | `applied.json` (what was installed, what was skipped, shadowing checks) |
| `artifacts.overlay` | `overlay/` (reversible; remove it from `OVERLAY_PYTHONPATH` to return to the original stack) |
| `metrics.e2e_delta_pct` | **whole-machine** change |
| `metrics.op_speedup_max` | **operator** peak speedup |
| `metrics.verdict` | `gain` / `no_change` / `regression` |

## Four rules that must not be changed

1. **Operator speedup and end-to-end delta are stated separately and never merged.** The
   real failure mode of this scenario is exactly an operator win with no system movement:
   MoE sorting at 1.80× corresponds to +0.2356% whole-machine, MoE split4 at 1.342× to
   **−0.0872%**. Printed side by side, neither can be mistaken for the other.
2. **Inside the noise band means no change.** Report `no_change` honestly, without
   rounding it upward. The band is calibrated from measurement (default 1.045), not
   quoted from documentation.
3. **No conclusion until the overlay is shown to shadow its targets.** Otherwise "no
   gain" may simply mean the patch was never loaded. `build_overlay.py` confirms each one
   with `overlay_setup.py check`; anything short of all-pass is a `fail`.
4. **The overlay must be loaded by a freshly started server** (`REUSE_SERVER=0`).
   Reusing an already-running server measures an unpatched stack.

For modules loaded lazily, verify the source path and hash in the actual serving workers
after a warmup request exercises that path. A parent-process import or a pre-warmup check
does not establish what each serving rank loaded. If a temporary loader observer is
needed, remove it after the real load and record that restoration; do not change the
kernel or leave instrumentation in steady-state timing.

Use the same cache placement and freshness policy on both sides of A/B. When a long
optimization run separates the candidate from its original reference, a fresh unchanged
control can expose environment drift. Record its drift from the original reference and
report throughput together with relevant TTFT/ITL tradeoffs. Keep valid slow repeats in
the reported statistic. Report invalid runs with their receipts and reasons rather than
deleting or reusing their benchmark IDs.

## Two environment constraints

- **Exclusive node**: end-to-end measurement needs all 8 GPUs at TP8/conc10 and cannot
  run alongside optimize.
- **`aiperf` / InferenceX are on neither `$PATH` nor `/opt`**, so `agentx_smoke.sh` runs
  first. When pieces are missing this stage returns `blocked` and names them, rather than
  producing a throughput number that actually measured something else.
