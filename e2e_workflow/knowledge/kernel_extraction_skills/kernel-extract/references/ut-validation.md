# Validate a generated UT (verify_ut.py)

A passing UT proves the candidate matched the baseline under that test. It does not
prove the test can see anything. `verify_ut.py` checks the test itself, independently
of the author, and its verdict is part of the delivery: the gates accept a generated
task only with a `run/verify_ut.json` whose verdict is `PASS` and whose
`contract_sha256` and `snapshot_sha256` equal the current ones.

```bash
cd "$TASK_DIR"
python3 "$SHARED_TOOLS/verify_ut.py" --task . --static-only          # CPU part, any time
bash "$SHARED_TOOLS/../../kernel_workflow/scripts/gpu_lock.sh" "$GPU_ID" \
  python3 "$SHARED_TOOLS/verify_ut.py" --task .                     # full verification
```

Run it after the UT passes on the GPU and before the source mutations are recorded
(a mutation must not be in `kernel_src/` while it runs), best in the same lease as the
UT and the mutations ([SKILL.md](../SKILL.md) step 5). A multi-rank task runs it
under the same group lease as its UT (`group:$SERVING_GPU`).

## What it checks

Static (CPU, never imports `cases.py`):

- the vendored `harness_lib.py`, `leg_runner.py`, `overlay_setup.py`, `ut_runner.py`
  are byte-identical to the shared ones, `unittest.py` is the generated one, and
  `run/ut_contract.json` still matches the test files;
- `generate_ut.py check`: hooks required by the declared capabilities, metadata
  (each graph-deployed phase's `graph_capture` and the `m` of its timing cases, a
  stateful task's `stateful_tiers`),
  case-manifest coverage, the GPU count against `run/requested_kernel.json`, and the
  reverse check of a single-GPU declaration (cross-device symbols in `kernel_src/`
  need `capabilities.single_rank_justification`), and every source file in
  `kernel_src/` is either copied source with a recorded origin or a declared binding
  file;
- every disabled control has a written reason; declared coverage gaps
  (`meta.coverage.gaps`) are empty.

GPU:

- **A/A.** The candidate side runs the task's own overlay and binding with the
  baseline's own source, so a bias of the binding itself (for example a cheaper
  dispatch path than the deployed wrapper) is measured too.
  - When every copied source in `kernel_src/` is byte-identical to its origin
    (`meta.kernel_extraction.source_origins`) and every other file is a declared
    binding file (`meta.kernel_extraction.binding_files`), the UT's own passing result
    for the same contract and snapshot is the first A/A measurement; otherwise a
    fresh run measures it.
  - When the copied source was modified, both legs run the baseline (`--aa`) and
    `verify_ut.json` notes that the A/A did not go through the candidate binding.
  - Every check must pass, and the baseline's run-to-run error must stay below half
    the tolerance. The measure is `native_noise.noise_rel` of the UT result: the native
    refs leg against the native oracle on the random draws, the largest
    |out − ref| / (|ref| + RMS(ref)) over the finite elements (RMS of the finite part,
    as `correct()` takes it), worst output and worst rank (with per-output tolerances,
    task-authoring.md step 0 (b): each output's `native_noise.per_output` noise_rel against
    half its own tolerance). On the same pair of tensors
    `correct()` passes at a tolerance t exactly when noise_rel ≤ t, so noise_rel ≤ 0.5 ×
    tol says the self-noise uses at most half of the tolerance band. `max_rel_err` is
    not the measure: its denominator is |ref| + tol·RMS, so on a reference element near
    0 it reads |Δ| / (tol·RMS) -- the share of the band used, not a number on tol's
    scale -- and holding it below tol/2 would demand |Δ| ≤ tol²·RMS/2. A result written
    before `noise_rel` existed is judged by its `max_rel_err` as before. `verify_ut.json`
    records for every A/A measurement `noise_rel`, `max_rel_err`, `tol` and 2 × noise_rel
    (`self_noise`), and notes "tolerance looser than the measured self-noise needs"
    (without changing the verdict) when tol is above the smallest of {1, 1.5, 2, 3, 5} ×
    10^k that holds 3 × noise_rel.
  - The weighted speedup AND the call-weighted speedup of every phase with call mass
    must lie within ±3%.
  - The A/A always runs all 3 timing pairs (no early stop). Outside the band it is
    measured once more and fails only when both measurements are outside. The failure
    names each out-of-band phase's worst case, with both legs' `ms`, `host_ms` and
    dispatch-inclusive latency.
  - A through-binding A/A outside the band also runs `--aa` once as a diagnostic,
    recorded in `verify_ut.json`. If the baseline against itself is in band, the bias
    is on the candidate side; if not, it is in the harness.
- **Timing receipt.** Read-evict cache preparation, the same timer on both legs,
  device-dominated readings for every positive-mass case of a graph deployment, the
  queued GPU spin on both legs of every eager-deployed single-GPU case (see
  task-authoring.md, "Timing an eager-deployed phase"), and for a multi-rank task
  every rank's time and the alignment method. Receipt `flags` (dispatch slower than
  the kernel, a spin that ran out) are recorded and never fail the verification.
- **Controls** (correctness only, no timing, one at a time; each wraps what the UT
  compares inside the worker, so module and rebind bindings are treated alike and no
  later normalization can cancel it):

| control | perturbation | expected |
|---|---|---|
| `noop` | wrapper only | PASS |
| `within_tol` | float outputs × (1 + tol/10) (per-output tolerances: each its own tol) | PASS |
| `scale_out` | float outputs × (1 + 4·tol) (each its own tol); integer/bool outputs moved off their values | rejected |
| `stale_output` | a call whose input shapes were seen before returns that shape's first output | rejected (random draws) |
| `skip_state_write` (stateful) | every tensor argument the call wrote is restored afterwards; in a `carry_state` replay group, the declared `state` is restored after every replay | rejected in the sequence check and in that group's graph_replay rows |
| `rank_skew` (multi-rank) | the last rank builds the previous rank's data | rejected on that rank |

A case whose call raised compared no output and never counts as a catch; a NaN or
inf output does. This applies to every control (stricter than the original
monokernel rule, which applied it to task controls only).

**Task controls** (`meta.controls.task[]`): `{"name", "file", "attr", "expect",
"equivalent_on"}` where `attr(call, h, meta, ctx)` returns a wrapped `call` that
perturbs inside what the kernel computes. Write the perturbation so that nothing
after it cancels it (a factor applied before a normalization is undone by it);
`equivalent_on` lists the cases on which it is equivalent to the baseline -- those
cases do not count, and a control seen only there fails. A control seen only below
1.5 × tol gets a warning. `expect: "below_tol"` declares a real bug the test cannot
see: every case is checked to stay below tol with the output actually changed, and
even then it is a coverage gap that blocks `PASS`.

To disable a generic control, give the reason in `meta.controls.disabled`
(`{"stale_output": "the op is stateless and returns ..."}`); the gates check that the
reason exists.

Recorded errors keep 5 decimals, and each record is checked against the static rule
before it is written: a case the UT failed reads above tol. `correct()` fails an
element when |Δ| > tol·(|ref| + RMS) and reads it as |Δ| / (|ref| + tol·RMS), which
is then above tol (above 1 when tol is above 1; with per-output tolerances, above the
row's own tolerance t, or t / T when the largest tolerance T is above 1). A failing
case whose error rounds to exactly tol is recorded rounded up and named in
`rounded_up`; one recorded lower failed on something its error does not show, and the
record is refused. A case that passed keeps its recorded error and is not held to
tol: on a reference element near 0 a passing comparison reads up to 1
(|Δ| / (tol·RMS), the share of the band used, as in the A/A above), so a passing
`max_rel_err` above tol is no reason to widen a tolerance sized from the self-noise
as in task-authoring.md step 0 (b).

## Verdicts

- `PASS`: every check above holds.
- `FAIL`: a named check or control failed, or the task's own code raised (a
  `cases.py` traceback is a FAIL, never a reason to retry). Fix the test, regenerate
  with `generate_ut.py all`, rerun the UT and verify again.
- `INCONCLUSIVE`: a run was stopped from outside its code: out of GPU memory or
  resources, a time limit, the GPU lock not acquired, foreign work on the GPU, a GPU IPC
  fault (`hipIpcOpenMemHandle ... invalid device pointer`), a rank killed by a signal.
  `verify_ut.py` already reran it (at most twice; a multi-rank task with half as many
  rank pools alive each time); run it again later. It is not a task defect.

## Division of labour with the source mutations

Controls prove the test is sensitive; the per-leaf source mutations
([source records](source-delivery-records.md)) prove the UT executes the extracted
source. Both are required; a mutation that only breaks the build is not a numerical
rejection.
