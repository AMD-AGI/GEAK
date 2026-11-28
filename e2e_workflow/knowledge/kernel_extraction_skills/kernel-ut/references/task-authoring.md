# Author a self-contained task

The agent authors task-specific code; the skill supplies the shared measurement
and overlay primitives. This is not a prebuilt implementation for every kernel.
Use the inspected live signature, outputs and math rather than fabricating an
operator just to satisfy a template.

```text
tasks/<name>/
  meta.json
  cases.py
  unittest.py
  harness_lib.py
  leg_runner.py
  overlay_setup.py
  kernel_src/             actual editable native/JIT/generated source
  baseline_overlay/       immutable baseline overlay configuration
  baseline_ref/           original hashes/source text, not imported as baseline
  run/                    declarations, raw traces, gate logs, results
```

Copy `harness_lib.py`, `leg_runner.py` and `overlay_setup.py` from
`$SHARED_TOOLS/` to the task unchanged. Record every vendored helper in
`meta.kernel_extraction.source_delivery_files` so the snapshot binds the harness.
Freeze the tests when authored; candidate edits belong under `kernel_src/`.

## Metadata and adapters

`meta.target_callable` is an exact `module:attribute` spec. For a Python module
replacement, `candidate_bind` has `kind="module"`, `module` and a task-relative
`file` under `kernel_src/`. For rebind, use `kind="rebind"`, `target`,
`impl_module`, `impl_attr` and `file`. For a native/JIT leaf, the task-local Python
binding must build/import and invoke that local implementation; ship build
scripts/resources, version/arch flags and distinct module/cache identities.
Do not vendor an entire Python package merely to shadow one file.

The baseline is the installed serving stack plus immutable accepted overlays;
its callable must resolve outside the task. `_cand_overlay` adds the declared
candidate to that same baseline. `h.build_candidate_overlay(task, meta)` returns
`(baseline_overlay, candidate_overlay)`; then call
`h.assert_legs_differ(task, base, candidate, meta)`. A path difference alone is
insufficient: perform each claimed computation leaf's invocation and numerical
negative-control checks in separate untimed diagnostic processes.

`cases.py` exposes the functions consumed by the leg runner:

- `call(args)` invokes the declared entry under the ambient overlay and returns
  the complete output, including auxiliary tensors and in-place outputs.
- `timing_specs(h, meta)` returns metadata only (`{sig, regime}` and optional `m`)
  for every declared timing case, and `make_timing_case(spec)` builds ONE case's
  `{sig, regime, m, args}` with actual operands before the timed call. The leg
  runner lists sigs without allocating anything and builds one shape at a time.
- `random_shapes(h, meta)` yields `{sig, regime, make_inputs}`; the factory
  accepts a seeded torch Generator and varies values at the fixed served shapes.
  Preserve required routing semantics and the observed work/access pattern; reuse
  saved structural metadata rather than redrawing it blindly.

For joint phases, include `meta.phase_target_callables` mapping each phase to
its live entry, and `meta.phase_regimes` mapping phase labels to observed
graph/compile settings. `cases.call` selects only those entries using frozen
phase metadata; a single candidate module may implement both. If multiple
modules/leaves are needed, use a tested local binding adapter and prove all
bindings. The stock harness's single `candidate_bind` is not automatic support
for arbitrary multi-module swapping.

Record `meta.workload` using the observed case model from the [UT skill](../SKILL.md) §3. Keep
evidence paths and full signatures in the manifest. Shapes reconstructed from
median context or synthetic values are approximations, not captured joint
distribution coverage. Match cache residency to deployment; the vendored timer
has a cache policy, so inspect `h.cache_policy` and explicitly record its chosen
condition rather than treating defaults as measured deployment facts.

## Seeded inputs by default

Build `cases.py` from a fixed case manifest: shape, dtype, stride/layout,
quantization parameters, phase, seed and observed call weight, all taken from the
profile handoff. Missing shape metadata is a request to profile; UT authoring and
execution never invoke `capture_shapes.py`, a profiler or the serving model.

Generate ordinary activations and weights directly with the supplied seeded
`torch.Generator`. Preserve each operand's valid range and format: scales may need
positive values, and packed quantized operands must be constructed with the real
packing/quantization contract. Use multiple fixed seeds and boundary-value cases;
random values alone do not cover ties, masks, empty rows or numerical extremes.
Keep construction outside the timed region. Freeze the generator code and seeds
so baseline/candidate processes reconstruct identical inputs on the pinned stack.

Use `h.baseline_random_outputs(task, meta, seed=0, draws=...)` to obtain reference
outputs in the isolated frozen-baseline process. It returns outputs keyed by
`sig|draw` and deletes its temporary file; neither inputs nor outputs persist. Build
the eager cases from the same seeded factory with those baseline outputs as `ref`, and
pass the recorded outputs to `h.run_correctness` for the additional draws. Use matching
seeds and draw counts on both sides (the shared `run_correctness` random gate uses
seed 0). Graph replay still needs its normal boundary cases and in-place input refresh.
Do not introduce a `reference_io.pt`, suite-wide golden outputs or a second correctness
runner.

For MoE routing, sparse indices, page tables and sequence state, reuse handoff
metadata or a validated reconstruction that preserves the observed work/access
pattern; when neither can, report the gap rather than capturing state.
Valid synthetic structures can test correctness, but random uniform routing can
change expert occupancy and gather locality, so report its timing scope honestly.

## Correctness qualification

Use separate subprocesses for baseline and candidate. Import task helpers by
explicit file path and remove the task directory from `sys.path` before importing
torch: a file named `unittest.py` otherwise shadows Python's standard library.
The shared leg runner already performs this isolation.

0. **Before any verdict, prove the comparison can produce one.** Verify reproducible
   inputs, a tolerance supported by baseline self-noise, and effective numerical
   negative controls, all on generated cases and frozen-baseline outputs.

   A tolerance below the installed kernel's own noise floor decides nothing.
   Establish the floor with ONE set of args reused across repeats, and only after
   the args are shown reproducible: on one deployment `aiter.fused_moe.moe_sorting`
   returns different routing metadata from bit-identical inputs, so rebuilding args
   inside the repeat loop reports the routing change as kernel nondeterminism. For
   packed sub-byte payloads (MXFP4 values + exponent bytes) the only meaningful
   criterion is byte-exactness; `rel_frobenius` over packed bytes is not a number
   about the computation. Record the floor in `run/self_noise.json`.

1. Run the frozen seeded/structural inputs through the baseline; verify output structure and full
   numerical/semantic contract. Reconstruct implicit state and all required glue.
2. Build candidate overlay; prove distinct baseline/candidate entries in every
   phase and actual execution of every claimed editable leaf. Record each
   leaf's launch census so the same check binds a candidate by what it
   launches ([candidate gates](candidate-gates.md)).
3. Obtain baseline outputs with `h.baseline_random_outputs` (see *Seeded inputs*).
4. In the candidate process, use `h.run_correctness(regime, eager_cases=...,
   current_call=..., random_shapes=..., tol=..., baseline_outputs=...,
   draws=..., replay=...)`. Read its docstring for the callable bundle contracts.
   Supply recorded per-phase regimes for joint tasks, covering all output rows.
   Use task-specific semantic checks for integer indices, ties, masks, padding,
   page mappings or in-place contracts; do not loosen tolerance to hide errors.
5. Graph deployments require replay tests with observed boundary cases and
   changed input values in the same static buffers. Verify the expected graph
   actually ran. A compile error reported as a diagnostic by the shared helper
   is not proof of deployed compiled correctness: resolve it in the actual
   deployed context or report that gate incomplete.
6. Apply a reversible bad-source mutation per editable leaf. It must execute
   the intended native/JIT computation and produce a numerical/semantic
   mismatch. Import/build/launch failure does not satisfy this negative control.
   **Record the deviation the mutation actually produced and check it clears the
   tolerance by a wide margin.** A control that "passed" without its deviation ever
   being measured is not evidence of anything.
   Preserve the mutation and logs, restore pristine source, then rerun final
   normal checks before recording completion. Record each mutation in
   `meta.leaves[]` and apply it to a copy of `kernel_src/`, so on a candidate the
   UT runs it only for leaves still launched, alongside the controls the
   candidate declares ([candidate gates](candidate-gates.md)).
7. Timing is not part of qualification. `unittest.py --validation-mode correctness`
   runs steps 1-6 only and records `performance_measured: false`.

The shared tensor checker assumes independent returned outputs. If the actual
serving contract permits persistent/in-place buffers, document that contract
and add an explicit state/output checker with independent snapshots. Do not
silently present an incompatible generic tensor gate as passing.

Write a single `python3 unittest.py` entry that orchestrates and combines these
checks, emits per-gate and per-case evidence and exits nonzero on failure or
incompleteness. A prototype with only replay or timing is not a completed UT.
It accepts `--validation-mode correctness|performance` (bare entry: `performance`).
Qualify with `correctness`.

In `performance` mode (run by the optimizer) the same entry then calls
`h.measure_legs(task, meta)` and `h.serving_weighted_speedup(per_case, meta)` and prints
`GEAK_WEIGHTED_SPEEDUP=<value>`, keeping per-case timings as diagnostics; the weighting
contract is in the [UT skill](../SKILL.md) §3.

## Delivery

The collective case requires the real rank harness described in
[collective UT](collective-ut.md); the single-process leg runner cannot test a
multi-rank collective by itself.

Freeze the source/test snapshot before GPU gates. The final `run/ut_result.json`
must include the snapshot, target, validation mode, exit code, completed gates,
`performance_measured` and coverage.
Generate leaf/source evidence as described in [source records](source-delivery-records.md).
Keep all raw logs; JSON declarations alone do not establish execution truth.
