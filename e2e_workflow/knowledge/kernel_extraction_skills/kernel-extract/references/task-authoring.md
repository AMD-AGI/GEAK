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
  captures/               optional minimal structural fixtures or fallback oracles
  run/                    declarations, raw traces, gate logs, results
```

Copy `harness_lib.py`, `leg_runner.py` and `overlay_setup.py` from
`$SHARED_TOOLS/` to the task unchanged. If a helper needs its sibling
capture/trace utilities, copy those too. Record every vendored helper in
`meta.kernel_extraction.source_delivery_files` so the snapshot binds the harness.
Freeze the tests/oracles when authored; candidate edits belong under `kernel_src/`.

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
- `timing_cases(h, meta)` yields `{sig, regime, args}` and optional `m` for every
  declared timing case. Construct actual operands before the timed call.
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

Record `meta.workload` using the observed case model from `SKILL.md`. Keep
evidence paths and full signatures in the manifest. Shapes reconstructed from
median context or synthetic values are approximations, not captured joint
distribution coverage. Match cache residency to deployment; the vendored timer
has a cache policy, so inspect `h.cache_policy` and explicitly record its chosen
condition rather than treating defaults as measured deployment facts.

## Seeded inputs by default

Build `cases.py` from a fixed case manifest: shape, dtype, stride/layout,
quantization parameters, phase, seed and observed call weight. Use existing logs,
trace metadata and verified source/configuration to establish that manifest. If
needed, collect missing shape metadata once during extraction; UT execution must
never invoke `capture_shapes.py`, a profiler or the serving model.

The manifest spans the **served distribution**, not one point in it. Derive its shape set
from `plan_sampling_points.py`'s points: one case per planned point of the phases this
task owns, within the ledger's shape tolerance, each carrying that point's
`effective_calls`. Then add the buckets the points do not already name — dispatch
boundaries, graph-padding sizes, chunk tails, the smallest and largest served shapes —
as their own cases, because a bucket that switches branch or kernel is a different test
regardless of its mass. One case per phase is a bootstrap, and a manifest that never grew
past it will pass every gate while testing a single batch size of a kernel served across
ten. Export the frozen manifest to `run/ut_cases.json` and grade it with
`coverage_ledger.py`; the resulting `shape_coverage` block is part of the task return and
the delivery gate rejects a task without it.

A case whose shape is reconstructed rather than observed stays in the manifest with zero
observed weight and is labelled as an envelope probe: it extends correctness coverage and
must not be weighted as served mass.

Generate ordinary activations and weights directly with the supplied seeded
`torch.Generator`. Preserve each operand's valid range and format: scales may need
positive values, and packed quantized operands must be constructed with the real
packing/quantization contract. Use multiple fixed seeds and boundary-value cases;
random values alone do not cover ties, masks, empty rows or numerical extremes.
Keep construction outside the timed region. Freeze the generator code and seeds
so baseline/candidate processes reconstruct identical inputs on the pinned stack.

Use `h.baseline_random_outputs(task, meta, seed=0, draws=...)` to obtain reference
outputs in the isolated frozen-baseline process. It records outputs keyed by
`sig|draw`; input tensors need not be saved. Build eager/reference cases from the
same seeded factory, and pass the recorded outputs to `h.run_correctness` for the
additional draws. Use matching seeds and draw counts on both sides (the shared
`run_correctness` random gate uses seed 0). Graph replay still needs its normal
boundary cases and in-place input refresh. A serving-captured `reference_io.pt`
is optional; generated reference outputs are not captures from the model.

Record which inputs are generated and which are structural fixtures. For MoE
routing, sparse indices, page tables and sequence state, first reuse existing
metadata or a validated reconstruction that preserves the observed work/access
pattern. Use a minimal structural capture only when those are insufficient.
Valid synthetic structures can test correctness, but random uniform routing can
change expert occupancy and gather locality, so report its timing scope honestly.

## Capture policy

Use `CAPTURE_PERSIST_POLICY=structure_only` when capture is necessary: retain
structural operands and reduce bulk float payloads to seeded descriptors. Integer
and boolean operands are structural by default; declare additional float structural
operands with `CAPTURE_STRUCTURAL_OPERANDS` / `meta.structural_operands`.

A captured output was computed from the original values, so it is not a reference
for regenerated random inputs. The reduced file is marked `oracle_status=needs_regen`:

```bash
python3 "$SKILL_ROOT/scripts/regen_reference.py" captures/reference_io.pt \
    --out captures/reference_io.regen.pt
python3 "$SKILL_ROOT/scripts/oracle_power.py" captures/reference_io.regen.pt --tol "$TOL"
```

This fallback persists reconstructed operands and baseline outputs. Prefer the
seeded factory above when it avoids capture entirely. Never regenerate structural
state through nondeterministic routing while pretending the inputs are unchanged.

## Correctness and timing

Use separate subprocesses for baseline and candidate. Import task helpers by
explicit file path and remove the task directory from `sys.path` before importing
torch: a file named `unittest.py` otherwise shadows Python's standard library.
The shared leg runner already performs this isolation.

0. **Before any verdict, prove the comparison can produce one.** Verify reproducible
   inputs, a tolerance supported by baseline self-noise, and effective numerical
   negative controls. For a seeded task, run these checks on generated cases and
   frozen-baseline outputs without collecting a serving oracle. The following
   helper commands apply when their input formats match: `oracle_power.py` for a
   saved capture and `kernel_self_noise.py` for a shape-by-M `_build_args` adapter.

   ```bash
   # (a) can the captured oracle reject anything?
   python3 "$SKILL_ROOT/scripts/oracle_power.py" captures/reference_io.pt --tol "$TOL"
   # (b) does the installed kernel reproduce itself, and are the inputs even stable?
   python3 "$SKILL_ROOT/scripts/kernel_self_noise.py" --repeats 5 [--packed] \
       --out "$TASK_DIR/run/self_noise.json"
   ```

   (a) A phase that runs entirely inside a CUDA graph fires python-level capture
   hooks only during graph **capture**, when the buffers are placeholders. On a
   real decode seam this recorded every float operand and the output as identically
   zero while the integer index operand looked real — so "is the file all zero"
   would have passed it, the control computed `0 * 0.5 == 0`, and the report said
   `max_rel_err: 0.0`, which reads like a perfect pass rather than an empty one.
   Capture during graph **replay** in steady state, or take a bounded eager
   diagnostic leg and revalidate the timing in deployment graph mode.

   If the capture turns out to carry no usable values, a reference generated by running
   the installed, unmodified kernel is a legitimate substitute — say so in
   `meta.oracle_coverage`, since gate 1 against it is then a determinism check rather
   than a deployment-value check. **Persist the exact operands it was computed from
   alongside the output.** Rebuilding them at UT time reintroduces (b) inside the
   correctness gate, where it is much harder to see than in a probe.

   (b) A tolerance below the installed kernel's own noise floor decides nothing.
   Establish the floor with ONE set of args reused across repeats, and only after
   phase 1 shows the args are reproducible: on this deployment
   `aiter.fused_moe.moe_sorting` returns different routing metadata from
   bit-identical inputs, so rebuilding args inside the repeat loop reports the
   routing change as kernel nondeterminism. For packed sub-byte payloads
   (MXFP4 values + exponent bytes) the only meaningful criterion is byte-exactness;
   `rel_frobenius` over packed bytes is not a number about the computation.

1. Run the frozen seeded/structural inputs through the baseline; verify output structure and full
   numerical/semantic contract. Reconstruct implicit state and all required glue.
2. Build candidate overlay; prove distinct baseline/candidate entries in every
   phase and actual execution of every claimed editable leaf.
3. Obtain independent baseline random outputs with
   `h.baseline_random_outputs(task, meta, draws=...)`.
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
   tolerance by a wide margin** — `oracle_power.py ... --tol T --control-rel-err
   <observed>`. A control that "passed" without its deviation ever being measured
   is not evidence of anything; see the zero-oracle case in step 0 below.
   Preserve the mutation and logs, restore pristine source, then rerun final
   normal checks before recording completion.
7. `h.measure_legs(task, meta)` measures interleaved baseline/candidate subprocess
   legs with the same mode and shared timer. Use
   `h.serving_weighted_speedup(per_case, meta)` for the observed-call aggregate.
   Print the aggregate as `GEAK_WEIGHTED_SPEEDUP=<value>` for downstream optimization;
   retain per-case timings and unweighted geomean as diagnostics.
   Every positive-mass signature must have finite timings on both sides.

The shared tensor checker assumes independent returned outputs. If the actual
serving contract permits persistent/in-place buffers, document that contract
and add an explicit state/output checker with independent snapshots. Do not
silently present an incompatible generic tensor gate as passing.

Write a single `python3 unittest.py` entry that orchestrates and combines these
checks, emits per-gate and per-case evidence and exits nonzero on failure or
incompleteness. A prototype with only replay or timing is not a completed UT.
The collective case requires the real rank harness described in
[collective UT](collective-ut.md); the single-process leg runner cannot test a
multi-rank collective by itself.

Freeze the source/test snapshot before GPU gates. The final `run/ut_result.json`
must include the snapshot, target, exit code, completed gates and coverage.
Generate leaf/source evidence as described in [source records](source-delivery-records.md).
Keep all raw logs; JSON declarations alone do not establish execution truth.
