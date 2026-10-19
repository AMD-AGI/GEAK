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
  captures/               phase-local input/oracle records
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
  Preserve captured routing where required by the math and dispatch contract.

For joint phases, include `meta.phase_target_callables` mapping each phase to
its live entry, and `meta.phase_regimes` mapping phase labels to observed
graph/compile settings. `cases.call` selects only those entries using captured
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

## Capture policy

A capture buys three separable things, and only the middle one has to be the deployment's
own bytes:

1. **Shape and call mass** — comes from the trace and the serving log, not from the oracle
   file at all. `CAPTURE_PERSIST_POLICY` never affects it.
2. **Structural operands** — the ones whose CONTENT selects work: `topk_ids`,
   `sorted_expert_ids`, `m_indices`, `cumsum`, gather `indices`, page/block tables,
   sequence lengths. Redrawing these changes *which* work runs. They must be real.
3. **Bulk floating-point payload** — activations and weights. Redrawing these changes the
   numbers inside the tiles, not the tiles that run.

`CAPTURE_PERSIST_POLICY=structure_only` keeps (2) verbatim and reduces (3) to a seeded
descriptor — on a real GLM-5.2 run, 138 GB of raw captures against ~30 MB of task captures.
Integer and boolean tensors are structural by default; float operands are structural when
named in `CAPTURE_STRUCTURAL_OPERANDS` / `meta.structural_operands` or when their name
matches a routing pattern (`topk_weights` is float but scales per expert).

The reduced operands are **persisted**, not omitted: the recorded seed reproduces the same
tensor in every process, forever. That is the property the oracle needs and the property a
rebuild-at-UT-time factory does not have — rebuilding through the serving stack re-runs
whatever nondeterminism the stack contains, inside the correctness gate.

Because the output recorded at capture time came from the real operands, a `structure_only`
file ships as `oracle_status="needs_regen"` and both `harness_lib.reconstruct_captured` and
`oracle_power.py` refuse it. Close the loop inside the serving container:

```bash
python3 "$SKILL_ROOT/scripts/regen_reference.py" captures/reference_io.pt \
    --out captures/reference_io.regen.pt
python3 "$SKILL_ROOT/scripts/oracle_power.py" captures/reference_io.regen.pt --tol "$TOL"
```

What it costs, and what to write in `meta.oracle_coverage`: gate 1 becomes a determinism
check against the installed kernel rather than a deployment-value check, and the operands
carry unit variance rather than the served activation's outlier distribution — so a kernel
with value-dependent cost (denormals, early exit, saturation) needs a few captured cases
kept as the control. It is still strictly more discriminating than what it replaces on a
graphed seam, where the capture records placeholder buffers and the gate cannot fail at all.

Per seam, as measured on this deployment: the MoE gemms keep `topk_ids` /
`sorted_expert_ids` / `m_indices` / `cumsum` and must **not** regenerate routing from random
logits — uniform routing is the slow end of the distribution (2.46x on decode), so a redraw
moves the measurement, not just the values. Sparse-MLA decode/prefill keep `indices` and
need a real-sized, densely populated pool. An indexer top-k keeps the score distribution,
because ties and the reachable range are what its quality gate reads. A score-only prefill
seam and a collective have no structural operands at all and can be fully random.

## Correctness and timing

Use separate subprocesses for baseline and candidate. Import task helpers by
explicit file path and remove the task directory from `sys.path` before importing
torch: a file named `unittest.py` otherwise shadows Python's standard library.
The shared leg runner already performs this isolation.

0. **Before any verdict, prove the comparison can produce one.** Two preconditions,
   both executable, both of which a real run of this skill violated silently:

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

1. Replay frozen inputs through the baseline; verify output structure and full
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
