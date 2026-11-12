# Author a fixed-contract task

Create tasks through `ut_contracts/scaffold_ut.py` and the shared
[UT contracts](../ut_contracts/README.md). Fill the family hooks and case metadata;
`ut_driver.py` owns execution and verdicts. An unsupported operator needs a custom
family in the same schema, with an explicit reason.

```text
tasks/<name>/
  cases.py, meta.json, unittest.py
  ut_driver.py, ut_contract.py, reference_f64.py
  harness_lib.py, leg_runner.py, overlay_setup.py, compile_watch.py
  kernel_src/             editable computation and build resources
  baseline_overlay/      immutable accepted baseline configuration
  baseline_ref/          original source/hash evidence; not an imported baseline
  run/                   case/contract manifests, source evidence, gate reports
```

The scaffold copies and freezes the helpers. Existing tasks use their own reviewed
snapshots; refreshing the skill does not rewrite those tasks. During optimization,
only the declared candidate computation may change.

## Metadata and hooks

`meta.target_callable` is the inspected `module:attribute` entry. Module replacement
uses `candidate_bind={kind:"module", module, file}`; rebind uses
`{kind:"rebind", target, impl_module, impl_attr, file}`. Files live under `kernel_src/`.
Native/JIT leaves must build and invoke the delivered computation, with its resources
and architecture flags. A copied dispatcher alone is not extracted computation.

The baseline resolves to the installed serving stack plus accepted overlays outside
the task. The candidate overlay adds only the declared task source. The driver proves
distinct binding and actual leaf execution in separate processes.

The generated `cases.py` supplies:

- `call(args)`: invoke the real entry and expose all contracted outputs, including
  auxiliary tensors and in-place results.
- Family hooks and case axes: construct valid layouts, quantization, routing and state.
- `replay_bundle()`: when required, refresh inputs and replay observed boundary cases
  through the same static buffers.
- `random_shapes()` and `timing_cases()`: template adapters for the leg runner. Keep
  all served signatures, phase/config IDs and observed weights.

Joint tasks retain `phase_target_callables` and `phase_regimes`; dispatch comes from
frozen phase metadata. A multi-module task needs a tested binding adapter and evidence
for every leaf. Review the complete operator boundary before generating cases.

## Seeded inputs by default

Shapes, dtypes, layouts, quantization, phase and weights come from the profile handoff.
Generate ordinary activations and weights from the supplied seeded generator. Preserve
valid packed encodings, positive scales and semantic boundary cases such as ties,
empty rows and padding. Baseline and candidate reconstruct the same inputs.

Routing, sparse indices, page tables and cache objects must retain the observed work
and access pattern. Reuse structural descriptors or validated reconstruction; report
missing evidence to profile. Uniform random routing is not a substitute for measured
expert occupancy. Keep any necessary structural fixtures small and source-backed.

Use the template's private bounded weight cache and the driver's
[bounded tensor lifecycle](../ut_contracts/README.md#temporary-tensor-lifecycle).
Baseline/candidate workers generate temporary output batches, compare them and release
them. Do not introduce a persistent `reference_io.pt`, suite-wide golden outputs or a
second correctness runner. The profile records call descriptors through `compile_watch`;
UT authoring does not launch serving to capture ordinary numerical payloads.

## Qualification

Run CPU lint before borrowing GPUs:

```bash
python3 "$TASK_DIR/ut_contract.py" "$TASK_DIR"
```

Inside a UT wave, iterate with `ut_wave.py gpu -- python3 "$TASK_DIR/unittest.py" --quick`,
then qualify with:

```bash
python3 "$SKILL_ROOT/scripts/ut_wave.py" gpu -- \
  python3 "$TASK_DIR/unittest.py" --validation-mode correctness --reuse
```

Outside a wave use the leases in [setup](setup.md). `--quick` is a debugging subset;
only full qualification can pass. The fixed driver requires:

- Contract/input validity, deterministic construction and all case/draw comparisons.
  `noise_relative` bounds measure repeated baseline outputs; bitwise contracts check
  repeatability. Packed formats use the family's numerical or exact-byte contract.
- Source binding, expected launches, recorded calls and compile variants, including
  startup and graph-capture variants. Both legs use the recorded serving preimports.
- Output isolation and required graph replay with changed values and boundary shapes.
- One executed numerical negative control per editable leaf. A build/import failure
  cannot qualify a control. Witness search may stop a control only after reaching its
  required margin; ordinary correctness still covers every case.
- Restored pristine source and fresh receipts tied to the frozen task snapshot.

Read failures and fix task code or missing evidence; do not loosen bounds, edit
receipts or omit cases. Record justified family overrides through the contract tooling.
Persistent/in-place outputs require the family's explicit state/output semantics.
Collectives need the real [rank harness](collective-ut.md), not a single-rank surrogate.

## Optimization interface and delivery

UT generation does not measure timing. The optimizer later runs
`unittest.py --validation-mode performance --reuse`; the bare entry also retains
performance mode. The same driver calls `measure_legs` and `serving_weighted_speedup`,
with isolated baseline/candidate legs and the recorded cache/graph policy. Every
positive-mass signature needs finite timings on both sides; per-config weighting and
`GEAK_WEIGHTED_SPEEDUP` remain part of this interface.

Retain the raw gate logs and `run/ut_result.json` / `run/ut_report.json`, including
`performance_measured:false` for correctness qualification. Generate the actual
[source delivery records](source-delivery-records.md) and `run/RETURN.json`; declarations
alone do not prove leaf execution. Missing cases, bindings or evidence remain explicit
failures/blockers rather than being renormalized away.
