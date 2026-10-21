# Serving metadata and reproducible kernel tests

The Kernel Extractor produces an immutable test harness for real serving shapes.
Correctness references come from the frozen baseline on deterministic inputs; input
and output tensors do not need to be saved. Keep the shape manifest, constructor,
seeds, baseline identity and validation evidence immutable during optimization.

## Task directory

```text
<task_dir>/
  kernel_src/         # editable candidate source
  baseline_overlay/   # frozen live serving stack
  cases.py            # fixed shapes and deterministic input constructors
  harness_lib.py      # vendored shared helper
  leg_runner.py       # matching isolated baseline/candidate runner
  unittest.py         # immutable correctness and timing driver
  meta.json           # source identity, regime, complete shapes/layouts/weights
  reference_io.pt     # OPTIONAL minimal structure or explicit legacy capture
```

## Shape and launch evidence

Reuse existing profiles and metadata. When a serving probe is necessary,
`scripts/capture_shapes.py` defaults to `CAPTURE_PERSIST_POLICY=metadata_only`.
It writes JSON with all observed shape/layout/scalar signatures, call counts,
phase hints and graph evidence; it never snapshots tensor contents. `max_cases`
limits optional tensor captures, not the full histogram. Validate launch attribution
and the complete histogram; `num_tensor_records=0` and `reference_io=null` are normal here.

Run the same workload/regime and preserve actual quantization, stride, scalar
parameters and phase boundaries. Do not infer full coverage from a few captured
cases or treat Python graph-capture calls as replay counts. Use trace-derived
weights and the shared coverage ledger for the full observed distribution.

## Inputs and correctness

Generate numerical payloads with fixed seeds. Construct packed quantized weights
with the actual packing and scale contract. Reconstruct routing, masks, indices,
page tables and sequence metadata with valid semantics and the observed access
pattern; synthetic structural probes must be labelled separately from serving cases.

Only when structural state cannot be reconstructed, explicitly use `structure_only`.
It retains necessary structure under 64 MiB per-process / 8 MiB per-case defaults,
checked before copying. It does not save output tensors. Integer packed weights
are numerical payloads, not structural metadata. Legacy full I/O capture is opt-in.

Use `h.baseline_random_outputs(..., bucket=sig)` to obtain independent references
in bounded anonymous memory, compare that batch, then release it. The default
reference payload budget is 256 MiB; oversize batches fail rather than skip cases.
Never write the returned mapping to `_baseline_random.pt` or another tensor file.
Use all manifest shapes and multiple seeds, full-output checks, negative controls,
and the required cross-shape independence and graph replay gates.

For details, read [task authoring](kernel_extraction_skills/kernel-extract/references/task-authoring.md).
Existing frozen task copies need a matched helper update before re-freezing or an
external ephemeral transport launcher; changing the shared source does not update them.

## Timing and delivery

Measure both legs against the frozen live stack with the shared harness, respecting
the deployed graph/compile mode. For observed workloads, aggregate time weighted by
observed call counts. Retain source paths/hashes, complete coverage and small result
reports. Deliver constructors and metadata, not generated inputs, golden outputs or
runtime tensor buffers. The e2e integrator still validates serving performance and
output parity before accepting a kernel change.
