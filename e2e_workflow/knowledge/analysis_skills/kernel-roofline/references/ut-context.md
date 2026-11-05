# Recover operation identity from a UT

When a standalone HIP/pybind UT produces device activity but TL lacks a usable
CPU operation, inspect the UT before declaring model coverage missing. The agent
does this source review as part of the analysis; the user need not author metadata
or approve routine mappings. Use `ut_context.py` to collect evidence and bind the
reviewed identity to the actual measurement.

**Recover evidence; do not construct plausible metadata.** Every supplied value
must come from this UT's selected case and executed source path. Missing fields
remain unknown. Never assume contiguous strides, equal input/output dtypes,
in-place/out-of-place behavior, or `quantization=false` to satisfy a TL parser.
The absence of a scale in an incomplete description does not prove no quantization.

## Establish the complete operation

Read `_make_args`, the timed `_launch`/`_timed` entry, the correctness reference,
and the selected implementation through its C++/HIP/Triton entry point. Check the
selected **case's branch**, output tensors, aliases, layout/strides, dtypes,
quantization/scales, masks, and other relevant flags. Keep required stages inside
the operator boundary and correctness-only casts/gathers outside it. Task labels
and kernel symbols are supporting evidence; they do not choose a model.

New UT-runner manifests include `input_metadata` (tensor shape, stride, dtype,
storage offset, alias groups and scalar configuration) plus the UT source hash.
This snapshot reads descriptors, not tensor contents, and occurs before timing.
Some UTs keep weights/scales in globals used by `_invoke`, outside `_make_args`'s
payload. A workload adapter can include those **same objects** in
`snapshot_inputs({'payload': payload, 'weight': ut.WEIGHT}, torch)` and save the
result as `metadata.input_metadata`; cite the actual global construction/use.
Do not allocate replacement tensors just to populate a descriptor. Validation
callbacks may add correctness/output evidence to their shared metadata dictionary.
Older measurements may use `parameter_basis: reviewed_source` when the exact
input-generating sources/case are available; say explicitly that their arguments
were reconstructed from source. If source changes or dynamic branches prevent a
reliable reconstruction, recapture. Do not claim a runtime snapshot was recorded.
For source reconstruction, cite the construction and the selected case/constants
used in the derivation. Shape alone does not establish stride: follow any view,
transpose, slice, contiguous or conversion in the actual UT. Input/output roles
come from the timed call, implementation and reference, not parameter-name guesses.

Resolve current upstream main and inspect its `torch_op_mapping.py` and the
selected model's parameter parser/semantics. Reuse a registered operation only
when its semantics and modeled argument layout match the measured boundary.
Do not add a local formula or change model FLOPs/bytes. Missing semantics or a
different operator boundary requires more evidence, a narrower measured scope,
or a native model extension.

For Chushi's `glm52_add_rmsnorm_quant_task`, the inspected UT has two branches:

| Selected branch | Source-backed identity | Relevant facts |
| --- | --- | --- |
| `add_residual=true` | `aiter::add_rmsnorm` | Read input/residual_in/weight; write out/residual_out; `FUSE_QUANT=false` |
| `add_residual=false` | `aiter::rmsnorm` | Read input/weight; write out; `FUSE_QUANT=false` |

Recheck these identities against the selected sources. An optimized candidate
may implement a different decomposition; its complete boundary still needs to
match the model. The task's `quant` suffix does not license a quantization model.

## Save reviewed context

```bash
python3 scripts/ut_context.py \
  --measurement results/case/measurement.json --ut /path/to/unittest.py \
  --output results/case/ut-context.draft.json
```

The helper reads source without importing/running the UT. Review its excerpts and
write a separate `ut-context.json`, retaining its binding and source hashes:

- Set `review_status` to `source_reviewed` after the agent checks the sources.
- Fill `semantic_name` and `semantics`: exact variant, outputs, alias facts and
  flags relevant to model selection. Keep the oracle/implementation evidence.
- Fill `native_operation` with the verified native registry name. Use `null` if
  only semantic identity is established; context remains useful without a model.
- Set `launch_frame` to the captured Python function implementing the complete
  timed call (normally the UT path and `_launch`). This is an observed frame, not
  an invented stack. The bridge requires exactly one such frame per call.
- Fill ordered `arguments` matching the registered TL parser. A tensor entry has
  `name`, `source` (its payload path, e.g. `input`), `kind: tensor`, `shape`,
  `stride`, TL-compatible `dtype`, and `access: read/write/read_write`.
  A scalar has `name`, `source`, `kind: scalar`, and `value`. For a literal present
  in the reviewed implementation, use e.g. `source: literal:false`, `value: false`.
  Preserve any model-relevant flags not parsed by TL in `semantics` as well.
- Preserve `expected_dispatches_per_call` and add hashes/citations for meta.json,
  the correctness reference and kernel bodies used in the review. The helper's
  `source_evidence(path, role)` can collect additional file evidence.
- `source_reviewed` and file hashes alone are insufficient. Populate `citations`
  with exact `path`, `first_line`, `last_line`, and `quote` records, produced by
  `source_citation(path, first_line, last_line)`. Paths must belong to the reviewed
  UT/case/reference/implementation chain; TL model code and this sidecar are not
  evidence of workload parameters.
- Populate `field_evidence` for `semantic_name`, `native_operation` (when present),
  every `semantics.<key>`, and each `arguments.<index>.binding` (name, position,
  payload/constant source) and tensor `.access`. Each entry has `citations: [ids]`
  and a `derivation` explaining how the selected UT branch establishes that field.
  Tensor `.shape`, `.dtype`, `.stride` and scalar `.value` additionally need this
  evidence whenever they cannot be checked against the actual runtime snapshot.
  `literal:false` is only a representation; it always needs source evidence for
  the real constant in the executed call. Put unresolved fields in `unknown_fields`
  and keep the draft unavailable; the model bridge rejects them.

For example, the residual UT's ordered tensor arguments are
`out, input, residual_in, residual_out, weight`, followed by epsilon and its
implementation flag. For `pre_m1137_res`, the four matrices are `[1137,6144]`,
stride `[6144,1]`, BF16; weight is `[6144]`, stride `[1]`. The actual scalar values
and separate outputs are established by the UT and implementation, not the name.

```bash
python3 scripts/tl_only.py \
  --measurement results/case/measurement.json --ut-context results/case/ut-context.json \
  --tl-source /path/to/current-main --tl-commit "$TL_COMMIT" \
  --arch /path/to/frozen_arch.json --output results/tl-only/case-with-ut-context
```

Use a fresh report directory. Context binds the task/case/arm, measurement hash,
trace hash and implementation hash. New runtime snapshots also check each
argument's shape/stride/dtype/value. Stale or contradictory context fails.
The validator checks runtime values, source hashes, exact quote ranges, and
per-field evidence completeness. The agent must still verify the semantic
derivation: an unrelated but real source quote does not justify a field's value.

## What the bridge changes

For an **eager** trace lacking CPU operators that own its measured GPU work, the bridge
adds one explicitly labeled semantic CPU event over each captured launch frame.
It preserves all original events, GPU durations, correlations and Python frames
in a separate `trace.ut-enriched.json`; original traces and reports are untouched.
Native TL then parses the supplied argument metadata, selects its own model and
generates `unified_perf_summary.csv` with `--include_call_stack`.

The bridge checks that all required device work belongs to the measured call and
is correlated with launches in that frame. It rejects existing CPU operations
that already own a measured GPU launch (which retain their native mapping), missing frames/correlations, partial
scope and changed dispatch multiplicity. The returned native report must contain
one complete operation, the measured call count, all subkernels and matched stacks.
One model may cover several required kernels; do not split its roof per kernel.
Host-only allocation/view operations such as `aten::empty`, `aten::transpose` and
`aten::as_strided` can coexist with the bridge when their recorded ranges own no
measured runtime-to-GPU launch. All original events remain in the analysis trace;
the decision uses actual correlation/range evidence, not an operator-name filter.

Results carry `metadata_origin: source_reviewed_ut_metadata`, the context and the
exact native mapping/parser source. Describe them as **TL native model/report on
UT-enriched metadata with captured device timings**, not raw profiler-generated
operator identity. `perf_params`, FLOPs, bytes and native percentages come from TL.
Input reconstruction does not prove the native traffic assumptions match HBM.

For **graph** execution, semantic context is retained but this bridge does not
inject replay metadata or copy an eager roof. Native capture/replay mapping is
still required. If there is no matched native model, use `native_operation: null`,
retain the context, and report native model coverage as unavailable.
