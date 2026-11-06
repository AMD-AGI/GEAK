# Hardware context across GEAK workflows

The shared CLI `scripts/hardware_context.py` reads product specifications from
`perf_knowledge/hardware/data/sku.json`. It separates observations from product
specifications and keeps the legacy `device_target` policy field unchanged.
The identity allowlist now comes from the registry instead of a second CU table.

Entry points: [shared CLI](hardware_context.py), [workflow contract](hardware_context_workflow.js),
[SKU registry](../perf_knowledge/hardware/data/sku.json).

Setup and Freeze run the probe in the workload environment, after the run directory
exists, and return its complete JSON as `hardware_context`, its `sku` as
`device_sku`, and the absolute filename as `hardware_context_path`:

```bash
python3 "$GEAK_ROOT/scripts/hardware_context.py" --output "$EVAL_DIR/hardware_context.json"
export GEAK_HARDWARE_CONTEXT="$EVAL_DIR/hardware_context.json"
python3 "$GEAK_ROOT/scripts/hardware_context.py" --check "$GEAK_HARDWARE_CONTEXT"
```

The workflow validates the returned identity against Setup/Freeze and any supplied
expectations, forwards the object to every role and child lane, and retains it in
e2e resume state. This applies with expert skills disabled and to HIP, Triton,
Gluon, backend tuning and serving configuration. The JS runtime cannot access
files, so the existing roles perform the CLI/file work; Python consumers validate
the catalog hash and content again before using numeric specifications.

`observed.physical_cu_count` is the resource count visible to one rocminfo agent.
`observed.memory_bytes`, when present, is that agent's GLOBAL pool capacity, not
free memory. GPU UUID/BDF/chip/node observations are retained in `observed.agents`;
they are not guessed PCI-to-product mappings. Grid/split-K planning uses visible
CUs. Serving must still query free memory before selecting batch/KV allocation.
Unknown memory, partition, cache or clock facts must not be filled from another
SKU. A smaller CU count alone does not identify MI308X. Mixed visible identities
are refused; use an appropriately isolated workload environment.
The probe reflects rocminfo/ROCR visibility; a HIP-only device mask does not
necessarily filter rocminfo. Partition mode is retained when explicitly supplied
by the execution environment. A declared non-SPX/non-NPS1 scope cannot select
whole-device peaks. The CLI does not guess partition mode from an ISA or memory size.

Matched product name + ISA + expected CU count selects that product's reference.
Conflicts, unknown names and reference-only products cannot supply legacy e2e
numeric peaks. MI308X uses its own historical/derived row with low confidence;
matching its name does not certify those rates. R9600D cache stays unknown.

The old `resolve_peaks(path, gfx, ...)` call signature remains available. New
callers pass `hardware_context=FILE_OR_OBJECT` (or export `GEAK_HARDWARE_CONTEXT`),
or pass `identity=GPU_IDENTITY_JSON`. Arch-only resolution returns `None`.
Explicit legacy product-only lookups are low-confidence reference data. All local
specification estimates remain planning-only and cannot establish acceptance.

`kernel_tools/calc_perf.py`, `hw_budget.py`, `mem_bw_probe.py` and `amd_occupancy.py` accept `--hardware-context FILE` or the same environment
variable. Product-based tools select the matched SKU and reject a conflicting `--sku`;
`calc_perf.py` also rejects conflicting `--arch`/`--cus`. `amd_occupancy.py` selects
the observed architecture and rejects conflicting CLI or assembly targets.
Unresolved contexts cannot supply product peaks; architecture-only resource
analysis remains available. `mem_bw_probe.py` checks live device scope before measurement.
Outside a run context, explicit `--arch` keeps the existing labelled reference
lookup for compatibility; it is not a detected product.

## Native TraceLens and frozen references

Native callers can validate scope with `hardware_context.py --check FILE --native-arch ARCH_JSON`
and verify frozen reference bindings with `--reference REFERENCE`. These checks preserve native
model/precision/peak values. GPU callers must probe in the actual execution environment under
GEAK's existing lease; a context file alone does not prove which GPU a process uses.

Reference producers bind a newly collected file to the observed hardware context using the
CLI below. The sidecar format is:

```json
{
  "hardware_context": {"schema_version": 1, "...": "complete captured context"},
  "reference_sha256": "SHA-256 of the exact reference file"
}
```

The binding filename is the reference filename plus `.hardware-context.json`.
It records the calibration producer's device scope; do not manufacture it from
the machine consuming an unrelated CSV. Missing/mismatched metadata is actionable
unavailability, never an automatic replacement with SKU peaks. Native numerical
calibration audits still run. Existing direct native calls without a context
retain their native validation behavior; they do not acquire a GEAK scope check.

Immediately after successful native calibration, on the producer in the same
workload environment and under its GEAK lease, create the binding with:

```bash
python3 "$GEAK_ROOT/scripts/hardware_context.py" --check "$GEAK_HARDWARE_CONTEXT" \
  --live --bind-reference "$ROOFLINE_MIBENCH_CACHE"
```

This records the producer's declared provenance and exact file digest; it does
not certify numerical accuracy or recover the origin of old files. Existing
bindings cannot be relabelled. Consumers must explicitly check a binding before using the associated reference. A catalog change deliberately invalidates a context; review and
recapture references rather than silently relabelling an archived calibration.

On the separate upstream `tl-roofline` branch, the old e2e `roofline/` engine and
its tests are removed and the generated product table moves to
`perf_knowledge/hardware/product_peaks.md`. Keep that retirement when integrating;
the shared context and local calculators do not depend on the old engine. Until that branch lands, the legacy resolver uses the same shared context.

## Maintaining the contract

Edit sourced specifications in the public JSON, then regenerate its Markdown
views. Update identity fixtures when adding a product. Changes to the pure JS
contract go in `scripts/hardware_context_workflow.js`, then run
`python3 scripts/sync_hardware_context_workflow.py`; the embedded copies support
both Claude Code's native sandbox and GEAK's interface runtime without imports.
Run `python3 -m pytest -q scripts/tests/test_hardware_context.py` and
`node kernel_workflow/scripts/test_hardware_context_routing.js` for regression
coverage. A changed catalog requires regenerating run contexts; archived contexts
remain provenance and must not silently select new numerical references.
