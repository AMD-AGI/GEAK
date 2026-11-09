# Shared hardware specifications

`sku.json` supplies product specifications to GEAK's local planning tools and
the legacy e2e roofline resolver. The legacy `peaks.md` is a generated view.
Native TraceLens architecture models and calibration CSVs remain independent. Local calculations remain `analysis_role: planning`
and `may_gate: false`, including when supplied measured bandwidth or manual peaks.

## Sources and product identity

- `sources` records official URLs, checked dates and the fields each source supports.
  `source` and `basis_note` retain explanations for historical or derived values;
  rows without a newly checked source are not newly verified by this update.
- `arch` describes the ISA. `identity_target`, `geak_support` and
  `roofline_default_for_arch` describe separate GEAK policies. R9700, R9600D and
  RX 9070 / XT share gfx1201 but differ in product specifications. RX 9070 / XT
  remain reference-only; gfx1201 has no default product peak.
- `source_conflicts` records unresolved official disagreements. The affected
  canonical value is `null`, which consumers must treat as unknown. R9600D's
  product page lists 64 MB Infinity Cache; ROCm's GPU table lists 48 MiB.
  A bandwidth probe cannot certify DRAM residency from an unknown cache capacity.
- Peaks are dense, with sparsity excluded. Vector rates are recorded separately.
  The client BF16 rate follows the architectural FP16 WMMA equivalence; it is not
  separately quoted on those product pages. Their FP32 rate is vector throughput.
  Rates and bandwidth use decimal TFLOP/s, TOPS and TB/s; cache capacities use MiB.

## Updating data and using the calculators

Edit the JSON, then run:

```bash
python3 kernel_workflow/scripts/kernel_tools/extract_sku.py --sync-docs
python3 kernel_workflow/scripts/kernel_tools/extract_sku.py --check --check-docs
```

Generated Markdown tables are views of the JSON. Ridge points are computed from
the requested precision and product; a missing precision has no substituted rate.
The recorded native MXFP rates are available only on gfx950 products.

Default bandwidth selection is shared by the calculators, hardware context,
bandwidth probe and generated tables through `kernel_tools/_hwdata.memory_peak`.
It selects the datasheet pin rate when recorded; derived specifications retain
their original basis. Historical measurements are not default specifications.
For example, `AI_MAX_395` defaults to 0.256 TB/s; its older 0.212 TB/s report stays
in the catalog as historical evidence. Pass a measured ceiling explicitly for
measurement-based analysis (`hw_budget.py --measured-hbm-tb-s` with `--measured-from`
for its source, or `calc_perf.py --peak-hbm-bw` as a labelled manual override).
Local measured inputs remain planning-only and do not replace TraceLens calibration.

The simple GEMM byte model assumes one read of each operand and one output write.
FP4/FP6 default to MX storage with one scale byte per 32 elements; explicit MXFP8
also counts scales. Low-precision floating-point inputs default to a 16-bit output,
while FP32/TF32 and FP64 use 4-byte and 8-byte outputs. This is an optimistic
storage model; use `hw_budget.py --tensors` to describe actual padded storage or
other output widths. It does not measure HBM traffic or attainable throughput.

## Runtime consumers

Setup/Freeze capture [the shared hardware context](../../../scripts/hardware_context.md),
which kernel, bakeoff and e2e/resume pass to their roles. Local tools accept
`--hardware-context FILE` or `GEAK_HARDWARE_CONTEXT`; inconsistent SKU/ISA inputs
are refused. The legacy Gluon occupancy entry forwards to the shared implementation.
No pack-local hardware JSON is maintained. Adding a catalog row does not enroll
a product in GEAK's serving policy.
