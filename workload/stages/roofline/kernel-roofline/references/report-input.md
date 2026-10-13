# Report input and portable example

This helper and input schema belong to the corrections workflow
([corrections.md](corrections.md)), not to any one mode. `--mode extended` below is
an **input-format label the calculator consumes**, not a mode of this skill -- the
`extended` mode no longer exists. For `tl-only`, run the native reporter described
in [tl-only.md](tl-only.md) instead of this helper.

Run from the skill directory with Python 3.10 or newer:

```bash
python3 scripts/roofline_report.py \
  --mode extended \
  --input assets/example_cases.json \
  --arch assets/example_arch.json \
  --output /tmp/tracelens-roofline-example
```

This is a **synthetic CPU-only arithmetic example**, not a GPU benchmark. It writes
`roofline.json`, `roofline.csv`, and `roofline.md`. Choose a fresh output directory
for each real run; those three filenames are replaced if they already exist.

The example keeps useful GEMM work and ideal bytes fixed. Latency drops from
100 to 80 us, so reference efficiency rises from 42.9497% to 53.6871% (1.25x
latency speedup). Counter-derived efficiency falls from 85.8993% to 53.6871%
because redundant executed work disappears. This demonstrates why the counter
ratio and the optimization score must remain separate.

## Input contract

Use [example_cases.json](../assets/example_cases.json) as the schema example.
Units are FLOPs/call, bytes/call, and microseconds/call; input FLOPs are not GFLOPs.
The architecture file uses TFLOP/s and decimal GB/s, as in TraceLens arch JSONs.

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1` |
| `mode` | Literal `extended` (the calculator's own format label); omitted in older inputs defaults to `extended`. `tl-only` is rejected by this helper. |
| `provenance` | Nonempty object: TL commit/import location, local model changes, device/runtime, artifact paths, and collection commands as applicable |
| `cases[].case_id` | Unique operating-point identifier; include phase when relevant |
| `contract` | Nonempty strings: `id`, `operator`, `boundary`, `precision`, `memory_level`, `cache_policy`, `model_version` |
| `workload` | Nonempty object with shape/layout/dtype and relevant routing/selection metadata |
| `model.kind` | `semantic_reference`, `legacy_tl`, or `implementation_model` |
| `model.source` | TL class/function and pinned version, or an explicitly handwritten model |
| `model.assumptions` | Nonempty list; state reuse and omissions, ideally with named byte/FLOP terms |
| `model.flops`, `model.bytes` | Finite nonnegative per-call values; at least one must be positive |
| `model.peak_key` | Exact key in the selected arch's `max_achievable_tflops`; no fallback |
| `measurements` | One record per implementation, all sharing this case's fixed model; empty/omitted for model-only |
| `implementation` | Unique name within the case, e.g. baseline or candidate |
| `latency_us` | Finite positive duration for the agreed scope |
| `timing_basis` | Device activity sum, operator elapsed time, graph replay interval, etc.; only compare equivalent bases |
| `source` | Raw timing artifact / source/build provenance; preserve raw samples and statistic as extra fields |
| `correctness` | `pass`, `fail`, or `not_checked`; report supplied validation status honestly |
| `counters` | Optional object; requires its own `latency_us`, `source`, `coverage`, and `timing_basis` |
| `counters.flops`, `counters.bytes` | Optional nonnegative per-call counts; omit or use null for unavailable coverage |

Extra fields such as `byte_terms`, `raw_samples_us`, `pass_timings`, `excluded_events`,
`implementation_sha256`, and `deployment_calls` are preserved in the input snapshot.
The calculator does not verify that those terms sum to the supplied model totals;
the modeling adapter must do that. Record architecture calibration/source metadata
inside the arch file. The output embeds that file and hashes its exact bytes.

Counter diagnostics use the model's selected hardware references for comparability.
Only use that derived counter roof when the counters' compute types are compatible
with this peak. For mixed or incompatible resources, omit that aggregate and build
an explicit analysis with resource/stage models outside this helper.

## Output behavior and limits

- Missing measured latency in model-only mode yields null rates and efficiency.
- Missing FLOP or byte counters still permit the corresponding available rate;
  the combined counter roof remains unavailable.
- Zero FLOPs can describe a copy-like workload; zero bytes leave AI undefined.
  Both zero means an unavailable model, which is rejected.
- Reference efficiency above 1 is preserved and flagged, never clamped.
- Failed/unchecked correctness and non-semantic models are visibly flagged.
- Exact equality of compute/memory time is labeled `balanced`.
- Malformed units cannot be inferred automatically. Invalid numerical values,
  missing required fields, and unknown peak keys are rejected with exit code 2.
- The script does not prove the FLOP/byte model, benchmark cleanliness, source
  binding, hardware peak validity, or matching counter coverage. No optimization
  ranking, stage schedule, plot, or deployment weighting is inferred.

For a real TL test, first compute the case model using the selected TL API plus
documented corrections, then export measured data from the workload adapter in
this format. The report helper performs deterministic arithmetic on those inputs.
