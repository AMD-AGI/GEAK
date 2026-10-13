# Mode: tl-only

Use TraceLens's native model and reporting path. The objective is to report what
the selected TL version says for the workload, retaining its assumptions and
coverage. Record `mode: tl-only` and `engine: TraceLens native` in the summary.

If TraceLens is absent or cannot be imported, first complete
[installation.md](installation.md), then return to this workflow.

## Keep this mode native

- Import/run a selected TraceLens installation and record its version, source path,
  commit, and local changes. Existing model classes/extensions shipped in that
  version are allowed. A checkout carrying custom model changes is not an
  unmodified TL result; identify the variant or use an unmodified installation.
- Use TL's FLOP/byte models, precision/peak selection, and roofline reporting code.
  An adapter may map real trace/input metadata to a supported TL schema; it must
  not invent a new physical model or alter the returned FLOPs/bytes.
- Do not apply the shared-KV, split-K, quantization, or MoE corrections from the
  corrections reference. Document suspected model limitations separately.
- Do not use `scripts/roofline_report.py`, handwritten substitute models,
  rocprof counter formulas, or rocprof-compute roofline calculations in this mode.
- Do not enable an optional external prediction engine such as Origami for a
  TL-only roofline. Keep TL's native roofline fields distinct from simulation fields.

Unsupported operators, absent shape metadata, missing peaks, and unavailable native
metrics remain explicitly unsupported/unavailable. Continue with supported cases;
do not silently fill the gaps with handwritten corrections or a second engine.

## 1. Prepare a trace or model input

Prefer an existing trace containing the target cases and matching device activity.
When collecting a new one, use the kernel's UT/framework profiler to capture CPU op
metadata, shapes/dtypes, and correlated device kernels after warmup. The native TL
report requires enough information to map an event to a registered performance
model. A bare device symbol may be insufficient, particularly for custom kernels.

Keep per-case identity, selected measurement windows, repetitions, cache state,
and correctness evidence. Run import-selected baseline and candidate bodies in
separate processes when necessary. Inspect the TL op-to-kernel mapping: an operator
roofline is not automatically a separate roofline for every device kernel it calls.

## 2. Choose the TL hardware configuration

Select a hardware configuration supported by the chosen version. Record the exact
file and SHA-256; keep it fixed across comparisons. Where TL requires
`mem_bw_gbps` and `max_achievable_tflops`, verify the needed fields are present.
Use the native peak-selection logic and report which `Compute Spec` it chose.
If the choice is unsuitable, flag the result instead of silently overriding it
with a correction.

If a configuration needs calibration, use that TL version's documented
microbenchmark entry point and preserve its output, or report that the required
reference is unavailable. Do not substitute another GPU's peak. Calibration is
separate from per-case timing and requires the matching GPU environment.

## 3. Run the native report

Inspect the installed entry point's `--help`. The following command shape was
checked against the local TraceLens checkout used for the September comparison;
flags and outputs are version-dependent:

```bash
python3 -m TraceLens.Reporting.generate_perf_report_pytorch --help

python3 -m TraceLens.Reporting.generate_perf_report_pytorch \
  --profile_json_path case_trace.json \
  --gpu_arch_json_path frozen_arch.json \
  --output_csvs_dir results/tl-only/native_csv
```

Use the input trace and architecture paths selected for the task. The installed
console entry point `TraceLens_generate_perf_report_pytorch` is another entry to
the same reporter in that checkout. If the selected version needs a different
reporter for its supported trace format, use its documented native entry point
and record the exact command.

Inspect emitted roofline tables and coverage. Do not assume a successful command
produced a model for every requested custom kernel. Preserve native CSVs/workbooks
and logs unchanged; a Markdown summary may extract rows and annotate limitations.

For **model-only** analysis, directly call a supported TL model's `flops()` /
`bytes()` or documented static functions with actual case metadata, recording the
exact API/arguments. Use native roofline APIs only where that version exposes
them for shape-only inputs. If native roof time is unavailable without a trace,
report the available model fields and leave it unavailable. Do not fabricate a
trace duration or substitute our calculator. A `get_simulation_time` result may
describe a different prediction model and must not be relabeled as native roofline.

## 4. Preserve native fields and explain their units

In the inspected `TreePerf/tree_perf.py` reporting implementation:

| Native field | Meaning / units |
| --- | --- |
| `GFLOPS` | Model work in billions of FLOPs, not a rate |
| `Data Moved (MB)` | Despite the label, bytes divided by 1024 squared (MiB) |
| `Kernel Time (µs)` | TL's linked GPU busy time; preserve its overlap treatment |
| `TFLOPS/s` | Model FLOPs divided by TL's measured busy time |
| `TB/s` | Model bytes/time in decimal TB/s |
| `FLOPS/Byte` | TL model arithmetic intensity |
| `Compute Spec` | TL-selected precision/resource key |
| `Roofline Time (µs)` | Native maximum of compute and memory reference times |
| `Roofline Bound` | Native `COMPUTE_BOUND` or `MEMORY_BOUND` label |
| `Pct Roofline` | Percent already: `50` means 50%, not a fraction of 50 |

Verify units in the installed version. If presenting decimal MB or GB/s, retain
native values and label the display conversion (`MiB * 1048576 / 1e6` to MB;
`TB/s * 1000` to GB/s). Such unit conversion is not model correction. Do not
replace native bound labels, tie behavior, aggregation, or reference values with
the standalone calculator's results.

The inspected native path requires positive modeled FLOPs to emit its roof fields.
A copy-like operation may therefore lack native roofline columns even if bytes
are available. Preserve that limitation instead of inventing a completed row.

## 4b. One kernel, many shapes: fold per [folding.md](folding.md)

A kernel exercised at twenty shapes still owes the report one number per regime, and
that collapse has exactly one correct arithmetic — time-domain, numerator and
denominator summed separately, divided once, weights normalized within a regime.
**`mean(roofline_pct)` over the shapes is wrong** and is the most common way a
`tl-only` table stops being comparable to anything, including a later run of itself.
[folding.md](folding.md) is engine-independent and binds this mode, and
`scripts/fold.py` implements it — call it rather than rewriting the sums, even though
this mode has no second leg to join:

```bash
python3 scripts/fold.py --rows results/tl-only/rows_candidate.json --label TL
```

Note what that rollup is *not*: it spans every shape, while a `combination` join
table re-folds TL over only the shapes rocprof profiled. The same kernel legitimately
reads 75.7% there and 61.6% here. Quote one or the other, never both as if they were
the same column.

## 5. Deliver and qualify the result

Report requested cases, supported cases, native data, correctness/measurement
provenance, and any unmapped kernels or questionable model assumptions. Keep the
raw TL report authoritative. Label any new derived field as a presentation-derived
value and state its formula; it must not change the underlying TL model or roof.

For a baseline/candidate comparison, align actual workload and operator scope and
show measured latency alongside native roofline fields. Do not describe TL-native
models as corrected semantic lower bounds without separately establishing that
claim. If a `rocprof` or `combination` leg was also requested, write it
independently under `results/rocprof/` or `results/combination/` and show the
disagreement without overwriting native results -- the two engines' percentages are
not comparable (see [combination.md](combination.md)).
