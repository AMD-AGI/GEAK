---
name: kernel-roofline
description: "Analyze GPU kernel or operator rooflines: tl-only uses current TraceLens main, native models/reports, and operation-to-kernel call stacks; rocprof uses rocprof-compute hardware counters. Use for modeled work, latency, arithmetic intensity, roofline comparisons, and model coverage. TL means TraceLens."
---

# Kernel Roofline

**TL means TraceLens.** Analyze kernels written in Triton, TileLang, HIP, or a
library using one of two explicitly separated engines. The engines are not
interchangeable and their percentages are not comparable — pick one on purpose.

When invoked by GEAK's Profiler through `ANALYSIS_SKILL_DIR`, also read
[geak-integration.md](references/geak-integration.md) for the advisory artifact
and Top-N handoff contract.

## Select the mode first

| Mode | Engine | Question it answers | Denominator | Instructions |
| --- | --- | --- | --- | --- |
| `tl-only` | Latest upstream TraceLens `main`, native models + reporter | Model a workload, analyze a trace, or compare measured implementations. Model-only and existing-trace analysis need no GPU; new device measurements do. | Selected TL arch configuration; record calibration/specification provenance | [tl-only.md](references/tl-only.md) |
| `rocprof` | `rocprof-compute profile --roof-only` + `analyze -b 4` | What is this **binary** actually issuing? A *diagnoser* — measured counters per leg. Needs a GPU and a runnable command. | empirical microbench peak or datasheet; derating is the caller's `target_eff` | [rocprof.md](references/rocprof.md) |

Honor an explicit mode. For an unqualified request to "do a roofline", default to
`tl-only` and say so. A request for hardware-counter-derived work or rates
selects `rocprof`; model FLOPs divided by measured TL time remains `tl-only`. These are the only two
supported modes. If both are requested, run them independently and preserve each
mode's case coverage, weights, and native results.
Within an explicit mode, report conflicting requirements or missing coverage; do
not silently switch engines.

These are skill invocation options, not arguments accepted by any CLI:

```text
Use $kernel-roofline with mode=tl-only to bound this shape.
Use $kernel-roofline with mode=rocprof to measure what the kernel issues.
```

### The two engines measure different things

TraceLens uses modeled work/bytes and linked device time; rocprof uses counter
measurements. Compare their scope, traffic assumptions, timing, and actual peak
references before explaining a difference. Peak ratios are configuration-specific,
not constants. Do not apply an unexplained discount to an empirical reference.
With identical TL reference times, cases, and weights across implementations,
the TL efficiency ratio equals the measured speedup; it is not independent evidence.

## There is no `extended` mode

The standalone calculator and [corrections.md](references/corrections.md) describe
separately requested custom-reference analyses. They must not replace or be labeled
as TL-native results. The calculator's `--mode extended` is a legacy input label,
not a skill mode. Within explicit `tl-only`, document limitations and use supported
native models; do not silently repair returned FLOPs/bytes or switch calculators.

## Select the operation within the mode

- **Model only:** report supported modeled work/traffic and reference metrics.
  Measured latency, achieved throughput, and measured efficiency stay unavailable.
- **Measure / compare:** collect or reuse a matching profile, validate correctness,
  report the selected engine's results for baseline/candidate.
- **Explain results:** inspect assumptions and scope. Counter evidence is
  `rocprof`-mode work.

An existing kernel UT or framework profiler may construct inputs, check correctness,
and capture device activity in any mode. In `tl-only`, TraceLens owns the modeling
and the roofline; ordinary trace capture does not introduce a second engine.
TraceLens does not universally benchmark arbitrary kernel source.

## Check or install the tooling

For each new TL analysis, resolve the latest `main` from
<https://github.com/AMD-AGI/TraceLens>, then freeze that full commit for the entire
comparison. Record the upstream URL, fetch time, commit, import location, and local
changes. An older installed package or vendored campaign copy is not evidence
about current support. Explicit historical reproduction may use its original pin
and must be labeled historical. Follow [installation.md](references/installation.md).

For measured TL analysis, generate reports with `--include_call_stack` and use
`unified_perf_summary.csv` as the native input to review. The row's `name` is the
**TL operation that selects the model**; device kernel names are in
`kernel_details_summary`. Preserve `perf_params` and every subkernel's call stack.
Read [unified-summary.md](references/unified-summary.md) before selecting a row,
interpreting a kernel, declaring it unsupported, or preparing an upstream issue.

When a standalone UT lacks CPU operator metadata, first recover its actual
operation from the input builder, timed entry, reference and selected kernel
source. Follow [ut-context.md](references/ut-context.md) to save reviewed identity,
arguments and evidence. The eager bridge can supply this metadata to a matching
native TL model without changing captured device timings or model formulas;
results explicitly distinguish UT-enriched metadata from captured identity.
Every supplied shape, dtype, stride, input/output role and quantization flag must
come from that UT's actual case/input construction and invoked source path.
Require per-field runtime evidence or exact source citations and derivation;
never fill gaps from model defaults, kernel names or plausible conventions.
Unknown or conflicting fields stay unknown and cannot enter a model mapping.

`rocprof` requires `rocprof-compute`, a GPU, and a verified frozen peak CSV.
On gfx95x, old microbenchmarks are unreliable; a compatible older profiler can
use independently verified peaks. See [rocprof.md](references/rocprof.md).
`scripts/roofline_report.py` and its synthetic example need only the standard
library. Follow [installation.md](references/installation.md); a missing dependency
is a setup problem, not a reason to silently change mode.

## Shared rules, both modes

For new measurements use [runner.md](references/runner.md): the bundled runner
owns cache conditioning, timed closure, graph/eager execution, markers and traces.
TL-only then runs `scripts/tl_only.py`; rocprof uses the same runner under
`scripts/run_roofline.py`. Do not reuse a campaign's handwritten TL roof calculator.

- **Every row carries its provenance.** Engine name, tool version/commit, arch JSON
  path + SHA-256 + its `source` field + the peak key actually resolved, device/partition,
  runtime/container digest, workload shape, layout/dtypes, operator boundary, cache
  policy, and the implementation actually dispatched. A row missing any of these is
  not quotable. The arch JSON is **not a constant** — see
  [assets/README.md](assets/README.md) for the four MI355X variants on this
  cluster and their 18.6% bf16 dispersion.
- **A roofline bound label is a model classification, not proof of a measured
  bottleneck.** Low efficiency alone does not establish bandwidth, occupancy, or
  launch limitation. Preserve efficiencies above 1 and investigate model, scope,
  cache, timing, and reference assumptions; empirical calibration is not a strict
  upper bound. Never clamp, auto-select a replacement model, or claim guaranteed
  headroom from a suspect reference.
- **One kernel, many shapes: fold per [folding.md](references/folding.md).** The
  collapse from per-shape rooflines to one number per kernel/regime uses
  time-domain arithmetic — numerator and denominator summed separately, divided
  once, with declared case weights within a regime. Preserve native report
  aggregation; label any additional workload rollup. Do not split an operation's
  TL roof across its subkernels. The additional rollup is implemented
  in `scripts/fold.py` — call it (`python3 scripts/fold.py --rows <rows.json>`)
  rather than rewriting the sums. Baseline/candidate comparisons within a mode
  require the same cases and weights. Preserve each mode's declared coverage when
  both modes are requested independently.
  Include the full requested-case manifest; report case coverage and kernel
  coverage separately. `ab_speedup` is measured latency improvement;
  `distance_to_reference` is not promised attainable speedup.
- **Results split by mode:** `results/tl-only/` and `results/rocprof/`.
  Each summary states its mode, engine, assumptions,
  limitations, and **whether a GPU was actually measured**.
- Match modeled work to measured scope. One operator can launch several kernels;
  never divide a whole operator's FLOPs/bytes by one stage's time. Preserve the
  engine's own timing semantics and document overlap/aggregation.
- Exclude JIT warmup and identified harness work without deleting required operator
  work. Keep correctness status, raw samples, exclusions, and coverage visible.
- Keep cases and phases separate. Benchmark repetitions are not deployment weights,
  a missing metric is not a zero, and a shape-only result is not a GPU measurement.
- Preserve source units and native values; label any presentation-only conversion.

## Worked examples of both modes

`examples/krf_0927/` contains baseline/candidate results for the two modes:
`tl-only/COMPARE.md` and its per-case JSON inputs, plus the independent
`rocprof/leg/` per-kernel reports. The modes cover different case sets;
retain the coverage and provenance of each report. The campaign-specific
`tl_leg_*.py` / `tl_compare.py` drivers are not part of this package.
These are historical campaign artifacts, not evidence that current main supports
or rejects a model, nor a substitute for a native report with operation identity.

Timing and cache-policy guidance is shared: read the first two sections of
[measurement.md](references/measurement.md) in any mode. Its counter section
belongs to `rocprof`. [pitfalls.md](references/pitfalls.md) is a one-page index of
every trap in this package, each anchored to where it is explained.
