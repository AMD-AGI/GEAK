# Folding many shapes into one number

**This file is engine-independent and binds both modes.** One kernel is
exercised at many shapes; a report wants one number per kernel per regime. How that
collapse is done decides whether two tables can be compared at all — and it is where
hand-built tables go wrong most often, because the wrong way (average the
percentages) looks obviously right.

The rules below are implemented in `scripts/fold.py`. Call the module or its
CLI for a standalone mode's rollup rather than rewriting the sums:

```bash
python3 scripts/fold.py --rows results/tl-only/rows_candidate.json --label TL
```

Its three primitives, in the order a leg uses them: `fold_kernels` (within one case:
per-kernel efficiencies -> that case's roof), `fold_cases` (across shapes: per-case
roofs -> one pct per kernel/regime), `fold_ratio` (across shapes: any ratio -- AI,
bound share). Nothing in that file parses a profiler, knows a peak, or cares which
engine produced `eff`; each mode uses it with its own inputs and provenance.

## The rule: sum numerator and denominator separately, divide once

```
roof_us(case)      = Σ_k  t_k · Eff_k                      # within a case
pct(kernel,regime) = Σ_cases w·roof_us / Σ_cases w·measured_us
distance_to_reference = 1 / pct  # not promised attainable speedup
```

**Never `mean(pct_case)`.** A percentage is a ratio, and averaging ratios gives a
shape carrying 0.3% of the calls the same vote as the one holding 90% of the clock.
The same applies to every other ratio in the report:

- **AI folds as `Σ(w·FLOPs) / Σ(w·bytes)`**, never as a mean of per-shape AIs.
- **Bound type is a share, not a label.** Report "fraction of weighted device time
  sitting on the compute axis". A seam is not one kernel, so "compute-bound" for a
  seam is a mixture, and a single label throws away the mixture.

## Per-kernel aggregation requires per-kernel models

In TL-only, a `unified_perf_summary.csv` row is a native operation, potentially
with several device kernels. Preserve its native model, roof, and timing. Do not
assign that whole roof to each child or manufacture child models from time shares.
The per-kernel procedure below applies only when independent per-kernel metrics
actually exist. Preserve native CSV statistics; any additional case-weighted
rollup is a labeled derived metric, not a replacement for native aggregation.

`attainable = min(compute_peak, AI · BW)` is a per-kernel quantity because the two
candidates sit on different axes. Summing first and taking the min afterwards gives a
seam whose kernels straddle the ridge two different peaks over one shared
denominator. Folding *time* is what makes the fold axis-safe. (Skipping this already
produced one unquotable number, on `mxfp4_fused_moe/prefill`.)

## Weights

- `w` is each case's own `share_of_op_calls` — how often that shape is actually
  dispatched in deployment. Benchmark repetition counts are not deployment weights.
- **Weights normalize WITHIN a regime.** Decode and prefill are never blended: they
  are two numbers, and a byte model built for one is fiction in the other.
- `t_k` belongs to the selected engine and declared measurement. TL uses native
  linked device time. The counter fold uses its selected counter-run dispatch
  times and labels them serialized; it does not borrow faster TL timings.

## Dispatch probes do not replace the native roof

There is no portable 2.3-us dispatch lower bound. Empty-kernel timing varies with
device, NUMA, cache conditioning, and eager/graph execution, and is an empirical
diagnostic rather than proof of a strict limit. Keep such probes separate from
the native TL roof. Do not score with `max(roof, n_dispatch * probe_time)`, fill an
unsupported row with a probe, or promise `1/pct` as attainable speedup. That ratio
is distance to the selected reference, subject to model and measurement limits.

## Coverage is reported, not imputed — and an unscorable kernel is dropped, not zeroed

A case whose dominant kernel got no verdict must not be presented as if it did; a
missing metric is not a zero. That is what the `cov` / `covds` columns and the
VALU/datasheet exclusion in [rocprof.md](rocprof.md) §9 are for.

`fold_kernels(..., missing=)` makes this an explicit choice because the two options
differ materially:

- **`drop`** — out of numerator *and* denominator; the case is scored on its covered
  time and the shortfall shows up as coverage. Neutral, what this file prescribes,
  and the module default.
- **`zero`** — out of the numerator, still in the denominator, i.e. scored as 0%
  efficient. This reduces the efficiency by a factor of coverage and increases
  its reciprocal by `1/coverage`; it is only a legacy reproduction option.

Use `drop` for empirical and datasheet results and publish coverage. A VALU
kernel may have no datasheet `mfma_*` peak, so it remains unscorable for that
reference. The module retains `missing="zero"` for reproducing legacy data;
state that choice explicitly if used.

## Comparing two tables: preserve case coverage

Baseline and candidate rollups within a mode need the same cases and weights.
Publish missing cases and coverage rather than treating missing metrics as zero.
Keep independently requested modes in separate reports. A rocprof run may cover
fewer cases because each case needs a profile; that coverage must not filter or
replace the standalone TL rollup.


## Executable coverage and comparison output

`fold_kernels` now returns the covered reference time without extrapolating it
onto unknown work. For `missing="drop"`, pass both `kernel_coverage` and
`covered_measured_us = measured_us * kernel_coverage` to `fold_cases`. Omitting
that matching time makes a partial row unavailable, not silently complete.

`case_coverage = sum(weights of scored cases) / sum(weights of requested cases)`.
Use `--cases requested_cases.json` to restore rows dropped by an upstream caller.
The file is a list or `{ "cases": [...] }` with kernel, regime, case, and weight.
Missing/invalid rows retain their weight, reason, and count. `coverage` remains a
legacy alias for kernel coverage only. Duplicate cases and mismatched A/B weights
are errors. A suspect individual case propagates even when the aggregate is <1.

```bash
python3 scripts/fold.py --rows rows_baseline.json --candidate rows_candidate.json \
  --cases requested_cases.json --out comparison.json
```

`ab_speedup` uses matched complete-operator measured times, even when reference
metrics are unavailable; its `latency_case_coverage` is separate. A fixed-reference
identity is flagged only when compatible covered cases and full kernel scopes
match. Numeric JSON preserves precision. `--target-efficiency 0.85` is optional;
`PARTIAL`, `SUSPECT`, and `UNAVAILABLE` rows are not certified. When requested
case coverage is not established by a manifest, the target verdict is
`UNVERIFIED_CASE_COVERAGE`. When raw repeated
samples are supplied, a target crossing their conservative envelope is
`BORDERLINE`; that envelope is not a statistical confidence interval.
