# Folding many shapes into one number

**This file is engine-independent and binds all three modes.** One kernel is
exercised at many shapes; a report wants one number per kernel per regime. How that
collapse is done decides whether two tables can be compared at all — and it is where
hand-built tables go wrong most often, because the wrong way (average the
percentages) looks obviously right.

The rules below are **implemented once**, in `scripts/fold.py`, which
`join_legs.py` and `compare_legs.py` both import. A rule stated in prose and
implemented twice is a rule that will drift, so prefer calling the module over
reimplementing it — including from a single-engine leg, which has no second leg to
join but still owes the report the same arithmetic:

```bash
python3 scripts/fold.py --rows results/tl-only/rows_candidate.json --label TL
```

Its three primitives, in the order a leg uses them: `fold_kernels` (within one case:
per-kernel efficiencies -> that case's roof), `fold_cases` (across shapes: per-case
roofs -> one pct per kernel/regime), `fold_ratio` (across shapes: any ratio -- AI,
bound share). Nothing in that file parses a profiler, knows a peak, or cares which
engine produced `eff`; that is what makes the two legs' columns comparable at all.

## The rule: sum numerator and denominator separately, divide once

```
roof_us(case)      = Σ_k  t_k · Eff_k                      # within a case
pct(kernel,regime) = Σ_cases w·roof_us / Σ_cases w·measured_us
attainable_speedup = 1 / pct
```

**Never `mean(pct_case)`.** A percentage is a ratio, and averaging ratios gives a
shape carrying 0.3% of the calls the same vote as the one holding 90% of the clock.
The same applies to every other ratio in the report:

- **AI folds as `Σ(w·FLOPs) / Σ(w·bytes)`**, never as a mean of per-shape AIs.
- **Bound type is a share, not a label.** Report "fraction of weighted device time
  sitting on the compute axis". A seam is not one kernel, so "compute-bound" for a
  seam is a mixture, and a single label throws away the mixture.

## Apply `attainable` per kernel, BEFORE the sum

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
- `t_k`, the within-case weight, is **device ms per kernel from a timeline that
  reflects deployment** — in practice TraceLens'. A profiler's own durations come
  from serialized dispatch and are not the deployment time distribution.

## The roof is `max(byte roof, dispatch floor)`

Every dispatched kernel costs ~2.3 us of device time before it moves a byte, so a
seam of `n` kernels cannot finish under `n · 2.3 us` however perfect its memory
behaviour. Small decode shapes cross that line easily — a 5-kernel seam with a
0.13 us byte roof measured at 8.9 us reads as "68x available" when the honest ceiling
is 11.5 us, i.e. *already there*. Carry the floor as its own field, flag the rows it
binds, and say what the lever is: those rows are a kernel-COUNT problem (fuse nodes),
not a bandwidth one. **Both engines get this wrong identically, because neither
models dispatch.**

## Coverage is reported, not imputed — and an unscorable kernel is dropped, not zeroed

A case whose dominant kernel got no verdict must not be presented as if it did; a
missing metric is not a zero. That is what the `cov` / `covds` columns and the
VALU/datasheet exclusion in [rocprof.md](rocprof.md) §9 are for.

`fold_kernels(..., missing=)` makes this an explicit choice because the two options
differ materially:

- **`drop`** — out of numerator *and* denominator; the case is scored on its covered
  time and the shortfall shows up as coverage. Neutral, what this file prescribes,
  and the default on both legs.
- **`zero`** — out of the numerator, still in the denominator, i.e. scored as 0%
  efficient. This deflates the roof and **inflates the reported headroom by exactly
  `(1 - coverage)`**.

Both legs of `join_legs.py` use `drop`. The datasheet leg always did — a VALU kernel
has no datasheet `mfma_*` peak at all, so scoring it 0 would gut the seam — and the
empirical leg was switched to match on 2026-10-12. `--missing-emp zero` reproduces
the older behaviour; it moves every row with coverage below 100%, by construction
*downward* in roof-% and upward in claimed headroom. Whichever you pick, publish the
coverage column — that is what makes the choice visible.

## Comparing two tables: fold over the same case set

Two columns are comparable only if they were folded over the **same cases with the
same weights**. This is the trap when one leg is cheaper than the other: a `tl-only`
leg costs no GPU and happily covers every shape, while a `rocprof` leg costs a full
profile per (shape, arm) and in practice covers the dominant case per seam. Folding
TL over 56 cases and rocprof over 8 and printing them side by side compares two
different workloads.

`join_legs.py` handles this by re-folding the TL column (through that same
`fold_cases`) over exactly the subset the rocprof leg covered, so both columns share
a denominator. The consequence is worth stating in the report: **the `TL%` in a join table is not the `TL%` from the
standalone tl-only rollup**, and the two must not be quoted interchangeably.

Note what this does and does not buy. Identical folding makes the two columns
*structurally* comparable; it does **not** make their percentages the same kind of
number, because the denominators differ by construction — see
[combination.md](combination.md) §1 for the gfx950 ratios. Fold identically, report
side by side, never divide one into the other.
