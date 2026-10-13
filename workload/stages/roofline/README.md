# Stage `roofline` — compute each selected kernel's ceiling, and set the optimization order

**Input** `$IN` = profile's out (`selection_targets.json` for shares, `tasks/<name>/` for
runnable unit tests).
**Output** `$OUT/roofline.json`: per-kernel `roofline_pct`, `attainable_speedup` and
`bound_type`, plus the **ranking key `expected_e2e_gain_pct`**.

## Implementations

| Directory | Notes |
| --- | --- |
| `kernel-roofline/` | default. External archive, vendored unmodified. Three modes: `tl-only` (TraceLens prediction, no GPUs), `rocprof` (measured counters, requires rocprof-compute ≥ 3.6.0), `combination` (reconciles both legs in the time domain). Integrity enforced by `../MANIFEST.kernel-roofline.json` |
| `stub-roofline/` | a minimal worked example of swapping this slot |

Agent-shaped. The mode comes from `params.roofline_mode`, defaulting to `tl-only` — no
GPUs, and sufficient for ranking. Set `rocprof` for measured counters, or `combination`
to reconcile the two legs.

## Why this stage exists

Not to produce a pretty table — to decide **which kernels get the 48 hours**.

```
expected_e2e_gain_pct = pct_gpu_time × (1 − 1 / attainable_speedup)
```

Historical calibration on this workload:

| Kernel | GPU share | Roofline | Attainable | Measured e2e |
| --- | --- | --- | --- | --- |
| `fused_moe_kernel` | 26.45% | 88% | 1.02× | **−0.064%** |
| `kernel_paged_attention_2d` | 8.86% | 18% | 2.8× | 1.56× in isolation |

The largest consumer of GPU time was already against its ceiling: another 48 hours buys
roughly half a percent. The kernel with a third of its share had 2.8× of headroom.
**Ranking by operator-speedup potential would have put every GPU on the former.**

## Artifacts → `result.json`

The slot's `rank.py` performs this mapping (it belongs to the slot, not to any
implementation):

```bash
python3 "$WORKLOAD_ROOT/stages/roofline/rank.py"
```

It requires the skill to leave **`rows.json`** in `$OUT` — the raw per-case table, with
fields `kernel`, `regime`, `case`, `measured_us`, `roofline_us` (or whichever column
`--roof-key` names), and optionally `case_weight` and `bound`. That is `kernel-roofline`'s
own `fold.py` input format; any other roofline implementation plugs in by emitting the
same `rows.json`.

What `rank.py` does:

1. **Folding calls the skill's own `fold.py`** rather than reimplementing a sum. Folding
   multiple shapes into one number has exactly one correct algorithm — stay in the time
   domain, sum numerator and denominator separately, divide once, normalize weights
   within a regime — and `mean(roofline_pct)` is not it. So this is a `subprocess` call to
   `$SKILL_ROOT/scripts/fold.py`; swap the implementation and it calls that one instead.
2. Distributes each kernel's overall share across regimes by their `measured_us` in
   `rows.json`, computes expected gain per regime, and sums.
3. Writes `roofline.json` (all kernels, descending by `expected_e2e_gain_pct`) and
   `result.json`.

Artifacts: `roofline.json`, `folded.json` (raw `fold.py` output) and `fold.log` (its
table, with coverage and suspect markers).

## Three rules that must not be broken

- **An efficiency above 1 falsifies the denominator; it does not mean 100% saturation.**
  Such rows are marked `suspect`, must **not** be clamped to 100%, and no speedup may be
  derived from them. `rank.py` sets their `attainable_speedup` and
  `expected_e2e_gain_pct` to null — listed, but excluded from the ranking.
- **`bound_type` is a model classification, not proof of a measured bottleneck.** Low
  efficiency on its own does not establish a bandwidth, occupancy or issue limit.
- **TraceLens percentages and rocprof percentages are different quantities and must not
  be divided by one another.** On gfx950 their denominators differ by HBM 0.856, bf16
  0.573, fp8 0.628, fp4 0.502; stacking a `target_eff=0.90` on top of TraceLens bandwidth
  that is already derated is a double derate and inflates `attainable_speedup` by roughly
  17%.
