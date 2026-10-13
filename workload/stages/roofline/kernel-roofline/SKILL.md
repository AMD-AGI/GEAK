---
name: kernel-roofline
description: "Analyze GPU kernel or operator rooflines in three explicit modes: tl-only predicts a shape's ceiling with TraceLens models (no GPU); rocprof measures a binary's achieved rate and arithmetic intensity with rocprof-compute hardware counters; combination runs both legs on aligned cases and folds them in the time domain to locate where they disagree. Use for per-case FLOPs, bytes, latency, throughput, AI, bound type, and roofline efficiency. TL means TraceLens."
---

# Kernel Roofline

**TL means TraceLens.** Analyze kernels written in Triton, TileLang, HIP, or a
library using one of three explicitly separated engines. The engines are not
interchangeable and their percentages are not comparable — pick one on purpose.

## Select the mode first

| Mode | Engine | Question it answers | Denominator | Instructions |
| --- | --- | --- | --- | --- |
| `tl-only` | TraceLens `PerfModel` + native reporter | How far is this **shape** from the hardware ceiling? A *predictor* — stable across implementations, right for setting a target before any code exists. No GPU needed. | arch JSON, **already derated** | [tl-only.md](references/tl-only.md) |
| `rocprof` | `rocprof-compute profile --roof-only` + `analyze -b 4` | What is this **binary** actually issuing? A *diagnoser* — measured counters per leg. Needs a GPU and a runnable command. | empirical microbench peak or datasheet; derating is the caller's `target_eff` | [rocprof.md](references/rocprof.md) |
| `combination` | Both legs over one aligned case set, folded in the time domain | Does the disagreement sit in the numerator (FLOPs/bytes) or the denominator (peaks)? | both reported side by side, **never divided into each other** | [combination.md](references/combination.md) |

Honor an explicit mode. For an unqualified request to "do a roofline", default to
`tl-only` and say so. A request for measured counters, achieved FLOP/s, or
"what is this kernel actually doing" selects `rocprof`. A request to reconcile two
roofline numbers, or to adjudicate which engine is wrong, selects `combination`.
Within an explicit mode, report conflicting requirements or missing coverage; do
not silently switch engines.

These are skill invocation options, not arguments accepted by any CLI:

```text
Use $kernel-roofline with mode=tl-only to bound this shape.
Use $kernel-roofline with mode=rocprof to measure what the kernel issues.
Use $kernel-roofline with mode=combination to reconcile the two legs.
```

### The two engines measure different things

This is the single most-repeated mistake, so it is stated up front. TraceLens'
`roofline_us` is a **prediction from a shape** against a bandwidth figure that is
already derated; rocprof-compute's `Roofline Eff.` is a **measurement of a binary**
against an undiscounted microbench or datasheet ceiling. On gfx950 the denominator
ratio TL/rocprof is HBM 0.856, bf16 0.573, fp8 0.628, fp4 0.502. Stacking a
`target_eff=0.90` on top of TL's already-derated bandwidth is a double derate and
inflates `attainable_speedup` by roughly 17%. The two columns go side by side and
never divide into each other.

## There is no `extended` mode

The predecessor skill had `tl-only` / `extended`, where `extended` bundled two
unrelated things: handwritten semantic corrections, and rocprof counter
diagnostics. They have been split. The corrections workflow is **orthogonal to the
engine** and now lives in [corrections.md](references/corrections.md), usable from
any of the three modes; the counter work became the `rocprof` mode. The standalone
`scripts/roofline_report.py` calculator still accepts `--mode extended` as an input
label — that is a string in a file format, not a mode of this skill.

## Select the operation within the mode

- **Model only:** report modeled work/traffic and reference metrics. Measured
  latency, achieved throughput, and measured efficiency stay unavailable. This is
  the whole of `tl-only`.
- **Measure / compare:** collect or reuse a matching profile, validate correctness,
  report the selected engine's results for baseline/candidate.
- **Explain results:** inspect assumptions and scope. Counter evidence is
  `rocprof`-mode work; reconciling two engines is `combination`-mode work.

An existing kernel UT or framework profiler may construct inputs, check correctness,
and capture device activity in any mode. In `tl-only`, TraceLens owns the modeling
and the roofline; ordinary trace capture does not introduce a second engine.
TraceLens does not universally benchmark arbitrary kernel source.

## Check or install the tooling

`tl-only` requires a working TraceLens install. `rocprof` requires
`rocprof-compute` — **≥ 3.6.0 on gfx95x**, see the version gate in
[rocprof.md](references/rocprof.md) — plus a GPU. `combination` requires both.
`scripts/roofline_report.py` and its synthetic example need only the standard
library. Follow [installation.md](references/installation.md); a missing dependency
is a setup problem, not a reason to silently change mode.

## Shared rules, all three modes

- **Every row carries its provenance.** Engine name, tool version/commit, arch JSON
  path + md5 + its `source` field + the peak key actually resolved, device/partition,
  runtime/container digest, workload shape, layout/dtypes, operator boundary, cache
  policy, and the implementation actually dispatched. A row missing any of these is
  not quotable. The arch JSON is **not a constant** — see
  [combination.md](references/combination.md) for the four MI355X variants on this
  cluster and their 18.6% bf16 dispersion, and `assets/arch_variants/README.md`.
- **A roofline bound label is a model classification, not proof of a measured
  bottleneck.** Low efficiency alone does not establish bandwidth, occupancy, or
  launch limitation. An efficiency above 1 is a disproof of the denominator:
  investigate it, mark it suspect, publish no speedup from it, and **never clamp**
  — a clamped 100% reads as "saturated", which is the opposite of what happened.
- **One kernel, many shapes: fold per [folding.md](references/folding.md).** The
  collapse from per-shape rooflines to one number per kernel/regime has exactly one
  correct arithmetic in every mode — time domain, numerator and denominator summed
  separately, divided once, weights normalized within a regime, `attainable` applied
  per kernel before the sum. `mean(roofline_pct)` is wrong. It is implemented once,
  in `scripts/fold.py` — call it (`python3 scripts/fold.py --rows <rows.json>`)
  rather than rewriting the sums, including from a single-engine leg. Two columns are
  comparable only if they were folded over the same cases with the same weights;
  identical folding is what makes two engines' tables structurally comparable, and it
  still does not make their percentages the same kind of number.
- **Results split by mode:** `results/tl-only/`, `results/rocprof/`,
  `results/combination/`. Each summary states its mode, engine, assumptions,
  limitations, and **whether a GPU was actually measured**.
- Match modeled work to measured scope. One operator can launch several kernels;
  never divide a whole operator's FLOPs/bytes by one stage's time. Preserve the
  engine's own timing semantics and document overlap/aggregation.
- Exclude JIT warmup and identified harness work without deleting required operator
  work. Keep correctness status, raw samples, exclusions, and coverage visible.
- Keep cases and phases separate. Benchmark repetitions are not deployment weights,
  a missing metric is not a zero, and a shape-only result is not a GPU measurement.
- Preserve source units and native values; label any presentation-only conversion.

## A worked example of all three modes

`examples/krf_0927/` is a complete two-engine, two-arm campaign over
nine seams: `tl-only/`, `rocprof/leg/`, and
`combination/REPORT.md`. Read it when you want to see what a finished
report looks like, which columns are quotable, and how the adjudication of
section 6 of [combination.md](references/combination.md) reads in practice. Its
`tl_leg_*.py` / `tl_compare.py` are campaign-specific seam drivers, not part of this
package — the per-seam driving is yours to write; the folding rules are not.

Timing and cache-policy guidance is shared: read the first two sections of
[measurement.md](references/measurement.md) in any mode. Its counter section
belongs to `rocprof`. [pitfalls.md](references/pitfalls.md) is a one-page index of
every trap in this package, each anchored to where it is explained.
