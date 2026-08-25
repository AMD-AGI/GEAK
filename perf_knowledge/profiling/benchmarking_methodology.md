---
title: profiling — benchmarking methodology (warmup, repeats, noise band, graphs, locked clocks)
kind: technique
gens: [gfx906, gfx90a, gfx942, gfx950]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.docs.amd.com/projects/amdsmi/en/latest/
---

# Benchmarking methodology on MI GPUs

## TL;DR
A trustworthy MI-GPU measurement is: **warm** (discard cold runs), **repeated** (median of ≥3; the
perf_knowledge e2e standard is **REPEATS=7**), inside a **noise band** (accept a change only if it clears the
**~0.5%** e2e band), with **clocks controlled** (or at least monitored), and done as a **same-session,
non-overlapping A/B** (ref vs candidate back-to-back). If the delta is inside the noise band, it is not
a result. **One scoped exception**: an isolated single-kernel sweep over many build variants should report
the **minimum** over interleaved replays, not the median — see §scope exception below. Profiling perturbs timing, so measure in a *separate, untraced* pass from your counter/trace
diagnosis ([`trace_analysis.md`](trace_analysis.md), [`rocprofv3_counters.md`](rocprofv3_counters.md)).

## Why MI300X is noisy (what you're fighting)
- **Peak ≠ sustained clock.** 2.1 GHz is a boost ceiling; under sustained load the engine clock settles
  lower and is power/thermal-capped ([`../hardware/cdna3_mi300/clocks_power.md`](../hardware/cdna3_mi300/clocks_power.md)).
- **Per-XCD clock variance 3–10%** across the 8 XCDs — different launches hit different clocks.
- **DVFS ramp lag** — a short kernel can finish before the clock ramps; warmup hides this.
Net: compute achieved TFLOP/s from *measured time*, never from assumed clock.

## The recipe
1. **Warmup.** Run the kernel/workload several times before timing to ramp clocks, warm caches, JIT/
   autotune. Discard warmup samples.
2. **Repeats.** Time **REPEATS=7** (perf_knowledge e2e standard; minimum median-of-3). Report **median + spread**.
3. **Noise band.** Treat anything within **~0.5%** e2e as noise — do not accept it as a win. Per-kernel
   microbench bands are tighter but still nonzero; quote the spread.
4. **Control clocks.** Lock or pin clocks for kernel microbenchmarks so DVFS variance doesn't masquerade
   as a speedup; e.g. `rocm-smi`/`amd-smi` to set a deterministic performance level. At minimum, monitor
   with `amd-smi metric` and reject runs where the clock drifted between ref and candidate.
5. **A/B same session, non-overlapping.** Run **ref** then **candidate** back-to-back in the *same*
   process/session on the *same* clocks; never compare numbers from two different sessions/boxes/days.
   The perf_knowledge aiter GEMM result used a 5-rep non-overlapping A/B (1548.9 → 1583.5 tok/s).
6. **Graphs for launch-bound work.** Use **HIP graphs / CUDA graphs** to replay a launch sequence with
   near-zero host overhead, both to *get* the real GPU-bound time and as a perf technique when the trace
   shows host-launch gaps ([`trace_analysis.md`](trace_analysis.md)).

## Scope exception: single-kernel variant sweeps use the MINIMUM, not the median
The median-with-spread rule above is the **e2e** standard and stays the rule for anything you report as an
e2e delta. It is the wrong statistic for an **isolated single-kernel sweep over many build variants**,
where the failure mode is different: sustained back-to-back replays throttle the clocks, so the median
drifts *within* a run. A measured instance (gfx942 MI300X, attention backward): the median of an
**unchanged** kernel drifted **438 → 534 µs** across runs — larger than most effects you would be trying
to measure. The minimum over replays reflects unthrottled capability and is stable enough to rank variants.

Two rules for this scope, and neither is optional:
1. **Report the minimum over replays**, never the mean or median.
2. **Build every variant up front, then time them round-robin in one process.** Clock drift then hits all
   variants equally. Numbers from separate processes are not comparable, and a table you produce is only
   internally comparable within one invocation — including the ablation rows, so re-derive any single
   comparison by building both sides in **one** interleaved sweep rather than quoting two old runs.

Print correctness per variant alongside the time, so a fast-but-wrong build is obvious, and print each
variant's delta against the first — that is the number to reason about. Keep ablation code paths alive
permanently for this reason: it is what makes a claim re-checkable on new hardware, and lets a regression
be bisected against a mechanism rather than a commit.

**Do not use graphs for this sweep**: capturing one graph per variant and replaying round-robin faults, and
at ~1 ms per call launch overhead is far under the noise floor anyway. Graphs remain correct (and
necessary) for a *separate* small-kernel harness — a 4 µs kernel is otherwise swamped by ~40 µs of Python
dispatch per launch, so without graphs you are measuring the launch, not the kernel.

## Per-leg vs 2-launch A/B
For an e2e serving change, prefer a **2-launch A/B** (full ref launch vs full candidate launch) over
summing **per-leg** microbenchmarks: per-leg sums miss overlap, caching, and dispatch interactions and
routinely disagree with e2e. The aiter GEMM tuning win (**+2.23% e2e** on Qwen3.5-27B/sglang) was
validated by a same-session 2-launch A/B, not by per-kernel sums
([`../operators/dense_gemm/tuning.md`](../operators/dense_gemm/tuning.md)).

## Reporting format
Follow conventions: `<value> @ <hw>, ROCm <ver>, <lib>@<commit/ver>, <date>`, e.g.
`+2.23% e2e @ MI300X gfx942, sglang 0.5.11 / aiter, 2026-06-08`. Median of ≥3 (preferably 7) warm
repeats, with spread; never present theoretical peak as achievable.

## Pitfalls
- Accepting a sub-0.5% delta as a win → noise ([`common_pitfalls.md`](common_pitfalls.md)).
- Timing a profiled/traced run → counter replay and tracer overhead inflate it.
- Cold-cache first run counted in the median; clock not yet ramped.
- Comparing across sessions/days — clocks, thermals, and background load differ.
- Trusting summed per-leg microbenchmarks over a real e2e 2-launch A/B.
- Using the median for a throttling single-kernel variant sweep, or comparing two variants timed in
  separate processes (§scope exception).

## Verify
A real win clears the 0.5% band across REPEATS=7, reproduces on a re-run of the same A/B, and is
accompanied by an engagement proof that the change is actually live
([`engagement_verification.md`](engagement_verification.md)).

## Sources
- Warmup / median-of-repeats / measure-don't-assume-clock discipline and MI300X clock variance: perf_knowledge [`../hardware/cdna3_mi300/clocks_power.md`](../hardware/cdna3_mi300/clocks_power.md) (ROCm MI300 arch docs) + ROCm workload-optimization guide.
- REPEATS=7 / 0.5% noise band / same-session 2-launch A/B / +2.23% e2e: perf_knowledge e2e run 2026-06-08 (see [`../backends/aiter/tuned_gemm.md`](../backends/aiter/tuned_gemm.md), [`../backends/aiter/overview.md`](../backends/aiter/overview.md)).
- Clock control via amd-smi/rocm-smi: AMD SMI docs.
