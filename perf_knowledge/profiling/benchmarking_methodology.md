---
title: profiling — benchmarking methodology (warmup, repeats, noise band, graphs, locked clocks)
kind: technique
gens: [gfx906, gfx90a, gfx942, gfx950]
updated: 2026-09-21
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.docs.amd.com/projects/amdsmi/en/latest/
  - ROCm/aiter@04c7b808:.claude/skills/review-pr/rules.md
  - ROCm/aiter@04c7b808:.claude/skills/aiter-op-test/SKILL.md
---

# Benchmarking methodology on MI GPUs

> **Canonical deep version:** [`../expert_skills/tuning/tuning-core/measurement.md`](../expert_skills/tuning/tuning-core/measurement.md)
> in the vendored tuning skillset. That copy is the one whose claims are executable
> (`perf_knowledge/expert_skills/tuning/validate/claims.py`) and re-checked per image, and it is
> what the e2e tuning phase actually runs. This card stays because benchmarking is not only a tuning activity: kernel authoring,
> profiling and e2e A/B all cite this card, and the single-kernel-sweep exception below is GEAK's own
> — where the two touch the same ground, the deep version wins and this one is the index entry
> into it. Do not grow tuning procedure here; send the fix upstream and re-sync the tree.

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

**Do not mix graph capture into this sweep unless launch overhead is material and identically
controlled for every variant.** One graph per build complicates round-robin replay and adds another
stateful variable. For kernels around 1 ms, first verify that host overhead is stable and too small
to change the ranking. For a *separate* small-kernel harness, graphs are necessary: a 4 µs kernel can
otherwise be swamped by ~40 µs of Python dispatch per launch, so without graphs you are measuring the
launch rather than the kernel.

## Reviewing someone else's number: base vs head, patch reversed
Everything above tells you how to produce a trustworthy measurement. Judging a **claimed** one is a
different job, and the rule is narrower: **the measurement that counts is base vs head, on this box,
back to back, from clean trees at the exact compared revisions.** Separate worktrees are preferable;
if one worktree is reused, restore it exactly and verify the diff before measuring the other side.

Running only the candidate against whatever baseline the change's author chose reproduces *their*
comparison. It cannot show a regression, and it silently inherits any staleness in their baseline.
Reduce each pair to a ratio oriented so **`<1` is always a regression**: candidate/base for
higher-is-better metrics such as throughput, and base/candidate for lower-is-better metrics such as
latency. If the verdict takes the minimum across columns, quote the column that set it. Require
several matched rows with both sides exiting cleanly—a truncated log yields a meaningless ratio, so
a nonzero exit on either side is never a "no regression".

When the comparison could not be made, say which of these it was rather than merging them: the
change has **no runtime surface** (nothing to measure); there is **no benchmark entry point** in the
target; **the change adds the target**, so base has nothing to compare against; the two sides
**measured different things**; or the run **could not happen** (no idle GPU, wrong arch, out of
time). The last is an environment gap, not a defect in the change. A single sample on a shared box is
weak evidence either way — report the sample count or the spread, never one bare number.

**What may be excluded from the timing window, and what may not.** Excluding a genuinely one-time,
amortizable cost from steady-state per-call latency is *correct* methodology, not a trick: weight
shuffle/preshuffle, model weight loading, and a first-call JIT whose result is cached for the
process's life are all paid once per deployment. `warmup_iters` before a steady-state loop is
standard. What may not be excluded is a cost that **recurs** — a first-call JIT on a path that is not
cached across calls, or setup running on the live stream on every cold start. Excluding a recurring
cost can turn a net regression into an apparent speedup; charging a one-time shuffle against a single
call to manufacture a regression is the same error in the other direction. Ask which it is before
either accepting or rejecting the claim. Full review procedure:
[`../workflows/review_kernels.md`](../workflows/review_kernels.md).

## Candidate-table hygiene (when you bench several backends at once)
A backend bake-off table is read as if every cell is comparable, so five rules keep it honest.

1. **The reference is not a candidate.** Compute the torch reference, compare against it, and keep
   it *out* of the timing table — an unoptimized reference in a perf column invites a meaningless
   speedup ratio. The exception is when torch genuinely is one of the kernels under test
   (`torch.mm`, `torch.einsum`).
2. **Drop a candidate in configs it does not support, and say why in the table.** A kernel that is
   only correct for some layouts or dtypes will still *produce a time* in the ones it isn't, and a
   wrong-but-fast number is worse than a blank: it wins the bake-off. Leave the cell empty and
   comment the reason. The error column is how you find these — a candidate sitting near `err ≈
   0.99` is signalling an unsupported config or a real bug, so never silently drop it before you
   know which.
3. **Record the arch in the row**, so one table stays self-describing when results from two cards
   end up side by side.
4. **Report both roofline metrics, not bare `ms`** — TFLOP/s *and* GB/s from the same timing, so a
   decode row's low TFLOP/s is readable as bandwidth-bound rather than as a failure. Worked
   example and the reading rules: [`../expert_skills/tuning/benchmark/README.md`](../expert_skills/tuning/benchmark/README.md).
5. **Store raw per-candidate values, not ratio columns.** Each candidate gets its own time,
   TFLOP/s, GB/s and error cells; a hand-written `a/b` column hides which side moved and goes stale
   when a candidate is added. Derive ratios from the raw cells when reading the table — the
   base-vs-head ratio convention above is a review verdict, not a table column.

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
