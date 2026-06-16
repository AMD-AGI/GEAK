---
title: profiling — common pitfalls (cold cache, throttling, fork-storm, wrong measurement)
kind: technique
gens: [gfx906, gfx90a, gfx942, gfx950]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.blogs.amd.com/software-tools-optimization/profilers/README.html
---

# Common profiling & benchmarking pitfalls on MI GPUs

## TL;DR
Most "results" die for one of these reasons: **cold cache / cold clock**, **clock throttling**,
**host fork-storm** (e.g. repeated `rocm_agent_enumerator`) starving launches, **measuring the wrong
thing** (timing a profiled run, or summing per-leg microbenchmarks instead of an e2e A/B), and
**no engagement proof** (the change never ran). Each has a concrete tell and a concrete fix.

## The pitfalls

### 1. Cold cache / cold clock
First run is slow: caches empty, clocks not ramped (DVFS lag), JIT/autotune not done. **Tell:** run 1
≫ runs 2–N. **Fix:** warmup and discard; median over **REPEATS=7** warm runs
([`benchmarking_methodology.md`](benchmarking_methodology.md)).

### 2. Clock throttling (power/thermal)
2.1 GHz is a boost ceiling; under sustained load the engine clock settles lower and is power-capped, and
the **8 XCDs vary 3–10%** in clock. **Tell:** achieved TFLOP/s drifts run-to-run; `amd-smi metric` shows
clock < boost during the hot region. **Fix:** lock/pin clocks for microbenchmarks, or monitor and reject
drifted runs; always compute achieved FLOP/s from *measured time*, not assumed clock
([`../hardware/cdna3_mi300/clocks_power.md`](../hardware/cdna3_mi300/clocks_power.md)).

### 3. Host fork-storm (`rocm_agent_enumerator` thrash)
Some tuning/dispatch paths repeatedly fork host helpers (notably `rocm_agent_enumerator`) — e.g. racing
~1365 hipBLASLt solutions per shape. **Tell:** CPU lane in a trace is saturated with short forked
processes; GPU lane has gaps; launches starve. **Fix:** bucket-reduce the shape set (e.g. pad/round M
via `get_padded_m`), cap the solution race, and avoid per-call enumeration in the hot loop
([`../operators/dense_gemm/tuning.md`](../operators/dense_gemm/tuning.md),
[`trace_analysis.md`](trace_analysis.md)).

### 4. Measuring the wrong thing
- **Timing a profiled/traced run.** Counter collection **replays the kernel** (multi-pass) and tracing
  adds host overhead → inflated, meaningless time. **Fix:** profile to *diagnose*, benchmark in a
  separate untraced pass to *measure* ([`rocprofv3_counters.md`](rocprofv3_counters.md)).
- **Per-leg vs e2e 2-launch A/B.** Summing per-kernel microbenchmarks misses overlap, caching, and
  dispatch interaction and routinely disagrees with e2e. **Fix:** prefer a same-session **2-launch A/B**
  of the full workload ([`benchmarking_methodology.md`](benchmarking_methodology.md)).
- **Wrong roofline dtype.** Plotting an FP8 kernel against the FP32 roof looks terrible for no reason.
  **Fix:** `--roofline-data-type` matching the kernel's MFMA dtype ([`roofline_on_mi.md`](roofline_on_mi.md)).
- **Slow ⇒ compute-bound assumption.** Slow is not a bottleneck class. **Fix:** place it on the roofline
  first ([`reading_a_kernel_bottleneck.md`](reading_a_kernel_bottleneck.md)).

### 5. Accepting noise as a win
Treat anything within the **~0.5%** e2e band as noise. **Tell:** the "win" doesn't reproduce on a
re-run. **Fix:** require the delta to clear the band across REPEATS=7, and re-run the A/B.

### 6. No engagement proof (the change never ran)
A flat A/B with **0 engagement** says nothing about your kernel. **Tells/fixes:** aiter →
`grep -c 'is tuned on cu_num' server.log` must be > 0 (`AITER_LOG_TUNED_CONFIG=1`); custom kernel → its
name must appear in `rocprofv3 --kernel-trace`. Note **TunableOp/`HIPBLASLT_TUNING_FILE` get 0
engagement on aiter/sglang/vllm** (PyTorch-dispatch bypass). Full gate:
[`engagement_verification.md`](engagement_verification.md).

### 7. Bias / cu_num / dtype key mismatch
A populated tuning DB can still 100%-miss if the key (bias flag, `cu_num`, dtype) doesn't match live
calls. **Tell:** DB exists, 0 hits. **Fix:** capture real shapes from a warm server; verify the grep
count ([`engagement_verification.md`](engagement_verification.md)).

## Quick checklist before quoting a number
- [ ] Warm (cold runs discarded), REPEATS=7, median+spread.
- [ ] Clocks locked or monitored; no throttle/drift between ref and candidate.
- [ ] Measured in a separate untraced/unprofiled pass.
- [ ] Same-session 2-launch A/B; delta clears the 0.5% band.
- [ ] Engagement proven (`is tuned on cu_num` > 0 or kernel in `--kernel-trace`).
- [ ] Roofline plotted against the correct dtype; achieved (not peak) reported.

## Sources
- Cold cache/clock, throttle, XCD clock variance, measure-don't-assume: perf_knowledge [`../hardware/cdna3_mi300/clocks_power.md`](../hardware/cdna3_mi300/clocks_power.md) + ROCm workload guide.
- Fork-storm `rocm_agent_enumerator` / ~1365 hipBLASLt solutions / bucket-reduce M: perf_knowledge [`../operators/dense_gemm/tuning.md`](../operators/dense_gemm/tuning.md) (e2e run 2026-06-08).
- Counter replay / profile-vs-measure separation: ROCm Blogs profilers intro.
- 0.5% noise band / REPEATS=7 / 2-launch A/B / engagement grep / TunableOp 0-engagement: perf_knowledge aiter + rocblas_tunableop cards (e2e run 2026-06-08).
