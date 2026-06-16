---
title: profiling — rocprof-compute (Omniperf) profile→analyze→roofline workflow
kind: technique
gens: [gfx942, gfx950]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html
  - https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/install/quickstart.html
  - https://rocm.blogs.amd.com/software-tools-optimization/profilers/README.html
---

# rocprof-compute: profile → analyze → roofline

## TL;DR
Three verbs: **`profile`** (collect counters into `workloads/<name>/<SoC>/`), **`analyze`** (render
SoL / memory-chart / roofline from that dir), and optionally a GUI/standalone mode. For a fast
compute-vs-BW verdict, profile with **`--roof-only`** (collects only roofline counters and drops a
standalone roofline PDF). To compare two kernels/configs, pass **two `-p` paths** to `analyze` for a
baseline A/B. This is the only tool that produces an *empirical* roofline on Instinct.

## The default two-stage profile
A plain `rocprof-compute profile` runs **two stages**: (1) collect the full analysis counter set
(multiple dispatch-replay passes), then (2) collect roofline data via on-device microbenchmarks.
Disable stage 2 with `--no-roof`; do *only* stage 2 with `--roof-only`.

```bash
# Full kernel analysis (slow: many counter groups, kernel replayed per group)
rocprof-compute profile --name myrun -- python bench.py --shape 4096

# Roofline only (fast: roofline counters + standalone PDF)
rocprof-compute profile --name myrun --roof-only -- python bench.py --shape 4096
```
Output lands in a **SoC-named** dir: `workloads/myrun/MI300X/` (gfx942) or `.../MI350X/` etc. The SoC
name comes from the family and does not distinguish all SKUs in a family.

## Analyze (CLI)
```bash
rocprof-compute analyze -p workloads/myrun/MI300X/                 # full report
rocprof-compute analyze -p workloads/myrun/MI300X/ --roofline-data-type FP16   # dtype-specific roof
rocprof-compute analyze -p workloads/myrun/MI300X/ --dispatch 1 --block 7.1.0 7.1.1   # one kernel, specific metric blocks
```
- `--block N.M.K` selects IP-block / metric sections (e.g. `7.x` LDS, `16.x`/`17.x` cache & HBM).
- `--dispatch K` isolates a single kernel launch — essential when one launch dominates.

## Baseline A/B (the comparison you actually want)
```bash
rocprof-compute analyze \
  -p workloads/baseline/MI300X \
  -p workloads/candidate/MI300X \
  --dispatch 1 --block 7.1.0 7.1.1 7.1.2
```
This diffs every metric side-by-side — the in-tool equivalent of the **same-session 2-launch A/B**
discipline in [`benchmarking_methodology.md`](benchmarking_methodology.md). Note the *counter* A/B is
for diagnosing *why*; the *timing* A/B for "did it get faster" should still be done in your own bench
harness with locked clocks, because profiling perturbs timing.

## Roofline shortcut
`--roof-only` emits `roofline.csv` (empirical peaks from microbench) plus `empirRoof_gpu-0_FP16.pdf`
etc., one PDF per dtype. Overlay dtypes/kernels with `--device`/`--kernel-names`. Interpreting the
plot: [`roofline_on_mi.md`](roofline_on_mi.md).

## GUI / standalone
`rocprof-compute analyze --gui` serves an interactive dashboard (Grafana-style panels) from a profiled
workload; useful for browsing SoL panels and the memory chart without memorizing block ids. CLI is the
scriptable, reproducible path; GUI is for exploration.

## Where it fits
1. Suspect a kernel from a [`trace_analysis.md`](trace_analysis.md) timeline (longest bar / biggest gap).
2. `profile --roof-only` → is it near the compute roof or the BW roof?
3. If unclear, full `profile` + `analyze` SoL/memory-chart → read counters per
   [`reading_a_kernel_bottleneck.md`](reading_a_kernel_bottleneck.md).
4. Apply a lever (e.g. [`../operators/dense_gemm/tuning.md`](../operators/dense_gemm/tuning.md)),
   re-profile, A/B with two `-p` paths.

## Pitfalls
- Profiling time ≠ real time: dispatch replay re-runs the kernel many times. Time elsewhere.
- Wrong SoC dir: analyze must point at the SoC subfolder, not the `workloads/<name>/` parent.
- Default roofline dtype is FP32; for LLM kernels pass `--roofline-data-type FP16/BF16/FP8`.

## Verify
After a `--roof-only` run, `ls workloads/<name>/<SoC>/` shows `roofline.csv` + `empirRoof_*.pdf`.

## Sources
- profile/analyze, `--roof-only`, two-stage routine, `--no-roof`, SoC dir naming, dtype/kernel filters: ROCm Compute Profiler profile-mode docs.
- Baseline A/B with two `-p` paths and `--block`: ROCm Compute Profiler analyze docs / training material (linked).
- GUI + tool role: ROCm Blogs profilers intro.
