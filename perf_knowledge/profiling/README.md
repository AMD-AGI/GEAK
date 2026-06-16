---
title: profiling — MI-series + ROCm tooling index
kind: technique
gens: [gfx906, gfx90a, gfx942, gfx950]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/profilers/README.html
  - https://rocm.docs.amd.com/en/docs-6.3.0/about/release-notes.html
---

# profiling — how to measure MI-GPU kernels and prove a win

## TL;DR
On AMD Instinct (CDNA) you have **three** modern tools, all built on **rocprofiler-sdk**: `rocprofv3`
(trace + counters CLI), `rocprof-compute` (per-kernel SoL/roofline analysis, ex-Omniperf), and
`rocprof-sys` (whole-system timeline, ex-Omnitrace). Pick the tool by question:
- "Is this kernel compute- or BW-bound?" → [`rocprof_compute_workflow.md`](rocprof_compute_workflow.md) + [`roofline_on_mi.md`](roofline_on_mi.md)
- "Which counter is pinned?" → [`rocprofv3_counters.md`](rocprofv3_counters.md) + [`reading_a_kernel_bottleneck.md`](reading_a_kernel_bottleneck.md)
- "Where are the gaps / host stalls?" → [`trace_analysis.md`](trace_analysis.md)
- "Is my measurement even trustworthy?" → [`benchmarking_methodology.md`](benchmarking_methodology.md) + [`common_pitfalls.md`](common_pitfalls.md)
- "Is my tuned kernel actually the one running?" → [`engagement_verification.md`](engagement_verification.md)

The single most important lesson encoded here: **a number you cannot reproduce A/B in the same session,
with locked clocks and a known engagement proof, is not evidence.** See engagement_verification.md.

## Files
| File | What it answers |
|---|---|
| [`tooling_overview.md`](tooling_overview.md) | The 4 tools (rocprofv3 / rocprof-compute / rocprof-sys / rocprofiler-sdk), what each is for, install + version map |
| [`rocprof_compute_workflow.md`](rocprof_compute_workflow.md) | profile → analyze → roofline; CLI/GUI; `--roof-only`; baseline A/B |
| [`rocprofv3_counters.md`](rocprofv3_counters.md) | key HW counters (VALU/MFMA busy, LDS/L2/HBM, occupancy, wavefronts) and how to collect |
| [`reading_a_kernel_bottleneck.md`](reading_a_kernel_bottleneck.md) | decision flow: compute- / BW- / latency- / occupancy-bound, and which counter says which |
| [`roofline_on_mi.md`](roofline_on_mi.md) | building a roofline on MI300X/MI350X; peak FLOPS per dtype; HBM BW; reading points |
| [`trace_analysis.md`](trace_analysis.md) | Perfetto / rocprof-sys traces, kernel timeline, gaps, HIP API trace, CPU↔GPU sync stalls |
| [`benchmarking_methodology.md`](benchmarking_methodology.md) | warmup, REPEATS=7, 0.5% noise band, HIP graphs, locked clocks, A/B same-session |
| [`engagement_verification.md`](engagement_verification.md) | proving the kernel/config is used: `AITER_LOG_TUNED_CONFIG`, TunableOp-0-engagement, ncu-equiv |
| [`common_pitfalls.md`](common_pitfalls.md) | cold cache, clock throttle, fork-storm host thrash, measuring the wrong thing |

## Cross-refs
- Tuning that these tools verify: [`../operators/dense_gemm/tuning.md`](../operators/dense_gemm/tuning.md),
  [`../backends/aiter/tuned_gemm.md`](../backends/aiter/tuned_gemm.md),
  [`../backends/rocblas_tunableop/tunableop.md`](../backends/rocblas_tunableop/tunableop.md).
- Hardware peaks/clocks these tools sit on top of:
  [`../hardware/cdna3_mi300/peak_tables.md`](../hardware/cdna3_mi300/peak_tables.md),
  [`../hardware/cdna4_mi350/peak_tables.md`](../hardware/cdna4_mi350/peak_tables.md),
  [`../hardware/cdna3_mi300/clocks_power.md`](../hardware/cdna3_mi300/clocks_power.md).

## Sources
- Tool roles + naming: ROCm Blogs "Introduction to profiling tools for AMD hardware".
- 6.3 rename (Omniperf→rocprof-compute, Omnitrace→rocprof-sys): ROCm 6.3.0 release notes.

<!-- MANIFEST: profiling/ — 10 technique docs (README, tooling_overview, rocprof_compute_workflow, rocprofv3_counters, reading_a_kernel_bottleneck, roofline_on_mi, trace_analysis, benchmarking_methodology, engagement_verification, common_pitfalls); MI300X/MI350X + ROCm-tool specific; cross-linked to operators/dense_gemm, backends/aiter, backends/rocblas_tunableop, hardware/cdna3_mi300, hardware/cdna4_mi350. -->
