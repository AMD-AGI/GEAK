---
title: profiling — ROCm tooling overview (rocprofv3 / rocprof-compute / rocprof-sys / SDK)
kind: technique
gens: [gfx906, gfx90a, gfx942, gfx950]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/profilers/README.html
  - https://rocm.docs.amd.com/en/docs-6.3.0/about/release-notes.html
  - https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html
---

# ROCm profiling tooling overview

## TL;DR
There are **three user-facing tools** and **one library** they all sit on. Memorize the rename: as of
**ROCm 6.3**, *Omniperf → rocprof-compute* (`rocprofiler-compute` package) and *Omnitrace →
rocprof-sys* (`rocprofiler-systems` package); the legacy `rocprof`/`rocprofv2` are superseded by
**`rocprofv3`**. `apt install omniperf` / `omnitrace` now fail — install the new package names.

| Tool | Binary | Package | Role | "Use it when…" |
|---|---|---|---|---|
| **rocprofiler-sdk** | (library/API) | `rocprofiler-sdk` | Counter + trace collection API; the engine under everything | you are writing a custom collector |
| **rocprofv3** | `rocprofv3` | `rocprofiler-sdk-tools` | CLI for GPU trace + HW counter collection (replaces rocprof/rocprofv2/roctracer) | you want raw counters or a HIP/kernel trace, scriptable |
| **rocprof-compute** | `rocprof-compute` | `rocprofiler-compute` | Per-kernel SoL, memory-chart, **roofline**, baseline A/B (ex-Omniperf) | "is this *one kernel* compute- or BW-bound?" |
| **rocprof-sys** | `rocprof-sys-*` | `rocprofiler-systems` | Whole-system CPU+GPU+MPI timeline → Perfetto (ex-Omnitrace) | "where are the gaps / host stalls across the whole run?" |

## The four, in detail

### rocprofiler-sdk (the foundation)
Modern collection library that replaces the legacy `rocprofiler` + `roctracer`. You rarely call it
directly; `rocprofv3`, `rocprof-compute` (beta in 6.4), and `rocprof-sys` are all clients. Knowing it
exists matters for one reason: counter availability and dispatch-replay behavior are SDK-defined, so a
counter that is missing in `rocprofv3` is missing everywhere.

### rocprofv3 (counters + traces, CLI)
The scriptable workhorse. Collects **HIP API trace, kernel dispatch trace, and hardware counters**.
Output: CSV (default today) / OTF2 / Perfetto. Counter selection is the path to the bottleneck
question — see [`rocprofv3_counters.md`](rocprofv3_counters.md). In ROCm 7.0 it will default to a DB
plus a new `rocpd` tool for export.

### rocprof-compute (per-kernel analysis + roofline)
The "what is wrong with this kernel" tool. Drives `rocprofv3`/SDK to collect the full counter set,
then renders **System Speed-of-Light**, **IP-block SoL**, **memory chart**, and **roofline**, with
**baseline comparison** (two `-p` paths). This is the *only* tool that gives empirical roofline on
Instinct. Full workflow: [`rocprof_compute_workflow.md`](rocprof_compute_workflow.md).

### rocprof-sys (system trace)
The "where did time go end-to-end" tool. Unified host + device + communication (MPI/RCCL) timeline,
exported to **Perfetto** for visual inspection of kernel gaps, host overhead and CPU↔GPU sync stalls.
See [`trace_analysis.md`](trace_analysis.md).

## Install / version map
- ROCm **6.2 and earlier**: `omniperf` / `omnitrace` (AMD-Research, separate install).
- ROCm **6.3**: renamed and folded in — `apt install rocprofiler-compute rocprofiler-systems`; binaries
  `rocprof-compute`, `rocprof-sys-*`; **`rocprofv3`** is the default CLI. Tool versions: ROCm Compute
  Profiler **3.0.0**, ROCm Systems Profiler **0.1.0**.
- ROCm **6.4**: `rocprof-compute` gains **rocprofv3 backend (beta)**; some MI300X features land only here.
- Upgrade gotcha: old `omniperf`/`omnitrace` dirs are **not auto-removed** on 6.2→6.3 — `apt remove` them.
- ROCm **7.0** (forward-looking): `rocprofv3` defaults to DB output, `rocpd` exports CSV/OTF2/Perfetto.

Confirm what you actually have: `rocprof-compute --version`, `rocprofv3 --version`, `rocminfo | grep gfx`
(the gfx target — gfx942=MI300, gfx950=MI350 — drives counter set and roofline peaks).

## Pitfalls
- Old habit `rocprof --hip-trace`: dead path; use `rocprofv3`. Scripts referencing `omniperf` break.
- Counter replay multiplies wall time: tools re-run the kernel per counter group; never time a kernel
  *while* profiling it. Profile and benchmark are separate passes ([`common_pitfalls.md`](common_pitfalls.md)).

## Verify
`which rocprof-compute rocprofv3 rocprof-sys-run` resolves; versions match the ROCm you booted.

## Sources
- Tool roles, "rocprofv3 for trace+counters / rocprof-sys for CPU+GPU / rocprof-compute for kernel analysis": ROCm Blogs profilers intro.
- 6.3 rename + package names + version numbers + apt-remove cleanup: ROCm 6.3.0 release notes.
- rocprof-compute being the only roofline path on Instinct: ROCm Compute Profiler profile-mode docs.
