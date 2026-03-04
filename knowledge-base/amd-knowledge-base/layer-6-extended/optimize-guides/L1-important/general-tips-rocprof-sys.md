---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "profiling", "rocprof-sys", "best-practices"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/general-tips-using-rocprof-sys.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# General Tips for Using ROCm Systems Profiler

## Overview

The ROCm Systems Profiler documentation provides guidance for effective performance analysis. Key recommendations include using `rocprof-sys-avail` to explore configuration settings and hardware counters, with the `-d` flag for descriptions.

## Configuration and Compilation

Users should generate default configurations via `rocprof-sys-avail -G ${HOME}/.rocprof-sys.cfg` and customize as needed. When preparing applications, compile with optimization flags (`-O2` or higher), disable assertions using `-DNDEBUG`, and include debug symbols (minimum `-g1`).

The documentation notes that debug information doesn't degrade runtime performance—it only affects build time and binary size. In CMake, this is typically achieved through `CMAKE_BUILD_TYPE=RelWithDebInfo` or `CMAKE_BUILD_TYPE=Release` combined with `CMAKE_<LANG>_FLAGS=-g1`.

## Profiling Strategies

The guide distinguishes between two primary approaches: binary instrumentation captures every function invocation's performance data, while statistical sampling characterizes overall application behavior with reduced overhead. Combining both methods allows instrumentation to focus on specific functions while sampling fills performance analysis gaps.

## Performance Analysis Workflow

When seeking optimization opportunities, the recommended process begins with flat profiling to identify functions exhibiting high call counts, substantial cumulative runtimes, or elevated standard deviations. Hierarchical profiling should follow to understand calling context and critical paths.

## MPI Considerations

For MPI applications using binary instrumentation, avoid runtime instrumentation due to incompatibility with process spawning. Instead, perform binary rewrites on executables and run instrumented versions through `rocprof-sys-run` rather than directly launching with `mpirun`.
