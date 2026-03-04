---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "profiling", "rocprof-sys", "instrumentation", "sampling"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/conceptual/data-collection-modes.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Data Collection Modes

ROCm Systems Profiler supports several modes for recording trace and profiling data:

## Overview Table

| Mode | Description |
|------|-------------|
| Binary Instrumentation | Locates functions (and loops, if desired) in the binary and inserts snippets at entry and exit |
| Statistical Sampling | Periodically pauses application at specified intervals and records metrics for the call stack |
| Callback APIs | Parallelism frameworks like ROCm, OpenMP, and Kokkos make callbacks to provide work information |
| Dynamic Symbol Interception | Wraps function symbols in position-independent dynamic libraries (e.g., `pthread_mutex_lock`) |
| User API | User-defined regions and controls for ROCm Systems Profiler |

The two primary modes are binary instrumentation and statistical sampling, both performable with `rocprof-sys-instrument`. For sampling alone, `rocprof-sys-sample` is recommended.

## Binary Instrumentation

This approach records deterministic measurements for every function invocation. It "adds instructions to the target application to collect required information," potentially causing performance changes. The overhead depends on what data is collected—wall-clock timing has less impact than collecting timing, memory usage, cache-misses, and instruction counts.

The key control mechanism is "the minimum number of instructions for selecting functions for instrumentation."

## Statistical Sampling

Statistical call-stack sampling "periodically interrupts the application at regular intervals using operating system interrupts." While less numerically precise than instrumentation, the application runs near full speed. The resulting data is "a statistical approximation" rather than exact values, but often more accurate overall due to reduced intrusiveness.

Overhead depends on the sampling rate and whether samples relate to CPU time and/or real time.

## Comparative Example

Given a recursive Fibonacci function, binary instrumentation records every invocation with high precision but significant overhead for small functions. Statistical sampling provides less detail but avoids the overhead problem and offers better cache behavior insights.
