---
tags: ["optimization", "performance", "profiling", "rocprof-sys", "sampling", "call-stack"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/sampling-call-stack.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Sampling the Call Stack

## Overview

The ROCm Systems Profiler implements call-stack sampling through multiple methods. Users can employ `rocprof-sys-sample` without instrumentation, use `rocprof-sys-instrument -M sampling` for binary rewriting, or apply runtime instrumentation. These approaches are functionally equivalent when sampling is the sole objective.

To activate call-stack sampling on an instrumented binary, set `ROCPROFSYS_USE_SAMPLING=ON`.

## Why rocprof-sys-sample is Recommended

The `rocprof-sys-sample` executable offers distinct advantages over instrumented-sampling approaches:

- **Command-line configuration**: Provides direct options for controlling profiler features rather than requiring configuration files or environment variables
- **Faster launch times**: Avoids unnecessary symbol parsing and processing that instrumented-sampling requires
- **MPI compatibility**: Works seamlessly with MPI distributions like OpenMPI, whereas binary instrumentation conflicts with MPI's forking restrictions

## The rocprof-sys-sample Executable

This tool accepts extensive command-line options organized into functional categories:

**General Options**: Output paths, trace/profile generation, and host/device sampling

**Tracing Options**: Buffer management, fill policies, timing, and clock selection

**Profiling Options**: Output formats and differential analysis

**Sampling Options**: Frequency, duration, thread targeting, and timer types (CPU-clock or real-clock)

**Backend Options**: Data collection from Kokkos, MPI, locks, OpenMP, and GPU profilers

**Hardware Counter Options**: CPU and GPU event configuration

Arguments preceding `--` belong to rocprof-sys-sample; arguments following it apply to the target application.

## Configuration Precedence

Command-line arguments override environment variables, which override configuration file settings. This enables flexible default configurations customizable per invocation.

## Example Usage

Running `rocprof-sys-sample -PTDH -E all -o rocprof-sys-output %tag% -- ./parallel-overhead-locks 30 4 100` enables profiling, tracing, device and host process-sampling while disabling all optional backends, demonstrating the tool's flexibility in collecting comprehensive performance data.
