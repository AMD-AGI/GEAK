---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "profiling", "rocprofiler", "grafana", "gui", "visualization"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/analyze/grafana-gui.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Grafana GUI Analysis for ROCm Compute Profiler

## Overview

The ROCm Compute Profiler's Grafana analysis dashboard provides comprehensive GPU performance profiling capabilities for MI accelerators. Key features include system and hardware component analysis, Speed-of-Light metrics, multiple normalization options, baseline comparisons, regex-based filtering, and roofline analysis.

> "The ROCm Compute Profiler Grafana analysis dashboard GUI supports the following features to facilitate MI accelerator performance profiling and analysis"

**Note:** Grafana and MongoDB functionality is deprecated and will be removed in a future release.

## Core Features

### Speed-of-Light (SOL)

Speed-of-Light panels appear at both system and individual hardware component levels. They compare workload performance metrics against theoretical maximums—such as floating-point operations and bandwidth—to identify optimization opportunities.

### Normalizations

Performance metrics support multiple normalization options:
- `per_wave`
- `per_cycle`
- `per_kernel`
- `per_second`

### Baseline Comparison

The system enables A/B testing by comparing current and baseline workloads on the same SoC. Independent filters for each workload include:
- Workload name selection
- GPU ID filtering (multi-select)
- Kernel name filtering (multi-select)
- Dispatch ID filtering (regex-based)
- Panel selection (multi-select)

### Regex-Based Dispatch ID Filtering

Users can employ regular expressions for flexible kernel invocation selection. For example, to examine dispatch IDs 17-48, the regex pattern is: `(1[7-9]|[23]\d|4[0-8])`.

### Incremental Profiling

This feature accelerates analysis by focusing on specific hardware components rather than profiling everything simultaneously. Prior results for other components remain intact and can be merged later.

### Color Coding

Standardized color coding applies across visualizations: yellow indicates over 50% utilization, red indicates over 90%.

## Database Import Process

The `--import` command adds profiling data to MongoDB. Default credentials are username `temp` and password `temp123`.

Database naming convention: `rocprofiler-compute_<team>_<database>_<soc>`

Example import command:
```
rocprof-compute database --import -H dummybox -u temp -t asw -w workloads/vcopy/mi200/
```

## Analysis Panels

The dashboard includes 18 main panel categories:

**Kernel Statistics:** Time histograms, top bottleneck kernels and dispatches, dispatch ID listings

**System Speed-of-Light:** Key metrics comparison against theoretical maximums

**Memory Chart Analysis:** Graphical GPU memory block performance representation

**Empirical Roofline Analysis:** Achieved performance versus benchmarked peak performance

**Command Processor:** CPU fetcher and compute controller metrics

**Shader Processor Input (SPI):** Workgroup manager statistics and resource allocation

**Wavefront Analysis:** Launch statistics, runtime metrics, occupancy data

**Compute Unit Instruction Mix:** VALU, MFMA, and VMEM instruction breakdowns

**Compute Unit Compute Pipeline:** Arithmetic operations and pipeline statistics

**Local Data Share (LDS):** Shared memory performance metrics

**Cache Hierarchies:** Instruction cache, scalar L1D, vector L1D, and L2 cache panels with detailed access patterns, stall analysis, and transaction tracking

**L2 Cache Per Channel:** Per-channel hit rates, transaction requests, and latency metrics
