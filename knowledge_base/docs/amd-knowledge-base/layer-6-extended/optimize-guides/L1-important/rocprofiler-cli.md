---
tags: ["optimization", "performance", "profiling", "rocprofiler", "cli", "analysis"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/analyze/cli.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# CLI Analysis - ROCm Compute Profiler 3.3.1

## Overview

The ROCm Compute Profiler's command-line interface offers several key analysis capabilities:

- **Derived metrics**: Access to all built-in profiler metrics
- **Baseline comparison**: Side-by-side analysis of multiple profiling runs
- **Metric customization**: Select specific metrics or create custom configurations
- **Filtering**: Focus on particular kernels, GPUs, or dispatch IDs
- **Per-kernel roofline analysis**: Detailed arithmetic intensity and performance metrics

## Walkthrough

### Step 1: Generate High-Level Analysis Reports

The `-b` (or `--block`) flag produces three primary GPU analysis views:

**System Speed-of-Light**: Displays key GPU performance metrics and overall utilization rates.

```bash
rocprof-compute analyze -p workloads/vcopy/MI200/ -b 2
```

**Memory Chart**: Shows memory transactions and throughput across cache hierarchy levels.

```bash
rocprof-compute analyze -p workloads/vcopy/MI200/ -b 3
```

**Empirical Hierarchical Roofline**: Compares achieved throughput with hardware limits, revealing peak compute and memory bandwidth constraints.

```bash
rocprof-compute analyze -p workloads/vcopy/MI200/ -b 4
```

### Step 2: List Available Metrics

Inspect all available metrics for your GPU architecture:

```bash
rocprof-compute analyze -p workloads/vcopy/MI200/ --list-available-metrics
```

This generates a hierarchical list of metrics organized by category, including compute utilization, cache performance, and memory bandwidth.

### Step 3: Customize Metric Selection

Select specific metrics using their index identifiers:

```bash
rocprof-compute analyze -p workloads/vcopy/MI200/ -b 2 5.1.0
```

This command shows System Speed-of-Light (block 2) and GPU Busy Cycles (metric 5.1.0).

### Step 4: Filter Kernels

First, identify available kernels:

```bash
rocprof-compute analyze -p workloads/vcopy/MI200/ --list-stats
```

Then apply kernel filtering by index:

```bash
rocprof-compute analyze -p workloads/vcopy/MI200/ -k 0
```

### Step 5: Per-Kernel Roofline Analysis

Generate detailed roofline metrics for specific kernels:

```bash
rocprof-compute analyze -p workloads/vcopy/MI200/ -k 0 -b 4
```

This produces performance rates and arithmetic intensity calculations for individual kernels, including:
- VALU/MFMA FLOPs (various precision levels)
- Memory bandwidth (HBM, L2, L1, LDS)
- Arithmetic intensity values

## Additional Analysis Options

### Single Run Analysis
```bash
rocprof-compute analyze -p workloads/vcopy/MI200/
```

### List Top Kernels and Dispatches
```bash
rocprof-compute analyze -p workloads/vcopy/MI200/ --list-stats
```

### List Metrics with Descriptions
```bash
rocprof-compute analyze -p workloads/vcopy/MI200/ --list-metrics gfx90a --include-cols Description
```

### Baseline Comparison
Compare two profiling runs:

```bash
rocprof-compute analyze -p workload1/path/ -p workload2/path/
```

Or compare specific kernels:

```bash
rocprof-compute analyze -p workload1/path/ -k 0 -p workload2/path/ -k 1
```

## Analysis Output Formats

The `--output-format <format>` option supports four formats:

| Format | Output | Notes |
|--------|--------|-------|
| `stdout` | Terminal display | Default; no file generated |
| `txt` | Text file `rocprof_compute_<uuid>.txt` | Useful for searching long reports |
| `csv` | Folder with multiple CSV files | Enables programmatic analysis |
| `db` | SQLite database `rocprof_compute_<uuid>.db` | Requires `--format-rocprof-output rocpd` profile option |

Override default file naming using `--output-name <name>`.

## Analysis Database Schema

The database format generates a structured SQLite database containing:

- **Tables**: Dispatch data, metrics, roofline information, and metadata
- **Views**: Pre-built queries for common analysis patterns

This schema enables complex programmatic analysis across multiple profiling runs.

### Merging Multiple Workloads

Combine analysis data from multiple profile runs:

```bash
rocprof-compute analyze --db test -p workloads/vmem/MI300X_A1 -p workloads/vmem1/MI300X_A1
```

## Important Notes

- Memory chart and roofline visualizations only display in single-run analysis mode
- Memory chart visualization requires terminal width ≥ 234 characters
- Roofline charts adapt to initial terminal size; resize and regenerate if clarity is poor
- Some metrics may be unavailable if corresponding hardware counters are missing
