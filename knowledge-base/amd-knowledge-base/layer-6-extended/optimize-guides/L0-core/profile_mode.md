---
tags: ["optimization", "performance", "profiling", "rocprofiler", "filtering"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/projects/rocprofiler-compute/how-to/profile/mode.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Profile Mode — ROCm Compute Profiler 3.3.1 Documentation

## Profile Mode

The ROCm Compute Profiler's profile mode acquires performance monitoring data through analysis of compute workloads. This chapter covers core profiling features with practical examples.

## Profiling

Use the `rocprof-compute` executable to collect performance monitoring data from compute workloads.

### Key Benefits

- **Automated counter collection**: Pre-configured input files handle profiling automatically
- **Flexible output formats**: Supports `csv` and `rocpd` formats (default is `csv`)
- **Runtime filtering**: Apply filters to accelerate profiling
- **Standalone roofline analysis**: Isolate specific metrics or create custom configurations

### Profiling Example

The sample `vcopy.cpp` workload demonstrates profiling on MI accelerators. Compile it with:

```bash
$ hipcc vcopy.cpp -o vcopy
$ ./vcopy -n 1048576 -b 256
```

Profile using:

```bash
$ rocprof-compute profile --name vcopy -- ./vcopy -n 1048576 -b 256
```

This executes two main stages:

1. Collects counters needed for analysis (respecting any applied filters)
2. Gathers roofline analysis data (disable with `--no-roof`)

Results appear in a SoC-specific directory (e.g., `MI200`, `MI300X`, `MI100`).

### Output Files

- `pmc_perf.csv`: Merged performance counter data
- `sysinfo.csv`: SoC parameters
- `roofline.csv`: Roofline benchmark results
- `empirRoof_gpu-0_[datatype].pdf`: Roofline plots
- `log.txt`: Complete profiling log

### Profiling Output Formats

**CSV format**:
- Rocprof dumps raw counter data as CSV
- Multiple CSV files merge into `pmc_perf.csv`

**ROCPD format**:
- Rocprof outputs rocpd database files
- Merges into single CSV; databases removed by default
- Use `--retain-rocpd-output` to preserve databases

## Filtering

Profiling filters reduce time and counter collection. The following describe filters available with ROCProfiler.

### Filtering Options

**`-b`, `--block <block-name>`**
Profile specific analysis report blocks. Cannot combine with `--roof-only` or `--set`.

**`-k`, `--kernel <kernel-substr>`**
Filter kernels by name substring.

**`-d`, `--dispatch <dispatch-id>`**
Filter by global dispatch index (zero-based).

**`--set <metric-set>`**
Collect grouped metrics in single pass. Cannot combine with `--roof-only` or `--block`.

### Analysis Report Block Filtering

Profile specific hardware blocks to skip unnecessary counters:

```bash
$ rocprof-compute profile --name vcopy -b 10 7 -- ./vcopy -n 1048576 -b 256
```

Collect individual metrics by ID:

```bash
$ rocprof-compute profile --name vcopy -b 11.1.0 12.1.0 -- ./vcopy -n 1048576 -b 256
```

List available metrics:

```bash
$ rocprof-compute profile --list-available-metrics
```

### Kernel Filtering

Isolate kernels by substring match:

```bash
$ rocprof-compute profile --name vcopy -k vecCopy -- ./vcopy -n 1048576 -b 256
```

### Dispatch Filtering

Profile specific kernel dispatches:

```bash
$ rocprof-compute profile --name vcopy -d 0 -- ./vcopy -n 1048576 -b 256
```

### Metric Sets Filtering

Collect related metrics efficiently:

```bash
$ rocprof-compute profile --name vcopy --set compute_thruput_util -- ./vcopy -n 1048576 -b 256
```

View available sets:

```bash
$ rocprof-compute profile --list-sets
```

Available sets include `compute_thruput_util`, `launch_stats`, and others organized by profiling scenario.

## Standalone Roofline

Roofline analysis runs automatically unless `--no-roof` is specified. Use `--roof-only` to focus on roofline data and reduce profiling time. This option checks for existing `pmc_perf.csv` and `roofline.csv`:

1. If found, uses existing data with provided arguments to generate roofline PDF
2. Otherwise, profiles with roofline counters only

Cannot combine `--roof-only` with `--block` or `--set`.

### Roofline Options

**`--sort <desired_sort>`**
Overlay top kernel or dispatch data in roofline plot

**`-m`, `--mem-level <cache_level>`**
Specify cache levels for roofline plot

**`--device <gpu_id>`**
Select GPU device ID for roofline benchmark

**`-k`, `--kernel <kernel-substr>`**
Filter kernels in roofline analysis

**`--roofline-data-type <datatype>`**
Specify data types for PDF output (default: FP32). Multiple types overlay on same plot.

**`--kernel-names`**
Distinguish different kernels with unique markers in PDF plots

### Roofline Only Example

```bash
$ rocprof-compute profile --name vcopy --roof-only -- ./vcopy -n 1048576 -b 256
```

Output includes `empirRoof_gpu-0_FP32.pdf` and other roofline-related files.

Multiple data types generate separate PDFs per type from same workload run by re-running with different `--roofline-data-type` values while `roofline.csv` exists.
