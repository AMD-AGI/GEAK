---
tags: ["optimization", "performance", "profiling", "rocprofiler", "usage"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/projects/rocprofiler-compute/how-to/use.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Basic Usage — ROCm Compute Profiler 3.3.1

## Command Line Profiler

The command line profiler launches your target application and collects profile results via the `rocprof` binary. When no specifications are provided, it gathers all available counters for all kernels and dispatches.

Basic profiling command:
```
$ rocprof-compute profile -n vcopy_data -- ./vcopy -n 1048576 -b 256
```

Results are written to a subdirectory named after your accelerator (e.g., `./workloads/vcopy_data/MI200/`).

**Note:** ROCm Compute Profiler may replay kernels multiple times to collect all requested profile information.

### Customize Data Collection

Filter options allow you to specify which kernels and metrics to collect:

- **`-k`, `--kernel`**: Filter kernels by name
- **`-d`, `--dispatch`**: Filter based on dispatch ID
- **`-b`, `--block`**: Collect metrics for specified analysis report blocks

View available metrics using:
```
$ rocprof-compute --list-metrics <sys_arch>
$ rocprof-compute profile --list-available-metrics
```

### Analyze in the Command Line

After profiling, use the CLI to quickly examine results:

```
$ rocprof-compute analyze -p <workload_path>
```

The `-p` or `--path` option lets you analyze existing profiling data from previous sessions.

### Analyze in the Grafana GUI

For deeper analysis, import data to MongoDB:

```
$ rocprof-compute database --import [CONNECTION_OPTIONS]
```

## Modes

ROCm Compute Profiler operates in three distinct modes:

### Profile Mode

```
$ rocprof-compute profile --help
```

Launches your application locally using ROCProfiler, collecting data for selected kernels, dispatches, and hardware components. Results are stored in `./workloads/<name>`.

### Analyze Mode

```
$ rocprof-compute analyze --help
$ rocprof-compute analyze --tui
```

Loads profiling data and generates metrics. Supports a lightweight GUI with the `--gui` flag and an interactive Text-based User Interface (TUI) with the `--tui` flag.

### Database Mode

```
$ rocprof-compute database --help
```

Manages MongoDB storage for Grafana visualization. Use `--import` to add workloads or `--remove` to delete them.

## Global Options

- **`-v`, `--version`**: Display version information
- **`-V`, `--verbose`**: Increase output verbosity
- **`-q`, `--quiet`**: Reduce output verbosity
- **`-s`, `--specs`**: Display system specifications

**Environment variable:** Set `ROCPROFCOMPUTE_COLOR=0` to disable colorful output.

## Basic Operations

| Operation | Mode | Required Arguments |
|-----------|------|-------------------|
| Profile a workload | `profile` | `--name`, `-- <profile_cmd>` |
| Standalone roofline analysis | `profile` | `--name`, `--roof-only`, `--roofline-data-type <data_type>`, `-- <profile_cmd>` |
| Import workload to database | `database` | `--import`, `--host`, `--username`, `--workload`, `--team` |
| Remove workload from database | `database` | `--remove`, `--host`, `--username`, `--workload`, `--team` |
| Launch standalone GUI | `analyze` | `--path`, `--gui` |
| Interact with results via CLI | `analyze` | `--path` |
