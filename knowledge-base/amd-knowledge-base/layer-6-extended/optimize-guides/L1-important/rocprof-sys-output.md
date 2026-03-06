Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "profiling", "rocprof-sys", "output", "visualization"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/understanding-rocprof-sys-output.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Understanding the Systems Profiler Output

## Output File Structure

ROCm Systems Profiler generates files following this pattern: `<OUTPUT_PATH>[/<TIMESTAMP>]/[<PREFIX>]<DATA_NAME>[-<OUTPUT_SUFFIX>].<EXT>`

Configuration options control naming:

| Setting | Purpose |
|---------|---------|
| `ROCPROFSYS_OUTPUT_PATH` | Directory for output files |
| `ROCPROFSYS_OUTPUT_PREFIX` | Custom prefix for filenames |
| `ROCPROFSYS_TIME_OUTPUT` | Enable timestamped subdirectories |
| `ROCPROFSYS_USE_PID` | Append process ID to filenames |

## Metadata

ROCm Systems Profiler automatically generates a `metadata.json` file containing:

- Hardware specifications (CPU model, cache sizes, concurrency)
- System information (OS, processor architecture)
- Launch timestamp and environment variables
- Configuration settings used for profiling
- Memory mapping data from shared libraries
- List of generated output files

This metadata provides comprehensive context for analyzing profiling results.

## Output Prefix Keys

When profiling multiple runs, use prefix keys to organize outputs systematically. These encodings substitute values into filenames:

- `%argv%` - Complete command line as single string
- `%argt%` - Basename of first argument
- `%pid%` - Process identifier
- `%rank%` - MPI rank or zero
- `%job%` - SLURM job ID if available
- `%m` - Shorthand for `%argt_hash%`
- `%p` - Shorthand for `%pid%`

Forward slashes in substituted values convert to underscores.

## ROCm Profiling Data (rocpd) Output

The emerging standard for profiling is the rocpd SQLite3 database format, available with ROCProfiler-SDK 1.0.0+.

**Key advantages:**
- Single database consolidates all profiling artifacts
- Standard SQL query interface via CLI or Python
- Integration with third-party analysis frameworks

To generate rocpd output:

```bash
export ROCPROFSYS_USE_ROCPD=ON
rocprof-sys-sample -- ./your_application
```

A Python conversion tool transforms rocpd databases into formats like OTF2, Perfetto, and CSV.

## Native Perfetto Visualization

Set `ROCPROFSYS_OUTPUT_FILE` with an absolute path to control output location. Open the generated `.proto` file in [ui.perfetto.dev](https://ui.perfetto.dev) for interactive visualization of GPU metrics, API calls, and execution flow.

## Timemory Text Output

Timemory generates human-readable text profiles showing function timing hierarchies. The format displays:

- Call stack with indentation
- Thread/rank identifiers (e.g., `|00>>>`)
- Execution statistics (count, mean, min, max, standard deviation)
- Percentage of self time

Control generation via `ROCPROFSYS_TEXT_OUTPUT` setting.

## Timemory JSON Output

JSON output supports two structures:

**Flat Layout:** All entries in a single array under `["timemory"][<metric>]["ranks"]`, easier for custom Python post-processing

**Hierarchical Layout:** Tree structure under `["timemory"][<metric>]["graph"]` with inclusive/exclusive metrics, compatible with Hatchet analysis tools

Use `ROCPROFSYS_JSON_OUTPUT` and `ROCPROFSYS_TREE_OUTPUT` to control generation of each format respectively.

## Advanced Configuration

- `ROCPROFSYS_COLLAPSE_THREADS` - Combine identical call stacks across threads
- `ROCPROFSYS_COLLAPSE_PROCESSES` - Combine data across MPI ranks
- `ROCPROFSYS_FLAT_PROFILE` - Remove call stack hierarchy for simpler output
- `ROCPROFSYS_TIMELINE_PROFILE` - Generate Perfetto-like timeline data
- `ROCPROFSYS_MAX_WIDTH` - Adjust text output truncation
