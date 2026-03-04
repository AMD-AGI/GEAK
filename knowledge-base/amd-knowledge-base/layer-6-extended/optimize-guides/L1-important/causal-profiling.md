---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "profiling", "rocprof-sys", "causal-profiling"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/performing-causal-profiling.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Performing Causal Profiling

## Overview

Causal profiling answers a fundamental question: "If you speed up a given block of code by X%, the application will run Y% faster". This methodology helps developers identify optimization priorities by quantifying potential performance improvements.

The core principle rests on this equivalence: accelerating one code section by X% mathematically equals slowing all concurrent code by X%. Causal profiling exploits this by inserting deliberate pauses during execution, then translating these experiments into speed-up predictions during analysis.

## Getting Started

### Progress Points

Progress points track code advancement between samples and must be triggered deterministically through:

- Callbacks from Kokkos-Tools, OpenMP-Tools, rocprofiler-sdk
- Runtime instrumentation capabilities for inserting custom progress points
- User APIs like `ROCPROFSYS_CAUSAL_PROGRESS`

**Important:** Binary rewriting cannot insert progress points, as Dyninst translates instruction pointer addresses, invalidating call stack samples.

### Key Concepts

| Concept | Setting | Options | Description |
|---------|---------|---------|-------------|
| Backend | `ROCPROFSYS_CAUSAL_BACKEND` | `perf`, `timer` | Sample recording mechanism |
| Mode | `ROCPROFSYS_CAUSAL_MODE` | `function`, `line` | Experiment granularity |
| End-to-end | `ROCPROFSYS_CAUSAL_END_TO_END` | Boolean | Single run-wide experiment |
| Fixed speed-up | `ROCPROFSYS_CAUSAL_FIXED_SPEEDUP` | Values [0, 100] | Virtual speed-up selection |
| Binary scope | `ROCPROFSYS_CAUSAL_BINARY_SCOPE` | Regex | Target dynamic binaries |
| Source scope | `ROCPROFSYS_CAUSAL_SOURCE_SCOPE` | Regex | Target source files/lines |
| Function scope | `ROCPROFSYS_CAUSAL_FUNCTION_SCOPE` | Regex | Restrict to matching functions |

### Backends

Two backends record samples needed for speed-up calculations:

**Perf Backend:**
- Requires Linux Perf and elevated privileges
- Interrupts less frequently
- More accurate call stacks

**Timer Backend:**
- No privilege requirements
- Interrupts 1000 times per second (realtime)
- Suffers from instruction pointer skid

#### Instruction Pointer Skid

IP skid represents instructions executing after an event occurs before the program halts. The timer backend exhibits pronounced skid due to thread-pausing overhead, reducing resolution especially in line mode.

#### Installing Linux Perf

Verify installation by checking for `/proc/sys/kernel/perf_event_paranoid`. On Debian systems, run:

```bash
apt-get install linux-tools-common linux-tools-generic linux-tools-$(uname -r)
```

Ensure the paranoid level is ≤ 2:

```bash
echo 2 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

Make persistent by adding `kernel.perf_event_paranoid=2` to `/etc/sysctl.conf`.

## rocprof-sys-causal Executable

The `rocprof-sys-causal` tool streamlines multiple application runs with varying configurations. Key usage patterns include:

```bash
rocprof-sys-causal -n 5 -- <exe>                    # 5 runs
rocprof-sys-causal -s 0,10,20 -m function -- <exe>  # function mode with speedups
rocprof-sys-causal -F func_A func_B -- <exe>        # multiple function scopes
```

### Using rocprof-sys-causal with MPI

When profiling MPI applications, use the `--launcher` option to target the actual executable rather than the launcher:

```bash
rocprof-sys-causal -l foo -n 3 -- mpirun -n 2 foo
```

This effectively runs:

```bash
mpirun -n 2 rocprof-sys-causal -- foo  # (3 times)
```

## Output Visualization

ROCm Systems Profiler generates `experiments.json` and `experiments.coz` files in the configured output path. Visualize results at [plasma-umass.org/coz](https://plasma-umass.org/coz/).

## ROCm Systems Profiler vs. Coz

Key advantages of ROCm Systems Profiler:

- Optional debug information (supports any DWARF version)
- Function-level experiments without debug info
- Customizable speed-up subsets rather than fixed ranges
- Binary, source, and function scope filtering with regex support
- Scope exclusion capabilities
- Alternative backend support beyond Linux Perf
