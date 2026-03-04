---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "profiling", "rocprofiler", "pc-sampling", "hotspots"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/pc_sampling.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Using PC sampling in ROCm Compute Profiler

Program Counter (PC) sampling is a profiling technique that periodically samples the GPU program counter during kernel execution to identify code execution patterns and performance hotspots.

## Overview

ROCm Compute Profiler supports two PC sampling approaches:

- **Host Trap PC sampling**: Available for AMD Instinct MI200 series and later accelerators
- **Stochastic (Hardware-Based) PC sampling**: Available for AMD Instinct MI300 series and later accelerators

Stochastic sampling provides enhanced insight by indicating whether a sampled wave issued an instruction at a particular program counter location, plus the reason for any instruction stalls—valuable for understanding kernel execution bottlenecks.

## Profiling Options

Configure PC sampling during profiling with these parameters:

- `--pc-sampling-method`: Set to `stochastic` or `host_trap` (default: stochastic)
- `--pc-sampling-interval`: For stochastic sampling, measured in cycles (minimum granularity: 1 cycle). For host_trap, measured in microseconds (default: 1048576). Values must be powers of 2; starting at 1048576 and reducing down to 65536 is recommended.

**Example command:**

```
rocprof-compute profile -n pc_test -b 21 --no-roof --pc-sampling-method stochastic --pc-sampling-interval 1048576 -VVV -- target_app
```

## Analysis Options

When analyzing PC sampling data, use:

- `--pc-sampling-sorting-type`: Choose `offset` (assembly instruction offset in code object) or `count` (default: offset)

**Example command:**

```
rocprof-compute analyze -p workloads/pc_test/MI300A_A1/ -b 21 -k 0 --pc-sampling-sorting-type offset
```

## Important Notes

- PC sampling is currently in beta; enable explicitly using block index 21
- Output displays only sampled instructions, not the complete compiled instruction set
- Build target applications with `-g` flag to retain symbols for associating PC sampling data with HIP source code; otherwise, associations remain at assembly level
