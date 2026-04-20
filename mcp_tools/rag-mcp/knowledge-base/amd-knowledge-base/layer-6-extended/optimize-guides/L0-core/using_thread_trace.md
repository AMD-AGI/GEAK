---
tags: ["optimization", "performance", "profiling", "thread-trace", "rocprofv3"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/rocprofiler-sdk/how-to/using-thread-trace.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Using Thread Trace

## Overview

Thread trace is a shader execution tracing technique that profiles wavefronts at the instruction timing level, targeting single or few kernel executions.

**Key features:**
- Near cycle-accurate instruction tracing
- Exact thread or wave execution path
- Wave scheduling and stall timing analysis
- Instruction and source level hotspots
- Extremely fast and granular counter collection (AMD Instinct)

**Supported devices:**
- AMD Instinct: MI200 and MI300 series
- AMD Radeon: gfx10, gfx11, and gfx12

## Prerequisites

- **aqlprofile:** Requires ROCm 7.x build or can be built from source via GitHub
- **ROCprof Trace Decoder:** Available at the release page; default install location is `/opt/rocm/lib`
- Use `--att-library-path` parameter or `ROCPROF_ATT_LIBRARY_PATH` environment variable for custom locations

## rocprofv3 Parameters for Thread Tracing

Basic command:
```
rocprofv3 --att -d <output_dir> -- <application_path>
```

| Parameter | Type | Range | Typical | Description |
|-----------|------|-------|---------|-------------|
| att-target-cu | Integer | 0-15 | 1 | Defines the CU for detail tokens |
| att-shader-engine-mask | Bitmask | 1-~0u | 0x1 | Defines Shader Engines to trace |
| att-simd-select | Integer | 0-0xF | gfx9: 0xF, Navi: 0x0 | Defines SIMDs to trace |
| kernel-iteration-range | List | — | — | Dispatch iteration to profile |
| kernel-include-regex | String | Any | — | Profile matching kernel names |
| kernel-exclude-regex | String | Any | — | Exclude matching kernel names |
| att-buffer-size | Bytes | 1MB-2GB | 96MB | Trace buffer size |
| att-serialize-all | Bool | — | False | Enable serialization for untraced kernels |
| att-perfcounter-ctrl | Integer | 1-32 | 2~8 | Stream SQ counters (gfx9 only) |
| att-activity | Integer | 1-16 | 5~10 | Shorthand for activity-related counters (gfx9 only) |
| att-gpu-index | Integer List | — | — | Profile specific GPU indexes |
| att-consecutive-kernels | Integer | ≥0 | — | Enable tracing for N consecutive kernel dispatches |

**For AMD Instinct:**
```
rocprofv3 --att --att-activity 8 -- <application_path>
```

**For AMD Radeon:**
```
rocprofv3 --att --att-simd-select 0x0 -- <application_path>
```

## Using Input File

Thread tracing parameters can be specified via JSON:

```json
{
    "jobs": [
        {
            "advanced_thread_trace": true,
            "att_target_cu": 1,
            "att_shader_engine_mask": "0x1",
            "att_simd_select": "0xF",
            "att_buffer_size": "0x6000000"
        }
    ]
}
```

## Thread Tracing for Multiple Kernel Instances

By default, thread trace executes once per kernel instance. Use `kernel-iteration-range` with `kernel-include-regex` to target multiple instances.

The `att-consecutive-kernels` parameter compiles multiple kernel profiles into a single output file, beginning tracing after encountering a targeted kernel and continuing until n kernels are profiled.

## rocprofv3 Output Files

After execution, the following files are generated:

- **stats_*.csv:** Summary of instruction latency per kernel
- **ui_output_agent_{agent_id}_dispatch_{dispatch_id}:** Detailed tracing information in JSON format; viewable with ROCprof Compute Viewer
- **.att:** Raw SQTT data for further analysis
- **.out:** Code object binaries for ISA analysis

### Stats CSV Columns

| Column | Description |
|--------|-------------|
| Codeobj | Code object load ID |
| Vaddr | ELF virtual address |
| Instruction | Assembly instruction |
| Hitcount | Execution count across all traced waves |
| Latency | Total cycles ("Stall + Issue" for gfx9, "Stall + Execute" for gfx10+) |
| Stall | Cycles when hardware couldn't issue instructions |
| Idle | Gap between instruction completion and next instruction start |
| Source | Original source code line (requires debug symbols) |

## Troubleshooting

If stats_*.csv is empty despite valid kernel dispatch, consider these solutions:

- Thread trace is limited to one CU per SE; kernels without enough waves may not assign any to the target CU
- Launch more waves or swap the target CU
- Increase shader engine mask: set to `0x11111111` or `0xFFFFFFFF` (higher values may cause packet losses)
- Use `HSA_CU_MASK` to mask all CUs except target (may impact performance on demanding kernels)
