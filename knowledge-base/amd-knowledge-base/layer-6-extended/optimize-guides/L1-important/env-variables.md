---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["reference", "hip", "environment-variables", "configuration"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# HIP Environment Variables

## Overview

The HIP environment variables documentation provides developers with configuration options for controlling GPU behavior across AMD platforms. These variables are organized by functionality to help users optimize their applications.

## GPU Isolation Variables

These variables restrict application access to specific GPUs:

| Variable | Purpose | Example |
|----------|---------|---------|
| `ROCR_VISIBLE_DEVICES` | Exposes device indices or UUIDs | `0,GPU-4b2c1a9f-8d3e-6f7a-b5c9-2e4d8a1f6c3b` |
| `GPU_DEVICE_ORDINAL` | Controls device exposure to OpenCL/HIP | `0,2` |
| `HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` | Specifies visible HIP devices | `0,2` |

**Recommendations:**
- Linux users should use `ROCR_VISIBLE_DEVICES`
- Windows users should use `HIP_VISIBLE_DEVICES`
- Cross-platform applications should use `CUDA_VISIBLE_DEVICES`

## Profiling Variables

The profiling options include:

- `HSA_CU_MASK`: "Sets the mask on a lower level of queue creation in the driver" (example: `1:0-8`)
- `ROC_GLOBAL_CU_MASK`: Controls queue masks for HIP/OpenCL runtimes (example: `0xf` enables 4 CUs)
- `HIP_FORCE_QUEUE_PROFILING`: Enables profiling mode (0=disabled, 1=enabled)

## Debug Variables

Key debugging configurations include logging levels via `AMD_LOG_LEVEL` (0-5 scale), output file specification, and log filtering with `AMD_LOG_MASK`. Additional debug controls encompass kernel serialization, code object dumping, and direct dispatch settings.

## Memory Management Variables

Important memory settings include initial heap sizing (`HIP_INITIAL_DM_SIZE` defaults to 8 MB), memory pool support toggles, and allocation limits through `GPU_SINGLE_ALLOC_PERCENT` and `GPU_MAX_HEAP_SIZE`.

## Compilation Options

The `HIPRTC_COMPILE_OPTIONS_APPEND` variable allows developers to "set compile options needed for hiprtc compilation," supporting parameters like `--gpu-architecture=gfx906:sramecc+:xnack`.
