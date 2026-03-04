---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "hip", "debugging", "rocgdb", "tracing"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/debugging.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Debugging with HIP

## Tracing

HIP debugging uses tools like `ltrace` and ROCgdb. The `ltrace` utility traces dynamic library calls to stderr, helping visualize the ROCm software stack's runtime behavior.

Example command for tracing HIP APIs:
```bash
$ ltrace -C -e "hip*" ./hipGetChanDesc
```

This displays function calls like:
```
hipGetChanDesc->hipCreateChannelDesc(0x7ffdc4b66860, 32, 0, 0) = 0x7ffdc4b66860
hipGetChanDesc->hipMallocArray(0x7ffdc4b66840, 0x7ffdc4b66860, 8, 8) = 0
```

Similarly, you can trace HSA APIs using `ltrace -C -e "hsa*"` to observe lower-level runtime operations.

## Debugging

ROCgdb serves as the ROCm source-level debugger for Linux, based on GNU debugger (GDB). It works with IDE frontends like Eclipse or Visual Studio Code.

To launch ROCgdb:
```bash
$ export PATH=$PATH:/opt/rocm/bin
$ rocgdb ./hipTexObjPitch
```

### Debugging HIP Applications

When a segmentation fault occurs, ROCgdb provides stack traces and thread information. Using the `bt` command shows the call stack, while `info thread` lists active threads. Switch between threads with `thread [number]` to identify which thread caused the issue.

## Useful Environment Variables

### Kernel Enqueue Serialization

Control kernel command serialization:

- `AMD_SERIALIZE_KERNEL`: Values 1 (wait before), 2 (wait after), or 3 (both)
- `AMD_SERIALIZE_COPY`: Same options for copy operations

These force the GPU to idle before/after commands for debugging synchronization problems.

### Making Device Visible

For multi-device systems, use `HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`:
```bash
$ HIP_VISIBLE_DEVICES=0,1
```

This restricts HIP to only specified device indices.

### Dump Code Object

Enable `GPU_DUMP_CODE_OBJECT=1` to analyze compiler-related issues.

### HSA-Related Variables (Linux)

- `HSA_ENABLE_SDMA=0`: Forces compute shader blits instead of DMA copy engines
- `HSA_ENABLE_INTERRUPT=0`: Uses memory polling instead of interrupts for completion signals

### Environment Variable Summary

| Variable | Default | Purpose |
|----------|---------|---------|
| `AMD_LOG_LEVEL` | 0 | Log levels 0–5; higher numbers include more detail |
| `AMD_LOG_MASK` | 0x7FFFFFFF | Filters logs (API calls, kernels, synchronization, etc.) |
| `HIP_LAUNCH_BLOCKING` | 0 | Serialize kernel execution when set to 1 |
| `AMD_SERIALIZE_KERNEL` | 0 | Serialize kernel enqueue (1, 2, or 3) |
| `GPU_MAX_HW_QUEUES` | 4 | Maximum hardware queues per device |

## General Debugging Tips

- Use `gdb --args` to pass executable and arguments to GDB
- Set environment variables in GDB with `set env AMD_SERIALIZE_KERNEL 3` (no equals sign)
- Backtrace faults may appear in the runtime due to asynchronous GPU commands
- Enable `AMD_SERIALIZE_KERNEL=3` and `AMD_SERIALIZE_COPY=3` to force synchronous execution and pinpoint faults
- VM faults stem from incorrect code, memory issues, synchronization problems, or compiler/runtime bugs
