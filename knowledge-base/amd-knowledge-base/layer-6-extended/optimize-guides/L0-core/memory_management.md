---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "hip", "memory"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/how-to/hip_runtime_api/memory_management.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Memory Management

Memory management represents a crucial component of the HIP runtime API for developing high-performance applications. Both memory allocation and data transfers can create performance bottlenecks that significantly affect overall application performance.

## Overview

The programming framework assumes a system architecture with two distinct memory spaces: a host and a device, each maintaining separate memory. Device kernels operate within "Device memory" while host functions work with "Host memory."

The runtime provides essential functions for:
- Allocating and freeing device memory
- Copying device memory
- Transferring data between host and device spaces

## Memory Management Techniques

The following approaches are available:

- Coherence control
- Unified memory management
- Virtual memory management
- Stream Ordered Memory Allocator

## Memory Allocation Overview

The table below summarizes API calls and their corresponding allocations:

| API | Data Location | Allocation Type | System Allocated |
|-----|---------------|-----------------|------------------|
| `hipMallocManaged()` | Host | Managed | Host |
| `hipHostMalloc()` | Host | Pinned | Host |
| `hipMalloc()` | Device | Pinned | Device |

The host manages pageable memory by default, while pinned memory provides faster transfer rates. Managed memory offers automatic migration between host and device through unified memory mechanisms.
