Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "hip", "memory", "host-memory", "pinned-memory"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/host_memory.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Host Memory

Host memory represents the standard RAM memory on a computer system. The documentation describes two allocation approaches:

## Overview

Host memory can be allocated through two methods: **pageable memory** and **pinned memory**. Each approach offers different performance characteristics and trade-offs.

The guide illustrates how data transfers differ between these memory types, with pageable memory requiring an intermediate step during transfers while pinned memory enables direct device access.

## Pageable Memory

Pageable memory exists in system RAM blocks that can be relocated to other storage locations, such as swap partitions when RAM becomes saturated. Standard C++ allocation functions like `malloc` and `new` create pageable memory.

The documentation provides a code example demonstrating typical usage patterns: allocating arrays with `new`, transferring data to device memory via `hipMemcpy()`, and freeing resources with `delete[]`.

## Pinned Memory

Pinned (or page-locked) memory resides in fixed RAM locations that cannot migrate. While device kernels can access pinned host memory directly, this approach generally degrades performance due to PCIe bandwidth limitations.

The primary advantage involves transfer speeds: "using pinned memory instead of pageable memory on the host can lead to a three times improvement in bandwidth" during copy operations.

The trade-off comes from reduced available system RAM, potentially affecting overall host performance.

### Allocation Flags

The `hipHostMalloc()` function supports several flags controlling memory behavior:

- **hipHostMallocPortable**: Allocation usable across multiple contexts
- **hipHostMallocMapped**: Enables device pointer retrieval via `hipHostGetDevicePointer()`
- **hipHostMallocNumaUser**: Follows user-specified NUMA policies
- **hipHostMallocWriteCombined**: Optimizes PCIe transfers at cost of read efficiency
- **hipHostMallocCoherent/NonCoherent**: Controls memory coherence behavior

These flags can combine freely except when specifying both coherence options simultaneously.
