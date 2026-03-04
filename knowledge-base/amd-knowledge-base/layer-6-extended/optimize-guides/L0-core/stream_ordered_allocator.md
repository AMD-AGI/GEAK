---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "hip", "memory", "stream"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Stream Ordered Memory Allocator

## Overview

The Stream Ordered Memory Allocator (SOMA) is a HIP runtime API component that provides asynchronous memory allocation with stream-ordering semantics. It ensures that all asynchronous accesses occur between stream executions of allocation and deallocation operations, preventing use-before-allocation or use-after-free errors.

### Key Advantages

- **Efficient reuse**: Enables memory reuse across streams, reducing allocation overhead
- **Fine-grained control**: Allows setting attributes and controlling caching behavior for memory pools
- **Inter-process sharing**: Enables secure memory allocation sharing between processes
- **Driver optimizations**: Permits optimization based on awareness of SOMA and stream management APIs

### Key Disadvantages

- **Temporal constraints**: Requires strict adherence to stream order
- **Complexity**: Involves intricate memory management in stream order
- **Learning curve**: Requires additional effort to understand and utilize effectively

## Using SOMA

The primary functions for stream-ordered memory allocation are:

- `hipMallocAsync()`: Allocates memory with stream-ordered semantics
- `hipFreeAsync()`: Frees memory from the pool with stream-ordered semantics

Memory access restrictions ensure that asynchronous operations occur only between allocation and deallocation execution points on the same stream.

## Memory Pools

Memory pools manage memory with stream-ordered behavior while ensuring proper synchronization. They can be created, configured, and monitored for resource usage.

### Creating and Using Pools

Use `hipMemPoolCreate()` to create custom pools and `hipMallocFromPoolAsync()` to allocate from specific pools. Note that HIP requires explicit memory allocation per stream, unlike CUDA's implicit approach.

### Pool Trimming

Control memory usage by setting the `hipMemPoolAttrReleaseThreshold` attribute, which specifies reserved memory to retain in bytes. The allocator attempts to release excess memory during synchronization operations. Use `hipMemPoolTrimTo()` to reclaim memory and optimize usage.

### Resource Usage Statistics

Query pool attributes to monitor memory consumption:

- `hipMemPoolAttrReservedMemCurrent`: Total physical GPU memory currently held
- `hipMemPoolAttrUsedMemCurrent`: Total allocated memory size
- `hipMemPoolAttrReservedMemHigh`: Peak physical GPU memory held since last reset
- `hipMemPoolAttrUsedMemHigh`: Peak allocated memory size since last reset

Reset these attributes using `hipMemPoolSetAttribute()`.

### Memory Reuse Policies

Configure reuse behavior with these policy flags:

- `hipMemPoolReuseFollowEventDependencies`: Checks event dependencies before allocating additional GPU memory
- `hipMemPoolReuseAllowOpportunistic`: Checks if stream order semantics from free operations have been satisfied
- `hipMemPoolReuseAllowInternalDependencies`: Manages reuse based on internal runtime dependencies, searching for memory awaiting another stream's progress

### Multi-GPU Device Accessibility

Allocations are initially accessible only from their resident device but can be made accessible to other devices as needed.

## Interprocess Memory Handling

**Note**: IPC API calls require an active `amdgpu-dkms` driver.

Interprocess capable (IPC) memory pools enable secure GPU memory sharing between processes using either direct device pointers or shareable handles.

### Device Pointer Approach

- `hipMemPoolExportPointer()`: Exports a memory pool pointer for direct sharing
- `hipMemPoolImportPointer()`: Imports a shared memory pool pointer from another process

### Shareable Handle Approach

- `hipMemPoolExportToShareableHandle()`: Exports a pool to a shareable handle (file descriptor or inter-process handle)
- `hipMemPoolImportFromShareableHandle()`: Imports and restores a memory pool from a shareable handle

The shareable handle approach provides information about pool size, location, and other metadata, enabling memory sharing across different contexts.
