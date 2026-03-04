---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "hip", "best-practices", "memory", "parallel-execution"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Performance Guidelines

## Overview

The AMD HIP performance guidelines offer best practices for optimizing applications on AMD GPUs. Four cornerstones form the foundation: parallel execution, memory bandwidth optimization, maximum throughput, and memory thrashing minimization.

## Parallel Execution

Effective performance requires revealing and efficiently providing substantial parallelism across three levels:

### Application Level
Use asynchronous calls and streams to enable parallel execution between host and devices. For thread synchronization within a block, employ `__syncthreads()`. For cross-block synchronization, use global memory with separate kernel invocations, though this adds overhead.

### Device Level
Optimize by executing multiple kernels concurrently. Streams manage kernel execution, overlapping computation and data transfers. Balance is essential—too many kernels cause resource contention.

### Multiprocessor Level
Maximize parallel execution within each multiprocessor by efficiently utilizing functional units. Ensure sufficient resident warps so instructions are ready every clock cycle, exploiting thread-level parallelism.

## Memory Throughput Optimization

### Data Transfer
Minimize host-device transfers by moving computations to the device. Batch small transfers into larger ones to reduce overhead. Use page-locked host memory when available. On integrated systems, use mapped page-locked memory to avoid explicit copying.

### Device Memory Access

Optimize memory access through:
- Coalescing thread accesses within warps into minimal transactions
- Following optimal access patterns
- Using properly sized and aligned data types
- Padding data when necessary

Global memory supports 1, 2, 4, 8, or 16-byte naturally aligned reads/writes. Misaligned access degrades performance.

**Key considerations:**
- 2D arrays should have widths as multiples of warp size
- Local memory resides in device memory (high latency)
- Shared memory offers higher bandwidth and lower latency with banking
- Texture and surface memory provide optimizations for 2D spatial locality

## Instruction Throughput Optimization

### Arithmetic Instructions
- Use efficient operations (multiplication faster than division)
- Leverage intrinsic functions
- Consider single-precision over double-precision when appropriate

### Control Flow
Minimize divergent warps caused by conditional statements. "The compiler might optimize loops, short ifs, or switch blocks using branch predication" to avoid unnecessary operations. Use `__builtin_expect` for predictable conditions.

### Synchronization
Use `__syncthreads()` to ensure all block threads reach the same code point. Consider streams for finer-grained execution control.

## Memory Thrashing Minimization

Optimize memory allocation patterns:
- Avoid allocating all available memory immediately
- Allocate suitably sized blocks early; deallocate only when finished
- Minimize `hipMalloc()` and `hipFree()` calls in performance-critical sections
- Consider `hipHostMalloc()` or `hipMallocManaged()` as alternatives
- Use `hipMallocManaged()` for supported platforms to enable oversubscription and reduce OS scheduler pressure
