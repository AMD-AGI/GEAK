---
tags: ["optimization", "performance", "hip", "memory", "throughput"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/how-to/performance_guidelines.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Performance Guidelines

## Overview

The AMD HIP performance guidelines provide best practices for optimizing application performance on AMD GPUs. Four main cornerstones support this optimization:

- Parallel execution
- Memory bandwidth usage optimization
- Maximum throughput optimization
- Memory thrashing minimization

## Parallel Execution

Effective performance requires revealing and efficiently providing maximum parallelism across application, device, and multiprocessor levels.

### Application Level

Enable parallel execution across host and devices using asynchronous calls and streams. Assign workloads based on efficiency—serial tasks to the host, parallel tasks to devices.

For thread synchronization within blocks, use `__syncthreads()`. For threads in different blocks, use global memory with separate kernel invocations, though this approach adds overhead and should be avoided when possible.

### Device Level

Maximize parallel execution across multiprocessors by running multiple kernels concurrently. Streams facilitate overlapping computation and data transfers while keeping multiprocessors busy. Balance is essential—too many kernels cause resource contention and performance degradation.

### Multiprocessor Level

Maximize execution within each multiprocessor by efficiently utilizing functional units. Ensure sufficient resident warps so instructions are ready for execution each clock cycle, exploiting both instruction-level and thread-level parallelism.

## Memory Throughput Optimization

Minimize low-bandwidth host-to-device transfers and maximize on-chip memory usage (shared memory and caches) while minimizing global memory transfers.

### Data Transfer Strategy

- Move computations from host to device, even when kernels don't fully utilize device parallelism
- Create, use, and discard intermediate data structures in device memory without copying to host
- Batch small transfers into larger transfers to reduce overhead
- Use page-locked host memory on systems with front-side buses
- Employ mapped page-locked memory for implicit data transfers with coalesced access patterns
- On integrated systems, use mapped page-locked memory since host and device memory are physically identical

### Device Memory Access

Optimize throughput by:

- Coalescing memory accesses of warp threads into minimal transactions
- Following optimal access patterns
- Using properly sized and aligned data types
- Padding data when necessary

Memory transactions require natural alignment (32-, 64-, or 128-byte). Global memory instructions support 1, 2, 4, 8, or 16-byte reads/writes with natural alignment. Misaligned access triggers multiple instructions, reducing performance.

For 2D array access (address: `BaseAddress + xIndex + width * yIndex`), align array and thread block widths to warp size. Pad rows if width isn't a multiple of warp size.

**Local Memory:** Organized for consecutive threads accessing consecutive 32-bit words, enabling full coalescing when warp threads access the same relative address.

**Shared Memory:** On-chip memory with high bandwidth and low latency, divided into banks for simultaneous access. Bank conflicts—where addresses map to the same bank—serialize access and reduce throughput.

**Constant Memory:** Cached in constant cache; requests split by address and serviced through cache hits or device memory throughput.

**Texture/Surface Memory:** Cached in texture cache, optimizing 2D spatial locality. Benefits include higher local bandwidth, offloaded addressing, data broadcasting, and optional 8/16-bit to 32-bit float conversion.

## Optimization for Maximum Instruction Throughput

Maximize throughput by minimizing low-throughput arithmetic instructions, reducing warp divergence from control flow, and maximizing instruction parallelism.

### Arithmetic Instructions

- Use efficient operations (multiplication faster than division; integer faster than floating-point)
- Trade precision for speed when appropriate
- Leverage intrinsic functions available in HIP
- Optimize memory access patterns

### Control Flow Instructions

Control flow (`if`, `else`, `for`, `do`, `while`, `break`, `continue`, `switch`) can cause warp divergence and reduce throughput. Minimize divergence by:

- Writing conditions that don't cause thread divergence
- Using conditions based on `threadIdx` or `warpSize` (compiler-optimizable)
- Leveraging branch predication for short conditionals and loops
- Using `__builtin_expect` or `[[likely]]` annotations for predictable conditions

#### Avoiding Divergent Warps

Warp divergence occurs when threads follow different execution paths due to conditional statements. This significantly reduces instruction throughput; structure code to minimize divergence.

### Synchronization

`__syncthreads()` synchronizes all block threads, ensuring they reach the same code point and can access shared memory afterward. However, synchronization causes performance overhead as threads wait, potentially leaving GPU resources idle.

Alternatively, use streams for fine-grained execution order control, allowing concurrent command execution.

## Minimizing Memory Thrashing

Applications frequently allocating and freeing memory experience slower allocation calls over time as memory fragments. Optimize by:

- Avoiding allocation of all available memory with `hipMalloc()` or `hipHostMalloc()`, as this reserves memory and strains OS schedulers
- Allocating suitably sized blocks early and deallocating only when unnecessary; minimize `hipMalloc()` and `hipFree()` calls in performance-critical sections
- Considering `hipHostMalloc()` or `hipMallocManaged()` alternatives if insufficient device memory exists
- Using `hipMallocManaged()` on supported platforms for oversubscription capability with performance approaching `hipMalloc()` when using appropriate policies
