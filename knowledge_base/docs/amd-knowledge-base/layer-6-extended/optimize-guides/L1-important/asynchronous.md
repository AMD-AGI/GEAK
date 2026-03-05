---
tags: ["optimization", "performance", "hip", "asynchronous", "streams", "concurrent-execution"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Asynchronous Concurrent Execution in HIP

## Overview

Asynchronous concurrent execution is essential for maximizing GPU performance through techniques like overlapping computation with data transfer, managing concurrent kernels with streams across single or multiple devices, and utilizing HIP graphs.

## Streams and Concurrent Execution

All asynchronous operations—kernel execution, data movement, and memory allocation/deallocation—occur within device streams. "Streams are FIFO buffers of commands to execute in order on a given device." Commands queued on streams return immediately, executing asynchronously. Multiple streams can target the same device, and commands across different streams are not guaranteed to execute sequentially.

### Managing Streams

Stream creation functions provide different levels of control:

- **hipStreamCreate()**: Establishes a stream with default settings
- **hipStreamCreateWithFlags()**: Creates a stream with specific flags:
  - `hipStreamDefault`: Standard blocking stream
  - `hipStreamNonBlocking`: Permits concurrent task execution without blocking dependencies
- **hipStreamCreateWithPriority()**: Enables priority-based stream creation

The `hipStreamSynchronize()` function blocks the host thread until all queued tasks in a stream complete.

### Host-Device Concurrency

"Kernels are launched asynchronously using `hipLaunchKernelGGL` or using the triple chevron with a stream, enabling the CPU to continue executing other code while the GPU processes the kernel." Memory operations like `hipMemcpyAsync()` similarly execute asynchronously without blocking the CPU.

### Concurrent Kernel Execution

Multiple kernels can run simultaneously on the GPU when sufficient registers and shared memory are available. Developers may need to reduce block sizes to enable concurrent execution. Dependencies between kernels are managed via `hipStreamWaitEvent()`. Note that concurrent execution is most beneficial when kernels underutilize GPU resources; full utilization can increase execution time due to resource contention.

## Data Transfer and Kernel Overlapping

Asynchronous operations enable overlapping data transfers with kernel execution for better resource utilization. This is particularly advantageous in iterative processes where input preparation doesn't depend on previous kernel results.

### Device Capabilities

Applications can check the `asyncEngineCount` device property to determine concurrent data transfer support. Devices with values greater than zero support asynchronous copies. Page-locked (pinned) host memory optimizes bandwidth for host-device transfers.

### Asynchronous Memory Operations

Functions like `hipMemcpyAsync()` and `hipMemcpyPeerAsync()` enable non-blocking data transfers. "This overlap of computation and data transfer ensures that the GPU is not idle while waiting for data." Multi-GPU communication is supported through peer-to-peer asynchronous transfers.

### Intra-Device Copies

Devices supporting `concurrentKernels` can perform simultaneous on-device copies and kernel execution. Those supporting `asyncEngineCount` enable concurrent GPU data transfers alongside kernel work.

## Synchronization and Event Management

### Synchronous Calls

Synchronous operations like `hipMemcpy()` block until completion. "When a synchronous function is called, control is not returned to the host thread before the device has completed the requested task." Host behavior during synchronization is configurable via `hipSetDeviceFlags()`.

### Event-Based Synchronization

Events coordinate operations across streams:
- `hipEventCreate()`: Establishes an event
- `hipEventRecord()`: Records an event on a stream
- `hipEventSynchronize()`: Waits for event completion

### Stream Dependencies

While CUDA supports programmatic dependent launches, HIP achieves similar results using streams and events. "By employing `hipStreamWaitEvent()`, it is possible to manage the execution order without explicit hardware support."

## Code Examples

The documentation provides three implementation patterns:

**Sequential**: Blocking memory transfers and kernel launches with full synchronization between operations

**Asynchronous**: Non-blocking operations across multiple streams, enabling data transfer and kernel overlap with explicit stream synchronization points

**hipStreamWaitEvent**: Event-based dependencies allowing fine-grained control over kernel launch ordering while maintaining concurrent execution

## HIP Graphs

HIP graphs provide an optimized framework comprising operation nodes and dependency edges. "By representing sequences of kernels and memory operations as a single graph, they simplify complex workflows and enhance performance, particularly for applications with intricate dependencies and multiple execution stages."
