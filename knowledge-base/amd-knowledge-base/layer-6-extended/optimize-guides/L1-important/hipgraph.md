---
tags: ["optimization", "performance", "hip", "graphs", "workflow"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# HIP Graphs Documentation

## Overview

HIP graphs represent an alternative execution model that can improve performance compared to traditional stream-based kernel launching. A graph consists of nodes representing operations and edges denoting dependencies between them.

## Node Types

Supported node types include:
- Empty nodes
- Nested graphs
- Kernel launches
- Host-side function calls
- HIP memory operations (copy, memset, etc.)
- HIP events
- Semaphore signaling and waiting operations

## Performance Benefits

"The standard method of launching kernels incurs a small overhead for each iteration" which becomes significant in frameworks with multiple redirection layers. Graphs address this by predefining operations and dependencies, enabling the runtime to execute with "a single call" after initial setup.

## General Workflow

1. Create a `hipGraph_t` template using either stream capture or explicit creation
2. Instantiate into `hipGraphExec_t` via `hipGraphInstantiate()`
3. Launch using `hipGraphLaunch()`
4. Free resources after execution

Graphs require setup overhead, so they benefit workloads requiring multiple iterations.

## Memory Management

Memory in graphs can be pre-allocated or managed internally. "The lifetime of memory managed in a graph begins when the execution reaches the node allocating the memory, and ends when either reaching the corresponding free node within the graph, or after graph execution."

The runtime can maintain memory pools for reuse optimization within graphs.

## Stream Capture Method

Use `hipStreamBeginCapture()` and `hipStreamEndCapture()` to capture existing code:

```cpp
HIP_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));
// ... operations to capture ...
HIP_CHECK(hipStreamEndCapture(captureStream, &graph));
```

Asynchronous variants (`hipMallocAsync`, `hipMemcpyAsync`, `hipFreeAsync`) are required for capturing memory operations.

## Explicit Graph Creation

For fine-grained control, directly create nodes with parameters:

```cpp
HIP_CHECK(hipGraphCreate(&graph, 0));
HIP_CHECK(hipGraphAddKernelNode(&nodeHandle, graph, dependencies,
                                 depCount, &kernelParams));
```

Each node type requires specific parameter structures like `hipKernelNodeParams` for kernel launches.
