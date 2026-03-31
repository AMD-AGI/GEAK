---
tags: ["optimization", "performance", "hip", "kernel", "graph-api", "streams"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/graph_api.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# HIP Graph API Tutorial

## Introduction

HIP graphs allow you to model GPU operations as nodes with dependency edges, enabling optimized batch execution. Rather than launching individual operations sequentially (like a movie director calling "action" for each shot), graphs pre-plan the entire sequence for a single coordinated launch.

### Modeling Dependencies

**Streams** organize operations sequentially in FIFO queues. Multiple streams can run independently, but manual synchronization is needed across stream boundaries. Each operation is scheduled independently, creating potential CPU overhead.

**Graphs** represent dependencies explicitly as nodes and edges. The runtime automatically inserts necessary synchronization, and launching all operations requires only one API call, dramatically reducing overhead. However, graphs require fixed workflows—the structure cannot change after instantiation.

### When to Use Graphs

Use graphs when:
- Workflow is fixed and repetitive
- Same kernels execute many times
- Launch overhead is significant (many small kernels)

Avoid graphs when:
- Workflow changes dynamically
- Operations are one-shot
- Kernels are long-running

### Application Context

This tutorial uses a CT reconstruction pipeline that processes X-ray projections into cross-sectional medical images. The fixed workflow makes this ideal for graph optimization since "CT scanners process hundreds of projections per scan."

## Prerequisites

**Required:** ROCm 6.2+, hipFFT library, understanding of HIP kernels, GPU memory management, streams/events, and CMake. At least 4 GiB GPU memory recommended.

**Optional:** FFT familiarity and medical imaging knowledge helpful but not essential.

## Building the Code

Clone the repository and navigate to the tutorial directory:

```bash
git clone https://github.com/ROCm/rocm-examples.git
cd rocm-examples/HIP-Doc/Tutorials/graph_api/
mkdir build && cd build
```

Configure with your GPU architecture:

```bash
cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_PLATFORM=amd \
  -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_C_COMPILER=amdclang \
  -DCMAKE_HIP_COMPILER=amdclang++ ..

cmake --build . --target hip_graph_api_tutorial_streams \
  hip_graph_api_tutorial_graph_capture hip_graph_api_tutorial_graph_creation
```

## Stream-Based Baseline

The baseline application (`main_streams.hip`) processes multiple projections in parallel batches. It queries device properties to determine available asynchronous engines, creating one stream per engine to maximize parallelism.

### Batched Processing

The application groups projections into parallel batches matching the stream count, ensuring every stream stays busy. Final synchronization occurs just once when "the first stream waits for all other streams to complete."

### Processing Pipeline

Each projection undergoes:

1. Normalization kernel
2. Logarithmic transformation
3. Pixel weighting
4. Memory expansion and clearing
5. Forward FFT (R2C)
6. Shepp-Logan filtering
7. Inverse FFT (C2R)
8. Shrinking and normalization
9. Back-projection into 3D volume

The back-projection kernel processes 512³ voxels (cubic complexity), making it the computational bottleneck.

### Trace Analysis

Using `rocprofv3`, visible gaps appear between operations, representing scheduling and launch overhead. Graph conversion eliminates these gaps.

## Converting to Graphs via Stream Capture

Stream capture records GPU operations into a graph without manual node creation. Call `hipStreamBeginCapture()` before operations and `hipStreamEndCapture()` after:

```cpp
auto graph = hipGraph_t{};
hip_check(hipStreamBeginCapture(streams.at(0), hipStreamCaptureModeGlobal));

// Processing pipeline executes normally
// ...

hip_check(hipStreamEndCapture(streams.at(0), &graph));
```

For multiple streams, use fork-join patterns with events to create dependencies. Once captured, instantiate the graph once and update it for subsequent batches using `hipGraphExecUpdate()`.

The capture variant produces a trace without visible gaps—operations execute as a cohesive block rather than individually scheduled.

## Manual Graph Creation

For fine-grained control, manually construct graphs by creating nodes and specifying dependencies. Kernel nodes require parameter arrays and dependency specifications:

```cpp
void* kernelParams[] = { /* pointers to arguments */ };
auto kernelNodeParams = hipKernelNodeParams{};
kernelNodeParams.blockDim = threadsPerBlock;
kernelNodeParams.func = reinterpret_cast<void*>(kernel_function);
kernelNodeParams.gridDim = blocksPerGrid;
kernelNodeParams.kernelParams = kernelParams;

auto kernelNode = hipGraphNode_t{};
hip_check(hipGraphAddKernelNode(
    &kernelNode, graph, &dependencyNode, 1, &kernelNodeParams
));
```

### Integrating hipFFT

Since hipFFT doesn't support manual node creation, use stream capture within the graph:

1. Record existing node set
2. Begin capture targeting the graph
3. Execute hipFFT operations
4. End capture
5. Find new nodes using set difference
6. Identify the leaf node for dependencies

Subsequent nodes then depend on the FFT leaf node.

## Performance Results

Both the capture and manual creation variants eliminate visible gaps in execution traces. "By capturing all operations of a batch into a single graph, you have successfully eliminated the launching and scheduling overhead previously observed in the stream-based variant."

Manual construction potentially offers better optimization opportunities, though at the cost of increased verbosity.

## Conclusion

Choose **stream capture** for quick conversions with minimal code changes. Choose **explicit construction** when maximum control and optimization are needed. Both approaches significantly reduce launch overhead for fixed, repetitive workflows like CT reconstruction.
