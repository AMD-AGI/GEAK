Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "hip", "cooperative", "kernel"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/how-to/hip_runtime_api/cooperative_groups.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Cooperative Groups

## Overview

The cooperative groups API extends HIP's programming model by providing developers with flexible, dynamic grouping mechanisms for thread communication. It enables custom thread group definitions that may better suit specific use cases than hardware-defined groups, allowing developers to specify communication granularity for more efficient parallel decompositions.

Access this API through the `cooperative_groups` namespace after including `hip_cooperative_groups.h`, which provides:

- Static functions for creating groups and subgroups
- Hardware-accelerated operations like shuffles
- Group data types
- Member functions for synchronization and property queries

## Cooperative Groups Thread Model

The thread hierarchy consists of nested levels:

**Grid Level**: The multi-grid represents potentially multiple simultaneous kernel launches across devices. A grid is a single kernel dispatch.

> "The ability to synchronize over a grid or multi grid requires the kernel to be launched using the specific cooperative groups API."

**Block Level**: Equivalent to the hierarchical thread model's block entity.

**Sub-block Level**: Introduces thread-block tiles and coalesced groups between block and individual threads.

## Group Types

### Thread-Block Group

Represents intra-block cooperative grouping where participating threads match the currently executing block.

```cpp
class thread_block;
thread_block g = this_thread_block();
```

Public members include: `group_index()`, `thread_index()`, `thread_rank()`, `size()`, `cg_type()`, `is_valid()`, `sync()`, and `group_dim()`.

### Grid Group

Represents inter-block grouping where threads span multiple blocks executing the same kernel on one device. Requires cooperative launch API.

```cpp
class grid_group;
grid_group g = this_grid();
```

Public members: `thread_rank()`, `size()`, `cg_type()`, `is_valid()`, `sync()`.

### Multi-Grid Group

Represents inter-device grouping spanning multiple devices running the same kernel.

```cpp
class multi_grid_group;
multi_grid_group g = this_multi_grid();
```

Public members: `num_grids()`, `grid_rank()`, `thread_rank()`, `size()`, `cg_type()`, `is_valid()`, `sync()`.

> "Multi grid deprecated since ROCm 5.0."

### Thread-Block Tile

Templated class defining compile-time tile size for sub-wave level operations.

```cpp
template <unsigned int Size, typename ParentT = void>
class thread_block_tile;

template <unsigned int Size, typename ParentT>
_CG_QUALIFIER thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g)
```

Requirements: Size must be a power of 2 and not exceed wavefront size. Supports shuffle and ballot operations for integer and float types.

### Coalesced Groups

Represents active threads within a warp when conditional branches cause thread divergence.

```cpp
class coalesced_group;
coalesced_group active = coalesced_threads();
```

> "AMD GPUs do not support independent thread scheduling. Some CUDA application can rely on this feature and the ported HIP version on AMD GPUs can deadlock."

## Code Example Comparison

**Traditional Block Model**:
```cpp
__device__ int reduce_sum(int *shared, int val) {
    const unsigned int thread_id = threadIdx.x;
    for(unsigned int i = blockDim.x / 2; i > 0; i /= 2) {
        shared[thread_id] = val;
        __syncthreads();
        if(thread_id < i)
            val += shared[thread_id + i];
        __syncthreads();
    }
}

__global__ void sum_kernel(...) {
    __shared__ unsigned int workspace[2048];
    output = reduce_sum(workspace, input);
}
```

**Cooperative Groups Model**:
```cpp
__device__ int reduce_sum(thread_group g, int *shared, int val) {
    const unsigned int group_thread_id = g.thread_rank();
    for(unsigned int i = g.size() / 2; i > 0; i /= 2) {
        shared[group_thread_id] = val;
        g.sync();
        if(group_thread_id < i)
            val += shared[group_thread_id + i];
        g.sync();
    }
}

__global__ void sum_kernel(...) {
    __shared__ unsigned int workspace[2048];
    thread_block thread_block_group = this_thread_block();
    output = reduce_sum(thread_block_group, workspace, input);
}
```

## Synchronization

### Device Capability Checks

**Grid Launch** (single GPU):
```cpp
int device = 0;
int supports_coop_launch = 0;
HIP_CHECK(hipGetDevice(&device));
HIP_CHECK(hipDeviceGetAttribute(&supports_coop_launch,
    hipDeviceAttributeCooperativeLaunch, device));
```

**Multi-Grid Launch** (multiple GPUs):
```cpp
for(int deviceID = 0; deviceID < device_count; deviceID++) {
    int supports_coop_launch = 0;
    HIP_CHECK(hipDeviceGetAttribute(&supports_coop_launch,
        hipDeviceAttributeCooperativeMultiDeviceLaunch, deviceID));
}
```

### Kernel Launch Methods

**Thread-Block** (standard launch):
```cpp
HIP_CHECK(hipLaunchKernelGGL(vector_reduce_kernel<partition_size>,
    dim3(num_blocks), dim3(threads_per_block), 0,
    hipStreamDefault, &d_vector, &d_block_reduced, &d_partition_reduced));
```

**Grid** (cooperative single-device):
```cpp
HIP_CHECK(hipLaunchCooperativeKernel(vector_reduce_kernel<partition_size>,
    dim3(num_blocks), dim3(threads_per_block), 0, 0, hipStreamDefault));
```

**Multi-Grid** (cooperative multi-device):
```cpp
HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList,
    (int)deviceIDs.size(), hipCooperativeLaunchMultiDeviceNoPreSync));
```

### Device-Side Synchronization

```cpp
// Thread-block
thread_block g = this_thread_block();
g.sync();

// Grid
grid_group grid = this_grid();
grid.sync();

// Multi-grid
multi_grid_group multi_grid = this_multi_grid();
multi_grid.sync();
```

## Unsupported NVIDIA CUDA Features

HIP does not support:

**Headers**:
- `cooperative_groups/memcpy_async.h`
- `cooperative_groups/reduce.h`
- `cooperative_groups/scan.h`

**Classes**: `cluster_group`

**Functions/Operators**: `synchronize`, `memcpy_async`, `wait`/`wait_prior`, `barrier_arrive`/`barrier_wait`, `invoke_one`/`invoke_one_broadcast`, `reduce`, `reduce_update_async`/`reduce_store_async`, reduce operators (`plus`, `less`, `greater`, `bit_and`, `bit_xor`, `bit_or`), `inclusive_scan`, `exclusive_scan`
