Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "3"
category: "communications"
tags: ["nccl", "multi-gpu", "distributed", "communication"]
cuda_version: "13.0+"
last_updated: 2025-11-17
---

# NCCL Usage Guide

*Nvidia Collective Communications Library for multi-GPU and multi-node training*

## Overview

NCCL (Nvidia Collective Communications Library) implements multi-GPU and multi-node collective communication primitives optimized for Nvidia GPUs and networking.

**Official Documentation**: [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)

## Installation

```bash
# Usually included with CUDA
ls /usr/lib/x86_64-linux-gnu/libnccl*

# Or install separately
sudo apt-get install libnccl2 libnccl-dev

# Check version
dpkg -l | grep nccl
```

## Basic Concepts

### Communicator

Group of participating processes:

```cpp
#include <nccl.h>

ncclComm_t comm;
int nranks = 8;  // Number of GPUs
int rank = 0;    // This GPU's rank

// Create communicator
ncclUniqueId id;
if (rank == 0) ncclGetUniqueId(&id);
// Broadcast id to all ranks...

ncclCommInitRank(&comm, nranks, id, rank);

// Use communicator...

ncclCommDestroy(comm);
```

### Collective Operations

| Operation | Description |
|-----------|-------------|
| `AllReduce` | Reduce and broadcast result to all |
| `Broadcast` | Send data from root to all |
| `Reduce` | Reduce data to single GPU |
| `AllGather` | Gather data from all GPUs |
| `ReduceScatter` | Reduce and scatter results |

## Common Operations

### AllReduce

Sum tensors across all GPUs:

```cpp
float *sendbuff, *recvbuff;
size_t count = 1024 * 1024;  // Number of elements

// Allocate buffers on each GPU
cudaMalloc(&sendbuff, count * sizeof(float));
cudaMalloc(&recvbuff, count * sizeof(float));

// AllReduce: sum across all GPUs
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);

// Result in recvbuff on all GPUs
```

### Broadcast

Send data from root to all ranks:

```cpp
float *buff;
cudaMalloc(&buff, count * sizeof(float));

// Rank 0 broadcasts to all
int root = 0;
ncclBroadcast(buff, buff, count, ncclFloat, root, comm, stream);
```

### AllGather

Gather tensors from all GPUs:

```cpp
float *sendbuff, *recvbuff;
cudaMalloc(&sendbuff, count * sizeof(float));
cudaMalloc(&recvbuff, count * nranks * sizeof(float));

// Each GPU contributes count elements
// recvbuff contains concatenated data from all GPUs
ncclAllGather(sendbuff, recvbuff, count, ncclFloat, comm, stream);
```

### ReduceScatter

Reduce and scatter results:

```cpp
float *sendbuff, *recvbuff;
cudaMalloc(&sendbuff, count * nranks * sizeof(float));
cudaMalloc(&recvbuff, count * sizeof(float));

// Reduce sendbuff and scatter to recvbuff
ncclReduceScatter(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);
```

## Multi-GPU Example

```cpp
#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define NRANKS 4

int main() {
    int size = 1024 * 1024;
    
    // Create NCCL communicator
    ncclComm_t comms[NRANKS];
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    
    for (int i = 0; i < NRANKS; i++) {
        cudaSetDevice(i);
        ncclCommInitRank(&comms[i], NRANKS, id, i);
    }
    
    // Allocate buffers on each GPU
    float **sendbuff = (float**)malloc(NRANKS * sizeof(float*));
    float **recvbuff = (float**)malloc(NRANKS * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(NRANKS * sizeof(cudaStream_t));
    
    for (int i = 0; i < NRANKS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&sendbuff[i], size * sizeof(float));
        cudaMalloc(&recvbuff[i], size * sizeof(float));
        cudaStreamCreate(&streams[i]);
    }
    
    // Initialize data on each GPU
    for (int i = 0; i < NRANKS; i++) {
        cudaSetDevice(i);
        cudaMemset(sendbuff[i], i + 1, size * sizeof(float));
    }
    
    // Launch AllReduce on all GPUs
    for (int i = 0; i < NRANKS; i++) {
        cudaSetDevice(i);
        ncclAllReduce(sendbuff[i], recvbuff[i], size, ncclFloat, 
                      ncclSum, comms[i], streams[i]);
    }
    
    // Wait for completion
    for (int i = 0; i < NRANKS; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
    
    // Cleanup
    for (int i = 0; i < NRANKS; i++) {
        cudaSetDevice(i);
        ncclCommDestroy(comms[i]);
        cudaFree(sendbuff[i]);
        cudaFree(recvbuff[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    free(sendbuff);
    free(recvbuff);
    free(streams);
    
    return 0;
}
```

Compile:
```bash
nvcc -lnccl multi_gpu.cu -o multi_gpu
```

## Python Usage (PyTorch)

NCCL is automatically used by PyTorch:

```python
import torch
import torch.distributed as dist

# Initialize process group with NCCL
dist.init_process_group(backend='nccl')

# Get rank and world size
rank = dist.get_rank()
world_size = dist.get_world_size()

# AllReduce
tensor = torch.randn(1000, 1000).cuda()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# Broadcast
if rank == 0:
    tensor = torch.randn(1000, 1000).cuda()
else:
    tensor = torch.zeros(1000, 1000).cuda()
dist.broadcast(tensor, src=0)

# AllGather
tensor_list = [torch.zeros(1000, 1000).cuda() for _ in range(world_size)]
tensor = torch.randn(1000, 1000).cuda()
dist.all_gather(tensor_list, tensor)
```

## Performance Optimization

### Use CUDA Streams

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// Launch NCCL operations on stream
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);

// Overlap with computation
kernel<<<grid, block, 0, stream>>>(args);

cudaStreamSynchronize(stream);
```

### Group Calls

Batch multiple operations:

```cpp
ncclGroupStart();

// Multiple operations execute as a group
ncclAllReduce(buff1, buff1, count, ncclFloat, ncclSum, comm, stream);
ncclAllReduce(buff2, buff2, count, ncclFloat, ncclSum, comm, stream);
ncclBroadcast(buff3, buff3, count, ncclFloat, 0, comm, stream);

ncclGroupEnd();
```

### Tuning Environment Variables

```bash
# Enable NVLink/NVSwitch
export NCCL_NET_GDR_LEVEL=5

# Tune buffer sizes
export NCCL_BUFFSIZE=8388608

# Enable debug logging
export NCCL_DEBUG=INFO

# Specify network interface
export NCCL_SOCKET_IFNAME=eth0

# Enable topology awareness
export NCCL_TOPO_FILE=/path/to/topo.xml
```

## Multi-Node Setup

### With MPI

```cpp
#include <mpi.h>
#include <nccl.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get NCCL unique ID from rank 0
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Create communicator
    ncclComm_t comm;
    cudaSetDevice(rank % 4);  // Assuming 4 GPUs per node
    ncclCommInitRank(&comm, size, id, rank);
    
    // NCCL operations...
    
    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
}
```

Compile and run:
```bash
mpicc -lcuda -lnccl multi_node.cu -o multi_node
mpirun -np 16 -hostfile hosts ./multi_node
```

### Environment Variables for Multi-Node

```bash
# InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0

# RoCE
export NCCL_IB_GID_INDEX=3

# TCP/IP fallback
export NCCL_SOCKET_IFNAME=eth0
```

## Debugging

### Enable Debugging

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### Check Topology

```python
import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl')

# Print NCCL version
print(f"NCCL version: {torch.cuda.nccl.version()}")

# Check if NCCL is available
print(f"NCCL available: {dist.is_nccl_available()}")
```

### Verify NVLink

```bash
# Check NVLink status
nvidia-smi nvlink --status

# Check topology
nvidia-smi topo -m
```

## Best Practices

1. **Use NCCL for multi-GPU**: Much faster than CPU-based communication
2. **Enable NVLink**: Set `NCCL_NET_GDR_LEVEL=5`
3. **Batch communications**: Use `ncclGroupStart/End`
4. **Overlap communication**: Use streams for async execution
5. **Check topology**: Ensure NVLink is properly configured

## Benchmarking

```cpp
// Measure AllReduce bandwidth
float *buff;
size_t sizes[] = {1<<20, 1<<22, 1<<24, 1<<26};  // 1MB to 64MB

for (int i = 0; i < 4; i++) {
    size_t size = sizes[i];
    cudaMalloc(&buff, size);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    ncclAllReduce(buff, buff, size/sizeof(float), ncclFloat, 
                  ncclSum, comm, stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    float bandwidth = (size * (nranks - 1) / nranks) / (ms / 1000) / 1e9;
    printf("Size: %zu, Bandwidth: %.2f GB/s\n", size, bandwidth);
    
    cudaFree(buff);
}
```

## External Resources

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [NCCL GitHub](https://github.com/NVIDIA/nccl)
- [NCCL Tests](https://github.com/NVIDIA/nccl-tests)
- [Multi-GPU Training Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html)

## Related Guides

- [FSDP Training](../../layer-5-llm/03-training/distributed/fsdp-training.md)
- [PyTorch with CUDA](../../layer-4-frameworks/pytorch/pytorch-cuda-basics.md)
- [Multi-GPU Best Practices](../../best-practices/performance/gpu-optimization.md)

