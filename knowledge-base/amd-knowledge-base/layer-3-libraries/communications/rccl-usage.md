---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "3"
category: "communications"
subcategory: "multi-gpu"
tags: ["rccl", "multi-gpu", "distributed", "collective", "nccl"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
---

# RCCL Usage Guide

RCCL (ROCm Communication Collectives Library) enables multi-GPU and multi-node communication for distributed computing.

## Installation

```bash
# Ubuntu/Debian
sudo apt install rccl rccl-dev

# Verify
ls /opt/rocm/lib/librccl.so
```

## Basic Collective Operations

### AllReduce Example

```cpp
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int nGPUs;
    hipGetDeviceCount(&nGPUs);
    
    // Initialize RCCL communicators
    ncclComm_t* comms = new ncclComm_t[nGPUs];
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    
    // Create communicators for all GPUs
    ncclGroupStart();
    for (int i = 0; i < nGPUs; i++) {
        hipSetDevice(i);
        ncclCommInitRank(&comms[i], nGPUs, id, i);
    }
    ncclGroupEnd();
    
    // Allocate data on each GPU
    const int N = 1024;
    float **d_data = new float*[nGPUs];
    hipStream_t* streams = new hipStream_t[nGPUs];
    
    for (int i = 0; i < nGPUs; i++) {
        hipSetDevice(i);
        hipMalloc(&d_data[i], N * sizeof(float));
        hipStreamCreate(&streams[i]);
        
        // Initialize with GPU index
        std::vector<float> h_data(N, i);
        hipMemcpy(d_data[i], h_data.data(), N * sizeof(float), 
                  hipMemcpyHostToDevice);
    }
    
    // AllReduce: Sum across all GPUs
    ncclGroupStart();
    for (int i = 0; i < nGPUs; i++) {
        hipSetDevice(i);
        ncclAllReduce(d_data[i], d_data[i], N, ncclFloat, ncclSum,
                      comms[i], streams[i]);
    }
    ncclGroupEnd();
    
    // Wait for completion
    for (int i = 0; i < nGPUs; i++) {
        hipStreamSynchronize(streams[i]);
    }
    
    // Verify results
    std::vector<float> result(N);
    hipSetDevice(0);
    hipMemcpy(result.data(), d_data[0], N * sizeof(float),
              hipMemcpyDeviceToHost);
    
    // Each element should be sum of all GPU indices
    float expected = 0;
    for (int i = 0; i < nGPUs; i++) expected += i;
    
    std::cout << "Expected: " << expected 
              << ", Got: " << result[0] << std::endl;
    
    // Cleanup
    for (int i = 0; i < nGPUs; i++) {
        hipSetDevice(i);
        hipFree(d_data[i]);
        hipStreamDestroy(streams[i]);
        ncclCommDestroy(comms[i]);
    }
    
    delete[] d_data;
    delete[] streams;
    delete[] comms;
    
    return 0;
}
```

Compile:
```bash
hipcc -I/opt/rocm/include rccl_example.cpp -lrccl -o rccl_example
```

## Collective Operations

### 1. AllReduce

```cpp
// Sum, Min, Max, Prod across all GPUs
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);
```

### 2. Broadcast

```cpp
// Broadcast from root to all GPUs
ncclBroadcast(sendbuff, recvbuff, count, ncclFloat, root, comm, stream);
```

### 3. Reduce

```cpp
// Reduce to single GPU
ncclReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, root, comm, stream);
```

### 4. AllGather

```cpp
// Gather from all GPUs to all GPUs
ncclAllGather(sendbuff, recvbuff, sendcount, ncclFloat, comm, stream);
```

### 5. ReduceScatter

```cpp
// Reduce and scatter results
ncclReduceScatter(sendbuff, recvbuff, recvcount, ncclFloat, ncclSum, 
                  comm, stream);
```

## PyTorch DDP with RCCL

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    # Initialize process group (uses RCCL on AMD GPUs)
    dist.init_process_group(
        backend='nccl',  # RCCL on AMD, NCCL on NVIDIA
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    # Create model and wrap with DDP
    model = nn.Linear(10, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Training loop
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    
    for epoch in range(10):
        # Forward pass
        outputs = ddp_model(torch.randn(20, 10).to(rank))
        loss = outputs.sum()
        
        # Backward (gradients sync via RCCL AllReduce)
        optimizer.zero_grad()
        loss.backward()  # RCCL communication here
        optimizer.step()
        
        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    cleanup()

# Launch with torchrun
# torchrun --nproc_per_node=4 train.py
```

## Multi-Node Training

```python
# Launch on each node:
# Node 0 (master):
# torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
#          --master_addr="192.168.1.1" --master_port=29500 train.py

# Node 1:
# torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
#          --master_addr="192.168.1.1" --master_port=29500 train.py

import os
import torch.distributed as dist

def setup_multinode():
    # Get rank and world_size from environment
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Initialize
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size
```

## Performance Optimization

### 1. Overlapping Communication and Computation

```python
# Enable bucketing for gradient communication
model = DDP(
    model, 
    device_ids=[rank],
    bucket_cap_mb=25,  # Adjust bucket size
    gradient_as_bucket_view=True  # Reduce memory copies
)
```

### 2. Use Async Operations

```cpp
// Launch multiple operations without blocking
ncclGroupStart();
ncclAllReduce(..., comm0, stream0);
ncclAllReduce(..., comm1, stream1);
ncclAllReduce(..., comm2, stream2);
ncclGroupEnd();

// Overlap with computation
my_kernel<<<...>>>(data);

// Wait for completion
hipStreamSynchronize(stream0);
```

### 3. Tune RCCL Environment Variables

```bash
# Enable tree algorithm for large messages
export NCCL_ALGO=Tree

# Tune buffer sizes
export NCCL_BUFFSIZE=8388608

# Enable IPC for intra-node communication
export NCCL_IPC_ENABLE=1

# Set network interface (multi-node)
export NCCL_SOCKET_IFNAME=eth0

# Debug information
export NCCL_DEBUG=INFO
```

## Benchmarking

```bash
# Install RCCL tests
git clone https://github.com/ROCmSoftwarePlatform/rccl-tests.git
cd rccl-tests
make

# Run all-reduce benchmark
./build/all_reduce_perf -b 8 -e 1G -f 2 -g 4

# Output shows bandwidth and bus bandwidth
# Example: 4 GPUs, 1GB data, ~100 GB/s
```

## Troubleshooting

### Check GPU Topology

```bash
# See GPU interconnects
rocm-smi --showtoponuma

# Check for PCIe/Infinity Fabric links
rocm-smi --showtopo
```

### Common Issues

```bash
# Issue: Hang during AllReduce
# Solution: Check all ranks call collective
# All GPUs must participate!

# Issue: Slow performance
# Solution: Check interconnect topology
# Prefer Infinity Fabric over PCIe

# Issue: "Too many communicators"
# Solution: Reuse communicators, don't recreate
```

## References

- [RCCL Documentation](https://rocm.docs.amd.com/projects/rccl/en/latest/)
- [RCCL GitHub](https://github.com/ROCmSoftwarePlatform/rccl)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

