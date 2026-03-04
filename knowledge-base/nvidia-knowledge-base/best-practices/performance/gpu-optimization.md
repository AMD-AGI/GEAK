---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "best-practices"
category: "performance"
tags: ["optimization", "performance", "gpu", "best-practices"]
cuda_version: "13.0+"
last_updated: 2025-11-17
---

# GPU Performance Optimization Best Practices

*General principles for optimizing Nvidia GPU performance*

## Key Principles

### 1. Maximize Occupancy

- Use 256-512 threads per block
- Minimize register usage per thread
- Optimize shared memory usage
- Check occupancy with Nsight Compute

### 2. Coalesced Memory Access

```cuda
// Good: Coalesced access
__global__ void coalesced(float *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[tid];  // Consecutive threads access consecutive memory
}

// Bad: Strided access
__global__ void strided(float *data, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[tid * stride];  // Non-coalesced
}
```

### 3. Use Tensor Cores

- Enable TF32/FP16/FP8 for matrix operations
- Ensure dimensions are multiples of 8 (FP16) or 16 (INT8)
- Use cuBLAS/cuDNN when possible

### 4. Minimize Host-Device Transfers

```python
# Bad: Many small transfers
for i in range(1000):
    data_gpu = torch.tensor([i]).cuda()
    result = model(data_gpu)
    result_cpu = result.cpu()

# Good: Batch transfers
data_gpu = torch.arange(1000).cuda()
results = model(data_gpu)
results_cpu = results.cpu()
```

### 5. Async Operations

```python
# Use streams for concurrency
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    output1 = model1(input1)

with torch.cuda.stream(stream2):
    output2 = model2(input2)

torch.cuda.synchronize()
```

## Architecture-Specific Optimizations

### Volta/Turing (V100, T4)
- Enable Tensor Cores with FP16
- Use mixed precision training
- CUDA 10.0+

### Ampere (A100, A30, RTX 30-series)
- Enable TF32 (automatic)
- Use BF16 for training
- Leverage sparsity (2:4)
- Use MIG for multi-tenancy

### Hopper (H100, H200)
- Enable FP8 with Transformer Engine
- Use 4th gen Tensor Cores
- Maximize NVLink 4.0 bandwidth
- Consider DPX instructions

## Common Bottlenecks

### Memory Bandwidth Limited
- Use shared memory
- Increase arithmetic intensity
- Compress data
- Use mixed precision

### Compute Limited
- Use Tensor Cores
- Optimize algorithms
- Reduce unnecessary operations
- Enable compiler optimizations

### PCIe Limited
- Minimize transfers
- Use pinned memory
- Batch operations
- Use NVLink when available

## Profiling Workflow

1. **Profile with Nsight Systems**: Find bottlenecks
2. **Analyze with Nsight Compute**: Optimize kernels
3. **Monitor with nvidia-smi**: Track utilization
4. **Iterate**: Measure impact of changes

## Best Practices Checklist

- [ ] Use latest CUDA version
- [ ] Enable Tensor Cores (TF32/FP16/FP8)
- [ ] Use cuBLAS/cuDNN/NCCL
- [ ] Minimize host-device transfers
- [ ] Use async operations and streams
- [ ] Profile before optimizing
- [ ] Measure actual impact
- [ ] Check GPU utilization >80%

## External Resources

- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/)
- [Nsight Tools](https://developer.nvidia.com/tools-overview)

## Related Guides

- [Kernel Optimization](kernel-optimization.md)
- [Memory Optimization](memory-optimization.md)
- [CUDA Profiling](../../layer-2-compute-stack/cuda/cuda-profiling.md)

