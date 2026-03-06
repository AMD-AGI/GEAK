Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "best-practices"
category: "performance"
tags: ["memory", "optimization", "cuda"]
cuda_version: "13.0+"
last_updated: 2025-11-17
---

# Memory Optimization for Nvidia GPUs

*Optimize memory usage and bandwidth on Nvidia GPUs*

## Memory Hierarchy

```
Registers:       Fastest, per-thread, limited
Shared Memory:   Fast, per-block, 48-164KB
L1/L2 Cache:     Fast, automatic
Global Memory:   Slow, large capacity, HBM/GDDR
```

## Reduce Memory Usage

### Use Lower Precision

```python
# FP16 instead of FP32 (50% memory)
model = model.half()

# INT8 quantization (75% memory)
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Trade compute for memory
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return self.layer3(x)
```

### Gradient Accumulation

```python
# Effective batch size = 4 * 8 = 32
for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / 4  # Normalize
    loss.backward()
    
    if (i + 1) % 4 == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Optimize Memory Bandwidth

### Coalesced Access

```cuda
// Good: Sequential access pattern
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    data[i] = i;
}

// Bad: Strided access
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    data[i * stride] = i;
}
```

### Use Shared Memory

```cuda
__shared__ float shared[256];

// Load once, use many times
shared[threadIdx.x] = global[idx];
__syncthreads();

// Multiple operations on shared data
float result = shared[threadIdx.x] * 2.0f;
result += shared[(threadIdx.x + 1) % 256];
```

## Monitor Memory Usage

```python
# PyTorch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# Reset peak stats
torch.cuda.reset_peak_memory_stats()

# Empty cache
torch.cuda.empty_cache()
```

## Best Practices

1. **Use pinned memory**: Faster transfers
2. **Minimize transfers**: Batch operations
3. **Use async copies**: Overlap with compute
4. **Profile memory access**: Check bandwidth utilization
5. **Use appropriate precision**: FP16/BF16/FP8

## External Resources

- [CUDA Memory Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)

## Related Guides

- [GPU Optimization](gpu-optimization.md)
- [Kernel Optimization](kernel-optimization.md)

