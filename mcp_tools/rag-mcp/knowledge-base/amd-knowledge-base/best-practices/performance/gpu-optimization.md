---
layer: "best-practices"
category: "performance"
tags: ["optimization", "performance", "profiling"]
rocm_version: "7.0+"
last_updated: 2025-11-01
---

# GPU Performance Optimization Best Practices

## General Principles

### 1. Profile First, Optimize Second

```bash
# Profile with rocprof
rocprof --hip-trace python train.py

# Profile PyTorch
python -m torch.utils.bottleneck train.py

# Monitor GPU utilization
watch -n 1 rocm-smi
```

### 2. Memory Hierarchy Optimization

```
Registers (fastest, smallest)
    ↓
L1 Cache / LDS
    ↓
L2 Cache
    ↓
HBM (slowest, largest)
    ↓
CPU Memory (avoid if possible)
```

### 3. Compute vs Memory Bound

```python
# Check if memory or compute bound
import torch

# Compute bound: High arithmetic intensity
# Memory bound: Low arithmetic intensity

# Example: Large matrix multiply (compute bound)
A = torch.randn(4096, 4096, device='cuda')
B = torch.randn(4096, 4096, device='cuda')
C = torch.matmul(A, B)  # High FLOPS, good GPU utilization

# Example: Element-wise ops (memory bound)
x = torch.randn(1000000, device='cuda')
y = x + 1  # Low FLOPS, memory bandwidth limited
```

## HIP Kernel Optimization

### Coalesced Memory Access

```cpp
// Good: Coalesced access
__global__ void coalesced(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        float value = data[tid];  // Adjacent threads access adjacent memory
        data[tid] = value * 2.0f;
    }
}

// Bad: Strided access
__global__ void strided(float* data, int N, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        float value = data[tid * stride];  // Poor memory coalescing
        data[tid * stride] = value * 2.0f;
    }
}
```

### Shared Memory Usage

```cpp
#define TILE_SIZE 16

__global__ void tiled_matmul(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Compute partial product from shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Occupancy Optimization

```cpp
// Check occupancy
hipDeviceProp_t prop;
hipGetDeviceProperties(&prop, 0);

int maxThreadsPerBlock = prop.maxThreadsPerBlock;
int regsPerBlock = prop.regsPerBlock;
int sharedMemPerBlock = prop.sharedMemPerBlock;

// Use rocprof to check actual occupancy
// Aim for 50-100% occupancy for compute-bound kernels
```

## PyTorch Optimization

### Data Loading

```python
# Optimized DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=2,  # Prefetch batches
)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass with mixed precision
    with autocast(dtype=torch.bfloat16):  # Use BF16 on CDNA2+
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # Backward with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Gradient Accumulation

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps
    
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Efficient Memory Management

```python
# Use in-place operations when possible
x.add_(1)  # In-place
x = x + 1  # Creates new tensor

# Delete intermediate tensors
del intermediate_output
torch.cuda.empty_cache()

# Use torch.no_grad() for inference
with torch.no_grad():
    outputs = model(inputs)

# Gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x
```

## ROCm Library Optimization

### Use Optimized Libraries

```python
# PyTorch automatically uses rocBLAS for matrix ops
import torch

# This uses rocBLAS internally
C = torch.matmul(A, B)

# For custom code, link with ROCm libraries
# hipcc -lrocblas -lrocfft -lmiopen-hip
```

### Batch Operations

```python
# Batch matrix multiply (more efficient)
batch_size = 100
A = torch.randn(batch_size, 128, 128, device='cuda')
B = torch.randn(batch_size, 128, 128, device='cuda')
C = torch.bmm(A, B)  # Batched matmul
```

## Profiling and Debugging

### ROCm Profiler

```bash
# Basic profiling
rocprof python script.py

# Detailed trace
rocprof --hip-trace --stats python script.py

# Specific metrics
rocprof --stats --timestamp on python script.py

# Output to CSV
rocprof --stats --csv python script.py
```

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    model(inputs)

# Print results
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=10
))

# Export for visualization
prof.export_chrome_trace("trace.json")
# View in chrome://tracing
```

### Memory Profiling

```python
import torch

# Track memory allocation
torch.cuda.memory._record_memory_history()

# Your training code here
model(inputs)

# Generate memory snapshot
torch.cuda.memory._dump_snapshot("memory.pickle")

# Visualize with
# python -m torch.cuda._memory_viz trace_plot memory.pickle -o memory.html
```

## Benchmarking

### Proper Timing

```python
import torch
import time

# Warm up
for _ in range(10):
    output = model(input)

# Synchronize before timing
torch.cuda.synchronize()

# Timing
start = time.time()
for _ in range(100):
    output = model(input)
torch.cuda.synchronize()
end = time.time()

print(f"Average time: {(end - start) / 100 * 1000:.2f} ms")
```

### Throughput Measurement

```python
# Measure throughput
batch_size = 32
num_iterations = 100

start = time.time()
for _ in range(num_iterations):
    outputs = model(inputs)
    torch.cuda.synchronize()
elapsed = time.time() - start

throughput = (batch_size * num_iterations) / elapsed
print(f"Throughput: {throughput:.2f} samples/sec")
```

## Common Issues and Solutions

### 1. Low GPU Utilization

**Causes:**
- Small batch size
- CPU bottleneck in data loading
- Frequent CPU-GPU synchronization

**Solutions:**
```python
# Increase batch size
batch_size = 64  # or larger

# More data loading workers
num_workers = 8

# Avoid .item() in training loop
# Bad: loss.item() every iteration
# Good: Accumulate losses, print every N steps
```

### 2. Out of Memory

**Solutions:**
```python
# Reduce batch size
batch_size = 16

# Gradient accumulation
accumulation_steps = 4

# Gradient checkpointing
model.gradient_checkpointing_enable()

# Mixed precision
with autocast():
    outputs = model(inputs)

# Clear cache
torch.cuda.empty_cache()
```

### 3. Slow Data Loading

**Solutions:**
```python
# Pin memory
pin_memory = True

# More workers
num_workers = 8

# Prefetching
prefetch_factor = 2

# Persistent workers
persistent_workers = True
```

## Checklist

- [ ] Profile before optimizing
- [ ] Use mixed precision (BF16 on CDNA2+)
- [ ] Optimize data loading (pin_memory, num_workers)
- [ ] Maximize batch size
- [ ] Use gradient accumulation if needed
- [ ] Leverage ROCm libraries (rocBLAS, MIOpen)
- [ ] Minimize CPU-GPU transfers
- [ ] Use in-place operations when possible
- [ ] Enable gradient checkpointing for large models
- [ ] Monitor GPU utilization with rocm-smi
- [ ] Profile memory usage
- [ ] Benchmark properly (warm-up + synchronization)

## References

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [CDNA Architecture Guide](https://www.amd.com/en/technologies/cdna)
- [ROCm Performance and Optimization](https://rocm.docs.amd.com/en/latest/how-to/performance-optimization.html)

