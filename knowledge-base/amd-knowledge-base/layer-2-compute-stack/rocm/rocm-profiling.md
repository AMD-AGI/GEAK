---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "2"
category: "rocm"
subcategory: "profiling"
tags: ["rocm", "profiling", "rocprof", "roctracer", "performance", "optimization"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
---

# ROCm Profiling and Performance Analysis

Comprehensive guide to profiling AMD GPU applications using ROCm tools.

## ROCm Profiling Tools

### 1. rocprof - ROCm Profiler

rocprof is the primary profiling tool for ROCm applications.

#### Basic Usage

```bash
# Profile an application
rocprof ./my_app

# Output: results.csv with performance metrics

# Profile with specific metrics
rocprof --stats ./my_app

# Profile HIP kernels only
rocprof --hip-trace ./my_app

# Profile with timestamps
rocprof --timestamp on ./my_app
```

#### Advanced Profiling

```bash
# Collect hardware counters
rocprof --hsa-trace ./my_app

# Profile specific metrics
rocprof -i input.txt ./my_app

# input.txt content:
# pmc : SQ_WAVES Wavefronts
# pmc : SQ_INSTS_VALU VALU Instructions
# pmc : TCC_HIT[0-15] L2 Cache Hits
```

#### Common Metrics

```bash
# GPU Utilization
rocprof --stats \
  -i metrics.txt \
  ./my_app

# metrics.txt:
pmc : GPUBusy              # GPU busy percentage
pmc : Wavefronts           # Number of wavefronts
pmc : VALUInsts            # VALU instructions
pmc : SALUInsts            # SALU instructions  
pmc : VALUUtilization      # VALU utilization
pmc : MemUnitBusy          # Memory unit busy
pmc : L2CacheHit           # L2 cache hit rate
pmc : WriteSize            # Memory write size
pmc : FetchSize            # Memory fetch size
```

### 2. roctracer - API Tracing

roctracer captures API calls and kernel execution.

```bash
# Trace HIP API calls
roctracer -t ./my_app

# Trace with JSON output
roctracer --json-trace ./my_app

# Trace specific APIs
export HSA_TOOLS_LIB=/opt/rocm/lib/libroctracer64.so
export ROCTRACER_DOMAIN="hip:hcc"
./my_app
```

#### Analyzing Traces

```python
import json
import pandas as pd

# Load trace
with open('results.json') as f:
    trace = json.load(f)

# Analyze kernel times
kernels = trace['traceEvents']
df = pd.DataFrame(kernels)

# Find slow kernels
slow_kernels = df[df['dur'] > 1000].sort_values('dur', ascending=False)
print(slow_kernels[['name', 'dur', 'args']])
```

### 3. Performance Analysis Workflow

#### Step 1: Collect Profile Data

```bash
#!/bin/bash
# profile.sh - Comprehensive profiling script

APP=$1
OUTPUT_DIR="profile_results"
mkdir -p $OUTPUT_DIR

# 1. Basic stats
echo "Collecting basic stats..."
rocprof --stats -o $OUTPUT_DIR/stats.csv $APP

# 2. Kernel trace
echo "Collecting kernel trace..."
rocprof --hip-trace -o $OUTPUT_DIR/kernel_trace.csv $APP

# 3. Memory access patterns
echo "Collecting memory metrics..."
rocprof -i memory_metrics.txt -o $OUTPUT_DIR/memory.csv $APP

# 4. API trace
echo "Collecting API trace..."
roctracer -o $OUTPUT_DIR/api_trace.json $APP

echo "Profiling complete. Results in $OUTPUT_DIR/"
```

#### Step 2: Analyze Bottlenecks

```python
#!/usr/bin/env python3
# analyze_profile.py

import pandas as pd
import matplotlib.pyplot as plt

# Load profiling data
stats = pd.read_csv('profile_results/stats.csv')
kernel_trace = pd.read_csv('profile_results/kernel_trace.csv')

# Find top time consumers
print("=== Top 10 Kernels by Duration ===")
top_kernels = kernel_trace.nlargest(10, 'Duration(ns)')
print(top_kernels[['Name', 'Duration(ns)', 'Grid', 'Workgroup']])

# Calculate occupancy
def calculate_occupancy(workgroup_size, num_workgroups):
    max_waves_per_cu = 40  # MI250X
    waves = (workgroup_size + 63) // 64  # Wave size = 64
    occupancy = min(100, (waves / max_waves_per_cu) * 100)
    return occupancy

kernel_trace['Occupancy'] = kernel_trace.apply(
    lambda row: calculate_occupancy(
        row['Workgroup'][0] * row['Workgroup'][1] * row['Workgroup'][2],
        row['Grid'][0] * row['Grid'][1] * row['Grid'][2]
    ), axis=1
)

# Find low occupancy kernels
low_occ = kernel_trace[kernel_trace['Occupancy'] < 50]
print("\n=== Kernels with Low Occupancy (<50%) ===")
print(low_occ[['Name', 'Occupancy', 'Workgroup']])

# Memory bandwidth utilization
mem_bw_peak = 1600  # GB/s for MI250X (per GCD)
mem_bw_achieved = stats['MemoryBandwidth'].mean()
mem_efficiency = (mem_bw_achieved / mem_bw_peak) * 100
print(f"\n=== Memory Bandwidth ===")
print(f"Achieved: {mem_bw_achieved:.2f} GB/s")
print(f"Peak: {mem_bw_peak} GB/s")
print(f"Efficiency: {mem_efficiency:.2f}%")
```

## Common Performance Issues

### 1. Low GPU Utilization

**Symptoms:**
- GPU busy < 80%
- Kernel launch overhead visible

**Solutions:**

```cpp
// BAD: Launching many small kernels
for (int i = 0; i < 1000; i++) {
    small_kernel<<<1, 64>>>(data + i * 64);
}

// GOOD: Batch into larger kernels
large_kernel<<<16, 64>>>(data, 1000);

// BETTER: Use streams for overlap
for (int i = 0; i < 4; i++) {
    hipMemcpyAsync(..., streams[i]);
    kernel<<<..., streams[i]>>>(...);
}
```

### 2. Memory Bandwidth Bottleneck

**Symptoms:**
- MemUnitBusy > 80%
- Low arithmetic intensity

**Solutions:**

```cpp
// BAD: Uncoalesced memory access
__global__ void uncoalesced(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Stride access - bad!
    data[idx * 128] = idx;
}

// GOOD: Coalesced memory access
__global__ void coalesced(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Sequential access - good!
    data[idx] = idx;
}

// BETTER: Use shared memory
__global__ void tiled(float* data) {
    __shared__ float tile[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    tile[threadIdx.x] = data[idx];
    __syncthreads();
    
    // Process from shared memory
    float result = tile[threadIdx.x] * 2.0f;
    __syncthreads();
    
    data[idx] = result;
}
```

### 3. Low Occupancy

**Symptoms:**
- Occupancy < 50%
- High register usage or shared memory

**Solutions:**

```bash
# Check resource usage
rocprof --stats ./my_app | grep -i "occupancy"

# Reduce register usage
hipcc -O3 --maxrregcount=64 kernel.cpp

# Reduce shared memory per block
# Adjust block size to fit more waves per CU
```

## Profiling PyTorch Models

```python
import torch
import torch.profiler as profiler

model = MyModel().to('cuda')
inputs = torch.randn(32, 3, 224, 224).to('cuda')

# Profile with ROCm
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,  # Uses HIP on ROCm
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    model(inputs)

# Print results
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=10
))

# Export to Chrome trace
prof.export_chrome_trace("trace.json")
```

## Advanced: Custom Metrics

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_profile.h>

// Manual instrumentation
void profile_kernel() {
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    my_kernel<<<grid, block>>>();
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    
    printf("Kernel time: %.3f ms\n", milliseconds);
    
    hipEventDestroy(start);
    hipEventDestroy(stop);
}
```

## Optimization Checklist

- [ ] Profile baseline performance with rocprof
- [ ] Identify top time-consuming kernels (top 20%)
- [ ] Check GPU utilization (target > 80%)
- [ ] Analyze memory bandwidth (compare to peak)
- [ ] Check occupancy (target > 50%)
- [ ] Look for uncoalesced memory accesses
- [ ] Profile memory allocations (hipMalloc overhead)
- [ ] Check for CPU-GPU synchronization points
- [ ] Analyze kernel launch overhead
- [ ] Verify efficient use of shared memory
- [ ] Check register pressure and spilling
- [ ] Profile multi-GPU communication (if applicable)

## Best Practices

1. **Always profile before optimizing**
   - Measure, don't guess
   - Focus on real bottlenecks

2. **Use appropriate tools**
   - rocprof for kernel performance
   - roctracer for API overhead
   - PyTorch profiler for DL models

3. **Iterate systematically**
   - Fix one bottleneck at a time
   - Measure after each change
   - Keep detailed notes

4. **Consider roofline analysis**
   - Compare achieved vs peak performance
   - Identify compute vs memory bound

## References

- [rocprof Documentation](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
- [roctracer Documentation](https://rocm.docs.amd.com/projects/roctracer/en/latest/)
- [AMD GPU Performance Tuning](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides.html)

