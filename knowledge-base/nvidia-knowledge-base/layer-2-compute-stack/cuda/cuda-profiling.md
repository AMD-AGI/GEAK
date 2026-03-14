---
layer: "2"
category: "cuda"
subcategory: "profiling"
tags: ["cuda", "profiling", "performance", "optimization", "nsight"]
cuda_version: "13.0+"
cuda_verified: "13.0"
last_updated: 2025-11-17
---

# CUDA Profiling and Performance Analysis

*Comprehensive guide to profiling CUDA applications and optimizing performance*

## Overview

Profiling is essential for understanding GPU application performance and identifying bottlenecks. Nvidia provides several tools for profiling CUDA applications.

**Official Documentation**: [Nvidia Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)

## Profiling Tools Overview

| Tool | Purpose | Best For |
|------|---------|----------|
| **Nsight Systems** | System-wide performance | Finding bottlenecks, CPU-GPU interaction |
| **Nsight Compute** | Kernel-level analysis | Optimizing individual kernels |
| **nvidia-smi** | Real-time monitoring | Resource utilization, multi-GPU |
| **nvprof** | Legacy profiler | Older CUDA versions (deprecated) |

## Nsight Systems

System-wide profiler for understanding application behavior.

### Installation

```bash
# Usually included with CUDA Toolkit
# Or download separately
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_5/nsight-systems-2024.5.1_2024.5.1.106-1_amd64.deb
sudo dpkg -i nsight-systems-*.deb
```

### Basic Usage

```bash
# Profile application
nsys profile --stats=true ./my_app

# With detailed output
nsys profile --stats=true --force-overwrite true -o report ./my_app

# Profile Python application
nsys profile python train.py

# Open in GUI
nsys-ui report.nsys-rep
```

### Command-Line Options

```bash
# Trace CUDA API calls
nsys profile --trace=cuda ./app

# Trace CUDA and NVTX markers
nsys profile --trace=cuda,nvtx ./app

# Set sampling rate
nsys profile --sample=cpu --cpuctxsw=false ./app

# Profile for specific duration
nsys profile --duration=30 ./app

# Export to different formats
nsys profile --export=sqlite ./app
```

### NVTX Markers

Add custom markers to your code:

```cpp
#include <nvtx3/nvToolsExt.h>

void process_data() {
    // Start range
    nvtxRangePush("Process Data");
    
    // Your code here
    kernel<<<grid, block>>>(args);
    
    // End range
    nvtxRangePop();
}

// Named ranges with colors
void train_step() {
    nvtxRangePushEx({
        .message = "Training Step",
        .color = 0xFF00FF00  // Green
    });
    
    forward_pass();
    backward_pass();
    
    nvtxRangePop();
}
```

**Python (PyTorch):**

```python
import torch.cuda.nvtx as nvtx

def training_step():
    nvtx.range_push("forward")
    output = model(input)
    nvtx.range_pop()
    
    nvtx.range_push("backward")
    loss.backward()
    nvtx.range_pop()
```

### Interpreting Results

Key metrics to look for:

1. **GPU Utilization**: Should be >80% for compute-bound apps
2. **Memory Transfer Time**: Minimize host-device transfers
3. **Kernel Launch Overhead**: Batch small kernels
4. **CPU-GPU Gaps**: Look for synchronization issues

## Nsight Compute

Detailed kernel-level profiling and optimization.

### Installation

```bash
# Included with CUDA Toolkit
which ncu

# Or download separately from Nvidia website
```

### Basic Usage

```bash
# Profile specific kernel
ncu ./my_app

# Profile all kernels
ncu --kernel-regex ".*" ./my_app

# With metrics
ncu --metrics all ./my_app

# Generate report
ncu --set full -o report ./my_app

# Open in GUI
ncu-ui report.ncu-rep
```

### Common Metrics

```bash
# Occupancy analysis
ncu --metrics sm__warps_active.avg.pct_of_peak ./app

# Memory bandwidth
ncu --metrics dram__throughput.avg.pct_of_peak ./app

# Compute utilization
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak ./app

# All important metrics
ncu --set full ./app
```

### Metric Sets

```bash
# Predefined sets
ncu --set basic ./app       # Basic metrics
ncu --set detailed ./app    # Detailed metrics
ncu --set full ./app        # All metrics (slow)

# Custom sections
ncu --section SpeedOfLight ./app      # Performance limiters
ncu --section MemoryWorkloadAnalysis ./app
ncu --section Occupancy ./app
```

### Analyzing Results

Key metrics to check:

**Occupancy:**
```bash
# Check theoretical occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak ./app

# Occupancy should be >50% for most kernels
```

**Memory Bandwidth:**
```bash
# DRAM utilization
ncu --metrics dram__throughput.avg.pct_of_peak ./app

# L2 cache hit rate
ncu --metrics lts__t_sectors_op_read.avg.pct_of_peak ./app
```

**Compute Utilization:**
```bash
# Tensor Core utilization (on Ampere/Hopper)
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak ./app

# CUDA core utilization
ncu --metrics sm__pipe_fma_cycles_active.avg.pct_of_peak ./app
```

## nvidia-smi

Real-time GPU monitoring.

### Basic Monitoring

```bash
# Current status
nvidia-smi

# Continuous monitoring (1 second updates)
watch -n 1 nvidia-smi

# Specific fields
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv

# Monitor processes
nvidia-smi pmon -i 0  # GPU 0 only
```

### Query Options

```bash
# GPU utilization over time
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory \
           --format=csv --loop=1

# Temperature monitoring
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader --loop=1

# Power consumption
nvidia-smi --query-gpu=power.draw,power.limit --format=csv --loop=1

# Multiple GPUs
nvidia-smi --query-gpu=index,name,utilization.gpu --format=csv
```

### Logging

```bash
# Log to file
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used \
           --format=csv --loop=1 > gpu_log.csv

# Run in background
nohup nvidia-smi --query-gpu=timestamp,utilization.gpu \
                  --format=csv --loop=1 > gpu_monitor.csv 2>&1 &
```

## Performance Analysis Workflow

### 1. Initial Profile

```bash
# Get overview with Nsight Systems
nsys profile --stats=true -o initial_profile ./app

# Look for:
# - Long CPU-GPU gaps
# - Small kernels launched frequently
# - Large memory transfers
# - Low GPU utilization
```

### 2. Identify Bottlenecks

```bash
# Profile hotspot kernels with Nsight Compute
ncu --kernel-name "my_kernel" --set full ./app

# Check:
# - Occupancy < 50%
# - Memory bandwidth < 60%
# - Compute utilization < 70%
```

### 3. Optimize

Based on profiling results:

**Low Occupancy:**
- Increase threads per block
- Reduce register usage
- Reduce shared memory usage

**Memory Bound:**
- Improve memory coalescing
- Use shared memory
- Reduce global memory access

**Compute Bound:**
- Use Tensor Cores
- Improve arithmetic intensity
- Optimize algorithm

### 4. Re-Profile

```bash
# Compare before/after
nsys profile -o optimized_profile ./app

# Use diff feature in nsight-ui to compare
```

## Code Instrumentation

### CUDA Events

Measure kernel execution time:

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Record start event
cudaEventRecord(start);

// Launch kernel
kernel<<<grid, block>>>(args);

// Record stop event
cudaEventRecord(stop);
cudaEventSynchronize(stop);

// Calculate elapsed time
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel time: %f ms\n", milliseconds);

// Cleanup
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### Custom Metrics in Python

```python
import torch
import time

# CUDA events
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
output = model(input)
end.record()

torch.cuda.synchronize()
print(f"Forward pass: {start.elapsed_time(end):.2f} ms")

# CPU timing
torch.cuda.synchronize()  # Important!
t0 = time.time()
output = model(input)
torch.cuda.synchronize()
t1 = time.time()
print(f"Kernel time: {(t1-t0)*1000:.2f} ms")
```

## Optimization Tips

### Memory Optimization

```bash
# Check memory access patterns
ncu --section MemoryWorkloadAnalysis kernel_name

# Look for:
# - Uncoalesced global memory access
# - Bank conflicts in shared memory
# - High L1/L2 cache miss rate
```

### Compute Optimization

```bash
# Check compute efficiency
ncu --section ComputeWorkloadAnalysis kernel_name

# Look for:
# - Low arithmetic intensity
# - Unused Tensor Cores
# - Divergent branches
```

### Occupancy Optimization

```bash
# Check occupancy limiters
ncu --section Occupancy kernel_name

# Common limiters:
# - Registers per thread
# - Shared memory per block
# - Threads per block
```

## Common Performance Issues

### Issue: Low GPU Utilization

**Symptoms:**
- GPU utilization < 60%
- Frequent CPU-GPU gaps

**Solutions:**
```bash
# Profile with Nsight Systems
nsys profile --trace=cuda,nvtx ./app

# Look for:
# - Small kernels → batch them
# - Synchronization → use streams
# - Memory transfers → use async copies
```

### Issue: Memory Bandwidth Limited

**Symptoms:**
- DRAM throughput > 80%
- Compute utilization < 60%

**Solutions:**
```cpp
// Use shared memory
__shared__ float shared[BLOCK_SIZE];
shared[threadIdx.x] = global[idx];
__syncthreads();
// Reuse shared[] multiple times

// Improve coalescing
// Bad: data[tid * stride]
// Good: data[tid]
```

### Issue: Low Occupancy

**Symptoms:**
- Warps active < 50%
- Register/shared memory limitations

**Solutions:**
```bash
# Check resource usage
ncu --metrics launch__registers_per_thread,launch__shared_mem_per_block ./app

# Reduce registers
nvcc --maxrregcount 64 kernel.cu

# Reduce shared memory
# Use dynamic shared memory sparingly
```

## Profiling Best Practices

1. **Profile in release mode**: Use `-O3` optimization
2. **Representative workload**: Use realistic input sizes
3. **Warm up GPU**: Run kernel once before profiling
4. **Multiple runs**: Average over several runs
5. **Isolate kernels**: Profile one kernel at a time
6. **Check correctness first**: Ensure output is correct

## Example: Complete Profiling Session

```bash
# 1. Initial profile
nsys profile --stats=true -o initial ./my_app

# 2. Identify slow kernel
# Look at nsight-ui initial.nsys-rep

# 3. Profile slow kernel in detail
ncu --kernel-name "slow_kernel" --set full -o kernel_analysis ./my_app

# 4. Check specific metrics
ncu --kernel-name "slow_kernel" \
    --metrics dram__throughput.avg.pct_of_peak,sm__warps_active.avg.pct_of_peak \
    ./my_app

# 5. Optimize code based on results

# 6. Re-profile
nsys profile --stats=true -o optimized ./my_app

# 7. Compare results in GUI
nsight-ui initial.nsys-rep optimized.nsys-rep
```

## External Resources

- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Profiling Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [NVTX Documentation](https://docs.nvidia.com/nvtx/)
- [GPU Performance Analysis](https://developer.nvidia.com/blog/gpu-pro-tip-nvprof-is-your-handy-universal-gpu-profiler/)

## Related Guides

- [CUDA Programming Basics](cuda-basics.md)
- [Kernel Optimization](../../best-practices/performance/kernel-optimization.md)
- [Memory Optimization](../../best-practices/performance/memory-optimization.md)
- [Nsight Systems Guide](../nsight/nsight-systems.md)
- [Nsight Compute Guide](../nsight/nsight-compute.md)

