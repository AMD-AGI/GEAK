---
layer: "1"
category: "amd-gpu-arch"
tags: ["cdna", "architecture", "mi-series", "mi200", "mi300", "mi350", "hpc", "ai-acceleration"]
rocm_version: "7.0+"
last_updated: 2025-11-02
---

# CDNA Architecture and MI Series Guide

*Comprehensive guide to AMD CDNA architecture and Instinct MI series accelerators for HPC and AI workloads*

## Overview

AMD CDNA (Compute DNA) architecture is specifically designed for high-performance computing and AI workloads. Unlike consumer RDNA GPUs, CDNA-based MI series prioritize compute performance, memory bandwidth, and reliability for datacenter deployments.

## CDNA Architecture Generations

### CDNA 1 (MI100)

The first generation CDNA architecture introduced key features for compute-focused workloads:
- **Matrix Cores**: First-generation hardware-accelerated matrix operations
- **HBM2 Memory**: High bandwidth memory
- **PCIe 4.0**: Fast host connectivity

Key specifications for MI100:
- 120 Compute Units
- 32GB HBM2 memory
- 11.5 TFLOPS FP64
- 184.6 TFLOPS FP16 (with matrix cores)

### CDNA 2 (MI200 Series)

Significant improvements in compute capabilities and connectivity:
- **Enhanced Matrix Cores**: Hardware-accelerated matrix operations for AI/ML
- **Infinity Fabric**: High-bandwidth chip-to-chip connectivity
- **HBM2e Memory**: High bandwidth memory up to 3.2TB/s
- **Multi-Die Design**: MI250X uses dual-die configuration

#### MI250X
```
Architecture: CDNA 2 (dual-die)
Process: 6nm TSMC
Compute Units: 220 CUs (110 per die)
Memory: 128GB HBM2E
Memory BW: 3.2TB/s
FP64: 95.7 TFLOPS (47.9 per die)
FP32: 95.7 TFLOPS
Matrix Performance: 383 TFLOPS (BF16)
Form Factor: OAM
TDP: 560W
Infinity Fabric: 200 GB/s per link
```

#### MI210
```
Architecture: CDNA 2 (single-die)
Compute Units: 104 CUs
Memory: 64GB HBM2E
Memory BW: 1.6TB/s
FP64: 45.3 TFLOPS
AI Performance: 181 TFLOPS (BF16)
Form Factor: PCIe dual-slot
TDP: 300W
```

### CDNA 3 (MI300 Series)

Revolutionary architecture with 3D stacking and massive memory:
- **3D Chiplet Design**: Integrated CPU and GPU dies with advanced packaging
- **Enhanced Matrix Cores**: Improved AI performance with MFMA instructions
- **HBM3 Memory**: Higher bandwidth and capacity (up to 5.3TB/s)
- **Unified Memory**: MI300A provides shared CPU/GPU memory space

#### MI300X
```
Architecture: CDNA 3
Process: 5nm + 6nm
Compute Units: 304 CUs
Matrix Cores: 1216 (AI workloads)
Memory: 192GB HBM3
Memory BW: 5.3TB/s
FP64: 163 TFLOPS
FP32: 163 TFLOPS  
FP16: 1307 TFLOPS
BF16: 1307 TFLOPS
INT8: 2614 TOPS
Form Factor: OAM (OCP Accelerator Module)
TDP: 750W
Infinity Fabric: 128 GB/s per link
```

#### MI300A
```
Architecture: CDNA 3 + Zen 4 CPU
CPU Cores: 24x Zen 4 cores
GPU CUs: 228 CUs  
Memory: 128GB HBM3 (shared CPU/GPU)
Memory BW: 5.3TB/s
FP64: 122 TFLOPS
AI Performance: 980 TFLOPS (FP16)
Unified Memory: CPU and GPU share same HBM3
TDP: 550W
```

### CDNA 4 (MI350 Series)

The latest generation of AMD Instinct accelerators, optimized for next-generation AI and HPC workloads.

- **Advanced Architecture**: Next-generation CDNA compute capabilities
- **Enhanced AI Performance**: Optimized for large-scale AI training and inference
- **Improved Memory Technology**: Enhanced HBM3/HBM3e support
- **Energy Efficiency**: Improved performance per watt

#### MI350X

The MI350X is designed for demanding AI and HPC applications:
- Advanced compute units with enhanced matrix operations
- High-bandwidth memory for large-scale workloads
- Optimized for data center deployments
- Enhanced connectivity and scalability

#### MI355X

The MI355X represents the flagship model with premium capabilities:
- Maximum compute performance for the MI350 series
- Enhanced specifications for the most demanding workloads
- Advanced features for AI training and inference
- Optimized for large language models and generative AI

**Note**: For detailed specifications, visit:
- [MI350X Product Page](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html)
- [MI355X Product Page](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x/platform.html)

### Legacy Products

#### MI50/MI60 (GCN/Vega Architecture)
```
Architecture: GCN 5.0 (Vega)
Purpose: Legacy compute cards (pre-CDNA)
Memory: 16GB/32GB HBM2
Note: Limited support in modern ROCm versions
```

## Memory Hierarchy

Understanding the memory hierarchy is critical for optimization:

```
HBM3/HBM2e (High Bandwidth Memory)
    ↓
L3 Cache (Infinity Cache) - Shared across dies
    ↓
L2 Cache (per CU group) - Shared by multiple CUs
    ↓
L1 Cache (per CU) - Per compute unit
    ↓
LDS (Local Data Share - 64KB per CU) - Explicitly managed
    ↓
Registers (per thread) - Per-thread storage
```

### Memory Access Patterns

```cpp
// Preferred: Coalesced memory access
__global__ void coalesced_access(float* data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float value = data[tid];  // Threads access consecutive memory
}

// Avoid: Strided access patterns
__global__ void strided_access(float* data, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float value = data[tid * stride];  // Poor memory coalescing
}
```

### Memory Bandwidth Optimization

```cpp
// Maximize HBM3 bandwidth on MI300X
__global__ void memory_bandwidth_test(float4* in, float4* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Use vector loads for peak bandwidth
        float4 data = in[idx];
        
        // Simple compute to avoid memory bottleneck
        data.x += 1.0f;
        data.y += 1.0f; 
        data.z += 1.0f;
        data.w += 1.0f;
        
        out[idx] = data;
    }
}
```

## Programming Models and Optimization

### ROCm HIP Programming

Query MI series device capabilities:

```cpp
#include <hip/hip_runtime.h>

void queryMIDevice() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", 
           prop.major, prop.minor);
    printf("Memory: %.1f GB\n", 
           prop.totalGlobalMem / (1024.0*1024*1024));
    printf("Memory Clock: %d MHz\n", prop.memoryClockRate/1000);
    printf("Memory Bus: %d-bit\n", prop.memoryBusWidth);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
}
```

### Leveraging Matrix Cores

For optimal performance with matrix operations, use MFMA (Matrix Fused Multiply-Add) instructions:

#### Using rocBLAS

```cpp
#include <rocblas.h>

// Optimized for MI series matrix cores
void mi_series_gemm() {
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    
    // Use BF16 for optimal MI300 performance
    rocblas_gemm_ex(handle,
                   rocblas_operation_none,
                   rocblas_operation_none,
                   m, n, k,
                   &alpha,
                   A, rocblas_datatype_bf16_r, lda,
                   B, rocblas_datatype_bf16_r, ldb,
                   &beta,
                   C, rocblas_datatype_bf16_r, ldc,
                   C, rocblas_datatype_bf16_r, ldc,
                   rocblas_datatype_f32_r,
                   rocblas_gemm_algo_standard,
                   0, 0);
}
```

#### Using PyTorch

```python
import torch

# Enable matrix cores through TF32 or BF16
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use BF16 for optimal performance on MI300X
model = model.to(dtype=torch.bfloat16)

# Multi-GPU training on MI300X cluster
model = torch.nn.DataParallel(model)

# Optimize for large memory (192GB on MI300X)
batch_size = 128  # Can use much larger batches than consumer GPUs
```

### Multi-GPU Programming

```cpp
// Leverage multiple MI300X in system
void multi_mi_setup() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        hipSetDevice(i);
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        
        if (strstr(prop.name, "MI300") || strstr(prop.name, "MI350")) {
            printf("MI series device detected on device %d: %s\n", i, prop.name);
            // Enable peer access for fast inter-GPU comms
            for (int j = 0; j < deviceCount; j++) {
                if (i != j) {
                    int canAccess;
                    hipDeviceCanAccessPeer(&canAccess, i, j);
                    if (canAccess) {
                        hipDeviceEnablePeerAccess(j, 0);
                    }
                }
            }
        }
    }
}
```

## AI Workload Optimization

### LLM Training Configuration

Optimal settings for MI300X with large memory:

```python
# Optimal settings for MI300X
training_config = {
    'precision': 'bf16-mixed',
    'batch_size': 64,  # Large memory enables big batches  
    'gradient_checkpointing': False,  # Less needed with 192GB
    'ddp_backend': 'nccl',
    'find_unused_parameters': False
}

# Memory-efficient attention for long sequences
attention_config = {
    'use_flash_attention': True,
    'max_sequence_length': 32768,  # Leverage large HBM3
    'attention_dropout': 0.1
}
```

## High-Performance Computing

### Double-Precision Scientific Computing

MI series excels at FP64 workloads:

```cpp
// Double-precision optimized for MI series
__global__ void mi_scientific_kernel(double* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // MI series excels at FP64 workloads
        double val = data[idx];
        data[idx] = sqrt(val * val + 1.0);
    }
}

// Launch with optimal occupancy for MI300X
void launch_scientific_kernel(double* d_data, int n) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // MI300X has massive parallelism (304 CUs)
    mi_scientific_kernel<<<gridSize, blockSize>>>(d_data, n);
}
```

## Performance Tuning

### Occupancy Optimization

- Target 50-100% occupancy for compute-bound kernels
- Use `rocminfo` to check device properties
- Profile with `rocprof` to identify bottlenecks
- Balance thread count with register usage

### Bandwidth Testing

```bash
# Check memory bandwidth with rocm-bandwidth-test
/opt/rocm/bin/rocm-bandwidth-test

# Monitor GPU utilization
rocm-smi --showmeminfo vram
```

### Infinity Fabric Topology

For multi-GPU systems, understand the topology:

```bash
# Check GPU topology
rocm-smi --showtopo

# Optimize for NUMA-aware allocation
export HSA_FORCE_FINE_GRAIN_PCIE=1
```

## System Configuration

### ROCm Installation for MI Series

```bash
# Add ROCm repository (Ubuntu/Debian)
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list

# Install ROCm for MI series
sudo apt update
sudo apt install rocm-dev rocm-libs miopen-hip rocblas

# Verify installation
rocminfo | grep "Marketing Name"
rocm-smi
```

### System Tuning

```bash
# Enable large pages for better performance
echo 'vm.nr_hugepages = 2048' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Increase memory limits
echo '* soft memlock unlimited' | sudo tee -a /etc/security/limits.conf
echo '* hard memlock unlimited' | sudo tee -a /etc/security/limits.conf

# GPU frequency scaling (set to performance mode)
sudo sh -c 'echo performance > /sys/class/drm/card*/device/power_dpm_force_performance_level'

# Add user to render and video groups
sudo usermod -a -G render,video $USER
```

## Monitoring and Debugging

### rocm-smi for MI Series

```bash
# Monitor MI300X/MI350X status
rocm-smi -a

# Temperature and power monitoring
rocm-smi -t -p

# Memory usage
rocm-smi -u

# Performance counters
rocm-smi --showpids

# Show topology
rocm-smi --showtopo

# Check ECC status
rocm-smi --showecc
```

### Profiling Tools

```bash
# Profile AI workloads
rocprof --hip-trace python train_llm.py

# Detailed kernel analysis  
rocprof --stats --hsa-trace ./mi_app

# Memory tracing for large models
rocprof --hip-trace --hsa-trace --roctx-trace ./large_model

# Generate timeline visualization
rocprof --stats --timestamp on ./app
```

## Troubleshooting

### Common Issues

#### Memory Allocation Failures

```cpp
// Check available memory before allocation
size_t free_mem, total_mem;
hipMemGetInfo(&free_mem, &total_mem);

if (required_mem > free_mem * 0.9) {  // Leave 10% headroom
    printf("Insufficient memory: need %zu MB, have %zu MB\n",
           required_mem/(1024*1024), free_mem/(1024*1024));
    // Handle error appropriately
}
```

#### Performance Issues

```bash
# Check GPU utilization
rocm-smi -u

# Verify ECC status (affects performance)
rocm-smi --showecc

# Check thermal throttling
rocm-smi -t

# Verify clock speeds
rocm-smi -c
```

#### Peer Access Issues

```bash
# Check peer-to-peer topology
rocm-smi --showtopo

# Verify Infinity Fabric connections
rocm-bandwidth-test
```

## Best Practices

### Memory Management

1. **Use Unified Memory on MI300A**: CPU and GPU share the same HBM3
2. **Large Batch Sizes**: Leverage 192GB HBM3 on MI300X for larger batches
3. **Memory Pooling**: Reduce allocation overhead with memory pools
4. **Pinned Memory**: Use `hipHostMalloc()` for optimal transfer speeds
5. **Async Memory Copies**: Overlap compute with data transfer using streams

### Compute Optimization

1. **BF16 Precision**: Optimal for AI workloads on MI series
2. **Use Optimized Libraries**: rocBLAS, MIOpen, rocFFT, etc.
3. **Kernel Fusion**: Reduce memory traffic by fusing operations
4. **Occupancy Balance**: Target 50-100% occupancy for compute-bound kernels
5. **Leverage Matrix Cores**: Use MFMA instructions through libraries
6. **Profile First**: Use rocprof before optimizing

### Multi-GPU Scaling

1. **RCCL**: Use for collective operations (AllReduce, Broadcast, etc.)
2. **Enable Peer Access**: Use `hipDeviceEnablePeerAccess()` for direct GPU-GPU transfers
3. **Load Balancing**: Distribute work evenly across GPUs
4. **Minimize Communication**: Reduce inter-GPU traffic where possible
5. **Topology Awareness**: Place communicating GPUs on same Infinity Fabric links

### Data Layout

1. **Coalesced Access**: Ensure threads access consecutive memory
2. **Use LDS Effectively**: Leverage Local Data Share for thread group communication
3. **Consider Memory Hierarchy**: Keep hot data in L1/L2 cache
4. **Vectorized Loads**: Use float4/double2 for better bandwidth utilization

## References

### Official Documentation

- [AMD CDNA Architecture Whitepaper](https://www.amd.com/en/technologies/cdna)
- [AMD Instinct MI Series](https://www.amd.com/en/products/accelerators.html)
- [ROCm Documentation](https://rocmdocs.amd.com/)
- [MI200 Series Documentation](https://www.amd.com/en/products/accelerators.html)
- [MI300 Series Documentation](https://www.amd.com/en/products/accelerators/instinct/mi300.html)
- [MI350X Documentation](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html)
- [MI355X Documentation](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x/platform.html)

### Related Guides

- [GPU Optimization Best Practices](../../best-practices/performance/gpu-optimization.md)
- [Memory Optimization Techniques](../../best-practices/performance/memory-optimization.md)
- [Distributed Training with FSDP](../../layer-5-llm/03-training/distributed/fsdp-training.md)

---

*Tags: cdna, mi-series, architecture, hpc, ai-acceleration, multi-gpu, datacenter, fp64, hbm3, matrix-cores*

*Estimated reading time: 60 minutes*
