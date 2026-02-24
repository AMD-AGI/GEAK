# RDNA Architecture Guide

*Complete guide to AMD RDNA GPU architecture for gaming and compute workloads*

## Overview

RDNA (Radeon DNA) is AMD's graphics architecture designed for gaming and consumer graphics, but also increasingly used for compute workloads. Unlike CDNA which is purely for compute, RDNA balances graphics and compute performance.

## RDNA Evolution

### RDNA 1 (2019)
- **Process**: 7nm TSMC
- **Key Features**: 
  - New compute unit design
  - Improved performance per watt
  - Enhanced memory subsystem
- **Products**: RX 5000 series

### RDNA 2 (2020) 
- **Process**: 7nm TSMC Enhanced
- **Key Features**:
  - Ray tracing acceleration
  - Variable Rate Shading
  - Infinity Cache
  - 50% performance/watt improvement over RDNA 1
- **Products**: RX 6000 series, PlayStation 5, Xbox Series X/S

### RDNA 3 (2022)
- **Process**: 5nm + 6nm chiplet design
- **Key Features**:
  - Chiplet architecture
  - AI acceleration units
  - AV1 encode/decode
  - DisplayPort 2.1 support
- **Products**: RX 7000 series

## Architecture Details

### Compute Units (CUs)
```
RDNA CU Structure:
├── 2x SIMD32 units (64 ALUs total)
├── 1x Scalar unit
├── 32KB L1 Vector Cache
├── 16KB L1 Scalar Cache
└── 1x Ray tracing unit (RDNA 2+)
```

### Memory Hierarchy
- **L0**: Register files
- **L1**: 32KB vector cache per CU
- **L2**: Shared across Shader Engines
- **Infinity Cache**: Large L3 cache (RDNA 2+)
- **VRAM**: GDDR6/GDDR6X

### Workgroup Dispatchers (WGPs)
Each WGP contains:
- 2 Compute Units
- Shared L1 instruction cache
- Local data share (LDS)

## Programming Model

### HIP/ROCm Support
```cpp
// Device query for RDNA GPUs
hipDeviceProp_t props;
hipGetDeviceProperties(&props, deviceId);

// Check for RDNA architecture
if (strstr(props.name, "gfx10") || strstr(props.name, "gfx11")) {
    printf("RDNA GPU detected: %s\n", props.name);
}
```

### Key Architectural IDs
- **RDNA 1**: gfx1010, gfx1012
- **RDNA 2**: gfx1030, gfx1031, gfx1032, gfx1033
- **RDNA 3**: gfx1100, gfx1101, gfx1102

## Optimization Guidelines

### Wavefront Execution
```cpp
// RDNA uses wave32 by default (vs wave64 in GCN)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
constexpr int WAVE_SIZE = 32;  // RDNA default
#pragma clang diagnostic pop

__global__ void optimized_kernel() {
    int lane_id = threadIdx.x % WAVE_SIZE;
    // Optimize for wave32 execution
}
```

### Memory Access Patterns
- **Coalesced Access**: 128-byte cache lines
- **Bank Conflicts**: Avoid in LDS access
- **Infinity Cache**: Leverage for data reuse (RDNA 2+)

### Compute Optimization
```cpp
// Optimal workgroup sizes for RDNA
const int OPTIMAL_WORKGROUP_SIZES[] = {64, 128, 256};

// Launch configuration
dim3 blockSize(128);  // Multiple of wave size
dim3 gridSize((totalWork + blockSize.x - 1) / blockSize.x);
```

## Performance Characteristics

### Peak Performance (RDNA 3 Example - RX 7900 XTX)
- **FP32**: 61 TFLOPS
- **FP16**: 123 TFLOPS  
- **Memory BW**: 960 GB/s
- **Infinity Cache**: 96MB

### Occupancy Considerations
- **Max Threads per CU**: 2048
- **Max Workgroups per CU**: 32
- **Registers per CU**: 65536

## Compute Workload Examples

### Matrix Multiplication
```cpp
__global__ void rdna_gemm(float* A, float* B, float* C, int N) {
    // Tile for optimal cache usage
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    
    float sum = 0.0f;
    
    // Leverage RDNA's cache hierarchy
    for (int k = 0; k < N; k += 16) {
        As[ty][tx] = A[(by*16 + ty)*N + k + tx];
        Bs[ty][tx] = B[(k + ty)*N + bx*16 + tx];
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }
    
    C[(by*16 + ty)*N + bx*16 + tx] = sum;
}
```

### Reduction with Wave Operations
```cpp
__global__ void rdna_reduction(float* data, float* result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (tid < n) ? data[tid] : 0.0f;
    
    // Use RDNA wave32 operations
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    
    if (threadIdx.x % 32 == 0) {
        atomicAdd(result, val);
    }
}
```

## Debugging and Profiling

### ROCm Tools for RDNA
```bash
# Check architecture
rocminfo | grep "Marketing Name"

# Profile RDNA kernel
rocprof --hip-trace ./rdna_app

# Check occupancy
rocprof --stats ./rdna_app
```

### Common Issues
1. **Wave32 vs Wave64**: Ensure code assumes correct wave size
2. **Cache Misses**: Profile memory access patterns
3. **Occupancy**: Balance registers vs threads per CU

## Best Practices

### Memory Management
```cpp
// Prefer pinned memory for transfers
float* h_data;
hipHostMalloc(&h_data, size, hipHostMallocDefault);

// Use memory pools for frequent allocations
hipMemPool_t pool;
hipMemPoolCreate(&pool, &props);
```

### Kernel Launch
```cpp
// Query device properties first
hipDeviceProp_t prop;
hipGetDeviceProperties(&prop, 0);

// Configure based on CU count
int numCUs = prop.multiProcessorCount;
dim3 grid(numCUs * 4);  // 4 waves per CU
dim3 block(128);        // Wave32 friendly
```

## Compatibility Notes

### HIP Code Portability
```cpp
#ifdef __HIP_PLATFORM_AMD__
    // RDNA-specific optimizations
    #define WAVE_SIZE 32
#else
    // CUDA fallback
    #define WAVE_SIZE 32
#endif
```

### ROCm Version Requirements

#### Minimum Versions
- **RDNA 1**: ROCm 3.0+ (Legacy support in ROCm 7.x)
- **RDNA 2**: ROCm 4.0+ (Full support in ROCm 7.x)
- **RDNA 3**: ROCm 5.4+ (Optimized in ROCm 7.x)
- **RDNA 4**: ROCm 7.0+ (Latest generation)

#### ROCm 7.x Enhancements
- **Improved Performance**: Up to 3.5x inference, 3x training speedup vs ROCm 6.x
- **Better Driver Support**: Enhanced stability for RDNA 2 and RDNA 3
- **Extended Features**: More complete AI/ML framework support
- **Power Management**: Better thermal and power optimization

**Note**: While RDNA GPUs work with ROCm 7.x, CDNA GPUs (MI-series) are the primary targets for production AI/ML workloads.

## Resources

### Documentation
- [AMD RDNA Architecture](https://www.amd.com/en/technologies/rdna-architecture)
- [ROCm Programming Guide](https://rocmdocs.amd.com)
- [HIP Programming Guide](https://rocmdocs.amd.com/projects/HIP/en/latest/)

### Performance Guides
- [GPU Optimization](../../best-practices/performance/gpu-optimization.md)
- [Kernel Optimization](../../best-practices/performance/kernel-optimization.md)
- [Memory Optimization](../../best-practices/performance/memory-optimization.md)

---
*Tags: rdna, architecture, gaming-gpu, compute, hip, rocm, wave32, infinity-cache*
*Estimated reading time: 45 minutes*