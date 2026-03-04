---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "2"
category: "hip"
subcategory: "debugging"
tags: ["hip", "debugging", "rocgdb", "error-handling", "troubleshooting"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
therock_included: true
last_updated: 2025-11-03
---

# HIP Debugging Guide

Comprehensive guide to debugging HIP applications on AMD GPUs.

**Official HIP Repository**: [https://github.com/ROCm/HIP](https://github.com/ROCm/HIP) (within rocm-systems)  
**Documentation**: [https://rocm.docs.amd.com/projects/HIP](https://rocm.docs.amd.com/projects/HIP)  
**ROCm Systems**: [https://github.com/ROCm/rocm-systems](https://github.com/ROCm/rocm-systems)

> **Note**: HIP is now part of the [rocm-systems monorepo](../rocm-systems/rocm-systems-usage.md) at `projects/hip/`.

## Debugging Tools

### 1. rocgdb - ROCm Debugger

rocgdb is GDB with ROCm GPU debugging support.

#### Basic Usage

```bash
# Compile with debug symbols
hipcc -g -O0 program.cpp -o program

# Start debugger
rocgdb ./program

# Common commands:
(gdb) break main
(gdb) run
(gdb) next
(gdb) continue
(gdb) quit
```

#### GPU Debugging

```bash
# Set breakpoint in kernel
(gdb) break vectorAdd

# Run until kernel
(gdb) run

# Show GPU threads (lanes)
(gdb) info threads

# Switch to GPU thread
(gdb) thread 2

# Print GPU variable
(gdb) print blockIdx.x
(gdb) print threadIdx.x

# Print local array
(gdb) print data[0]@10
```

#### Example Session

```bash
$ rocgdb ./vector_add

(gdb) break vectorAdd
Breakpoint 1 at 0x401234: file vector_add.cpp, line 15.

(gdb) run
Thread 1 "vector_add" hit Breakpoint 1, vectorAdd (a=0x7fff...) at vector_add.cpp:15

(gdb) print blockIdx.x
$1 = 0

(gdb) print threadIdx.x  
$2 = 0

(gdb) print a[threadIdx.x]
$3 = 1.0

(gdb) continue
```

### 2. HIP Error Checking

#### Comprehensive Error Handling

```cpp
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

// Error checking macro
#define HIP_CHECK(call) \
do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) \
                  << " (" << hipGetErrorName(err) << ")" \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel launch error checking
#define HIP_CHECK_LAST() \
do { \
    hipError_t err = hipGetLastError(); \
    if (err != hipSuccess) { \
        std::cerr << "HIP Kernel Launch Error: " << hipGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Usage example
int main() {
    float *d_data;
    
    // Check memory allocation
    HIP_CHECK(hipMalloc(&d_data, 1024 * sizeof(float)));
    
    // Launch kernel
    myKernel<<<10, 256>>>(d_data);
    HIP_CHECK_LAST();
    
    // Check synchronization
    HIP_CHECK(hipDeviceSynchronize());
    
    // Check memory copy
    float h_data[1024];
    HIP_CHECK(hipMemcpy(h_data, d_data, 1024 * sizeof(float), 
                        hipMemcpyDeviceToHost));
    
    HIP_CHECK(hipFree(d_data));
    
    return 0;
}
```

### 3. GPU Sanitizer (Compute Sanitizer)

Detects memory errors, race conditions, and other issues.

```bash
# Compile with debug info
hipcc -g -lineinfo program.cpp -o program

# Run with sanitizer
compute-sanitizer ./program

# Check for:
# - Out of bounds access
# - Race conditions
# - Uninitialized memory
# - Memory leaks
```

## Common HIP Errors

### 1. Out of Bounds Access

```cpp
// BAD: Buffer overflow
__global__ void kernel_bug(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // No bounds check - can access beyond array!
    data[idx] = idx * 2.0f;
}

// GOOD: Bounds checking
__global__ void kernel_safe(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {  // Bounds check
        data[idx] = idx * 2.0f;
    }
}
```

### 2. Race Conditions

```cpp
// BAD: Race condition
__global__ void race_bug(float* result) {
    __shared__ float sum;
    
    // Multiple threads writing same location!
    sum += threadIdx.x;  // RACE!
    
    if (threadIdx.x == 0) {
        *result = sum;
    }
}

// GOOD: Proper reduction
__global__ void race_fixed(float* result) {
    __shared__ float shared[256];
    
    // Each thread writes to its own location
    shared[threadIdx.x] = threadIdx.x;
    __syncthreads();
    
    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        *result = shared[0];
    }
}
```

### 3. Synchronization Errors

```cpp
// BAD: Missing synchronization
__global__ void sync_bug(float* data) {
    __shared__ float tile[256];
    
    tile[threadIdx.x] = data[threadIdx.x];
    // Missing __syncthreads() here!
    
    // May read uninitialized data!
    float val = tile[(threadIdx.x + 1) % 256];
}

// GOOD: Proper synchronization
__global__ void sync_fixed(float* data) {
    __shared__ float tile[256];
    
    tile[threadIdx.x] = data[threadIdx.x];
    __syncthreads();  // Wait for all threads
    
    float val = tile[(threadIdx.x + 1) % 256];
}
```

### 4. Memory Leaks

```cpp
// BAD: Memory leak
void leak_memory() {
    float *d_data;
    hipMalloc(&d_data, 1024 * sizeof(float));
    
    // Do work...
    
    // Forgot to free!
    // hipFree(d_data);  // Missing!
}

// GOOD: RAII wrapper
class HipMemory {
    void* ptr;
    size_t size;
public:
    HipMemory(size_t bytes) : size(bytes) {
        HIP_CHECK(hipMalloc(&ptr, bytes));
    }
    
    ~HipMemory() {
        hipFree(ptr);  // Automatic cleanup
    }
    
    void* get() { return ptr; }
};

void no_leak() {
    HipMemory mem(1024 * sizeof(float));
    // Use mem.get()...
    // Automatically freed on scope exit
}
```

## Debugging Strategies

### 1. Printf Debugging

```cpp
__global__ void debug_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Debug first few threads
    if (idx < 5) {
        printf("Thread %d: blockIdx=%d, threadIdx=%d, value=%f\n",
               idx, blockIdx.x, threadIdx.x, data[idx]);
    }
    
    // Process...
}
```

### 2. Assertions

```cpp
#include <cassert>

__global__ void kernel_with_asserts(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Runtime assertion
    assert(idx < N && "Index out of bounds!");
    assert(data != nullptr && "Null pointer!");
    
    data[idx] = idx * 2.0f;
}
```

### 3. Incremental Testing

```cpp
// Test 1: CPU version
void cpu_version(float* result, const float* input, int N) {
    for (int i = 0; i < N; i++) {
        result[i] = input[i] * 2.0f;
    }
}

// Test 2: Simple GPU version
__global__ void simple_gpu(float* result, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        result[idx] = input[idx] * 2.0f;
    }
}

// Test 3: Optimized GPU version
__global__ void optimized_gpu(float* result, const float* input, int N) {
    // Add optimizations here
    // Compare results with simple version
}

// Validation
void validate() {
    // Compare CPU vs GPU results
    bool passed = true;
    for (int i = 0; i < N; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > 1e-5) {
            printf("Mismatch at %d: CPU=%f, GPU=%f\n", 
                   i, cpu_result[i], gpu_result[i]);
            passed = false;
        }
    }
    assert(passed && "Validation failed!");
}
```

## Debugging Checklist

### Before Running
- [ ] Compile with `-g` flag for debug symbols
- [ ] Add comprehensive error checking (HIP_CHECK macros)
- [ ] Add bounds checking in kernels
- [ ] Add assertions for invariants
- [ ] Initialize all variables and arrays

### Common Issues to Check
- [ ] Array bounds (off-by-one errors)
- [ ] Race conditions (shared memory access)
- [ ] Missing __syncthreads()
- [ ] Memory leaks (hipMalloc/hipFree pairs)
- [ ] Null pointer dereferences
- [ ] Integer overflow in grid calculations
- [ ] Uncoalesced memory access patterns
- [ ] Bank conflicts in shared memory

### Debugging Tools to Use
- [ ] rocgdb for step-through debugging
- [ ] compute-sanitizer for memory errors
- [ ] rocprof for performance issues
- [ ] printf for kernel debugging
- [ ] Assertions for invariants

## Performance Debugging

```cpp
// Add timing to find slow sections
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();

// Code to time
kernel<<<grid, block>>>();
hipDeviceSynchronize();

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
std::cout << "Kernel time: " << duration.count() << " ms\n";
```

## Troubleshooting Guide

### GPU Not Detected
```bash
# Check GPU visibility
rocm-smi

# Check driver
lsmod | grep amdgpu

# Check permissions
ls -l /dev/kfd /dev/dri/render*
```

### Out of Memory Errors
```bash
# Check memory usage
rocm-smi --showmeminfo

# Reduce batch size or use memory pooling
# Free memory when not needed
hipFree(large_buffer);
```

### Slow Performance
```bash
# Profile first
rocprof --stats ./program

# Check for CPU-GPU sync points
# Use async operations
# Batch small operations
```

## References

- [rocgdb Documentation](https://rocm.docs.amd.com/projects/ROCgdb/en/latest/)
- [HIP Error Handling](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/debugging.html)
- [AMD GPU Debugging Guide](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides.html)

