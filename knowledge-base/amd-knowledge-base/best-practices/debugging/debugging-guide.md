---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "best-practices"
category: "debugging"
subcategory: "troubleshooting"
tags: ["debugging", "troubleshooting", "rocgdb", "errors"]
rocm_version: "7.0+"
last_updated: 2025-11-01
---

# Debugging Best Practices

## Systematic Debugging Approach

1. **Reproduce consistently**
2. **Isolate the problem**
3. **Form hypothesis**
4. **Test hypothesis**
5. **Fix and verify**

## Common GPU Debugging Techniques

### 1. Printf Debugging

```cpp
__global__ void debug_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Print first few threads
    if (idx < 5) {
        printf("Thread %d: block=%d, thread=%d, value=%f\n",
               idx, blockIdx.x, threadIdx.x, data[idx]);
    }
}
```

### 2. Assertions

```cpp
__global__ void kernel_with_checks(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    assert(idx < N && "Index out of bounds!");
    assert(data != nullptr && "Null pointer!");
    assert(data[idx] >= 0 && "Invalid value!");
}
```

### 3. Validation Against CPU

```cpp
// Run same computation on CPU and GPU
void validate_gpu_results() {
    // CPU version
    std::vector<float> cpu_result(N);
    cpu_compute(cpu_result.data());
    
    // GPU version
    float *d_data;
    hipMalloc(&d_data, N * sizeof(float));
    gpu_kernel<<<...>>>(d_data);
    
    std::vector<float> gpu_result(N);
    hipMemcpy(gpu_result.data(), d_data, N * sizeof(float),
              hipMemcpyDeviceToHost);
    
    // Compare
    for (int i = 0; i < N; i++) {
        float diff = std::abs(cpu_result[i] - gpu_result[i]);
        if (diff > 1e-5) {
            printf("Mismatch at %d: CPU=%f, GPU=%f\n",
                   i, cpu_result[i], gpu_result[i]);
        }
    }
}
```

### 4. Use rocgdb

```bash
# Compile with debug symbols
hipcc -g -O0 program.cpp -o program

# Debug
rocgdb ./program
(gdb) break kernel_name
(gdb) run
(gdb) print threadIdx.x
(gdb) print data[0]@10  # Print array
```

## Common Errors and Solutions

### Segmentation Fault
- Check array bounds
- Verify memory allocation
- Check for null pointers

### Wrong Results
- Validate against CPU version
- Check for race conditions
- Verify synchronization points
- Check floating point precision

### Hangs
- Check for deadlocks
- Verify all GPUs in collective operations
- Check for infinite loops

### Performance Issues
- Profile with rocprof
- Check memory bandwidth utilization
- Verify GPU occupancy
- Look for CPU-GPU sync points

## References

- [ROCm Debugging Guide](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides.html)
