Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "3"
category: "fft"
subcategory: "transforms"
tags: ["rocfft", "fft", "signal-processing", "performance"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
---

# rocFFT Usage Guide

rocFFT is AMD's Fast Fourier Transform library for ROCm, providing highly optimized FFT implementations.

## Installation

```bash
# Ubuntu/Debian
sudo apt install rocfft rocfft-dev

# Verify
ls /opt/rocm/lib/librocfft.so
```

## Basic 1D FFT

```cpp
#include <rocfft/rocfft.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <complex>

int main() {
    const size_t N = 1024;
    
    // Setup
    rocfft_setup();
    
    // Create plan for 1D complex-to-complex FFT
    rocfft_plan plan = nullptr;
    size_t lengths[1] = {N};
    
    rocfft_plan_create(&plan,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_forward,
                       rocfft_precision_single,
                       1,      // dimensions
                       lengths,
                       1,      // number of transforms
                       nullptr);
    
    // Allocate GPU memory
    size_t buffer_size = N * sizeof(float) * 2;  // Complex = 2 floats
    void* d_data;
    hipMalloc(&d_data, buffer_size);
    
    // Copy input data
    std::vector<std::complex<float>> input(N);
    for (size_t i = 0; i < N; i++) {
        input[i] = std::complex<float>(i, 0);
    }
    hipMemcpy(d_data, input.data(), buffer_size, hipMemcpyHostToDevice);
    
    // Execute FFT
    rocfft_execution_info info = nullptr;
    rocfft_execution_info_create(&info);
    
    void* in_buffer[1] = {d_data};
    void* out_buffer[1] = {d_data};  // in-place
    
    rocfft_execute(plan, in_buffer, out_buffer, info);
    
    hipDeviceSynchronize();
    
    // Copy result back
    std::vector<std::complex<float>> output(N);
    hipMemcpy(output.data(), d_data, buffer_size, hipMemcpyDeviceToHost);
    
    // Cleanup
    rocfft_execution_info_destroy(info);
    rocfft_plan_destroy(plan);
    hipFree(d_data);
    rocfft_cleanup();
    
    return 0;
}
```

Compile:
```bash
hipcc fft_example.cpp -lrocfft -o fft_example
```

## 2D FFT

```cpp
// 2D FFT for image processing
void fft_2d_example() {
    const size_t N = 512, M = 512;
    
    rocfft_setup();
    
    // Create 2D plan
    size_t lengths[2] = {N, M};
    rocfft_plan plan = nullptr;
    
    rocfft_plan_create(&plan,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_forward,
                       rocfft_precision_single,
                       2,      // 2D
                       lengths,
                       1,
                       nullptr);
    
    // Allocate
    size_t total_size = N * M * sizeof(float) * 2;
    void* d_data;
    hipMalloc(&d_data, total_size);
    
    // Execute
    rocfft_execution_info info = nullptr;
    rocfft_execution_info_create(&info);
    
    void* buffers[1] = {d_data};
    rocfft_execute(plan, buffers, buffers, info);
    
    hipDeviceSynchronize();
    
    // Cleanup
    rocfft_execution_info_destroy(info);
    rocfft_plan_destroy(plan);
    hipFree(d_data);
    rocfft_cleanup();
}
```

## Real-to-Complex FFT

```cpp
// More efficient for real-valued inputs
void real_fft_example() {
    const size_t N = 1024;
    
    rocfft_setup();
    
    // Real-to-complex forward transform
    size_t lengths[1] = {N};
    rocfft_plan forward_plan = nullptr;
    
    rocfft_plan_create(&forward_plan,
                       rocfft_placement_notinplace,
                       rocfft_transform_type_real_forward,
                       rocfft_precision_single,
                       1,
                       lengths,
                       1,
                       nullptr);
    
    // Allocate
    void *d_input, *d_output;
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, (N/2 + 1) * sizeof(float) * 2);  // Hermitian symmetry
    
    // Execute
    rocfft_execution_info info = nullptr;
    rocfft_execution_info_create(&info);
    
    void* in_buffer[1] = {d_input};
    void* out_buffer[1] = {d_output};
    
    rocfft_execute(forward_plan, in_buffer, out_buffer, info);
    
    // Cleanup
    rocfft_execution_info_destroy(info);
    rocfft_plan_destroy(forward_plan);
    hipFree(d_input);
    hipFree(d_output);
    rocfft_cleanup();
}
```

## Batched FFT

```cpp
// Process multiple FFTs efficiently
void batched_fft() {
    const size_t N = 256;
    const size_t batch = 100;
    
    rocfft_setup();
    
    size_t lengths[1] = {N};
    rocfft_plan plan = nullptr;
    
    rocfft_plan_create(&plan,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_forward,
                       rocfft_precision_single,
                       1,
                       lengths,
                       batch,  // Number of FFTs
                       nullptr);
    
    // Allocate for all batches
    size_t total_size = N * batch * sizeof(float) * 2;
    void* d_data;
    hipMalloc(&d_data, total_size);
    
    // Execute all FFTs
    rocfft_execution_info info = nullptr;
    rocfft_execution_info_create(&info);
    
    void* buffers[1] = {d_data};
    rocfft_execute(plan, buffers, buffers, info);
    
    hipDeviceSynchronize();
    
    // Cleanup
    rocfft_execution_info_destroy(info);
    rocfft_plan_destroy(plan);
    hipFree(d_data);
    rocfft_cleanup();
}
```

## PyTorch Integration

```python
import torch
import torch.fft

# rocFFT is automatically used on AMD GPUs
def torch_fft_example():
    # Create signal on GPU
    x = torch.randn(1024, device='cuda', dtype=torch.complex64)
    
    # FFT (uses rocFFT)
    X = torch.fft.fft(x)
    
    # 2D FFT for images
    img = torch.randn(512, 512, device='cuda')
    img_fft = torch.fft.fft2(img)
    
    # Inverse FFT
    img_recovered = torch.fft.ifft2(img_fft).real
    
    return X, img_fft
```

## Performance Optimization

### 1. Choose Efficient Sizes

```python
# Good FFT sizes (powers of 2, 3, 5)
good_sizes = [256, 512, 1024, 2048, 4096]  # Powers of 2
also_good = [243, 432, 864]  # Powers of 3
mixed = [384, 640, 768]  # Mixed factors

# Avoid prime numbers > 13
bad_sizes = [1021, 2027]  # Prime numbers (slow!)
```

### 2. Reuse Plans

```cpp
// BAD: Creating plan every time
for (int i = 0; i < 1000; i++) {
    rocfft_plan plan;
    rocfft_plan_create(&plan, ...);
    rocfft_execute(plan, ...);
    rocfft_plan_destroy(plan);  // Expensive!
}

// GOOD: Reuse plan
rocfft_plan plan;
rocfft_plan_create(&plan, ...);
for (int i = 0; i < 1000; i++) {
    rocfft_execute(plan, ...);  // Fast!
}
rocfft_plan_destroy(plan);
```

### 3. Use Batched Transforms

```cpp
// BAD: Sequential FFTs
for (int i = 0; i < 100; i++) {
    rocfft_execute(plan, &data[i * N], ...);
}

// GOOD: Batched FFT
rocfft_plan_create(&plan, ..., 100);  // batch = 100
rocfft_execute(plan, data, ...);  // All at once
```

## Common Applications

### Signal Processing

```cpp
// Convolution via FFT
void fft_convolution(const float* signal, const float* kernel,
                     float* result, size_t N) {
    // 1. FFT of signal and kernel
    // 2. Point-wise multiplication in frequency domain
    // 3. Inverse FFT
    // O(N log N) instead of O(N^2)
}
```

### Image Filtering

```python
def frequency_domain_filter(image, filter_func):
    # Transform to frequency domain
    img_fft = torch.fft.fft2(image)
    
    # Apply filter
    filtered_fft = filter_func(img_fft)
    
    # Transform back
    result = torch.fft.ifft2(filtered_fft).real
    
    return result
```

## References

- [rocFFT Documentation](https://rocm.docs.amd.com/projects/rocFFT/en/latest/)
- [rocFFT GitHub](https://github.com/ROCmSoftwarePlatform/rocFFT)
- [FFT Best Practices](https://rocm.docs.amd.com/projects/rocFFT/en/latest/)

