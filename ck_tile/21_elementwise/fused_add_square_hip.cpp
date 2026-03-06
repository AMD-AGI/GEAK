// Simple HIP-based fused Add-Square kernel
// Demonstrates kernel fusion: Y = (A + B)^2 in one pass
// Saves 40% memory bandwidth vs separate Add + Square kernels

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// Baseline: Separate Add and Square kernels
__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void square_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        out[idx] = val * val;
    }
}

// Fused: Add-Square in single kernel
__global__ void fused_add_square_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = a[idx] + b[idx];
        out[idx] = sum * sum;
    }
}

// Helper function to measure bandwidth
double measure_bandwidth_gbps(size_t bytes, double time_ms) {
    return (bytes / 1e9) / (time_ms / 1000.0);
}

int main(int argc, char** argv) {
    int M = 4096;
    int N = 4096;
    if (argc > 2) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
    }
    
    size_t n = M * N;
    size_t bytes = n * sizeof(float);
    
    std::cout << "=== Kernel Fusion Demonstration ===" << std::endl;
    std::cout << "Problem size: " << M << " x " << N << " = " << n << " elements" << std::endl;
    std::cout << "Data size: " << bytes / 1e6 << " MB" << std::endl << std::endl;
    
    // Allocate host memory
    std::vector<float> h_a(n), h_b(n), h_out1(n), h_out2(n);
    for (size_t i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_temp, *d_out1, *d_out2;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_temp, bytes);
    hipMalloc(&d_out1, bytes);
    hipMalloc(&d_out2, bytes);
    
    hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        fused_add_square_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_out1, n);
    }
    hipDeviceSynchronize();
    
    // Benchmark baseline (unfused): Add + Square
    std::cout << "--- Baseline (Unfused): Add + Square ---" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    const int repeats = 100;
    for (int i = 0; i < repeats; i++) {
        add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_temp, n);
        square_kernel<<<gridSize, blockSize>>>(d_temp, d_out1, n);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double unfused_time = std::chrono::duration<double, std::milli>(end - start).count() / repeats;
    
    // Memory traffic: Read A, Read B, Write Temp, Read Temp, Write Out = 5 arrays
    size_t unfused_bytes = 5 * bytes;
    double unfused_bw = measure_bandwidth_gbps(unfused_bytes, unfused_time);
    
    std::cout << "  Time: " << unfused_time << " ms" << std::endl;
    std::cout << "  Memory traffic: " << unfused_bytes / 1e6 << " MB" << std::endl;
    std::cout << "  Bandwidth: " << unfused_bw << " GB/s" << std::endl << std::endl;
    
    // Benchmark fused: Add-Square in one kernel
    std::cout << "--- Optimized (Fused): Add-Square ---" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeats; i++) {
        fused_add_square_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_out2, n);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double fused_time = std::chrono::duration<double, std::milli>(end - start).count() / repeats;
    
    // Memory traffic: Read A, Read B, Write Out = 3 arrays
    size_t fused_bytes = 3 * bytes;
    double fused_bw = measure_bandwidth_gbps(fused_bytes, fused_time);
    
    std::cout << "  Time: " << fused_time << " ms" << std::endl;
    std::cout << "  Memory traffic: " << fused_bytes / 1e6 << " MB" << std::endl;
    std::cout << "  Bandwidth: " << fused_bw << " GB/s" << std::endl << std::endl;
    
    // Calculate speedup
    double speedup = unfused_time / fused_time;
    double memory_reduction = 100.0 * (1.0 - (double)fused_bytes / unfused_bytes);
    
    std::cout << "=== Results ===" << std::endl;
    std::cout << "  Speedup: " << speedup << "x (" 
              << ((speedup - 1.0) * 100.0) << "% faster)" << std::endl;
    std::cout << "  Memory bandwidth reduction: " << memory_reduction << "%" << std::endl;
    std::cout << "  Theoretical max speedup from fusion: " 
              << (double)unfused_bytes / fused_bytes << "x" << std::endl;
    
    // Verify correctness
    hipMemcpy(h_out1.data(), d_out1, bytes, hipMemcpyDeviceToHost);
    hipMemcpy(h_out2.data(), d_out2, bytes, hipMemcpyDeviceToHost);
    
    bool correct = true;
    float max_error = 0.0f;
    for (size_t i = 0; i < n && i < 1000; i++) {
        float expected = (h_a[i] + h_b[i]) * (h_a[i] + h_b[i]);
        float error1 = std::abs(h_out1[i] - expected);
        float error2 = std::abs(h_out2[i] - expected);
        max_error = std::max(max_error, std::max(error1, error2));
        if (error1 > 1e-4 || error2 > 1e-4) {
            correct = false;
            break;
        }
    }
    
    std::cout << "  Correctness: " << (correct ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  Max error: " << max_error << std::endl;
    
    // Cleanup
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_temp);
    hipFree(d_out1);
    hipFree(d_out2);
    
    return correct ? 0 : 1;
}
