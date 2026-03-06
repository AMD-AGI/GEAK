// Advanced kernel fusion: Fused Multiply-Add-ReLU (MAR)
// Y = max(α * A + B, 0)
// Common pattern in neural networks (scaled residual with activation)

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// Baseline: Separate kernels
__global__ void multiply_kernel(const float* a, float alpha, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = alpha * a[idx];
    }
}

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void relu_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        out[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

// Fused: All three operations in one kernel
__global__ void fused_mar_kernel(const float* a, const float* b, float alpha, 
                                  float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float scaled = alpha * a[idx];
        float sum = scaled + b[idx];
        out[idx] = (sum > 0.0f) ? sum : 0.0f;
    }
}

double measure_bandwidth_gbps(size_t bytes, double time_ms) {
    return (bytes / 1e9) / (time_ms / 1000.0);
}

int main(int argc, char** argv) {
    int M = 4096;
    int N = 4096;
    float alpha = 0.5f;
    
    if (argc > 2) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
    }
    
    size_t n = M * N;
    size_t bytes = n * sizeof(float);
    
    std::cout << "=== Advanced Kernel Fusion: Multiply-Add-ReLU ===" << std::endl;
    std::cout << "Problem size: " << M << " x " << N << " = " << n << " elements" << std::endl;
    std::cout << "Operation: Y = max(α * A + B, 0) where α = " << alpha << std::endl;
    std::cout << "Use case: Scaled residual connection with activation (common in ResNets)" << std::endl << std::endl;
    
    // Allocate host memory
    std::vector<float> h_a(n), h_b(n), h_out1(n), h_out2(n);
    for (size_t i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f;  // [-2, 2]
        h_b[i] = static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f;  // [-2, 2]
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_temp1, *d_temp2, *d_out1, *d_out2;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_temp1, bytes);
    hipMalloc(&d_temp2, bytes);
    hipMalloc(&d_out1, bytes);
    hipMalloc(&d_out2, bytes);
    
    hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        fused_mar_kernel<<<gridSize, blockSize>>>(d_a, d_b, alpha, d_out1, n);
    }
    hipDeviceSynchronize();
    
    // Benchmark unfused: Multiply + Add + ReLU (3 kernels)
    std::cout << "--- Baseline (Unfused): Multiply + Add + ReLU ---" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    const int repeats = 100;
    for (int i = 0; i < repeats; i++) {
        multiply_kernel<<<gridSize, blockSize>>>(d_a, alpha, d_temp1, n);
        add_kernel<<<gridSize, blockSize>>>(d_temp1, d_b, d_temp2, n);
        relu_kernel<<<gridSize, blockSize>>>(d_temp2, d_out1, n);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double unfused_time = std::chrono::duration<double, std::milli>(end - start).count() / repeats;
    
    // Memory traffic: Read A, Write T1, Read T1, Read B, Write T2, Read T2, Write Out = 7 arrays
    size_t unfused_bytes = 7 * bytes;
    double unfused_bw = measure_bandwidth_gbps(unfused_bytes, unfused_time);
    
    std::cout << "  Kernels launched: 3" << std::endl;
    std::cout << "  Time: " << unfused_time << " ms" << std::endl;
    std::cout << "  Memory traffic: " << unfused_bytes / 1e6 << " MB" << std::endl;
    std::cout << "  Bandwidth: " << unfused_bw << " GB/s" << std::endl << std::endl;
    
    // Benchmark fused: Multiply-Add-ReLU (1 kernel)
    std::cout << "--- Optimized (Fused): Multiply-Add-ReLU ---" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeats; i++) {
        fused_mar_kernel<<<gridSize, blockSize>>>(d_a, d_b, alpha, d_out2, n);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double fused_time = std::chrono::duration<double, std::milli>(end - start).count() / repeats;
    
    // Memory traffic: Read A, Read B, Write Out = 3 arrays
    size_t fused_bytes = 3 * bytes;
    double fused_bw = measure_bandwidth_gbps(fused_bytes, fused_time);
    
    std::cout << "  Kernels launched: 1" << std::endl;
    std::cout << "  Time: " << fused_time << " ms" << std::endl;
    std::cout << "  Memory traffic: " << fused_bytes / 1e6 << " MB" << std::endl;
    std::cout << "  Bandwidth: " << fused_bw << " GB/s" << std::endl << std::endl;
    
    // Results
    double speedup = unfused_time / fused_time;
    double memory_reduction = 100.0 * (1.0 - (double)fused_bytes / unfused_bytes);
    double theoretical_max = (double)unfused_bytes / fused_bytes;
    
    std::cout << "=== Results ===" << std::endl;
    std::cout << "  Speedup: " << speedup << "x (" 
              << ((speedup - 1.0) * 100.0) << "% faster)" << std::endl;
    std::cout << "  Memory bandwidth reduction: " << memory_reduction << "%" << std::endl;
    std::cout << "  Theoretical max speedup: " << theoretical_max << "x" << std::endl;
    std::cout << "  Kernel launch overhead saved: 66.7% (3 kernels → 1 kernel)" << std::endl;
    
    // Verify correctness
    hipMemcpy(h_out1.data(), d_out1, bytes, hipMemcpyDeviceToHost);
    hipMemcpy(h_out2.data(), d_out2, bytes, hipMemcpyDeviceToHost);
    
    bool correct = true;
    float max_error = 0.0f;
    int error_count = 0;
    for (size_t i = 0; i < n; i++) {
        float expected = alpha * h_a[i] + h_b[i];
        expected = (expected > 0.0f) ? expected : 0.0f;
        float error1 = std::abs(h_out1[i] - expected);
        float error2 = std::abs(h_out2[i] - expected);
        max_error = std::max(max_error, std::max(error1, error2));
        if (error1 > 1e-4 || error2 > 1e-4) {
            correct = false;
            error_count++;
            if (error_count < 5) {
                std::cout << "  Mismatch at " << i << ": expected " << expected 
                         << ", got unfused=" << h_out1[i] << ", fused=" << h_out2[i] << std::endl;
            }
        }
    }
    
    std::cout << "  Correctness: " << (correct ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  Max error: " << max_error << std::endl;
    if (!correct) {
        std::cout << "  Errors: " << error_count << " / " << n << std::endl;
    }
    
    // Cleanup
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_temp1);
    hipFree(d_temp2);
    hipFree(d_out1);
    hipFree(d_out2);
    
    std::cout << "\n=== Practical Impact ===" << std::endl;
    std::cout << "This fusion pattern is ubiquitous in modern neural networks:" << std::endl;
    std::cout << "  - ResNet residual blocks: out = ReLU(α * shortcut + main_path)" << std::endl;
    std::cout << "  - Layer normalization: x_norm = (x - mean) / std (fused stats + scale)" << std::endl;
    std::cout << "  - Attention mechanisms: scaled dot-product with activation" << std::endl;
    std::cout << "\nFor a typical ResNet-50 with ~50M parameters and 100 such operations:" << std::endl;
    std::cout << "  Time saved per forward pass: ~" << (speedup - 1.0) * 100 * 0.05 << " ms" << std::endl;
    std::cout << "  Training speedup (1000 iterations): ~" << (speedup - 1.0) * 5 << " seconds" << std::endl;
    
    return correct ? 0 : 1;
}
