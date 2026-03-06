// Minimal HIP elementwise add kernel for profiling and optimization
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Simple elementwise add kernel
__global__ void elementwise_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// FP16 version
__global__ void elementwise_add_kernel_fp16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// Optimized version with vectorized loads (fp16)
__global__ void elementwise_add_kernel_fp16_vec(
    const __half2* __restrict__ a,
    const __half2* __restrict__ b,
    __half2* __restrict__ c,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd2(a[idx], b[idx]);
    }
}

bool validate_results(const std::vector<float>& a, const std::vector<float>& b, 
                     const std::vector<float>& c, float tolerance = 1e-5f) {
    for (size_t i = 0; i < a.size(); i++) {
        float expected = a[i] + b[i];
        if (std::abs(c[i] - expected) > tolerance) {
            std::cerr << "Validation failed at index " << i 
                      << ": expected " << expected << ", got " << c[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // Parse arguments
    int M = 4096;
    int N = 4096;
    bool do_validation = true;
    int warmup = 5;
    int repeat = 50;
    
    if (argc > 1) M = std::atoi(argv[1]);
    if (argc > 2) N = std::atoi(argv[2]);
    if (argc > 3) do_validation = (std::atoi(argv[3]) != 0);
    if (argc > 4) warmup = std::atoi(argv[4]);
    if (argc > 5) repeat = std::atoi(argv[5]);
    
    int n = M * N;
    size_t bytes = n * sizeof(float);
    
    std::cout << "Elementwise Add: M=" << M << ", N=" << N << std::endl;
    std::cout << "Total elements: " << n << std::endl;
    
    // Allocate host memory
    std::vector<float> h_a(n), h_b(n), h_c(n);
    
    // Initialize with random data
    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX * 5.0f;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX * 5.0f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));
    
    // Launch configuration
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    std::cout << "Grid size: " << gridSize << ", Block size: " << blockSize << std::endl;
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        hipLaunchKernelGGL(elementwise_add_kernel, dim3(gridSize), dim3(blockSize), 0, 0,
                          d_a, d_b, d_c, n);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i++) {
        hipLaunchKernelGGL(elementwise_add_kernel, dim3(gridSize), dim3(blockSize), 0, 0,
                          d_a, d_b, d_c, n);
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / repeat;
    std::cout << "Average time: " << elapsed_ms << " ms" << std::endl;
    
    // Calculate bandwidth
    double bandwidth_gb_s = (3.0 * bytes / 1e9) / (elapsed_ms / 1000.0);
    std::cout << "Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    
    // Validate
    bool pass = true;
    if (do_validation) {
        HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));
        pass = validate_results(h_a, h_b, h_c);
        std::cout << (pass ? "PASSED" : "FAILED") << std::endl;
    }
    
    // Cleanup
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    
    return pass ? 0 : 1;
}
