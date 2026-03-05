# rocRAND Usage Guide

*Comprehensive guide to AMD's random number generation library for GPU computing*

## Overview

rocRAND is AMD's GPU-accelerated random number generation library, part of the ROCm ecosystem. It provides high-performance pseudorandom and quasi-random number generation for scientific computing, AI, and simulation workloads.

## Features

- **Multiple Generators**: XORWOW, MRG32k3a, Philox4x32-10, MTGP32, SOBOL32/64
- **GPU Acceleration**: Parallel generation on AMD GPUs
- **Host & Device APIs**: Generate on CPU or GPU
- **Multiple Distributions**: Uniform, normal, log-normal, Poisson
- **Backward Compatibility**: Drop-in replacement for cuRAND

## Installation

### ROCm Installation
```bash
# Install ROCm (includes rocRAND)
sudo apt update
sudo apt install rocm-dev rocrand-dev

# Verify installation
pkg-config --modversion rocrand
```

### From Source
```bash
git clone https://github.com/ROCmSoftwarePlatform/rocRAND.git
cd rocRAND
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

## Basic Usage

### Header and Initialization
```cpp
#include <rocrand/rocrand.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// Initialize generator
rocrand_generator generator;
rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT);

// Set seed for reproducibility
rocrand_set_seed(generator, 12345ULL);
```

### Generating Random Numbers
```cpp
// Generate uniform random floats [0.0, 1.0)
void generate_uniform_floats() {
    const size_t n = 1000000;
    float* d_data;
    
    // Allocate device memory
    hipMalloc(&d_data, n * sizeof(float));
    
    // Generate random numbers
    rocrand_generate_uniform(generator, d_data, n);
    
    // Copy to host
    std::vector<float> h_data(n);
    hipMemcpy(h_data.data(), d_data, n * sizeof(float), 
              hipMemcpyDeviceToHost);
    
    hipFree(d_data);
}

// Generate normal distribution (mean=0, stddev=1)
void generate_normal_floats() {
    const size_t n = 1000000;
    float* d_data;
    
    hipMalloc(&d_data, n * sizeof(float));
    
    // Generate normally distributed numbers
    rocrand_generate_normal(generator, d_data, n, 0.0f, 1.0f);
    
    std::vector<float> h_data(n);
    hipMemcpy(h_data.data(), d_data, n * sizeof(float),
              hipMemcpyDeviceToHost);
    
    hipFree(d_data);
}
```

## Random Number Generators

### XORWOW Generator
```cpp
// Fast, good quality for most applications
rocrand_generator xorwow_gen;
rocrand_create_generator(&xorwow_gen, ROCRAND_RNG_PSEUDO_XORWOW);
rocrand_set_seed(xorwow_gen, 1234ULL);

// Generate integers
unsigned int* d_ints;
hipMalloc(&d_ints, n * sizeof(unsigned int));
rocrand_generate(xorwow_gen, d_ints, n);
```

### Philox Generator  
```cpp
// High quality, cryptographically secure
rocrand_generator philox_gen;
rocrand_create_generator(&philox_gen, ROCRAND_RNG_PSEUDO_PHILOX4_32_10);
rocrand_set_seed(philox_gen, 5678ULL);

// Generate with subsequence support
rocrand_set_offset(philox_gen, 1000000ULL);
```

### SOBOL Quasi-Random Generator
```cpp
// For quasi-Monte Carlo methods
rocrand_generator sobol_gen;
rocrand_create_generator(&sobol_gen, ROCRAND_RNG_QUASI_SOBOL32);

// Generate quasi-random sequence
float* d_quasi;
hipMalloc(&d_quasi, n * sizeof(float));
rocrand_generate_uniform(sobol_gen, d_quasi, n);
```

## Probability Distributions

### Uniform Distribution
```cpp
// Integer uniform distribution [0, 2^32)
rocrand_generate(generator, d_uint_data, n);

// Float uniform distribution [0.0, 1.0)
rocrand_generate_uniform(generator, d_float_data, n);

// Double uniform distribution
rocrand_generate_uniform_double(generator, d_double_data, n);
```

### Normal Distribution
```cpp
// Standard normal (mean=0, stddev=1)
rocrand_generate_normal(generator, d_data, n, 0.0f, 1.0f);

// Custom mean and standard deviation
float mean = 10.0f, stddev = 2.5f;
rocrand_generate_normal(generator, d_data, n, mean, stddev);

// Double precision normal
rocrand_generate_normal_double(generator, d_double_data, n, 0.0, 1.0);
```

### Log-Normal Distribution
```cpp
// Log-normal distribution
float log_mean = 1.0f, log_stddev = 0.5f;
rocrand_generate_log_normal(generator, d_data, n, log_mean, log_stddev);
```

### Poisson Distribution
```cpp
// Poisson distribution with lambda parameter
double lambda = 3.5;
unsigned int* d_poisson;
hipMalloc(&d_poisson, n * sizeof(unsigned int));
rocrand_generate_poisson(generator, d_poisson, n, lambda);
```

## Advanced Usage

### Device API (Kernel-Level Generation)
```cpp
#include <rocrand/rocrand_kernel.h>

__global__ void kernel_random_generation(float* output, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize per-thread generator
    rocrand_state_xorwow state;
    rocrand_init(seed, idx, 0, &state);
    
    // Generate random numbers in kernel
    output[idx] = rocrand_uniform(&state);
}

// Launch kernel
dim3 block(256);
dim3 grid((n + block.x - 1) / block.x);
kernel_random_generation<<<grid, block>>>(d_output, seed);
```

### Multiple Streams for Parallel Generation
```cpp
void parallel_generation() {
    const int num_streams = 4;
    rocrand_generator generators[num_streams];
    hipStream_t streams[num_streams];
    
    // Create generators and streams
    for (int i = 0; i < num_streams; i++) {
        rocrand_create_generator(&generators[i], ROCRAND_RNG_PSEUDO_XORWOW);
        rocrand_set_seed(generators[i], 1000 + i);
        hipStreamCreate(&streams[i]);
        rocrand_set_stream(generators[i], streams[i]);
    }
    
    // Generate in parallel
    for (int i = 0; i < num_streams; i++) {
        rocrand_generate_uniform(generators[i], d_data[i], n);
    }
    
    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        hipStreamSynchronize(streams[i]);
        rocrand_destroy_generator(generators[i]);
        hipStreamDestroy(streams[i]);
    }
}
```

## Performance Optimization

### Optimal Buffer Sizes
```cpp
// Use large buffers for better performance
const size_t optimal_size = 1024 * 1024;  // 1M numbers

// Reuse buffers to avoid allocation overhead
class RandomNumberPool {
    float* d_buffer;
    size_t buffer_size;
    rocrand_generator gen;
    
public:
    RandomNumberPool(size_t size) : buffer_size(size) {
        hipMalloc(&d_buffer, size * sizeof(float));
        rocrand_create_generator(&gen, ROCRAND_RNG_PSEUDO_XORWOW);
    }
    
    void generate_batch(float* output, size_t count) {
        size_t remaining = count;
        size_t offset = 0;
        
        while (remaining > 0) {
            size_t batch = std::min(remaining, buffer_size);
            rocrand_generate_uniform(gen, d_buffer, batch);
            
            hipMemcpy(output + offset, d_buffer, 
                     batch * sizeof(float), hipMemcpyDeviceToDevice);
            
            offset += batch;
            remaining -= batch;
        }
    }
};
```

### Memory Coalescing
```cpp
__global__ void coalesced_random_kernel(float* output, 
                                       unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    rocrand_state_xorwow state;
    rocrand_init(seed, tid, 0, &state);
    
    // Coalesced memory access
    output[tid] = rocrand_uniform(&state);
}
```

## Scientific Computing Applications

### Monte Carlo Integration
```cpp
__global__ void monte_carlo_pi(float* x, float* y, int* hits, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x_val = x[idx];
        float y_val = y[idx];
        
        // Check if point is inside unit circle
        if (x_val * x_val + y_val * y_val <= 1.0f) {
            atomicAdd(hits, 1);
        }
    }
}

void estimate_pi() {
    const int n = 10000000;
    float *d_x, *d_y;
    int *d_hits, h_hits = 0;
    
    // Allocate memory
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));
    hipMalloc(&d_hits, sizeof(int));
    hipMemcpy(d_hits, &h_hits, sizeof(int), hipMemcpyHostToDevice);
    
    // Generate random points
    rocrand_generate_uniform(generator, d_x, n);
    rocrand_generate_uniform(generator, d_y, n);
    
    // Run Monte Carlo
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    monte_carlo_pi<<<grid, block>>>(d_x, d_y, d_hits, n);
    
    // Calculate pi estimate
    hipMemcpy(&h_hits, d_hits, sizeof(int), hipMemcpyDeviceToHost);
    float pi_estimate = 4.0f * h_hits / n;
    
    std::cout << "Pi estimate: " << pi_estimate << std::endl;
}
```

### Random Matrix Initialization
```cpp
void initialize_random_matrix(float* matrix, int rows, int cols, 
                             float mean = 0.0f, float stddev = 1.0f) {
    const size_t size = rows * cols;
    
    // Generate normally distributed values
    rocrand_generate_normal(generator, matrix, size, mean, stddev);
}

// Xavier initialization for neural networks
void xavier_init(float* weights, int fan_in, int fan_out) {
    float stddev = sqrt(2.0f / (fan_in + fan_out));
    initialize_random_matrix(weights, fan_in, fan_out, 0.0f, stddev);
}
```

## AI/ML Applications

### Dropout Implementation
```cpp
__global__ void dropout_kernel(float* input, float* output, 
                              float* mask, float dropout_rate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        if (mask[idx] < dropout_rate) {
            output[idx] = 0.0f;
        } else {
            output[idx] = input[idx] / (1.0f - dropout_rate);
        }
    }
}

void apply_dropout(float* input, float* output, int n, float rate) {
    float* d_mask;
    hipMalloc(&d_mask, n * sizeof(float));
    
    // Generate random mask
    rocrand_generate_uniform(generator, d_mask, n);
    
    // Apply dropout
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    dropout_kernel<<<grid, block>>>(input, output, d_mask, rate, n);
    
    hipFree(d_mask);
}
```

### Data Augmentation
```cpp
void random_crop_augmentation(float* images, int* crop_x, int* crop_y, 
                             int batch_size, int img_width, int img_height,
                             int crop_width, int crop_height) {
    // Generate random crop positions
    rocrand_generate_uniform(generator, temp_floats, batch_size * 2);
    
    // Convert to crop coordinates
    convert_to_crop_coords<<<grid, block>>>(temp_floats, crop_x, crop_y,
                                          batch_size, img_width - crop_width,
                                          img_height - crop_height);
}
```

## Error Handling

### Status Checking
```cpp
void check_rocrand_status(rocrand_status status) {
    if (status != ROCRAND_STATUS_SUCCESS) {
        const char* error_msg;
        switch (status) {
            case ROCRAND_STATUS_NOT_INITIALIZED:
                error_msg = "Generator not initialized";
                break;
            case ROCRAND_STATUS_ALLOCATION_FAILED:
                error_msg = "Memory allocation failed";
                break;
            case ROCRAND_STATUS_TYPE_ERROR:
                error_msg = "Wrong generator type";
                break;
            case ROCRAND_STATUS_OUT_OF_RANGE:
                error_msg = "Parameter out of range";
                break;
            default:
                error_msg = "Unknown error";
        }
        throw std::runtime_error(std::string("rocRAND error: ") + error_msg);
    }
}

// Usage with error checking
rocrand_status status = rocrand_generate_uniform(generator, d_data, n);
check_rocrand_status(status);
```

## Best Practices

### Memory Management
```cpp
// RAII wrapper for rocRAND generator
class RocrandGenerator {
    rocrand_generator gen;
    
public:
    RocrandGenerator(rocrand_rng_type type = ROCRAND_RNG_PSEUDO_DEFAULT) {
        rocrand_create_generator(&gen, type);
    }
    
    ~RocrandGenerator() {
        rocrand_destroy_generator(gen);
    }
    
    rocrand_generator get() { return gen; }
    
    void set_seed(unsigned long long seed) {
        rocrand_set_seed(gen, seed);
    }
};
```

### Performance Tips
1. **Buffer Reuse**: Avoid frequent allocations
2. **Large Batches**: Generate many numbers at once
3. **Appropriate Generator**: Choose based on quality needs
4. **Stream Usage**: Overlap generation with computation
5. **Device API**: Use for small amounts in kernels

### Reproducibility
```cpp
// Ensure reproducible results
void setup_reproducible_generation() {
    // Fixed seed
    rocrand_set_seed(generator, 42ULL);
    
    // Fixed offset for subsequences
    rocrand_set_offset(generator, 0ULL);
    
    // Consistent generator type
    rocrand_generator repro_gen;
    rocrand_create_generator(&repro_gen, ROCRAND_RNG_PSEUDO_XORWOW);
}
```

## Resources

### Documentation
- [rocRAND Documentation](https://rocm.docs.amd.com/projects/rocRAND/en/latest/)
- [ROCm Math Libraries](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/3rd-party-support-matrix.html)

### Related Libraries
- [rocBLAS](../blas/rocblas-usage.md) - Linear algebra operations
- [MIOpen](../ml-primitives/miopen-usage.md) - Machine learning primitives

### Performance Guides
- [GPU Optimization](../../best-practices/performance/gpu-optimization.md)
- [Memory Optimization](../../best-practices/performance/memory-optimization.md)

---
*Tags: rocrand, random-numbers, monte-carlo, gpu, rocm, pseudo-random, quasi-random*
*Estimated reading time: 35 minutes*