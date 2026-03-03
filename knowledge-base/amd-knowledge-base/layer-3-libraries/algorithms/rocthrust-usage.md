# rocThrust Usage Guide

*Comprehensive guide to AMD's parallel algorithms library for GPU computing*

## Overview

rocThrust is AMD's GPU-accelerated parallel algorithms library, equivalent to NVIDIA's Thrust. It provides high-level C++ template algorithms for parallel computing on AMD GPUs using HIP. rocThrust enables rapid development of GPU applications with STL-like syntax.

## Features

- **Parallel Algorithms**: Sort, reduce, scan, transform, and more
- **STL-Like Interface**: Familiar C++ template syntax  
- **GPU Acceleration**: Optimized for AMD GPUs via HIP
- **Memory Management**: Automatic device memory handling
- **Iterator Support**: Compatible with custom iterators
- **Backend Abstraction**: Works with HIP backend

## Installation

### ROCm Installation
```bash
# Install ROCm (includes rocThrust)
sudo apt update
sudo apt install rocm-dev rocthrust-dev

# Verify installation
find /opt/rocm -name "*thrust*"
```

### From Source
```bash
git clone https://github.com/ROCmSoftwarePlatform/rocThrust.git
cd rocThrust
mkdir build && cd build
cmake -DBUILD_TEST=OFF ..
make -j$(nproc)
sudo make install
```

## Basic Usage

### Headers and Namespace
```cpp
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <hip/hip_runtime.h>

using namespace thrust;
```

### Vector Operations
```cpp
// Host and device vectors
thrust::host_vector<int> h_vec(1000000);
thrust::device_vector<int> d_vec(1000000);

// Fill with data
thrust::fill(h_vec.begin(), h_vec.end(), 42);

// Copy between host and device
d_vec = h_vec;  // Host to device
h_vec = d_vec;  // Device to host

// Direct device initialization
thrust::device_vector<float> d_floats(1000, 3.14f);
```

## Core Algorithms

### Transform Operations
```cpp
// Simple transformation
struct square_functor {
    __device__ float operator()(float x) const {
        return x * x;
    }
};

void transform_example() {
    thrust::device_vector<float> input(1000000);
    thrust::device_vector<float> output(1000000);
    
    // Fill with sequence
    thrust::sequence(input.begin(), input.end(), 1.0f);
    
    // Transform: square all elements
    thrust::transform(input.begin(), input.end(), 
                     output.begin(), square_functor());
    
    // Using lambda (C++11)
    thrust::transform(input.begin(), input.end(), output.begin(),
                     [] __device__ (float x) { return x * x; });
}
```

### Reductions
```cpp
void reduction_examples() {
    thrust::device_vector<int> d_vec(1000000);
    thrust::sequence(d_vec.begin(), d_vec.end(), 1);
    
    // Sum all elements
    int sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0);
    
    // Find maximum
    int max_val = thrust::reduce(d_vec.begin(), d_vec.end(), 
                                INT_MIN, thrust::maximum<int>());
    
    // Custom reduction (product)
    struct multiply_op {
        __device__ float operator()(float a, float b) const {
            return a * b;
        }
    };
    
    thrust::device_vector<float> small_vec(10, 2.0f);
    float product = thrust::reduce(small_vec.begin(), small_vec.end(),
                                  1.0f, multiply_op());
}
```

### Sorting
```cpp
void sorting_examples() {
    const int n = 1000000;
    thrust::device_vector<int> keys(n);
    thrust::device_vector<float> values(n);
    
    // Generate random data
    thrust::sequence(keys.begin(), keys.end());
    thrust::random_shuffle(keys.begin(), keys.end());
    
    // Simple sort
    thrust::sort(keys.begin(), keys.end());
    
    // Sort by key-value pairs
    thrust::fill(values.begin(), values.end(), 1.0f);
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    
    // Custom comparator
    struct greater_than_op {
        __device__ bool operator()(int a, int b) const {
            return a > b;
        }
    };
    
    thrust::sort(keys.begin(), keys.end(), greater_than_op());
}
```

### Scan (Prefix Sum)
```cpp
void scan_examples() {
    thrust::device_vector<int> input(8);
    thrust::device_vector<int> output(8);
    
    // Input: [1, 2, 3, 4, 5, 6, 7, 8]
    thrust::sequence(input.begin(), input.end(), 1);
    
    // Inclusive scan (prefix sum)
    // Output: [1, 3, 6, 10, 15, 21, 28, 36]
    thrust::inclusive_scan(input.begin(), input.end(), output.begin());
    
    // Exclusive scan
    // Output: [0, 1, 3, 6, 10, 15, 21, 28]
    thrust::exclusive_scan(input.begin(), input.end(), output.begin());
    
    // Custom scan operation
    thrust::inclusive_scan(input.begin(), input.end(), output.begin(),
                          thrust::maximum<int>());
}
```

## Advanced Algorithms

### Gather and Scatter
```cpp
void gather_scatter_example() {
    // Source data
    thrust::device_vector<float> source = {10, 20, 30, 40, 50};
    
    // Indices for gathering
    thrust::device_vector<int> indices = {4, 1, 3, 0, 2};
    thrust::device_vector<float> gathered(5);
    
    // Gather: result = [50, 20, 40, 10, 30]
    thrust::gather(indices.begin(), indices.end(),
                   source.begin(), gathered.begin());
    
    // Scatter (inverse of gather)
    thrust::device_vector<float> scattered(5, 0);
    thrust::scatter(gathered.begin(), gathered.end(),
                   indices.begin(), scattered.begin());
}
```

### Unique and Remove Operations
```cpp
void unique_remove_example() {
    thrust::device_vector<int> data = {1, 2, 2, 3, 3, 3, 4, 5, 5};
    
    // Sort first (required for unique)
    thrust::sort(data.begin(), data.end());
    
    // Remove duplicates
    auto new_end = thrust::unique(data.begin(), data.end());
    data.resize(new_end - data.begin());
    // Result: [1, 2, 3, 4, 5]
    
    // Remove specific values
    data = {1, 2, 3, 2, 4, 2, 5};
    auto new_end2 = thrust::remove(data.begin(), data.end(), 2);
    data.resize(new_end2 - data.begin());
    // Result: [1, 3, 4, 5]
}
```

### Partition Operations
```cpp
struct is_even {
    __device__ bool operator()(int x) const {
        return x % 2 == 0;
    }
};

void partition_example() {
    thrust::device_vector<int> data(10);
    thrust::sequence(data.begin(), data.end(), 1); // [1,2,3,...,10]
    
    // Partition even/odd numbers
    auto partition_point = thrust::partition(data.begin(), data.end(), is_even());
    
    int even_count = partition_point - data.begin();
    std::cout << "Even numbers: " << even_count << std::endl;
}
```

## Custom Functors and Lambdas

### Complex Functors
```cpp
template<typename T>
struct saxpy_functor {
    const T a;
    
    saxpy_functor(T _a) : a(_a) {}
    
    __device__ T operator()(const T& x, const T& y) const {
        return a * x + y;
    }
};

void saxpy_example() {
    const int n = 1000000;
    thrust::device_vector<float> x(n, 1.0f);
    thrust::device_vector<float> y(n, 2.0f);
    thrust::device_vector<float> result(n);
    
    float a = 3.0f;
    
    // Compute: result = a * x + y
    thrust::transform(x.begin(), x.end(), y.begin(), result.begin(),
                     saxpy_functor<float>(a));
}
```

### Zip Iterators
```cpp
void zip_iterator_example() {
    thrust::device_vector<float> x(4, 1.0f);
    thrust::device_vector<float> y(4, 2.0f);
    thrust::device_vector<float> z(4);
    
    // Zip vectors together for parallel processing
    auto first = thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin()));
    auto last  = thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end()));
    
    // Transform using zip iterator
    thrust::transform(first, last, z.begin(),
        [] __device__ (thrust::tuple<float, float> t) {
            return thrust::get<0>(t) + thrust::get<1>(t);
        });
}
```

## Memory Management

### Memory Pools
```cpp
class ThrustMemoryPool {
    thrust::device_vector<char> pool;
    size_t pool_size;
    char* pool_ptr;
    
public:
    ThrustMemoryPool(size_t size) : pool_size(size), pool(size) {
        pool_ptr = thrust::raw_pointer_cast(pool.data());
    }
    
    template<typename T>
    thrust::device_ptr<T> allocate(size_t n) {
        // Simple linear allocator
        static size_t offset = 0;
        if (offset + n * sizeof(T) > pool_size) {
            throw std::bad_alloc();
        }
        
        T* ptr = reinterpret_cast<T*>(pool_ptr + offset);
        offset += n * sizeof(T);
        return thrust::device_ptr<T>(ptr);
    }
};
```

### Raw Pointers Integration
```cpp
void raw_pointer_integration() {
    // Allocate with HIP
    float* d_raw_ptr;
    hipMalloc(&d_raw_ptr, 1000 * sizeof(float));
    
    // Wrap with thrust::device_ptr
    thrust::device_ptr<float> d_thrust_ptr(d_raw_ptr);
    
    // Use thrust algorithms
    thrust::fill(d_thrust_ptr, d_thrust_ptr + 1000, 42.0f);
    
    // Convert back to raw pointer if needed
    float* raw_again = thrust::raw_pointer_cast(d_thrust_ptr);
    
    hipFree(d_raw_ptr);
}
```

## Performance Optimization

### Avoiding Temporary Allocations
```cpp
// Inefficient: creates temporary vectors
void inefficient_version() {
    thrust::device_vector<float> a(1000000), b(1000000), result(1000000);
    
    // This creates temporary vectors
    result = a + b * 2.0f;  // Avoid this pattern
}

// Efficient: explicit operations
void efficient_version() {
    thrust::device_vector<float> a(1000000), b(1000000), result(1000000);
    
    // Step by step to avoid temporaries
    thrust::transform(b.begin(), b.end(), result.begin(),
                     thrust::multiplies<float>(), 2.0f);
    thrust::transform(a.begin(), a.end(), result.begin(), result.begin(),
                     thrust::plus<float>());
}
```

### Custom Allocators
```cpp
template<typename T>
class pinned_allocator {
public:
    typedef T value_type;
    
    T* allocate(size_t n) {
        T* ptr;
        hipHostMalloc(&ptr, n * sizeof(T), hipHostMallocDefault);
        return ptr;
    }
    
    void deallocate(T* ptr, size_t) {
        hipHostFree(ptr);
    }
};

// Use pinned memory for faster transfers
typedef thrust::host_vector<float, pinned_allocator<float>> pinned_vector;
```

## Scientific Computing Applications

### Numerical Integration (Trapezoidal Rule)
```cpp
struct trapezoidal_integration {
    float h;  // step size
    
    trapezoidal_integration(float _h) : h(_h) {}
    
    __device__ float operator()(float f_prev, float f_curr) const {
        return 0.5f * h * (f_prev + f_curr);
    }
};

float integrate_function() {
    const int n = 1000000;
    thrust::device_vector<float> x(n), y(n);
    
    // Generate x values
    thrust::sequence(x.begin(), x.end(), 0.0f, 1.0f / (n - 1));
    
    // Compute y = sin(x)
    thrust::transform(x.begin(), x.end(), y.begin(),
        [] __device__ (float x) { return sin(x); });
    
    // Trapezoidal integration
    float h = 1.0f / (n - 1);
    thrust::device_vector<float> integrals(n - 1);
    
    thrust::transform(y.begin(), y.end() - 1, y.begin() + 1,
                     integrals.begin(), trapezoidal_integration(h));
    
    return thrust::reduce(integrals.begin(), integrals.end(), 0.0f);
}
```

### Histogram Computation
```cpp
void compute_histogram() {
    const int n = 1000000;
    const int num_bins = 100;
    
    thrust::device_vector<float> data(n);
    thrust::device_vector<int> histogram(num_bins, 0);
    
    // Generate random data [0, 1)
    thrust::sequence(data.begin(), data.end());
    thrust::transform(data.begin(), data.end(), data.begin(),
        [] __device__ (float x) { return fmod(x * 0.123456f, 1.0f); });
    
    // Compute bin indices
    thrust::device_vector<int> bin_indices(n);
    thrust::transform(data.begin(), data.end(), bin_indices.begin(),
        [num_bins] __device__ (float x) { 
            return min((int)(x * num_bins), num_bins - 1);
        });
    
    // Sort by bin index
    thrust::sort(bin_indices.begin(), bin_indices.end());
    
    // Count occurrences
    thrust::device_vector<int> bin_keys(num_bins);
    thrust::device_vector<int> counts(num_bins);
    
    auto end_pair = thrust::reduce_by_key(
        bin_indices.begin(), bin_indices.end(),
        thrust::make_constant_iterator(1),
        bin_keys.begin(), counts.begin()
    );
}
```

## Machine Learning Applications

### K-Means Clustering Helper
```cpp
struct distance_to_centroid {
    float cx, cy;  // centroid coordinates
    
    distance_to_centroid(float _cx, float _cy) : cx(_cx), cy(_cy) {}
    
    __device__ float operator()(thrust::tuple<float, float> point) const {
        float px = thrust::get<0>(point);
        float py = thrust::get<1>(point);
        return sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
    }
};

void kmeans_distance_computation() {
    const int n_points = 100000;
    const int k = 5;
    
    thrust::device_vector<float> points_x(n_points);
    thrust::device_vector<float> points_y(n_points);
    thrust::device_vector<float> centroids_x(k);
    thrust::device_vector<float> centroids_y(k);
    
    // For each centroid, compute distances to all points
    for (int i = 0; i < k; i++) {
        thrust::device_vector<float> distances(n_points);
        
        auto points_zip = thrust::make_zip_iterator(
            thrust::make_tuple(points_x.begin(), points_y.begin()));
        
        thrust::transform(points_zip, points_zip + n_points,
                         distances.begin(),
                         distance_to_centroid(centroids_x[i], centroids_y[i]));
    }
}
```

### Feature Normalization
```cpp
void normalize_features() {
    const int n_samples = 10000;
    const int n_features = 100;
    
    thrust::device_vector<float> features(n_samples * n_features);
    
    // Normalize each feature column
    for (int f = 0; f < n_features; f++) {
        // Extract feature column using strided iterator
        auto feature_begin = thrust::make_permutation_iterator(
            features.begin() + f,
            thrust::make_transform_iterator(
                thrust::counting_iterator<int>(0),
                [n_features] __device__ (int i) { return i * n_features; }
            )
        );
        
        // Compute mean and std
        float mean = thrust::reduce(feature_begin, feature_begin + n_samples) / n_samples;
        
        // Compute variance
        thrust::device_vector<float> deviations(n_samples);
        thrust::transform(feature_begin, feature_begin + n_samples,
                         deviations.begin(),
                         [mean] __device__ (float x) { 
                             float dev = x - mean;
                             return dev * dev; 
                         });
        
        float variance = thrust::reduce(deviations.begin(), deviations.end()) / n_samples;
        float std_dev = sqrt(variance);
        
        // Normalize: (x - mean) / std
        thrust::transform(feature_begin, feature_begin + n_samples,
                         feature_begin,
                         [mean, std_dev] __device__ (float x) {
                             return (x - mean) / std_dev;
                         });
    }
}
```

## Error Handling and Debugging

### Exception Handling
```cpp
void safe_thrust_operations() {
    try {
        thrust::device_vector<float> vec(1000000);
        
        // Operations that might throw
        thrust::fill(vec.begin(), vec.end(), 1.0f);
        float sum = thrust::reduce(vec.begin(), vec.end());
        
    } catch (const thrust::system_error& e) {
        std::cerr << "Thrust system error: " << e.what() << std::endl;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
    }
}
```

### Performance Debugging
```cpp
void profile_thrust_operations() {
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    thrust::device_vector<float> vec(10000000);
    
    // Time sorting operation
    hipEventRecord(start);
    thrust::sort(vec.begin(), vec.end());
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    
    float ms;
    hipEventElapsedTime(&ms, start, stop);
    std::cout << "Sort time: " << ms << " ms" << std::endl;
    
    hipEventDestroy(start);
    hipEventDestroy(stop);
}
```

## Best Practices

### Algorithm Selection
```cpp
// Choose appropriate algorithm based on data size
template<typename Iterator>
void conditional_sort(Iterator first, Iterator last) {
    size_t n = last - first;
    
    if (n < 1000) {
        // Use CPU for small datasets
        thrust::sort(thrust::host, first, last);
    } else {
        // Use GPU for large datasets  
        thrust::sort(thrust::device, first, last);
    }
}
```

### Memory Access Optimization
```cpp
// Prefer structure of arrays over array of structures
struct Point3D {
    float x, y, z;
};

// Less efficient (AoS)
thrust::device_vector<Point3D> points_aos(n);

// More efficient (SoA)
thrust::device_vector<float> x_coords(n);
thrust::device_vector<float> y_coords(n);
thrust::device_vector<float> z_coords(n);
```

### Iterator Reuse
```cpp
// Create iterators once and reuse
class OptimizedProcessor {
    thrust::counting_iterator<int> count_begin;
    thrust::constant_iterator<float> const_begin;
    
public:
    OptimizedProcessor() 
        : count_begin(0)
        , const_begin(1.0f) {}
    
    void process_data(thrust::device_vector<float>& data) {
        // Reuse pre-created iterators
        thrust::transform(count_begin, count_begin + data.size(),
                         const_begin, data.begin(),
                         thrust::multiplies<float>());
    }
};
```

## Resources

### Documentation
- [rocThrust Documentation](https://rocm.docs.amd.com/projects/rocThrust/en/latest/)
- [Thrust Quick Start Guide](https://nvidia.github.io/cccl/thrust/getting_started.html)

### Related Libraries
- rocPRIM - Low-level parallel primitives (see ROCm documentation)
- hipCUB - Collective primitives (see ROCm documentation)

### Performance Guides
- [GPU Optimization](../../best-practices/performance/gpu-optimization.md)
- [Memory Optimization](../../best-practices/performance/memory-optimization.md)

---
*Tags: rocthrust, parallel-algorithms, gpu, stl, thrust, hip, rocm, sorting, reduction*
*Estimated reading time: 45 minutes*