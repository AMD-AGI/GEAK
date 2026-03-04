---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: BVH_Node_Load

## Variant Context
- Input semantic type: BVH node memory access during tree traversal
- Datatype(s): BVHPackedNodeHalf (16-byte struct with fp32 bounds and int32 indices)
- Data representation: Packed 16-byte aligned BVH nodes
- Target architecture: gfx90a, gfx942 (AMD GPUs via HIP)

## Functionality
This kernel function loads BVH nodes from global memory during tree traversal. The `bvh_load_node` function is called frequently during ray-mesh intersection queries, making its performance critical. The function loads a 16-byte `BVHPackedNodeHalf` structure containing:
- 3 floats (x, y, z) for bounding box coordinates
- 1 packed integer with child index (31 bits) and leaf flag (1 bit)

---

## Optimization 1: Vectorized 16-byte Load for HIP (Experimental Branch)
- Commit ID: a9bb974
- Optimization type: Memory
- Summary: Added HIP-specific vectorized 16-byte load for better memory coalescing

- Detailed explanation:
  The original implementation used a simple struct load which may not utilize the full memory bandwidth on AMD GPUs. This optimization adds a HIP-specific path that uses a 16-byte aligned load helper to ensure the compiler generates efficient vectorized memory access instructions. The `BVHPackedNodeHalf` struct is exactly 16 bytes, making it ideal for a single 128-bit load operation.

- Code excerpt:
    ```cpp
    #elif defined(__HIPCC__)
    // HIP path with vectorized load
    __device__ inline wp::BVHPackedNodeHalf bvh_load_node(const wp::BVHPackedNodeHalf* nodes, int index)
    {
    #ifdef USE_LOAD4
        // AMD: Use vectorized 16-byte load for better memory coalescing
        // BVHPackedNodeHalf is exactly 16 bytes, so we can load it efficiently
        // Using a 16-byte aligned load improves memory bandwidth utilization
        struct alignas(16) LoadHelper { float data[4]; };
        const LoadHelper* __restrict__ ptr = reinterpret_cast<const LoadHelper* __restrict__>(nodes) + index;
        LoadHelper temp = *ptr;
        return *reinterpret_cast<const wp::BVHPackedNodeHalf*>(&temp);
    #else
        return nodes[index];
    #endif // USE_LOAD4
    }
    ```

- Evidence mapping:
  - 16-byte alignment: `struct alignas(16) LoadHelper` ensures proper alignment for vectorized load
  - Restrict pointer: `__restrict__` hint helps compiler optimize memory access
  - Reinterpret cast: Converts between LoadHelper and BVHPackedNodeHalf without copying
  - Conditional compilation: `#ifdef USE_LOAD4` allows toggling the optimization

---

## Optimization 2: Disable Shared Stack for Native Library Compilation
- Commit ID: 0c6157e
- Optimization type: Compute / Correctness
- Summary: Fixed incorrect shared memory usage in native library by disabling BVH_SHARED_STACK

- Detailed explanation:
  The original code conditionally enabled shared memory stack for BVH traversal when compiling for GPU (`__CUDA_ARCH__` or `__HIP_DEVICE_COMPILE__`). However, this caused issues because the native library (mesh.cpp, bvh.cpp) is compiled for the host, not the device. The actual ray tracing queries are executed by HIPRTC-compiled kernels at runtime. This fix disables the shared stack for native library compilation to avoid incorrect behavior.

- Code excerpt:
    ```cpp
    // BVH_SHARED_STACK: Disabled for native library compilation
    // The native library (mesh.cpp, bvh.cpp) only builds meshes and BVHs on the host.
    // The actual ray tracing queries are executed by HIPRTC-compiled kernels at runtime,
    // where we enable shared memory directly in the mesh_query_ray function.
    // Note: mesh.cpp defines __HIPRTC__ temporarily which would enable this incorrectly.
    #define BVH_SHARED_STACK 0
    ```

- Code excerpt (before - problematic):
    ```cpp
    #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    #define BVH_SHARED_STACK 1
    #else
    #define BVH_SHARED_STACK 0
    #endif
    ```

- Evidence mapping:
  - Unconditional disable: `#define BVH_SHARED_STACK 0` prevents shared memory issues
  - Documentation: Comments explain why this is necessary for the build model
  - HIPRTC handling: Shared memory is enabled in HIPRTC-compiled kernels instead

---

## Optimization 3: Separate CUDA and HIP Load Paths
- Commit ID: 0c6157e
- Optimization type: Compute / Portability
- Summary: Separated CUDA and HIP device code paths for bvh_load_node

- Detailed explanation:
  The original code used a combined `__CUDA_ARCH__ || __HIP_DEVICE_COMPILE__` check, but CUDA's `__ldg()` texture cache intrinsic is not available on HIP. This fix separates the paths: CUDA uses `__ldg()` for texture cache access, while HIP uses direct load (with optional vectorized load in experimental branches).

- Code excerpt:
    ```cpp
    // Device-optimized version with texture cache (only for actual device code)
    #if defined(__CUDA_ARCH__)
    __device__ inline wp::BVHPackedNodeHalf bvh_load_node(const wp::BVHPackedNodeHalf* nodes, int index)
    {
    #ifdef USE_LOAD4
        float4 f4 = __ldg((const float4*)(nodes)+index);
        return  (const wp::BVHPackedNodeHalf&)f4;
    #else
        return  nodes[index];
    #endif
    }
    #elif defined(__HIP_DEVICE_COMPILE__)
    __device__ inline wp::BVHPackedNodeHalf bvh_load_node(const wp::BVHPackedNodeHalf* nodes, int index)
    {
        // HIP device code - just use direct load (HIP doesn't have __ldg texture cache like CUDA)
        return  nodes[index];
    }
    #else
    // Host code and HIPRTC fallback
    CUDA_CALLABLE inline wp::BVHPackedNodeHalf bvh_load_node(const wp::BVHPackedNodeHalf* nodes, int index)
    {
        return  nodes[index];
    }
    #endif
    ```

- Evidence mapping:
  - CUDA path: Uses `__ldg()` for texture cache optimization
  - HIP path: Uses direct load (no `__ldg` equivalent)
  - Host path: Simple array access for CPU execution
  - Clear separation: Each platform has its own optimized implementation

---

## Note on Experimental Status

These optimizations are from experimental branches and may not be in the main branch. The vectorized load optimization (a9bb974) shows a promising direction for improving memory bandwidth utilization on AMD GPUs, but requires further testing and validation before merging to main.

The current main branch uses a simpler implementation that works correctly but may not achieve optimal memory bandwidth: