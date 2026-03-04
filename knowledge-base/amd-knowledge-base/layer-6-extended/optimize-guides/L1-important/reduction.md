---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "hip", "kernel", "reduction", "parallel-computing"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/reduction.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Reduction

Reduction is a fundamental parallel computing technique that condenses arrays into smaller outputs or single values using binary operations. This tutorial extends work by Mark Harris, demonstrating how modern GPU hardware has evolved while exploring key optimization strategies.

## The Algorithm

Reduction operates across multiple domains—functional programming calls it a "fold," C++ knows it as `std::accumulate`, and C++17 introduced `std::reduce`. The operation requires a neutral element (identity) that doesn't alter results when applied. Common applications include computing sums, normalizing datasets, and finding maximum values.

The technique supports parallelization by allowing identity elements to be inserted strategically, enabling partial parallel computations whose results can later be combined.

## Reduction on GPUs

Successful GPU reduction implementations require understanding HIP's threading model and synchronization mechanics. Unlike CPU code, GPU synchronization is expensive; efficient algorithms minimize cross-block synchronization by enabling independent multiprocessor progress.

### Naive Shared Reduction

This foundational approach distributes work across blocks using a tree structure. All threads load global data into shared memory, perform tree-like reduction, then write partial results back to global memory. Subsequent kernel launches combine these partials until a single value remains.

The factor calculation determines new sizes after each pass:

```cpp
std::size_t factor = block_size;
auto new_size = [factor](const std::size_t actual) {
    return actual / factor + (actual % factor == 0 ? 0 : 1);
};
```

The kernel structure loads data safely using zero-padding for threads without unique inputs, synchronizes across the block, performs reduction, and writes results:

```cpp
template<typename T, typename F>
__global__ void kernel(T* front, T* back, F op, T zero_elem, uint32_t front_size) {
    extern __shared__ T shared[];

    const uint32_t tid = threadIdx.x, bid = blockIdx.x, gid = bid * blockDim.x + tid;
    shared[tid] = (gid < front_size) ? front[gid] : zero_elem;
    __syncthreads();

    for (uint32_t i = 1; i < blockDim.x; i *= 2) {
        if (tid % (2 * i) == 0)
            shared[tid] = op(shared[tid], shared[tid + i]);
        __syncthreads();
    }

    if (tid == 0)
        back[bid] = shared[0];
}
```

However, the conditional `tid % (2 * i) == 0` creates significant thread divergence—warps remain partially active longer than necessary.

### Reducing Thread Divergence

Restructuring the indexing scheme keeps memory access patterns consistent while reassigning thread roles:

```cpp
for (uint32_t i = 1; i < blockDim.x; i *= 2) {
    if (uint32_t j = 2 * i * tid; j < blockDim.x)
        shared[j] = op(shared[j], shared[j + i]);
    __syncthreads();
}
```

Inactive threads now accumulate uniformly toward higher indices, though this introduces bank conflicts in shared memory.

### Resolving Bank Conflicts

Both AMD and NVIDIA organize shared memory into banks. When different threads simultaneously access the same bank, the hardware serializes accesses, degrading performance. The solution forms continuous thread activity ranges with coalesced memory access:

```cpp
for (uint32_t i = blockDim.x / 2; i != 0; i /= 2) {
    if (tid < i)
        shared[tid] = op(shared[tid], shared[tid + i]);
    __syncthreads();
}
```

This maintains uniform memory access patterns—consecutive threads access consecutive locations.

### Utilize Upper Half of the Block

The previous implementation wastes half the block initially. By having each thread process two inputs from global memory:

```cpp
const uint32_t gid = bid * (blockDim.x * 2) + tid;
shared[tid] = op(read_global_safe(gid), read_global_safe(gid + blockDim.x));
```

All threads perform meaningful work from the start, improving utilization.

### Unroll All Loops

Since warps execute in lockstep, once only a single warp participates meaningfully in reduction, loop unrolling eliminates unnecessary synchronization. Making block size a compile-time constant enables both static shared memory allocation and compiler optimizations:

```cpp
template<uint32_t BlockSize, uint32_t WarpSize, typename T, typename F>
__global__ __launch_bounds__(BlockSize) void kernel(...) {
    __shared__ T shared[BlockSize];
    // ... loop unrolling for warp-level operations
}
```

### Communicate Using Warp-Collective Functions

Instead of shared memory for intra-warp communication, shuffle instructions directly copy register values between lanes—faster than going through memory:

```cpp
if (tid < WarpSize) {
    T res = op(shared[tid], shared[tid + WarpSize]);
    // Unrolled warp reduction using __shfl_down()
}
```

### Prefer Warp Communication Over Shared

Rather than keeping shared communication until the end, perform parallel warp reductions across all active threads, combining only final warp results through shared memory:

```cpp
static constexpr uint32_t WarpCount = BlockSize / WarpSize;
__shared__ T shared[WarpCount];

T res = op(read_global_safe(gid), read_global_safe(gid + blockDim.x));
// Multiple warp reductions in parallel, then combine results
```

### Amortize Bookkeeping Variable Overhead

Processing multiple items per thread in a single kernel launch reduces register overhead from repeated kernel bookkeeping. The `ItemsPerThread` template parameter multiplies workload within one kernel:

```cpp
template<uint32_t BlockSize, uint32_t WarpSize, uint32_t ItemsPerThread>
__global__ __launch_bounds__(BlockSize) void kernel(...)
```

Each thread loads and reduces `ItemsPerThread` values before participating in block-level reduction, minimizing the register cost of kernel iteration indices.

### Two-Pass Reduction

Launching only as many blocks as a subsequent kernel can process in a single block, then combining multiple input passes before tree reduction, eliminates one or two kernel launches for large inputs.

### Global Data Share

On AMD hardware, the Global Data Share (GDS) provides on-chip memory accessible by all multiprocessors. Exploiting scheduling predictability and AMD-specific Global Wave Sync features, the last block can collect results from all others in GDS without additional kernel launches—though this requires inline AMDGCN assembly.

## Conclusion

GPU reduction optimization balances numerous competing resource constraints. While this tutorial explores techniques beyond typical practical limits, real-world applications often employ algorithm libraries providing near-optimal solutions across diverse workloads. The ideal approach depends on context: single-device full reductions differ from scenarios with multiple blocks performing parallel reductions or multi-device scenarios requiring different considerations.
