# HIP Kernel Optimization Patterns for AMD MI300X

Detection: `.hip`, `.cu`, `.cpp` files containing `__global__` kernel functions.

## Priority Hierarchy

**Priority 0 — Algorithmic rewrites** (highest impact, try first):
Fundamentally change how the kernel computes its result. Template parameterization, warp-cooperative algorithms, work decomposition, data structure changes.

**Priority 1 — Data reuse and tiling**:
LDS tiling to cache shared data, multi-query workgroups, register blocking, loop tiling.

**Priority 2 — Memory access optimization**:
Coalesced access, vectorized loads/stores, memory layout transformation, prefetching.

**Priority 3 — Compute optimization**:
Strength reduction, FMA chains, branchless patterns, loop unrolling, MFMA utilization.

**Priority 4 — Parallelism and occupancy**:
Block size tuning, occupancy optimization, persistent kernels, grid-stride loops.

**Priority 5 — Tuning and dispatch** (lowest, try last):
Launch config tuning, shape-specialized dispatch, compiler hints.

---

## Algorithmic Patterns (Priority 0)

### Template Parameterization
Fixed-size arrays like `float data[100]` waste registers when runtime size is small. Use C++ templates:

```cpp
template <int K>
__global__ void kernel(...) {
    float data[K];  // Compiler knows exact size → optimal register allocation
    #pragma unroll
    for (int i = 0; i < K; i++) { ... }
}

// Dispatcher
void launcher(int k, ...) {
    switch(k) {
        case 3:  kernel<3><<<grid, block>>>(...); break;
        case 5:  kernel<5><<<grid, block>>>(...); break;
        case 10: kernel<10><<<grid, block>>>(...); break;
        default: kernel<MAX_K><<<grid, block>>>(...); break;
    }
}
```

Impact: 3-10x speedup when baseline uses oversized arrays. Enables `#pragma unroll`, reduces register pressure, improves occupancy.

### Warp-Cooperative Algorithms
Instead of 1 thread per work item, use a full wavefront (64 threads) cooperatively:

```cpp
template <int K>
__global__ __launch_bounds__(64)
void warp_cooperative_kernel(...) {
    int lane = threadIdx.x;  // 0..63
    
    // Phase 1: Each lane processes N/64 elements
    float local_results[K];
    for (int i = lane; i < N; i += 64) {
        process(data[i], local_results);
    }
    
    // Phase 2: Merge via shared memory reduction
    __shared__ float s_results[64 * K];
    store_to_shared(s_results, lane, local_results);
    __syncthreads();
    
    for (int stride = 32; stride >= 1; stride >>= 1) {
        if (lane < stride)
            merge(s_results, lane, lane + stride);
        __syncthreads();
    }
    
    // Lane 0 writes final result
    if (lane == 0) write_output(s_results);
}
```

Impact: Distributes O(N) work across 64 threads → 64x fewer iterations per thread. Especially effective for search, selection, and reduction kernels.

### Work Decomposition
Split a kernel's work along a new dimension to increase parallelism:

```cpp
// Before: 1 workgroup per output element, serial inner loop of N iterations
// After: K workgroups per output element, each processes N/K iterations, then reduce

template <int SPLIT_K>
__global__ void split_kernel(float *partial_results, ...) {
    int output_idx = blockIdx.x;
    int split_idx = blockIdx.y;  // Which slice of the inner loop
    int start = split_idx * (N / SPLIT_K);
    int end = min(start + (N / SPLIT_K), N);
    
    float result = 0;
    for (int i = start; i < end; i++)
        result += compute(i, output_idx);
    
    partial_results[output_idx * SPLIT_K + split_idx] = result;
}
// Follow-up kernel reduces partial_results
```

### Insertion Sort vs Heap for Small K
For top-K selection with K ≤ 16, insertion sort outperforms heap:
- Fewer branches → better for `#pragma unroll`
- Simpler control flow → compiler optimizes better
- Sorted order maintained → early exit possible

---

## Data Reuse and Tiling (Priority 1)

### LDS Tiling for Shared Data
When multiple threads/queries read the same global data, cache it in LDS:

```cpp
// Multiple queries in one workgroup share reference data tiles
__shared__ float s_tile[TILE_SIZE * DIM];

for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
    // Cooperative load: all threads load part of the tile
    for (int i = threadIdx.x; i < TILE_SIZE * DIM; i += blockDim.x)
        s_tile[i] = global_data[tile_start * DIM + i];
    __syncthreads();
    
    // All threads process against the shared tile
    for (int j = 0; j < TILE_SIZE; j++)
        process(s_tile + j * DIM, my_query);
    __syncthreads();
}
```

Impact: Reduces global memory reads by `queries_per_workgroup` factor. Converts memory-bound kernel to compute-bound.

### Multi-Query Workgroups
Process multiple independent queries per workgroup, sharing the same data tile:

```cpp
#define QUERIES_PER_WG 4
#define THREADS_PER_QUERY 64  // 1 wavefront per query

__global__ __launch_bounds__(QUERIES_PER_WG * THREADS_PER_QUERY)
void multi_query_kernel(...) {
    int wave_id = threadIdx.x / THREADS_PER_QUERY;   // which query
    int lane_id = threadIdx.x % THREADS_PER_QUERY;    // lane within wave
    int query_idx = blockIdx.x * QUERIES_PER_WG + wave_id;
    
    __shared__ float s_ref_tile[TILE_SIZE * 3];  // Shared across all queries
    
    for (int tile = 0; tile < N; tile += TILE_SIZE) {
        // All threads cooperate to load tile
        for (int i = threadIdx.x; i < TILE_SIZE * 3; i += blockDim.x)
            s_ref_tile[i] = ref_points[tile * 3 + i];
        __syncthreads();
        
        // Each wave processes its query against the tile
        for (int j = lane_id; j < TILE_SIZE; j += THREADS_PER_QUERY)
            process_point(s_ref_tile + j * 3, query_data[query_idx]);
        __syncthreads();
    }
}
```

### Register Blocking
Keep frequently accessed data in registers, not LDS or global memory:

```cpp
// Load query coordinates once into registers
float qx = query[0], qy = query[1], qz = query[2];

// Process multiple elements per iteration
for (int i = 0; i < N; i += 4) {
    float4 rx = load_float4(ref + i * 3 + 0);  // 4 x-coords
    float4 ry = load_float4(ref + i * 3 + 4);  // 4 y-coords  
    // ... compute 4 distances simultaneously
}
```

---

## Memory Access (Priority 2)

### Coalesced Access
Consecutive threads must access consecutive addresses:

```cpp
// BAD: strided — each thread reads 3 apart
float x = data[tid * 3 + 0];  // stride-3 → bandwidth waste

// GOOD: SoA layout
float x = data_x[tid];  // consecutive → fully coalesced

// ALTERNATIVE: vectorized AoS
float3 point = ((float3*)data)[tid];  // single 12-byte load
```

### Vectorized Loads
Use wide loads to reduce memory instruction count:

```cpp
// 128-bit load: 4 floats in 1 instruction
float4 val = *reinterpret_cast<const float4*>(&data[tid * 4]);

// For 3D points with padding to float4:
float4 point = *reinterpret_cast<const float4*>(&points[i * 4]); // x,y,z,pad
float x = point.x, y = point.y, z = point.z;
```

### Prefetching
For kernels with predictable access patterns:

```cpp
// Software prefetch the next tile while processing current
__builtin_nontemporal_load(&data[next_tile_offset]);
```

---

## Compute Optimization (Priority 3)

### FMA Chains
Use fused multiply-add for distance computations:

```cpp
// Compiler usually fuses, but be explicit:
float d2 = __fmaf_rn(dx, dx, __fmaf_rn(dy, dy, dz * dz));
```

### Branchless Patterns
Replace branches with conditional moves:

```cpp
// BAD: branch divergence
if (d2 < best_dist) { best_dist = d2; best_idx = i; }

// GOOD: branchless (compiler may do this, but helps ensure it)
int cond = (d2 < best_dist);
best_dist = cond ? d2 : best_dist;
best_idx = cond ? i : best_idx;
```

### Loop Unrolling
```cpp
#pragma unroll           // Full unroll (only for small, known trip count)
#pragma unroll 4         // Partial unroll by factor 4
for (int i = 0; i < K; i++) { ... }
```

---

## Parallelism and Occupancy (Priority 4)

### Block Size Selection
- 64 threads = 1 wavefront (minimum, good for warp-cooperative kernels)
- 256 threads = 4 wavefronts (good default, hides latency)
- 512 threads = 8 wavefronts (max for many kernels due to LDS/register limits)

Use `__launch_bounds__(MAX_THREADS_PER_BLOCK)` to help the compiler optimize register allocation.

### Grid-Stride Loops
For kernels that need more work per thread:

```cpp
__global__ void kernel(int N, ...) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        process(i);
}
```

---

## Search and Selection Workloads

These patterns are especially effective for KNN, ball query, point-in-box, and similar spatial search kernels:

1. **Size-specialized variants**: Template on K/nsample to eliminate oversized arrays
2. **Wavefront-cooperative search**: 64 threads cooperate per query, each scanning N/64 elements
3. **LDS-tiled reference data**: Cache reference points in LDS, share across queries
4. **Sorted insertion over heap**: For K ≤ 16, insertion sort beats heap
5. **Two-phase search**: Coarse spatial filter → fine exact search
6. **Multi-query workgroups**: Group nearby queries to share data loads

## Wrapper-Level Patterns

### Handle Transposed Input Natively
Instead of transposing in Python wrapper (which launches GPU memcpy kernels), pass a flag and handle both layouts:
```cpp
__device__ void load_point(const float *base, int idx, int stride,
                           bool transposed, float &x, float &y, float &z) {
    if (transposed) {            // SoA: (B, 3, N)
        x = base[idx]; y = base[stride + idx]; z = base[2*stride + idx];
    } else {                     // AoS: (B, N, 3)
        x = base[idx*3]; y = base[idx*3+1]; z = base[idx*3+2];
    }
}
```
The `transposed` branch is uniform (all threads take the same path) — zero cost on AMD CDNA.

### Direct Output Layout
Write output in the format the consumer expects:
```cpp
// Instead of: idx[bs*m*k + query*k + i]  (B,M,K) — needs Python transpose
// Write:      idx[bs*k*m + i*m + query]   (B,K,M) — consumer-ready
```

## Build Cache Warning

JIT-compiled kernels (via `torch.utils.cpp_extension.load()`) cache compiled binaries. After editing kernel source:
```bash
rm -rf $REPO_ROOT/build
rm -rf ~/.cache/torch_extensions/*/KERNEL_NAME/
```
Always clear both caches before re-benchmarking.
