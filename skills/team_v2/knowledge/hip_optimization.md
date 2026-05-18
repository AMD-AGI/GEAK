# HIP Kernel Optimization Patterns

Patterns are ranked by priority. Higher priority (P0) = higher expected impact. Always start with P0 strategies before moving to lower priorities.

## P0: Algorithm Restructuring (Highest Impact)

### Template Parameterization
Replace runtime-sized arrays with compile-time template parameters. This eliminates register spilling from oversized local arrays and lets the compiler optimize aggressively.

**Pattern:**
```cpp
// BAD: Runtime-sized array forces worst-case register allocation
__global__ void kernel(int k, ...) {
    float vals[MAX_K];  // MAX_K=100 but actual k=5 → 95 wasted slots, massive spill
}

// GOOD: Template parameter → compiler knows exact size
template <int K>
__global__ void kernel(...) {
    float vals[K];  // Compiler allocates exactly K registers, no spill
}

// Dispatch with switch or if-constexpr
void launch(int k, ...) {
    switch(k) {
        case 3:  kernel<3><<<grid, block>>>(...); break;
        case 5:  kernel<5><<<grid, block>>>(...); break;
        case 10: kernel<10><<<grid, block>>>(...); break;
        default: kernel<16><<<grid, block>>>(...); break;  // Generic fallback
    }
}
```
**Expected speedup**: 2-10x when original uses oversized arrays.

### Warp-Cooperative Algorithms (HIGHEST PRIORITY for search/scan kernels)

**THIS IS THE MOST IMPACTFUL OPTIMIZATION FOR KERNELS WHERE EACH THREAD SCANS A LARGE ARRAY.**
Instead of 1 thread per work item, use 1 wavefront (64 threads) per work item. Each lane processes a strided subset of the data, then results are merged across lanes via shared memory.

**When to use**: Any kernel where a single thread iterates over N elements (brute-force search, KNN, nearest neighbor, argmin/argmax over arrays, histogram, top-K selection). Expected speedup: **5-30x**.

**Architecture**: With 256 threads per block and wavefront size 64:
- 4 wavefronts per block → 4 work items processed per block
- Grid: `dim3(DIVUP(M, 4), B)` where M = number of work items
- Each lane scans `N/64` elements (strided: `for (i = lane; i < N; i += 64)`)

**Complete pattern for Top-K search with warp-cooperative merge:**

```cpp
#include <float.h>

// Template on K (number of results) for register efficiency
template <int K>
__global__ __launch_bounds__(256)
void topk_search_warp(int N, int M,
                      const float *__restrict__ data,
                      const float *__restrict__ queries,
                      int *__restrict__ result_idx) {
    // 4 wavefronts per block, 64 lanes per wavefront
    const int warp_id = threadIdx.x >> 6;        // 0..3
    const int lane    = threadIdx.x & 63;         // 0..63
    const int query_id = blockIdx.x * 4 + warp_id;
    if (query_id >= M) return;

    // ---- Step 1: Each lane finds its local top-K ----
    float best_d[K];
    int   best_i[K];
    #pragma unroll
    for (int j = 0; j < K; j++) { best_d[j] = FLT_MAX; best_i[j] = 0; }

    // Strided scan: lane 0 checks indices 0,64,128,...; lane 1 checks 1,65,129,...
    for (int i = lane; i < N; i += 64) {
        float d = compute_distance(query_id, i);  // Your distance/score function
        if (d < best_d[K - 1]) {
            best_d[K - 1] = d;
            best_i[K - 1] = i;
            // Insertion sort to maintain sorted order (O(K), fine for small K)
            #pragma unroll
            for (int p = K - 1; p > 0; p--) {
                if (best_d[p] < best_d[p - 1]) {
                    float td = best_d[p]; best_d[p] = best_d[p-1]; best_d[p-1] = td;
                    int   ti = best_i[p]; best_i[p] = best_i[p-1]; best_i[p-1] = ti;
                } else break;
            }
        }
    }

    // ---- Step 2: Merge top-K results across 64 lanes via shared memory ----
    // Each lane has K sorted results. We do a log2(64)=6 step merge tree.
    __shared__ float s_dist[4][64 * 16];  // Max K=16 per lane, 4 warps
    __shared__ int   s_idx[4][64 * 16];

    // Write local results to shared memory
    #pragma unroll
    for (int j = 0; j < K; j++) {
        s_dist[warp_id][lane * K + j] = best_d[j];
        s_idx[warp_id][lane * K + j]  = best_i[j];
    }
    __syncthreads();

    // Tree reduction: merge pairs, then pairs of pairs, etc.
    for (int stride = 1; stride < 64; stride *= 2) {
        if ((lane & (stride * 2 - 1)) == 0 && (lane + stride) < 64) {
            float *a_d = &s_dist[warp_id][lane * K];
            int   *a_i = &s_idx[warp_id][lane * K];
            float *b_d = &s_dist[warp_id][(lane + stride) * K];
            int   *b_i = &s_idx[warp_id][(lane + stride) * K];

            // Merge two sorted K-arrays, keep top K
            float merged_d[16];  // Stack buffer (max K)
            int   merged_i[16];
            int ai = 0, bi = 0;
            #pragma unroll
            for (int j = 0; j < K; j++) {
                if (ai < K && (bi >= K || a_d[ai] <= b_d[bi])) {
                    merged_d[j] = a_d[ai]; merged_i[j] = a_i[ai]; ai++;
                } else {
                    merged_d[j] = b_d[bi]; merged_i[j] = b_i[bi]; bi++;
                }
            }
            #pragma unroll
            for (int j = 0; j < K; j++) {
                a_d[j] = merged_d[j]; a_i[j] = merged_i[j];
            }
        }
        __syncthreads();
    }

    // ---- Step 3: Lane 0 writes final result ----
    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < K; j++) {
            result_idx[query_id * K + j] = s_idx[warp_id][j];
        }
    }
}
```

**Key implementation details:**
- Grid is `dim3(DIVUP(M, 4), B)` — each block handles 4 queries
- Shared memory size: `4 * 64 * K * sizeof(float+int)` — fits in 64KB LDS for K≤16
- The merge tree runs in 6 steps (log2(64)=6), each step halving active lanes
- Use `__syncthreads()` after each merge step (block-level sync needed because shared memory is block-scoped)
- Template K so the compiler can unroll the merge and eliminate dead code
- Use `__launch_bounds__(256)` to help compiler optimize register allocation

**Launcher with template dispatch:**
```cpp
void launch(int b, int n, int m, int k, ..., hipStream_t stream) {
    dim3 blocks(DIVUP(m, 4), b);
    dim3 threads(256);
    switch (k) {
        case 3:  topk_search_warp<3><<<blocks, threads, 0, stream>>>(...); break;
        case 5:  topk_search_warp<5><<<blocks, threads, 0, stream>>>(...); break;
        case 10: topk_search_warp<10><<<blocks, threads, 0, stream>>>(...); break;
        default: topk_search_warp<16><<<blocks, threads, 0, stream>>>(...); break;
    }
}
```

**Expected speedup**: 5-30x for brute-force search kernels. The speedup scales with N because each lane only scans N/64 elements.

### Algorithmic Complexity Reduction
Replace O(N) brute-force with O(log N) or O(1) approaches where possible: spatial hashing, KD-tree traversal, bitonic sort, prefix scan.

## P1: Data Reuse (Shared Memory / LDS Tiling)

### Tiled Data Loading
When multiple threads read the same global data, tile it into LDS (shared memory) to amortize global memory cost.

**Pattern:**
```cpp
__shared__ float tile[TILE_SIZE][3];

for (int t = 0; t < N; t += TILE_SIZE) {
    // Cooperative load: each thread loads one element
    if (threadIdx.x < TILE_SIZE && t + threadIdx.x < N) {
        tile[threadIdx.x][0] = xyz[(t + threadIdx.x) * 3 + 0];
        tile[threadIdx.x][1] = xyz[(t + threadIdx.x) * 3 + 1];
        tile[threadIdx.x][2] = xyz[(t + threadIdx.x) * 3 + 2];
    }
    __syncthreads();

    // All threads read from LDS (fast) instead of global (slow)
    for (int j = 0; j < min(TILE_SIZE, N - t); j++) {
        float dx = qx - tile[j][0];
        // ...
    }
    __syncthreads();
}
```
**Expected speedup**: 2-5x for memory-bound kernels with data reuse.

### Register Blocking
Keep frequently accessed data in registers across loop iterations. Unroll inner loops to maximize register reuse.

## P2: Memory Access Optimization

### Coalesced Access
Ensure adjacent threads access adjacent memory addresses. Stride-1 access is ideal.

**Pattern:**
```cpp
// BAD: Stride-3 access (AoS layout)
float x = data[tid * 3 + 0];
float y = data[tid * 3 + 1];

// GOOD: Stride-1 access (SoA layout)
float x = data_x[tid];
float y = data_y[tid];
```

### Vectorized Loads
Use `float2`, `float4`, or `int4` for wide loads that reduce instruction count.

```cpp
// BAD: 4 separate loads
float a = data[i]; float b = data[i+1]; float c = data[i+2]; float d = data[i+3];

// GOOD: 1 vectorized load
float4 v = *reinterpret_cast<float4*>(&data[i]);
```

### Non-Temporal Hints
For streaming access patterns (data used once), use `__builtin_nontemporal_load` to avoid polluting cache.

## P3: Compute Optimization

### Branchless Patterns
Replace data-dependent branches with predicated operations to avoid wavefront divergence.

```cpp
// BAD: Divergent branch
if (a < b) { result = a; } else { result = b; }

// GOOD: Branchless
result = fminf(a, b);

// For conditional swap
float lo = fminf(a, b);
float hi = fmaxf(a, b);
```

### Instruction-Level Parallelism
Interleave independent operations to hide latency. The compiler does this partially, but manual interleaving helps.

### FMA (Fused Multiply-Add)
Use `fmaf(a, b, c)` instead of `a * b + c` for better precision and throughput (1 instruction vs 2).

### Loop Unrolling
Use `#pragma unroll` for small, fixed-trip-count loops. Use `#pragma unroll N` to partially unroll large loops.

## P4: Launch Configuration

### Block Size Tuning
- Must be multiple of 64 (wavefront size on AMD)
- Common sweet spots: 64, 128, 256
- Use `__launch_bounds__(max_threads, min_waves)` to guide compiler

### Occupancy vs Register Pressure
Higher occupancy hides latency but limits registers per thread. For register-heavy kernels, lower occupancy with more registers can be faster.

### Grid Size
- Ensure enough blocks to fill all 304 CUs
- For small problems: use persistent threads (fewer blocks, each does more work)

## P5: Autotuning

### Parameter Search
Tune block size, tile size, unroll factor, waves_per_eu via compile-time dispatch.

```cpp
template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void kernel(...) { /* ... */ }

// Try multiple configurations
void launch(...) {
    // Best config found by profiling
    kernel<256, 32><<<grid, 256>>>(...);
}
```

## Hipify Safety Rules

When writing HIP kernels, the build system may use `hipify-perl` to convert CUDA syntax to HIP. This tool rewrites `<<<>>>` kernel launch syntax into `hipLaunchKernelGGL()` calls. Several code patterns break during this transformation:

### NEVER: Macros with if/else around kernel launches
```cpp
// BAD: hipify mangles the else clause into "elsehipLaunchKernelGGL(...)"
#define LAUNCH(K) \
    if (transposed) kernel<K, true><<<grid, block>>>(...); \
    else kernel<K, false><<<grid, block>>>(...)

// GOOD: Use a template function instead
template <int K>
static void launch_dispatch(bool transposed, ..., hipStream_t stream) {
    dim3 blocks(...);
    dim3 threads(256);
    if (transposed) {
        kernel<K, true><<<blocks, threads, 0, stream>>>(...);
    } else {
        kernel<K, false><<<blocks, threads, 0, stream>>>(...);
    }
}

// Then dispatch:
switch (k) {
    case 3:  launch_dispatch<3>(transposed, ..., stream); break;
    case 5:  launch_dispatch<5>(transposed, ..., stream); break;
    default: launch_dispatch<16>(transposed, ..., stream); break;
}
```

### NEVER: Ternary operators with kernel launches
```cpp
// BAD: ternary with <<<>>> gets mangled
(flag ? kernel_a : kernel_b)<<<grid, block>>>(...);

// GOOD: explicit if/else in a function
if (flag) kernel_a<<<grid, block>>>(...);
else kernel_b<<<grid, block>>>(...);
```

### SAFE: Template functions with if/else (no macros)
Regular C++ template functions with if/else and `<<<>>>` inside the function body are fine — hipify correctly transforms each launch independently.

### SAFE: `if constexpr` inside kernels
`if constexpr (TRANSPOSED)` inside `__global__` kernel functions works correctly because `<<<>>>` is not involved at the if-level.

## Compound Strategy Compatibility

Strategies that compose well together (apply both):
- Template parameterization + Warp-cooperative → excellent (both reduce register pressure)
- LDS tiling + Coalesced access → excellent (tiling enables coalescing)
- Template parameterization + Launch bounds → good (compiler optimizes better)
- Vectorized loads + LDS tiling → good (faster tile loading)
- Loop unrolling + Register blocking → good (enables reuse)

Strategies that conflict (pick one):
- Two different tiling schemes → conflict (LDS size limit)
- Warp-cooperative + High occupancy → may conflict (more registers needed)
- Aggressive unrolling + High occupancy → conflict (register pressure)
