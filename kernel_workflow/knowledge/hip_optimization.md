# HIP Kernel Optimization Patterns

Patterns are ranked by priority. Higher priority (P0) = higher expected impact. Always start with P0 strategies before moving to lower priorities.

## P0: Algorithm Restructuring (Highest Impact)

### Template Parameterization
Replace runtime-sized arrays with compile-time template parameters. This eliminates register spilling from oversized local arrays and lets the compiler optimize aggressively.

**Pattern:**
```cpp
// BAD: Runtime-sized array forces worst-case register allocation
__global__ void kernel(int param, ...) {
    float vals[MAX_PARAM];  // MAX_PARAM=100 but actual param=5 → 95 wasted slots, massive spill
}

// GOOD: Template parameter → compiler knows exact size
template <int PARAM>
__global__ void kernel(...) {
    float vals[PARAM];  // Compiler allocates exactly PARAM registers, no spill
}

// Dispatch common values, generic fallback for the rest
void launch(int param, ...) {
    switch(param) {
        case 4:  kernel<4><<<grid, block>>>(...); break;
        case 8:  kernel<8><<<grid, block>>>(...); break;
        case 16: kernel<16><<<grid, block>>>(...); break;
        default: kernel<32><<<grid, block>>>(...); break;  // Generic fallback
    }
}
```
**Expected speedup**: 2-10x when original uses oversized arrays.

### Warp-Cooperative Algorithms (HIGHEST PRIORITY for search/scan kernels)

**THIS IS THE MOST IMPACTFUL OPTIMIZATION FOR KERNELS WHERE EACH THREAD SCANS A LARGE ARRAY.**
Instead of 1 thread per work item, use 1 wavefront (64 threads) per work item. Each lane processes a strided subset of the data, then results are merged across lanes via shared memory.

**When to use**: Any kernel where a single thread iterates over N elements (brute-force search, argmin/argmax over arrays, top-K selection, reduction over large arrays). Expected speedup: **5-30x**.

**Architecture**: With 256 threads per block and wavefront size 64:
- 4 wavefronts per block → 4 work items processed per block
- Grid: `dim3(DIVUP(M, 4), B)` where M = number of work items
- Each lane scans `N/64` elements (strided: `for (i = lane; i < N; i += 64)`)

**Pseudocode for warp-cooperative search:**

```
1. Thread indexing:
   warp_id = threadIdx.x / 64     (which wavefront within the block)
   lane    = threadIdx.x % 64     (which lane within the wavefront)
   item_id = blockIdx.x * 4 + warp_id  (which work item this wavefront handles)

2. Local scan: Each lane scans elements [lane, lane+64, lane+128, ...] up to N.
   Maintains a local best result (or local top-K sorted array for top-K problems).

3. Shared-memory merge: Write per-lane results to shared memory.
   Tree reduction in log2(64)=6 steps — each step merges pairs of results.
   For top-K: merge two sorted K-arrays, keep best K.

4. Final write: Lane 0 of each wavefront writes the merged result to global memory.
```

**Key implementation details:**
- Grid is `dim3(DIVUP(M, 4), B)` — each block handles 4 work items
- Shared memory: `4 * 64 * RESULT_SIZE * sizeof(...)` — must fit in 64KB LDS
- The merge tree runs in 6 steps (log2(64)=6), each step halving active lanes
- Use `__syncthreads()` after each merge step (block-level sync, shared memory is block-scoped)
- Template the result size so the compiler can unroll the merge and eliminate dead code
- Use `__launch_bounds__(256)` to help compiler optimize register allocation

**Expected speedup**: 5-30x for brute-force search/scan kernels. The speedup scales with N because each lane only scans N/64 elements.

### Algorithmic Complexity Reduction
Replace O(N) brute-force with O(log N) or O(1) approaches where possible: spatial hashing, KD-tree traversal, bitonic sort, prefix scan.

## P1: Data Reuse (Shared Memory / LDS Tiling)

### Tiled Data Loading
When multiple threads read the same global data, tile it into LDS (shared memory) to amortize global memory cost.

**Pattern:**
```cpp
__shared__ float tile[TILE_SIZE][D];  // D = feature dimension

for (int t = 0; t < N; t += TILE_SIZE) {
    // Cooperative load: each thread loads one element
    if (threadIdx.x < TILE_SIZE && t + threadIdx.x < N) {
        for (int d = 0; d < D; d++)
            tile[threadIdx.x][d] = data[(t + threadIdx.x) * D + d];
    }
    __syncthreads();

    // All threads read from LDS (fast) instead of global (slow)
    for (int j = 0; j < min(TILE_SIZE, N - t); j++) {
        float diff = query_val - tile[j][0];
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
- Ensure enough blocks to fill all CUs (detect the count with `rocminfo` — 304 on MI300X/MI325X, 228 on
  MI300A, 256 on MI350/MI355, reduced on MI308X; do not hard-code 304)
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
switch (param) {
    case 4:  launch_dispatch<4>(transposed, ..., stream); break;
    case 8:  launch_dispatch<8>(transposed, ..., stream); break;
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
