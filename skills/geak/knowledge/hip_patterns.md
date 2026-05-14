# HIP Kernel Optimization Patterns for AMD GPUs

## Backend Detection
A HIP kernel is identified by:
- File extensions: `.hip`, `.cu`, `.cpp` with ROCm/HIP includes
- Contains `__global__` or `__device__` functions
- Uses `hipLaunchKernelGGL` or `<<<blocks, threads>>>` launch syntax

## Planning Policy

When optimizing HIP kernels:
- Fill most task slots with "Prefer First" strategies
- Only add launch/dispatch/wrapper tasks after at least 3 kernel-body tasks
- Leave GPUs idle rather than padding with low-priority work

## Prefer First (High Priority)

1. **Algorithmic HIP kernel-body rewrites** that change the search/reduction/tiling structure.
2. **Common kernel optimizations** driven by the hottest profiled path, not by generic occupancy or launch heuristics.

### Memory-Bound Additions
- Coalescing, vectorized access, or LDS staging when they directly raise effective bandwidth on the hot path
- Global-memory traffic reduction by fusing steps or recomputing cheap values instead of reloading them

### Compute-Bound Additions
- Instruction-count reduction, branch simplification, and cheaper per-thread math in hottest loops
- Wave intrinsics, MFMA-friendly decomposition, or unrolled inner loops

### Latency-Bound Additions
- Branchless/control-flow simplification that reduces serialized decision cost in short kernels
- Operation-specific specialization so the hot path doesn't pay for generic functionality
- Wavefront-cooperative or persistent-work patterns that amortize per-launch overhead

### LDS-Bound Additions
- LDS-bank-conflict reduction and staged-access redesign inside the kernel body
- Register-vs-LDS tradeoff changes that lower LDS pressure on the hot path

## Consider Next (Medium Priority)

- Kernel-body memory-layout, register-pressure, or LDS-usage cleanup
- Size-specialized kernel variants when one generic implementation serves mismatched regimes
- Wavefront-level memory-access reordering or bank-conflict reduction (memory-bound)

## Deprioritize Until Later (Low Priority)

- Launch-config or occupancy-only tuning
- Wrapper/dispatch/copy-path edits unless profiling shows they dominate total time

## HIP-Specific Optimization Techniques

### Thread Block Sizing
```cpp
// Common block sizes for AMD GPUs
dim3 block(256);      // 4 wavefronts
dim3 block(512);      // 8 wavefronts (max per CU without occupancy limits)
dim3 grid((N + block.x - 1) / block.x);
```

### Memory Coalescing
```cpp
// GOOD: Coalesced - adjacent threads access adjacent addresses
int idx = blockIdx.x * blockDim.x + threadIdx.x;
output[idx] = input[idx];

// BAD: Strided - causes multiple cache line fetches
int idx = threadIdx.x * stride;
output[idx] = input[idx];
```

### Vectorized Loads/Stores
```cpp
// Use float4 for 128-bit loads (maximum HBM efficiency)
float4 data = *reinterpret_cast<float4*>(&input[idx * 4]);

// Process 4 elements per thread
output[idx*4 + 0] = process(data.x);
output[idx*4 + 1] = process(data.y);
output[idx*4 + 2] = process(data.z);
output[idx*4 + 3] = process(data.w);
```

### LDS (Shared Memory) Usage
```cpp
__shared__ float smem[BLOCK_SIZE + 1];  // +1 padding to avoid bank conflicts

// Stage global reads through LDS
smem[threadIdx.x] = input[global_idx];
__syncthreads();
// Now access smem with better patterns
```

### Wave Intrinsics
```cpp
// Cross-lane operations (AMD wavefront = 64 threads)
float val = __shfl_xor(my_val, lane_mask);     // Butterfly reduction
float sum = __reduce_add_sync(0xFFFFFFFF, val); // Warp-level reduce

// Ballot and vote
uint64_t mask = __ballot(predicate);
bool any = __any(predicate);
```

### Branchless Patterns (for latency-bound kernels)
```cpp
// GOOD: Branchless min/max
int result = (a < b) ? a : b;  // Compiles to v_min

// GOOD: Branchless select
float result = predicate * a + (1 - predicate) * b;

// BAD: Branch-heavy
if (a < b) result = a;
else result = b;
```

### Search Workload Optimizations
For binary search, KNN, and pointer-chasing workloads:
- **Size-specialized variants**: Separate kernels for small/medium/huge data
- **Wavefront-cooperative search**: Multiple threads collaborate on one search query
- **Branchless binary search**: Use conditional move instead of branch
- **Coarse-index narrowing**: Pre-build pivot table to reduce search depth
- **Items-per-thread tuning**: Balance between thread count and work per thread

### Register Pressure Management
- Use `__launch_bounds__(max_threads, min_blocks)` to hint register allocation
- Reduce live variables in hot loops
- Consider splitting large kernels into multiple passes if register pressure limits occupancy
- **CRITICAL**: Watch for fixed-size arrays in per-thread storage (e.g., `float arr[100]`). If the actual
  runtime size is much smaller (e.g., k=5), the wasted registers destroy occupancy. Use C++ templates
  with constexpr size parameters to eliminate this waste.

### Template Parameterization (High Impact)
When kernel code uses fixed-size arrays that could be much smaller at runtime:
```cpp
// BAD: Fixed-size array wastes registers when k is small
__global__ void kernel(int k, ...) {
    float data[100];  // Always uses 100 registers even when k=5
}

// GOOD: Template parameter eliminates waste
template <int K>
__global__ void kernel(int n, ...) {
    float data[K];  // Only K registers used
}

// Dispatch via switch:
switch (k) {
    case 1:  kernel<1><<<...>>>(...); break;
    case 5:  kernel<5><<<...>>>(...); break;
    case 16: kernel<16><<<...>>>(...); break;
    default: kernel_generic<<<...>>>(...); break;
}
```
This pattern often yields 3-10x speedup due to dramatically improved occupancy.

### Warp-Cooperative Algorithms (High Impact)
Instead of 1 thread per work item, use a full wavefront (64 threads) cooperatively:
```cpp
template <int K>
__global__ __launch_bounds__(64)
void kernel_warp_cooperative(int n, ...) {
    int work_item = blockIdx.x;  // 1 block = 1 wavefront = 1 work item
    int lane_id = threadIdx.x;   // 0..63

    // Phase 1: Each thread processes N/64 elements in strided pattern
    float local_best[K];
    for (int i = lane_id; i < n; i += 64) {
        // process element i, maintain local top-K
    }

    // Phase 2: Merge via LDS reduction (log2(64) = 6 rounds)
    __shared__ float s_data[K * 64];  // Transposed layout for bank-conflict-free access
    // Write local results to LDS
    for (int i = 0; i < K; i++)
        s_data[i * 64 + lane_id] = local_best[i];
    __syncthreads();

    // Log-reduction merge
    for (int stride = 32; stride >= 1; stride >>= 1) {
        if (lane_id < stride) {
            // Merge partner's sorted list into mine
        }
        __syncthreads();
    }

    // Lane 0 writes final result
    if (lane_id == 0) { /* write output */ }
}
```
Key benefits:
- Divides work across 64 threads (64x fewer iterations per thread)
- LDS merge is O(K * log(64)) = very fast for small K
- Bank-conflict-free LDS layout: `s_data[k_idx * WARP_SIZE + lane_id]`
- Use `__launch_bounds__(64)` for optimal register allocation

### Insertion Sort vs Heap for Small K
For small K values (K <= 16), insertion sort into a sorted array outperforms heap:
```cpp
// Insert d2 into sorted array best_dist[0..K-1] (ascending)
if (d2 < best_dist[K - 1]) {
    int pos = K - 1;
    #pragma unroll
    for (int j = K - 2; j >= 0; j--) {
        if (d2 < best_dist[j]) {
            best_dist[j + 1] = best_dist[j];
            best_idx[j + 1] = best_idx[j];
            pos = j;
        }
    }
    best_dist[pos] = d2;
    best_idx[pos] = new_idx;
}
```
Advantages over heap: fewer branches, better for `#pragma unroll`, simpler control flow.
