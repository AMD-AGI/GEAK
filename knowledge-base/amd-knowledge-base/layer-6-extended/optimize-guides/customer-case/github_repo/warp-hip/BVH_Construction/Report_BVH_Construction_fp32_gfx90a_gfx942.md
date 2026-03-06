Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: BVH_Construction

## Variant Context
- Input semantic type: 3D mesh geometry (triangle bounding boxes)
- Datatype(s): fp32 (vec3 for bounds), uint64_t (Morton codes), int32 (indices)
- Data representation: Packed BVH nodes (BVHPackedNodeHalf with 16-byte alignment)
- Target architecture: gfx90a, gfx942 (AMD GPUs via HIP)

## Functionality
This kernel constructs a Linear Bounding Volume Hierarchy (LBVH) for accelerating ray-mesh intersection queries. The construction involves:
1. Computing Morton codes for spatial sorting of primitives
2. Radix sorting primitives by Morton code
3. Building BVH topology using Karras algorithm
4. Refitting node bounds from leaves to root
5. Marking packed leaf nodes for efficient traversal

---

## Optimization 1: Fix Morton Code Delta Computation Type Mismatch
- Commit ID: 92c37b1
- Optimization type: Correctness / Compute
- Summary: Fixed incorrect type usage in Morton code delta computation that caused incorrect BVH topology

- Detailed explanation: 
  The original implementation incorrectly used `int` type for 64-bit Morton keys when computing deltas between adjacent keys. This caused truncation of the upper 32 bits, leading to incorrect common prefix length calculations and poor BVH quality. The fix uses proper `uint64_t` types and `__clzll()` for 64-bit leading zero count.

- Code excerpt (before):
    ```cpp
    int a = keys[index];
    int b = keys[index+1];
    int x = a^b;
    deltas[index] = x;
    ```

- Code excerpt (after):
    ```cpp
    const uint64_t a = keys[index];
    const uint64_t b = keys[index+1];
    const uint64_t diff = a ^ b;
    // Count leading zeros to find common prefix length
    // If keys are identical (diff==0), use 64 (maximum common prefix)
    deltas[index] = (diff == 0) ? 64 : __clzll(diff);
    ```

- Evidence mapping:
  - Type correction: `int` → `uint64_t` prevents truncation of 64-bit Morton codes
  - Proper CLZ: `__clzll(diff)` correctly counts leading zeros in 64-bit value
  - Edge case handling: `(diff == 0) ? 64` handles identical keys correctly

---

## Optimization 2: Configurable Block Dimension for AMD GPUs
- Commit ID: 5f102b9
- Optimization type: Launch configuration
- Summary: Made block dimension configurable via WP_BVH_BLOCK_DIM macro, optimized for AMD GPU wavefront size

- Detailed explanation:
  The original implementation used hardcoded 256 threads per block. This was changed to use a configurable WP_BVH_BLOCK_DIM macro, initially set to 680 for AMD GPUs to better utilize the GPU's compute units. AMD GPUs have 64-thread wavefronts, and using larger block sizes can improve occupancy.

- Code excerpt:
    ```cpp
    #ifndef WP_BVH_BLOCK_DIM
    #define WP_BVH_BLOCK_DIM 768
    #endif
    
    // In LinearBVHBuilderGPU::build():
    const int num_threads = WP_BVH_BLOCK_DIM;
    const int nb1 = (num_items+num_threads-1)/num_threads;
    
    // All kernel launches use configurable block size:
    compute_morton_codes<<<nb1, num_threads, 0, 0>>>(...);
    build_leaves<<<nb1, num_threads, 0, 0>>>(...);
    build_karras_topology<<<nb1, num_threads, 0, 0>>>(...);
    ```

- Evidence mapping:
  - Configurable macro: `WP_BVH_BLOCK_DIM` allows tuning for different AMD GPU architectures
  - Consistent usage: All kernel launches use `num_threads` variable derived from macro
  - Default value 768: Optimized for AMD GPU wavefront size (multiple of 64)

---

## Optimization 3: Deterministic Karras-style LBVH Topology Builder
- Commit ID: 18bf8c6
- Optimization type: Compute / Robustness
- Summary: Replaced atomic-based bottom-up hierarchy builder with deterministic Karras algorithm

- Detailed explanation:
  The original bottom-up hierarchy builder used atomics and `__threadfence()` for synchronization, which can have non-deterministic behavior on HIP/AMD GPUs. The new implementation uses the Karras algorithm which is fully deterministic - each thread independently computes its internal node's children based on Morton code prefixes, without requiring inter-thread synchronization during topology construction.

- Code excerpt:
    ```cpp
    __global__ void build_karras_topology(
        int n,
        int* root,
        const uint64_t* __restrict__ keys,
        volatile int* __restrict__ range_lefts,
        volatile int* __restrict__ range_rights,
        volatile int* __restrict__ parents,
        volatile BVHPackedNodeHalf* __restrict__ lowers,
        volatile BVHPackedNodeHalf* __restrict__ uppers)
    {
        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= n - 1) return;  // n-1 internal nodes

        const int internal_offset = n;

        // Determine direction of the range (+1 or -1)
        const int delta_next = delta_prefix(keys, n, i, i + 1);
        const int delta_prev = delta_prefix(keys, n, i, i - 1);
        const int d = (delta_next - delta_prev) >= 0 ? 1 : -1;

        // Binary search for range end and split position
        // ... (deterministic computation without atomics)
        
        const int left_child = (split == first) ? first : (internal_offset + split);
        const int right_child = (split + 1 == last) ? last : (internal_offset + split + 1);

        // Write internal node children (no atomics needed)
        lowers[internal_index].i = left_child;
        uppers[internal_index].i = right_child;
    }
    ```

- Evidence mapping:
  - No atomics: Each thread writes to unique memory locations (its own internal node)
  - Deterministic split: Binary search finds split position based on Morton code prefixes
  - Independent computation: `delta_prefix()` function computes common prefix length without shared state

---

## Optimization 4: Deterministic Bounds Reduction (Avoiding vec3 Atomics)
- Commit ID: 18bf8c6
- Optimization type: Memory / Robustness
- Summary: Replaced atomic vec3 operations with two-phase block reduction for deterministic bounds computation

- Detailed explanation:
  The original implementation used `atomic_max` and `atomic_min` on vec3 types for computing total bounds, which is not portable or deterministic on HIP. The new implementation uses a two-phase approach: first, each block computes its local bounds using hipcub::BlockReduce, then a single thread performs the final reduction across blocks.

- Code excerpt:
    ```cpp
    __global__ void compute_block_bounds(
        const vec3* item_lowers,
        const vec3* item_uppers,
        vec3* block_lowers,
        vec3* block_uppers,
        int num_items)
    {
        typedef hipcub::BlockReduce<vec3, WP_BVH_BLOCK_DIM> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;

        // ... load data ...
        
        vec3 block_upper = BlockReduce(temp_storage).Reduce(upper, Vec3Max, numValid);
        __syncthreads();
        vec3 block_lower = BlockReduce(temp_storage).Reduce(lower, Vec3Min, numValid);

        if (threadIdx.x == 0)
        {
            block_lowers[blockIdx.x] = block_lower;
            block_uppers[blockIdx.x] = block_upper;
        }
    }

    __global__ void reduce_block_bounds(
        const vec3* block_lowers,
        const vec3* block_uppers,
        int num_blocks,
        vec3* total_lower,
        vec3* total_upper)
    {
        // single thread deterministic reduction
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            vec3 lo = vec3(FLT_MAX);
            vec3 hi = vec3(-FLT_MAX);
            for (int i = 0; i < num_blocks; ++i)
            {
                lo = min(lo, block_lowers[i]);
                hi = max(hi, block_uppers[i]);
            }
            total_lower[0] = lo;
            total_upper[0] = hi;
        }
    }
    ```

- Evidence mapping:
  - No vec3 atomics: Replaced `atomic_max/atomic_min` with explicit block reduction
  - Two-phase reduction: Block-level parallel reduction followed by single-thread final reduction
  - Deterministic: Single thread performs final reduction, ensuring consistent results

---

## Optimization 5: Deterministic Depth-based BVH Refit
- Commit ID: 18bf8c6
- Optimization type: Compute / Robustness
- Summary: Replaced atomic-based bottom-up refit with level-by-level deterministic refit

- Detailed explanation:
  The original refit used atomics to track when both children of a node were complete. The new approach first computes the depth of each node, then refits nodes level-by-level from deepest to root. This ensures all children are processed before their parent, without requiring atomics.

- Code excerpt:
    ```cpp
    __global__ void compute_node_depths(
        int n_nodes,
        const int* __restrict__ parents,
        int* __restrict__ depths,
        int max_steps)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n_nodes) return;

        int depth = 1;
        int p = parents[idx];
        while (p != -1 && steps < max_steps)
        {
            p = parents[p];
            depth++;
            steps++;
        }
        depths[idx] = depth;
    }

    __global__ void refit_nodes_at_depth(
        int n_nodes,
        int target_depth,
        const int* __restrict__ depths,
        BVHPackedNodeHalf* __restrict__ node_lowers,
        BVHPackedNodeHalf* __restrict__ node_uppers)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n_nodes || depths[idx] != target_depth) return;
        
        // Skip leaves
        BVHPackedNodeHalf lower = bvh_load_node(node_lowers, idx);
        if (lower.b) return;

        const int left = lower.i;
        const int right = bvh_load_node(node_uppers, idx).i;

        // Union child bounds
        const vec3 out_lower = min(vec3(ll.x, ll.y, ll.z), vec3(rl.x, rl.y, rl.z));
        const vec3 out_upper = max(vec3(lu.x, lu.y, lu.z), vec3(ru.x, ru.y, ru.z));

        make_node(node_lowers + idx, out_lower, left, false);
        make_node(node_uppers + idx, out_upper, right, false);
    }
    
    // Host-side iteration from deepest to root:
    for (int d = max_depth_host - 1; d >= 1; --d)
    {
        refit_nodes_at_depth<<<nb_nodes, num_threads, 0, 0>>>(
            bvh.max_nodes, d, num_children, bvh.node_lowers, bvh.node_uppers);
    }
    ```

- Evidence mapping:
  - Level-by-level processing: Host loop iterates from max depth to root
  - No atomics: Each kernel invocation processes independent nodes at same depth
  - Guaranteed ordering: Children at depth d+1 are processed before parents at depth d

---

## Optimization 6: Parallel Max Reduction for Depth Computation
- Commit ID: 42cc7c4
- Optimization type: Compute
- Summary: Parallelized the max reduction for finding maximum tree depth

- Detailed explanation:
  The original implementation used a single-thread sequential loop to find the maximum depth. The optimized version uses parallel reduction with shared memory and grid-stride loops to efficiently compute the maximum across all nodes.

- Code excerpt:
    ```cpp
    __global__ void reduce_max_int(const int* __restrict__ values, int n, int* __restrict__ out_max)
    {
        extern __shared__ int sdata[];
        
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int gridSize = blockDim.x * gridDim.x;
        
        // Grid-stride loop to handle arrays larger than grid size
        int thread_max = INT_MIN;
        while (i < n)
        {
            thread_max = max(thread_max, values[i]);
            i += gridSize;
        }
        
        sdata[tid] = thread_max;
        __syncthreads();
        
        // Parallel reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                sdata[tid] = max(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }
        
        // Write result using atomicMax
        if (tid == 0)
        {
            atomicMax(out_max, sdata[0]);
        }
    }
    ```

- Evidence mapping:
  - Grid-stride loop: `while (i < n) { ... i += gridSize; }` handles large arrays
  - Shared memory reduction: `sdata[tid] = max(sdata[tid], sdata[tid + s])` parallel tree reduction
  - Final atomic: `atomicMax(out_max, sdata[0])` combines results from multiple blocks

---

## Optimization 7: Group-aware Morton Code and Leaf Packing
- Commit ID: 18bf8c6
- Optimization type: Memory / Compute
- Summary: Added group support to Morton codes and prevented packed leaves from straddling group boundaries

- Detailed explanation:
  The optimization encodes group IDs in the upper 32 bits of the 64-bit Morton key, ensuring primitives from different groups are separated in the sorted order. The packed leaf marking kernel also checks group boundaries to avoid creating leaves that span multiple groups.

- Code excerpt:
    ```cpp
    __global__ void compute_morton_codes(
        const vec3* __restrict__ item_lowers,
        const vec3* __restrict__ item_uppers,
        int n,
        const vec3* grid_lower,
        const vec3* grid_inv_edges,
        int* __restrict__ indices,
        uint64_t* __restrict__ keys,
        const int* __restrict__ item_groups)
    {
        // ...
        const uint64_t morton_code = static_cast<uint64_t>(morton3<1024>(local[0], local[1], local[2]));
        // Group in upper 32 bits, morton in lower 32 bits
        const uint64_t group = item_groups ? static_cast<uint32_t>(item_groups[index]) : 0u;
        const uint64_t key = (group << 32) | morton_code;
        // ...
    }

    __global__ void mark_packed_leaf_nodes(...)
    {
        // ...
        // Avoid creating packed leaves that straddle group boundaries
        bool single_group = true;
        const uint64_t group_left = keys[left] >> 32;
        const uint64_t group_right = keys[right - 1] >> 32;
        single_group = (group_left == group_right);

        if (single_group && (right - left <= leaf_size || depth >= BVH_QUERY_STACK_SIZE))
        {
            lowers[node_index].b = 1;
            lowers[node_index].i = left;
            uppers[node_index].i = right;
        }
    }
    ```

- Evidence mapping:
  - Group encoding: `(group << 32) | morton_code` places group in upper bits for sort priority
  - Group boundary check: `group_left == group_right` ensures packed leaves don't span groups
  - Configurable leaf size: `leaf_size` parameter allows tuning leaf packing threshold
