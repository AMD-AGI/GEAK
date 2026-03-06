Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: floyd_warshall

## Variant Context
- Input semantic type: All-pairs shortest path (Floyd-Warshall algorithm)
- Datatype(s): uint32 (unsigned int)
- Data representation: Dense adjacency matrix (nodes x nodes)
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel implements one step (k-th iteration) of the Floyd-Warshall all-pairs shortest path algorithm. For each pair of vertices (x, y), it checks if the path through intermediate vertex k is shorter than the current shortest path, and updates the distance and next-hop matrices accordingly.

## Optimization 1: Read-Only Cache Loads with __ldg Intrinsic
- Commit ID: baseline → optimized
- Optimization type: Memory
- Summary: Use __ldg intrinsic for read-only memory accesses to leverage texture cache
- Detailed explanation: The baseline performs regular global memory loads. The optimized version uses the `__ldg` (load global) intrinsic for reading the adjacency matrix values. This intrinsic hints to the hardware that the data is read-only and can be cached in the read-only texture cache (L1 texture cache on AMD GPUs), which can provide better cache hit rates for broadcast-style access patterns where multiple threads read the same k-th row/column.

- Code excerpt (baseline):
    ```cpp
    int d_x_y   = part_adjacency_matrix[y * nodes + x];
    int d_x_k_y = part_adjacency_matrix[y * nodes + k] + part_adjacency_matrix[k * nodes + x];
    ```

- Code excerpt (optimized):
    ```cpp
    // Vectorized load to improve memory throughput while preserving result.
    const unsigned int d_x_y = __ldg(&part_adjacency_matrix[y * nodes + x]);
    const unsigned int d_x_k_y = __ldg(&part_adjacency_matrix[y * nodes + k]) + __ldg(&part_adjacency_matrix[k * nodes + x]);
    ```

- Evidence mapping:
  - "__ldg intrinsic" → `__ldg(&part_adjacency_matrix[y * nodes + x])`
  - "Applied to all reads" → All three matrix reads use `__ldg`
  - "Read-only cache hint" → `__ldg` routes through texture/read-only cache

## Optimization 2: Const Qualifiers for Index Variables
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Added const qualifiers to thread index variables
- Detailed explanation: The optimized version marks the x and y index variables as const, helping the compiler understand these values won't change and potentially enabling better register allocation.

- Code excerpt (baseline):
    ```cpp
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    ```

- Code excerpt (optimized):
    ```cpp
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    ```

- Evidence mapping:
  - "Const qualifiers" → `const int x` and `const int y`
