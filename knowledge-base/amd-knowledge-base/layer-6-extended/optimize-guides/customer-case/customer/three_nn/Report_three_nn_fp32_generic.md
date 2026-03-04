---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: three_nn (Three Nearest Neighbors)

## Variant Context
- Input semantic type: Three nearest neighbors search in point cloud
- Datatype(s): fp32 (with double for distance accumulation)
- Data representation: Dense point cloud tensors (xyz coordinates)
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel finds the three nearest neighbors for each query point. For each point in `unknown`, it searches through all points in `known` to find the 3 closest points, outputting their indices and squared distances. This is commonly used in point cloud upsampling and interpolation.

## Optimization 1: Shared Memory Tiling with SoA Layout
- Commit ID: baseline → optimized
- Optimization type: Memory
- Summary: Introduced cooperative tiled loading of known points into shared memory using Structure-of-Arrays layout
- Detailed explanation: The baseline reads known points directly from global memory for each query point, causing redundant global memory accesses when multiple threads in a block process different query points. The optimized version loads tiles of 4096 points into shared memory cooperatively, with separate arrays for x, y, z coordinates (SoA layout). This reduces global memory traffic significantly and the SoA layout avoids bank conflicts.

- Code excerpt (baseline):
    ```cpp
    for (int k = 0; k < m; ++k) {
        float x = known[k * 3 + 0];
        float y = known[k * 3 + 1];
        float z = known[k * 3 + 2];
        float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
        // ...
    }
    ```

- Code excerpt (optimized):
    ```cpp
    // Cooperative tiling of known points into LDS to reduce global memory traffic.
    // Use SoA LDS layout to simplify addressing and keep LDS footprint modest.
    const int TILE_POINTS = 4096; // 4096 * 3 * 4 bytes = 48 KB per block
    __shared__ float s_x[TILE_POINTS];
    __shared__ float s_y[TILE_POINTS];
    __shared__ float s_z[TILE_POINTS];

    // Iterate over known points in tiles
    for (int base = 0; base < m; base += TILE_POINTS) {
        int tile_points = m - base;
        if (tile_points > TILE_POINTS) tile_points = TILE_POINTS;

        // Cooperative, coalesced load: each thread loads multiple points
        for (int t = threadIdx.x; t < tile_points; t += blockDim.x) {
            int k = base + t;
            int g = k * 3;
            s_x[t] = known_ptr[g + 0];
            s_y[t] = known_ptr[g + 1];
            s_z[t] = known_ptr[g + 2];
        }
        __syncthreads(); // ensure tile is fully loaded before compute

        // Compute distances to points in the current tile
        // ...
        float dx = ux - s_x[t + 0];
        float dy = uy - s_y[t + 0];
        float dz = uz - s_z[t + 0];
    }
    ```

- Evidence mapping:
  - "Shared memory allocation" → `__shared__ float s_x[TILE_POINTS]; __shared__ float s_y[TILE_POINTS]; __shared__ float s_z[TILE_POINTS];`
  - "SoA layout" → Separate arrays for x, y, z instead of interleaved
  - "Cooperative loading" → `for (int t = threadIdx.x; t < tile_points; t += blockDim.x)`
  - "Tile size" → `const int TILE_POINTS = 4096; // 4096 * 3 * 4 bytes = 48 KB per block`

## Optimization 2: Manual Loop Unrolling by Factor of 8
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Manually unrolled the distance computation loop by a factor of 8
- Detailed explanation: The optimized version manually unrolls the inner loop that computes distances to 8 points at a time. This reduces loop overhead and enables better instruction-level parallelism. The unrolled code processes 8 consecutive points from shared memory before moving to the next group.

- Code excerpt (optimized):
    ```cpp
    int t = 0;
    int limit8 = tile_points & ~7; // largest multiple of 8
    #pragma unroll 8
    for (; t < limit8; t += 8) {
        // Manually unrolled 8 iterations
        {
            float dx = ux - s_x[t + 0];
            float dy = uy - s_y[t + 0];
            float dz = uz - s_z[t + 0];
            float d = dx * dx + dy * dy + dz * dz;
            int k = base + (t + 0);
            if (d < best1) { /* update top-3 */ }
            // ...
        }
        {
            float dx = ux - s_x[t + 1];
            // ... repeat for t+1 through t+7
        }
        // ... 6 more blocks for t+2 through t+7
    }

    // Tail
    for (; t < tile_points; ++t) {
        float dx = ux - s_x[t];
        // ...
    }
    ```

- Evidence mapping:
  - "Unroll by 8" → `#pragma unroll 8` and `for (; t < limit8; t += 8)`
  - "8 explicit blocks" → 8 separate code blocks for t+0 through t+7
  - "Tail handling" → `for (; t < tile_points; ++t)` for remaining elements

## Optimization 3: Precomputed Base Pointers
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Precompute base pointers once instead of modifying input pointers
- Detailed explanation: The baseline modifies the input pointers directly with `+=` operations. The optimized version creates dedicated base pointer variables, making the code clearer and potentially helping the compiler with alias analysis.

- Code excerpt (baseline):
    ```cpp
    unknown += bs_idx * n * 3 + pt_idx * 3;
    known += bs_idx * m * 3;
    dist2 += bs_idx * n * 3 + pt_idx * 3;
    idx += bs_idx * n * 3 + pt_idx * 3;
    ```

- Code excerpt (optimized):
    ```cpp
    // Base pointers for this (batch, point) triple
    const float* __restrict__ unknown_ptr = unknown + bs_idx * n * 3 + pt_idx * 3;
    const float* __restrict__ known_ptr   = known + bs_idx * m * 3;
    float* __restrict__ dist2_ptr         = dist2 + bs_idx * n * 3 + pt_idx * 3;
    int* __restrict__ idx_ptr             = idx + bs_idx * n * 3 + pt_idx * 3;
    ```

- Evidence mapping:
  - "Dedicated pointer variables" → `unknown_ptr`, `known_ptr`, `dist2_ptr`, `idx_ptr`
  - "__restrict__ qualifiers" → Hints no aliasing for better optimization
