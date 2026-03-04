---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: points_in_boxes

## Variant Context
- Input semantic type: 3D point-in-box detection for LiDAR point clouds
- Datatype(s): fp32
- Data representation: Dense tensors for boxes (B, N, 7) and points (B, M, 3)
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel determines which 3D bounding boxes contain each point in a LiDAR point cloud. Each box is defined by center coordinates, dimensions, and rotation angle. The kernel transforms points to box-local coordinates and checks if they fall within the box boundaries.

## Optimization 1: Shared Memory Tiling with Precomputed Box Invariants
- Commit ID: baseline → optimized
- Optimization type: Memory / Compute
- Summary: Load boxes into shared memory with precomputed rotation values and half-dimensions
- Detailed explanation: The baseline reads box parameters from global memory for each point-box test and recomputes sin/cos for each test. The optimized version loads tiles of 256 boxes into shared memory, precomputing cos(-rz), sin(-rz), half-dimensions, and center-adjusted z coordinate. This reduces global memory traffic and eliminates redundant trigonometric computations.

- Code excerpt (baseline):
    ```cpp
    for (int k = 0; k < boxes_num; k++) {
        cur_in_flag = check_pt_in_box3d(pts, boxes + k * 7, local_x, local_y);
        if (cur_in_flag) {
            box_idx_of_points[k] = 1;
        }
        cur_in_flag = 0;
    }
    ```

- Code excerpt (optimized):
    ```cpp
    // Shared-memory tiling for boxes with precomputed invariants
    const int TILE = 256; // 8 arrays * 256 floats = 8192 floats (~32 KB)
    __shared__ float s_cx[TILE];
    __shared__ float s_cy[TILE];
    __shared__ float s_cz_center[TILE];
    __shared__ float s_hx[TILE];
    __shared__ float s_hy[TILE];
    __shared__ float s_hz[TILE];
    __shared__ float s_cosa[TILE];
    __shared__ float s_sina[TILE];

    for (int tile_start = 0; tile_start < boxes_num; tile_start += TILE) {
        // Precompute per-box invariants directly from global into LDS
        for (int j = threadIdx.x; j < tile_boxes; j += blockDim.x) {
            const float* b = boxes_base + (tile_start + j) * 7;
            // ...
            const float cosa = cosf(-rz);
            const float sina = sinf(-rz);
            s_cx[j] = cx;
            s_cosa[j] = cosa;
            s_sina[j] = sina;
            // ...
        }
        __syncthreads();
        // ... use precomputed values from shared memory
    }
    ```

- Evidence mapping:
  - "Shared memory arrays" → `__shared__ float s_cx[TILE];` and 7 other arrays
  - "Precomputed trig" → `const float cosa = cosf(-rz); const float sina = sinf(-rz);`
  - "Cooperative loading" → `for (int j = threadIdx.x; j < tile_boxes; j += blockDim.x)`

## Optimization 2: Point Coordinates Cached in Registers
- Commit ID: baseline → optimized
- Optimization type: Memory
- Summary: Cache point coordinates in registers to avoid repeated global memory loads
- Detailed explanation: The baseline passes the point pointer to the check function, which loads coordinates for each box test. The optimized version loads point coordinates once into registers (px, py, pz) before the box loop, eliminating redundant global memory accesses.

- Code excerpt (optimized):
    ```cpp
    // Cache point coordinates in registers to avoid repeated global loads
    const float px = pts_base[0];
    const float py = pts_base[1];
    const float pz = pts_base[2];

    // ... later in the loop:
    const float dz = pz - czc;
    const float dx = px - cx;
    const float dy = py - cy;
    ```

- Evidence mapping:
  - "Register caching" → `const float px = pts_base[0];` loaded once before loop
  - "Reused in loop" → `const float dx = px - cx;` uses cached value

## Optimization 3: Inlined Box Check with Early Z-Rejection
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Inline the box check logic with early Z-axis rejection before rotation computation
- Detailed explanation: The baseline calls a separate function for each box check. The optimized version inlines the check and performs the Z-axis test first (which doesn't require rotation). If the point fails the Z test, it skips the expensive rotation computation entirely.

- Code excerpt (optimized):
    ```cpp
    // First check Z-range for early reject
    const float dz = pz - czc;
    if (fabsf(dz) > hz) {
        cur_in_flag = 0;
    } else {
        // Transform point to box-local coordinates using precomputed rotation
        const float dx = px - cx;
        const float dy = py - cy;
        const float local_x = dx * cosa + dy * (-sina);
        const float local_y = dx * sina + dy *  cosa;

        // In-plane strict checks
        const int in_x = (local_x > -hx) & (local_x < hx);
        const int in_y = (local_y > -hy) & (local_y < hy);
        cur_in_flag = in_x & in_y;
    }
    ```

- Evidence mapping:
  - "Early Z rejection" → `if (fabsf(dz) > hz)` checked before rotation
  - "Skip rotation on reject" → Rotation only computed in else branch
  - "Inlined logic" → No function call, all logic in main kernel

## Optimization 4: Loop Unrolling for Box Iteration
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Added #pragma unroll 8 for the inner box iteration loop
- Detailed explanation: The optimized version adds a `#pragma unroll 8` directive to the loop that iterates over boxes within a tile. This reduces loop overhead and enables better instruction scheduling.

- Code excerpt (optimized):
    ```cpp
    // Iterate over the tile in LDS and test point against each box (inline check)
    int cur_in_flag = 0;
    #pragma unroll 8
    for (int j = 0; j < tile_boxes; ++j) {
        // Load precomputed values from LDS
        const float cx = s_cx[j];
        // ...
    }
    ```

- Evidence mapping:
  - "Unroll directive" → `#pragma unroll 8`
  - "Applied to tile loop" → `for (int j = 0; j < tile_boxes; ++j)`
