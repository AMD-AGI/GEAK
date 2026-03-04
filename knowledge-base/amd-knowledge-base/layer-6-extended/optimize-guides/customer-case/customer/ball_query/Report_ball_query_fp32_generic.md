---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: ball_query

## Variant Context
- Input semantic type: Point cloud ball query (radius-based neighbor search)
- Datatype(s): fp32
- Data representation: Dense point cloud tensors (xyz coordinates)
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel performs ball query operation for point cloud processing. For each query point in `new_xyz`, it finds up to `nsample` neighboring points from `xyz` that fall within a spherical shell defined by `min_radius` and `max_radius`. The output is an index array pointing to the found neighbors.

## Optimization 1: Shared Memory Tiling for Reduced Global Memory Traffic
- Commit ID: baseline → optimized
- Optimization type: Memory
- Summary: Introduced cooperative tiled loading of xyz points into shared memory to reduce redundant global memory accesses
- Detailed explanation: The baseline kernel has each thread independently reading xyz coordinates from global memory for all N points. When multiple threads in a block process different query points, they redundantly load the same xyz data. The optimized version uses shared memory tiling where threads cooperatively load a tile of 256 points (768 floats, ~3KB) into shared memory, then all threads read from the faster shared memory. This significantly reduces global memory bandwidth consumption.

- Code excerpt (baseline):
    ```cpp
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                   (new_z - z) * (new_z - z);
        // ...
    }
    ```

- Code excerpt (optimized):
    ```cpp
    const int TILE = 256; // 256 points -> 768 floats (~3 KB)
    __shared__ float shx[TILE];
    __shared__ float shy[TILE];
    __shared__ float shz[TILE];

    for (int kBase = 0; kBase < n; kBase += TILE) {
        int tileCount = n - kBase;
        if (tileCount > TILE) tileCount = TILE;

        // Cooperative load: threads fill LDS in strided fashion
        for (int t = threadIdx.x; t < tileCount; t += blockDim.x) {
            const float* p = xyz_base + (kBase + t) * 3;
            shx[t] = p[0];
            shy[t] = p[1];
            shz[t] = p[2];
        }
        __syncthreads();

        // Process the tile from LDS
        #pragma unroll 4
        for (int t = 0; t < tileCount; ++t) {
            const float x = shx[t];
            const float y = shy[t];
            const float z = shz[t];
            // ...
        }
    }
    ```

- Evidence mapping:
  - "Shared memory allocation" → `__shared__ float shx[TILE]; __shared__ float shy[TILE]; __shared__ float shz[TILE];`
  - "Cooperative loading" → `for (int t = threadIdx.x; t < tileCount; t += blockDim.x)` with strided access pattern
  - "Reading from shared memory" → `const float x = shx[t]; const float y = shy[t]; const float z = shz[t];`
  - "Tile size choice" → `const int TILE = 256; // 256 points -> 768 floats (~3 KB)`

## Optimization 2: Loop Unrolling for Inner Tile Processing
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Added #pragma unroll 4 directive for the inner tile processing loop
- Detailed explanation: The inner loop that processes points within a tile is unrolled by a factor of 4 using the pragma directive. This reduces loop overhead and enables better instruction-level parallelism by allowing the compiler to schedule multiple iterations' instructions together.

- Code excerpt (optimized):
    ```cpp
    // Process the tile from LDS
    #pragma unroll 4
    for (int t = 0; t < tileCount; ++t) {
        const float x = shx[t];
        const float y = shy[t];
        const float z = shz[t];

        const float dx = new_x - x;
        const float dy = new_y - y;
        const float dz = new_z - z;
        float d2 = dx * dx + dy * dy + dz * dz;
        // ...
    }
    ```

- Evidence mapping:
  - "Loop unrolling directive" → `#pragma unroll 4`
  - "Applied to tile processing loop" → `for (int t = 0; t < tileCount; ++t)`

## Optimization 3: Early Termination with Synchronization Safety
- Commit ID: baseline → optimized
- Optimization type: Compute / Scheduling
- Summary: Implemented early termination flag with proper synchronization handling
- Detailed explanation: The optimized kernel uses a `done` flag to track when a thread has found enough samples. However, it carefully maintains synchronization correctness by not breaking out of the tile loop prematurely (which would cause __syncthreads deadlock). The thread continues to participate in synchronization but skips computation. Only after the sync point can it safely exit the outer loop.

- Code excerpt (optimized):
    ```cpp
    bool done = false;

    for (int kBase = 0; kBase < n; kBase += TILE) {
        // ... cooperative load ...
        __syncthreads();

        if (!done) {
            // Process the tile from LDS
            #pragma unroll 4
            for (int t = 0; t < tileCount; ++t) {
                // ...
                if (d2 == 0.0f || (d2 >= min_radius2 && d2 < max_radius2)) {
                    // ...
                    if (cnt >= nsample) {
                        done = true; // Mark done; finish tile to keep __syncthreads balanced
                        // Do not break here; we must still iterate to the end of the tile for sync safety
                    }
                }
            }
            if (cnt >= nsample) {
                done = true;
            }
        }

        __syncthreads();
        if (done) {
            // We cannot break before this sync; now safe to exit the outer loop
            break;
        }
    }
    ```

- Evidence mapping:
  - "Early termination flag" → `bool done = false;` and `done = true;`
  - "Skip computation when done" → `if (!done) { /* process tile */ }`
  - "Sync-safe exit" → Comment: "Do not break here; we must still iterate to the end of the tile for sync safety"
  - "Safe loop exit after sync" → `__syncthreads(); if (done) { break; }`

## Optimization 4: Separate XYZ Component Storage in Shared Memory
- Commit ID: baseline → optimized
- Optimization type: Memory
- Summary: Stored x, y, z coordinates in separate shared memory arrays instead of interleaved
- Detailed explanation: The baseline reads xyz as interleaved (x,y,z,x,y,z,...) from global memory. The optimized version stores them in separate arrays (shx[], shy[], shz[]) in shared memory. This structure-of-arrays (SoA) layout in shared memory can provide better access patterns and avoid bank conflicts when threads access the same component across different points.

- Code excerpt (optimized):
    ```cpp
    __shared__ float shx[TILE];
    __shared__ float shy[TILE];
    __shared__ float shz[TILE];

    // Load into separate arrays
    for (int t = threadIdx.x; t < tileCount; t += blockDim.x) {
        const float* p = xyz_base + (kBase + t) * 3;
        shx[t] = p[0];
        shy[t] = p[1];
        shz[t] = p[2];
    }

    // Access from separate arrays
    const float x = shx[t];
    const float y = shy[t];
    const float z = shz[t];
    ```

- Evidence mapping:
  - "Separate arrays for each component" → `__shared__ float shx[TILE]; __shared__ float shy[TILE]; __shared__ float shz[TILE];`
  - "SoA access pattern" → `shx[t] = p[0]; shy[t] = p[1]; shz[t] = p[2];`
