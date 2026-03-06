Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: point_to_voxel

## Variant Context
- Input semantic type: Point cloud to voxel grid conversion
- Datatype(s): fp32
- Data representation: Point cloud (N, 3) to voxel grid
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel converts point cloud data into a voxel grid representation. Each point is assigned to a voxel based on its 3D coordinates, commonly used in 3D object detection and LiDAR processing pipelines.

## Optimization 1: Atomic Operations for Voxel Assignment
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Use atomic operations for thread-safe voxel point counting
- Detailed explanation: When multiple points fall into the same voxel, atomic operations ensure correct counting and assignment without race conditions.

- Code excerpt:
    ```cpp
    // Atomic increment for voxel point count
    atomicAdd(&voxel_count[voxel_idx], 1);
    ```

- Evidence mapping:
  - "Thread-safe counting" → atomicAdd for voxel point accumulation
