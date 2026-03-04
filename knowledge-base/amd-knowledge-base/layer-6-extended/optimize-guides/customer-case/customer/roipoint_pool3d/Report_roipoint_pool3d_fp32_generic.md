---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: roipoint_pool3d

## Variant Context
- Input semantic type: ROI point pooling for 3D object detection
- Datatype(s): fp32
- Data representation: Point cloud with ROI boxes
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel performs point-wise pooling within 3D regions of interest. For each ROI, it samples and aggregates point features, producing fixed-size feature representations for downstream processing in 3D detection networks.

## Optimization 1: Grid-Based Point Sampling
- Commit ID: baseline → optimized
- Optimization type: Memory / Compute
- Summary: Efficient point sampling within ROI using grid subdivision
- Detailed explanation: The ROI is subdivided into a grid, and points are sampled from each grid cell to ensure uniform coverage of the ROI volume.

- Code excerpt:
    ```cpp
    // Compute grid cell for each point
    int grid_x = (local_x + box_dx/2) / (box_dx / grid_size);
    int grid_y = (local_y + box_dy/2) / (box_dy / grid_size);
    int grid_z = (local_z + box_dz/2) / (box_dz / grid_size);
    ```

- Evidence mapping:
  - "Grid subdivision" → Dividing ROI into uniform cells
  - "Uniform sampling" → Selecting points from each grid cell
