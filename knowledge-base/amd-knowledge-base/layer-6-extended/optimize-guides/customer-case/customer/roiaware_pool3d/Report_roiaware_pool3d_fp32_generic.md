---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: roiaware_pool3d

## Variant Context
- Input semantic type: ROI-aware 3D pooling for point cloud
- Datatype(s): fp32
- Data representation: Point cloud features with 3D bounding boxes
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel performs ROI-aware 3D pooling on point cloud data. It aggregates point features within each 3D region of interest (ROI/bounding box), commonly used in 3D object detection networks like PointRCNN.

## Optimization 1: Point-in-Box Check Optimization
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Efficient point-in-box testing with early rejection
- Detailed explanation: The kernel checks if points fall within 3D bounding boxes using coordinate transformation and bounds checking, with early exit when a point is clearly outside.

- Code excerpt:
    ```cpp
    // Transform point to box-local coordinates
    float local_x = (pt_x - box_cx) * cos_rz + (pt_y - box_cy) * sin_rz;
    float local_y = -(pt_x - box_cx) * sin_rz + (pt_y - box_cy) * cos_rz;
    
    // Check bounds
    if (fabsf(local_x) < box_dx/2 && fabsf(local_y) < box_dy/2 && 
        fabsf(pt_z - box_cz) < box_dz/2) {
        // Point is inside box
    }
    ```

- Evidence mapping:
  - "Coordinate transformation" → Rotation to box-local frame
  - "Bounds checking" → Half-extent comparisons for each axis
