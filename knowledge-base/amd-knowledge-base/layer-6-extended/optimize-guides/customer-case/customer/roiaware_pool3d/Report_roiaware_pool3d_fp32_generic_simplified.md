# Kernel: roiaware_pool3d (Simplified)

## Variant Context
- Input: Point features, voxelized ROIs
- Datatype: fp32
- Architecture: Generic HIP/AMD GPU

## Key Optimizations

1. **Precomputed Subexpressions**: Computes `yz = out_y * out_z`, `voxels_per_box`, `features_per_voxel` once. Reuses for index calculations.

2. **Restrict Pointers**: Added `__restrict__` to `pIdx`, `pOutFeature`, `pOutArgmax` for compiler optimization.

## Performance Impact
- Reduced redundant multiplications
- Better compiler optimization with restrict hints
