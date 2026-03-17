# Kernel: roipoint_pool3d (Simplified)

## Variant Context
- Input: Point xyz/features, ROI indices
- Datatype: fp32
- Architecture: Generic HIP/AMD GPU

## Key Optimizations

1. **64-bit Offset Computation**: Uses `static_cast<size_t>()` for all stride and offset calculations to prevent overflow with large tensors.

2. **Vectorized Feature Copying**: Checks 16-byte alignment, uses float4 vector copies when possible for feature vectors.

3. **Restrict Pointers**: Added `__restrict__` to dst_ptr, src_xyz_ptr, src_feat_ptr, dst_feat_ptr.

4. **Const Index Variables**: Added `const` to sample_pt_idx, box_idx, bs_idx.

## Performance Impact
- Prevents overflow for large point clouds
- Faster feature copying with vectorization
- Better compiler optimization
