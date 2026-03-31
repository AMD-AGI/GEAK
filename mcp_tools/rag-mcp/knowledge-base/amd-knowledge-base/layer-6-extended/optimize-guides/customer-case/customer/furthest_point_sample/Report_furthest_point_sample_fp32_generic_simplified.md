# Kernel: furthest_point_sample (Simplified)

## Variant Context
- Input: 3D point cloud coordinates (B, N, 3)
- Datatype: fp32
- Architecture: Generic HIP/AMD GPU

## Key Optimizations

1. **Loop Unrolling (4x)**: Added `#pragma unroll 4` to main distance computation loop over N points. Reduces loop overhead and enables better instruction scheduling.

2. **Intermediate Variables**: Introduced `dx, dy, dz` for coordinate differences before squaring. Cleaner code structure may help compiler optimization.

3. **fminf/fmaxf Intrinsics**: Replaced generic `min()` with `fminf()` and conditional max with `fmaxf()`. Maps directly to hardware instructions for faster floating-point comparisons.

4. **Const Qualifiers**: Added `const` to coordinate variables (x1, y1, z1, x2, y2, z2). Helps compiler with register allocation and optimization.

5. **Compact Reduction**: Reformatted block-wide reduction to single-line statements for better readability.

## Performance Impact
- Loop unrolling reduces iteration overhead for large N
- Intrinsics may provide faster min/max operations
- Better compiler optimization with const hints
