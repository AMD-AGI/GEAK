# Kernel: prefix_sum (Simplified)

## Variant Context
- Input: Dense float array
- Datatype: fp32
- Architecture: Generic HIP/AMD GPU

## Key Optimizations

1. **Warp-Level Shuffle Scan**: Replaced shared memory tree reduction with `__shfl_up` intrinsics. `#pragma unroll` loop over delta powers of 2. No sync needed within warp.

2. **Register-Based 2-Element Scan**: Each thread loads 2 elements, performs local `a1 += a0` in registers before warp-level scan. Reduces warp-level work.

3. **LDS Bank Conflict Padding**: Uses `idx_pad(i) = i + (i >> 5)` to add padding every 32 elements, reducing bank conflicts.

4. **Loop Unrolling**: `#pragma unroll` on shuffle scan loop for better ILP.

## Performance Impact
- Shuffle instructions much faster than shared memory tree reduction
- Reduced synchronization overhead
- Better memory access patterns with padding
