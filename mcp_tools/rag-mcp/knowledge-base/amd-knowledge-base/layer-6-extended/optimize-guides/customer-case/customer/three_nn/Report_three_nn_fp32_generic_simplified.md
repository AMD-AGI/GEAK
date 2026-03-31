# Kernel: three_nn (Simplified)

## Variant Context
- Input: Unknown points (B, N, 3) and known points (B, M, 3)
- Datatype: fp32
- Architecture: Generic HIP/AMD GPU

## Key Optimizations

1. **Shared Memory Tiling**: Loads 4096 known points (~48KB) into LDS per tile. Cooperative loading across threads. All query points in block reuse same tile data.

2. **SoA Layout in LDS**: Separate `s_x[4096]`, `s_y[4096]`, `s_z[4096]` arrays instead of interleaved. Simpler addressing.

3. **8x Loop Unrolling**: `#pragma unroll 8` processes 8 known points per iteration. Reduces loop overhead, increases ILP.

4. **Restrict Base Pointers**: Precomputed `unknown_ptr`, `known_ptr`, `dist2_ptr`, `idx_ptr` with `__restrict__`.

## Performance Impact
- Shared memory tiling dramatically reduces global memory bandwidth
- Data reuse across query points in same block
- Better ILP with unrolling
