# Kernel: convolution (Simplified)

## Variant Context
- Input: 2D padded image/grid
- Datatype: fp32
- Architecture: Generic HIP/AMD GPU

## Key Optimizations

1. **Loop Unrolling**: Added `#pragma unroll` to both outer and inner convolution loops. With MaskWidth=5 as compile-time constant, compiler fully unrolls to 25 sequential MACs, eliminating loop overhead.

2. **Hoisted Row Computations**: Moved `conv_offset_y = mask_index_y * padded_width` and `mask_base = mask_index_y * MaskWidth` outside inner loop. Inner loop now only adds `mask_index_x` instead of full multiplication.

## Performance Impact
- Full unrolling eliminates 25 loop iterations' control overhead
- Hoisting saves 5 multiplications per outer iteration (25 total per thread)
- Better instruction scheduling with unrolled code
