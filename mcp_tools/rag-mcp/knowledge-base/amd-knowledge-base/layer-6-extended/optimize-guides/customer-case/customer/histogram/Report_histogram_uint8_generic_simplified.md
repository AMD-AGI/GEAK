# Kernel: histogram (Simplified)

## Variant Context
- Input: Unsigned char array
- Datatype: uint8 input, uint32 output
- Architecture: Generic HIP/AMD GPU

## Key Optimizations

1. **32-bit Initialization**: Cast shared memory to `unsigned int*`, write 64 zeros instead of 256 bytes. 4x fewer store operations.

2. **uchar4 Vector Loading**: Load 4 bytes per memory transaction using `uchar4` type. Process v.x, v.y, v.z, v.w separately. Better coalescing, fewer loads.

3. **32-bit Reduction**: Load 4 bytes as uint32, extract each byte with shifts/masks: `(x >> 8) & 0xFF`. 4x fewer load operations in reduction phase.

4. **Loop Unrolling**: `#pragma unroll` on initialization, data loading (`#pragma unroll 4`), and reduction loops.

5. **Restrict Pointers**: Added `__restrict__` to all pointer variables for compiler optimization.

## Performance Impact
- 4x reduction in memory operations for init and reduction
- Better memory coalescing with vector loads
- Reduced loop overhead with unrolling
