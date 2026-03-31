# Kernel: assign_score_withk (Simplified)

## Variant Context
- Input: Point cloud with KNN indices
- Datatype: fp32
- Architecture: Generic HIP/AMD GPU

## Key Optimizations

1. **Atomic Elimination**: Replaced atomicAdd with local accumulation + single direct store. Each thread computes complete M-dimension sum independently, eliminating memory contention.

2. **Loop Unrolling (8x)**: Manual unroll with `#pragma unroll 8` processes 8 M-iterations per loop cycle, reducing branch overhead and enabling ILP.

3. **Pointer Arithmetic**: Precomputed base addresses and strides outside loop. Uses pointer increments (`ptr += stride`) instead of full index recalculation each iteration.

4. **__restrict__ Pointers**: Added `__restrict__` qualifier enabling compiler to assume no aliasing, allowing better instruction scheduling.

5. **Early Bounds Check**: Moved `kn` bounds validation before stride computation, avoiding wasted work for invalid indices.

## Performance Impact
- Eliminates atomic serialization (major speedup)
- Reduces integer arithmetic per iteration
- Better instruction-level parallelism via unrolling
- Improved compiler optimization opportunities
