# Kernel: points_in_boxes (Simplified)

## Variant: fp32, AMD MI300 (gfx942)

## Functionality
Determines which 3D bounding box each LiDAR point belongs to for autonomous driving perception.

## Key Optimizations

### 1. Shared Memory Tiling
- Loads boxes into shared memory in tiles of 32
- `__shared__ float s_boxes[BOXES_PER_TILE][11]`
- Cooperative loading: `for (i = threadIdx.x; i < tile_size; i += blockDim.x)`
- Reduces global memory traffic ~256x

### 2. Precomputed Values
- Sin/cos computed once during load: `__sincosf(-rz, &sina, &cosa)`
- Half-sizes precomputed: `x_size * 0.5f`, `y_size * 0.5f`, `z_size * 0.5f`
- Center z adjusted: `cz + half_z`
- Stored in shared memory for reuse

### 3. Fast Math Intrinsics
- `__sincosf()`: Combined sin/cos computation
- `__fmaf_rn()`: FMA for coordinate transform
- `__ldg()`: Cached point coordinate reads

### 4. Early Z-Check
- Z bounds checked first (most common rejection)
- `if (dz > half_z || dz < -half_z) return 0`
- Avoids expensive coordinate transform when possible

### 5. Compiler Hints
- `__forceinline__` on check function eliminates call overhead
- `__restrict__` pointers enable aliasing optimizations
