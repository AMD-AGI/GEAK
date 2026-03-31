# Kernel: point_to_voxel, point_to_voxelidx

## Variant: int32, AMD MI300 (gfx942)

## Functionality
Maps points to voxel indices for 3D point cloud voxelization.

## Key Optimizations

### 1. Warp-Level Parallelism
- 64-thread-warp-per-point instead of 1-thread-per-point
- Strided scan: `for (i = lane_id; i < index; i += WARP_SIZE)`
- 64x speedup for O(n) inner loop

### 2. Cached Memory Reads
- `__ldg(&coor[i])` for texture cache path on read-only data

### 3. Warp Shuffle Reduction
- Sum: `local_num += __shfl_down(local_num, offset)`
- Min: `local_first_match = min(local_first_match, __shfl_down(...))`
- No shared memory needed

### 4. Compiler Hints
- `__restrict__` pointers for aliasing optimization
- `__launch_bounds__(256)` for register allocation

### 5. Fixed Grid (256 blocks)
- Work loop: `for (index = warp_id; index < num_points; index += num_warps)`

### 6. Single-Writer
- Only lane 0 writes: `if (lane_id == 0) { ... }`
