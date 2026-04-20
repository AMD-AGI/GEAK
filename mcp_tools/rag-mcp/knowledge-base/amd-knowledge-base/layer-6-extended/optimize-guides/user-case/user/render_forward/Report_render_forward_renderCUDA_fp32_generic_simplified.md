# Kernel: render_forward - 3D Gaussian Splatting (Simplified)

## Variant: fp32, Generic HIP

## Functionality
Tile-based forward rendering for 3D Gaussian Splatting. Collaboratively fetches Gaussian data to shared memory, then each thread alpha-blends overlapping Gaussians for its pixel.

## Key Optimizations

### 1. Loop Unrolling (Compute)
Add `#pragma unroll 32` to inner batch loop. Restructure `!done` condition to `if(done) continue` inside loop to enable unrolling. Reduces branch overhead, improves ILP.

### 2. Fast Math Intrinsic (Precision)
Replace `exp(power)` with `__expf(power)`. Trades IEEE-754 precision for speed in Gaussian falloff computation. Visually imperceptible difference.

### 3. Register-Based Color Accumulation (Memory)
Replace `float C[CHANNELS]` array with explicit `float C0, C1, C2` scalars. Prevents register spilling, eliminates indexing overhead. Precompute `weight = alpha * T` and `feat_idx`.

### 4. Const Qualifiers (Compute)
Add `const` to immutable variables (horizontal_blocks, pix, pixf, dx, dy, con_o). Enables compiler optimizations: constant propagation, better register allocation.

## APIs/Intrinsics
- `__expf()` fast exponential
- `#pragma unroll 32`
- `__syncthreads_count()` for early termination
- Cooperative groups: `cg::this_thread_block()`
