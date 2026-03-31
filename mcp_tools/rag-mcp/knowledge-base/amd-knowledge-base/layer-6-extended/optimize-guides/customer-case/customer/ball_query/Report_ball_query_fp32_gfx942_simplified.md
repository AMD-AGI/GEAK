# Kernel: ball_query (Simplified)

## Variant: fp32, AMD MI300 (gfx942)

## Functionality
Finds points within radius range around query points for PointNet++ neighborhood aggregation.

## Key Optimizations

### 1. Warp-Cooperative Processing
- Changed from 1-thread-per-query to 64-thread-warp-per-query
- Each warp thread scans strided subset: `for (k = lane_id; k < n; k += WARP_SIZE)`
- Provides 64x parallelism for inner loop scanning

### 2. Warp Intrinsics for Aggregation
- `__ballot(in_range)`: Creates 64-bit mask of valid threads
- `__popcll(mask & lower_mask)`: Counts valid points before current thread for write position
- `__ffsll(mask)`: Finds first valid lane for tracking first match
- Enables lock-free parallel result aggregation without atomics

### 3. Grid Configuration
- Adjusted: `DIVUP(m, warps_per_block)` instead of `DIVUP(m, THREADS_PER_BLOCK)`
- Ensures proper work distribution for warp-based algorithm

### 4. Optimized Fill Pattern
- Post-fill remaining slots instead of pre-fill all
- Only lane 0 performs fill: `if (lane_id == 0) { for (l = cnt; l < nsample; ++l) ... }`
- Reduces redundant memory writes
