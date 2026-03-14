---
name: rende_forward-optimization
description: This skill should be used when optimizing rende_forward kernel on AMD GPUs.
---

# Rende Forward Kernel Optimization

## Variant Context
- Input semantic type: 3D Gaussian Splatting rendering (point cloud / image synthesis)
- Datatype(s): fp32
- Data representation: Tile-based rasterization with Gaussian primitives
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel performs forward rendering for 3D Gaussian Splatting. It processes tiles of pixels, fetching Gaussian primitive data cooperatively into shared memory, then each thread computes alpha blending and color accumulation for its assigned pixel. The kernel implements the core rendering equation from the 3D Gaussian Splatting paper.

## Optimization 1: Loop Unrolling with Pragma Directive
- Commit ID: N/A (directory comparison)
- Optimization type: compute
- Summary: Added explicit loop unrolling hint to the inner batch processing loop
- Detailed explanation: The optimized kernel adds `#pragma unroll 32` to the inner loop that iterates over Gaussian primitives in each batch. This instructs the compiler to unroll the loop by a factor of 32, reducing loop overhead (branch instructions, counter updates) and enabling better instruction-level parallelism. The unroll factor of 32 is chosen to balance between code size and performance.
- Code excerpt:
    ```cpp
    // Iterate over current batch
    const int batch_size = min(BLOCK_SIZE, toDo);
    
    #pragma unroll 32
    for (int j = 0; j < batch_size; j++)
    {
      if (done)
        continue;
        
      // Keep track of current position in range
      contributor++;
      // ... rest of loop body
    }
    ```
- Evidence mapping:
  - "Loop unrolling" → `#pragma unroll 32` directive
  - "Reduced loop overhead" → Compiler generates unrolled code with fewer branch instructions

## Optimization 2: Fast Math Intrinsic for Exponential
- Commit ID: N/A (directory comparison)
- Optimization type: compute / precision
- Summary: Replace standard exp() with fast math intrinsic __expf() for faster exponential computation
- Detailed explanation: The optimized kernel uses `__expf()` instead of `exp()` for computing the Gaussian falloff. The intrinsic `__expf()` is a hardware-accelerated fast math function that trades some precision for significant speed improvement. For rendering applications where visual quality is the goal rather than numerical precision, this tradeoff is acceptable.
- Code excerpt:
    ```cpp
    // Baseline:
    // float alpha = min(0.99f, con_o.w * exp(power));
    
    // Optimized:
    // Use __expf for faster exponential
    float alpha = min(0.99f, con_o.w * __expf(power));
    ```
- Evidence mapping:
  - "Fast math intrinsic" → `__expf(power)` instead of `exp(power)`
  - "Hardware-accelerated" → `__expf` maps directly to GPU special function unit

## Optimization 3: Register-Based Color Accumulation
- Commit ID: N/A (directory comparison)
- Optimization type: memory
- Summary: Use explicit scalar registers for color channels instead of array indexing
- Detailed explanation: The baseline uses an array `float C[CHANNELS]` with loop-based access, while the optimized version uses explicit scalar variables `C0, C1, C2`. This eliminates array indexing overhead and ensures the compiler places these values in registers rather than local memory. Register access is significantly faster than even L1 cache.
- Code excerpt:
    ```cpp
    // Baseline:
    // float C[CHANNELS] = { 0 };
    // for (int ch = 0; ch < CHANNELS; ch++)
    //   C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
    
    // Optimized:
    float C0 = 0.0f, C1 = 0.0f, C2 = 0.0f;
    // ...
    const float weight = alpha * T;
    const int feat_idx = collected_id[j] * CHANNELS;
    C0 += features[feat_idx] * weight;
    C1 += features[feat_idx + 1] * weight;
    C2 += features[feat_idx + 2] * weight;
    ```
- Evidence mapping:
  - "Explicit scalar registers" → `float C0 = 0.0f, C1 = 0.0f, C2 = 0.0f`
  - "Eliminated array indexing" → Direct variable access `C0 +=` instead of `C[ch] +=`
  - "Precomputed weight" → `const float weight = alpha * T` computed once, used three times

## Optimization 4: Precomputed Weight Factor
- Commit ID: N/A (directory comparison)
- Optimization type: compute
- Summary: Precompute the weight factor (alpha * T) once and reuse for all color channels
- Detailed explanation: Instead of computing `alpha * T` three times (once per color channel), the optimized kernel computes it once into a local variable `weight`. This reduces redundant floating-point multiplications from 6 to 4 per Gaussian contribution.
- Code excerpt:
    ```cpp
    // Compute weight and accumulate colors
    const float weight = alpha * T;
    const int feat_idx = collected_id[j] * CHANNELS;
    C0 += features[feat_idx] * weight;
    C1 += features[feat_idx + 1] * weight;
    C2 += features[feat_idx + 2] * weight;
    ```
- Evidence mapping:
  - "Precomputed weight" → `const float weight = alpha * T` computed once
  - "Reused for all channels" → `weight` used in C0, C1, C2 accumulations

## Optimization 5: Const Qualifiers for Compiler Optimization
- Commit ID: N/A (directory comparison)
- Optimization type: compute
- Summary: Added const qualifiers to local variables to enable better compiler optimizations
- Detailed explanation: The optimized kernel adds `const` qualifiers to variables that don't change after initialization (e.g., `const uint32_t horizontal_blocks`, `const uint2 pix_min`, `const bool inside`). This helps the compiler perform better register allocation and enables certain optimizations like constant propagation.
- Code excerpt:
    ```cpp
    // Optimized version with const qualifiers:
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = { (float)pix.x, (float)pix.y };
    const bool inside = pix.x < W && pix.y < H;
    ```
- Evidence mapping:
  - "Const qualifiers" → `const` keyword on multiple variable declarations
  - "Compiler optimization hints" → Enables constant propagation and better register allocation

## Optimization 6: Unrolled Output Writes
- Commit ID: N/A (directory comparison)
- Optimization type: memory
- Summary: Explicit unrolled writes for output color channels instead of loop-based writes
- Detailed explanation: The baseline uses a loop to write output colors, while the optimized version writes each channel explicitly. This eliminates loop overhead and allows the compiler to potentially schedule these writes more efficiently.
- Code excerpt:
    ```cpp
    // Baseline:
    // for (int ch = 0; ch < CHANNELS; ch++)
    //   out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
    
    // Optimized:
    out_color[pix_id] = C0 + T * bg_color[0];
    out_color[H * W + pix_id] = C1 + T * bg_color[1];
    out_color[2 * H * W + pix_id] = C2 + T * bg_color[2];
    ```
- Evidence mapping:
  - "Unrolled writes" → Three explicit write statements instead of loop
  - "Eliminated loop overhead" → No loop counter, no branch instructions for loop control