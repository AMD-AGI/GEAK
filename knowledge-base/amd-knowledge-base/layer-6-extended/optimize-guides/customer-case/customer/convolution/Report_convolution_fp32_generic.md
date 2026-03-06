Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: convolution

## Variant Context
- Input semantic type: 2D image convolution with 5x5 filter
- Datatype(s): fp32
- Data representation: Dense 2D grid with padding
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel performs 2D convolution on an input grid using a 5x5 filter stored in constant memory. Each thread computes one output pixel by applying the convolution filter to the corresponding input region.

## Optimization 1: Loop Unrolling with #pragma unroll
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Added #pragma unroll directives to both nested loops for complete unrolling
- Detailed explanation: The optimized version adds `#pragma unroll` to both the outer (mask_index_y) and inner (mask_index_x) loops. Since MaskWidth is a compile-time constant (5), the compiler can fully unroll both loops, eliminating loop overhead and enabling better instruction scheduling. This results in 25 multiply-add operations being explicitly scheduled.

- Code excerpt (baseline):
    ```cpp
    for(size_t mask_index_y = 0; mask_index_y < MaskWidth; ++mask_index_y)
    {
        for(size_t mask_index_x = 0; mask_index_x < MaskWidth; ++mask_index_x)
        {
            const size_t mask_index         = mask_index_y * MaskWidth + mask_index_x;
            const size_t convolution_offset = mask_index_y * padded_width + mask_index_x;
            sum += input[convolution_base + convolution_offset] * d_mask[mask_index];
        }
    }
    ```

- Code excerpt (optimized):
    ```cpp
    #pragma unroll
    for(size_t mask_index_y = 0; mask_index_y < MaskWidth; ++mask_index_y)
    {
        const size_t conv_offset_y = mask_index_y * padded_width;
        const size_t mask_base    = mask_index_y * MaskWidth;

        #pragma unroll
        for(size_t mask_index_x = 0; mask_index_x < MaskWidth; ++mask_index_x)
        {
            const size_t conv_offset_x = mask_index_x;
            const size_t convolution_offset = conv_offset_y + conv_offset_x;
            const size_t mask_index         = mask_base + mask_index_x;

            sum += input[convolution_base + convolution_offset] * d_mask[mask_index];
        }
    }
    ```

- Evidence mapping:
  - "Outer loop unroll" → `#pragma unroll` before `for(size_t mask_index_y = 0; ...)`
  - "Inner loop unroll" → `#pragma unroll` before `for(size_t mask_index_x = 0; ...)`

## Optimization 2: Hoisted Row-Invariant Computations
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Move row-dependent offset computations outside the inner loop
- Detailed explanation: The optimized version computes `conv_offset_y` (the y-component of the convolution offset) and `mask_base` (the base index for the mask row) once per outer loop iteration, rather than recomputing them in the inner loop. This reduces redundant multiplications.

- Code excerpt (baseline):
    ```cpp
    for(size_t mask_index_y = 0; mask_index_y < MaskWidth; ++mask_index_y)
    {
        for(size_t mask_index_x = 0; mask_index_x < MaskWidth; ++mask_index_x)
        {
            const size_t mask_index         = mask_index_y * MaskWidth + mask_index_x;
            const size_t convolution_offset = mask_index_y * padded_width + mask_index_x;
            // ...
        }
    }
    ```

- Code excerpt (optimized):
    ```cpp
    #pragma unroll
    for(size_t mask_index_y = 0; mask_index_y < MaskWidth; ++mask_index_y)
    {
        const size_t conv_offset_y = mask_index_y * padded_width;
        const size_t mask_base    = mask_index_y * MaskWidth;

        #pragma unroll
        for(size_t mask_index_x = 0; mask_index_x < MaskWidth; ++mask_index_x)
        {
            const size_t conv_offset_x = mask_index_x;
            const size_t convolution_offset = conv_offset_y + conv_offset_x;
            const size_t mask_index         = mask_base + mask_index_x;
            // ...
        }
    }
    ```

- Evidence mapping:
  - "Hoisted y-offset" → `const size_t conv_offset_y = mask_index_y * padded_width;` outside inner loop
  - "Hoisted mask base" → `const size_t mask_base = mask_index_y * MaskWidth;` outside inner loop
  - "Simplified inner computation" → `convolution_offset = conv_offset_y + conv_offset_x` uses precomputed values
