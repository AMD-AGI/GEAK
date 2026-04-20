# Kernel: rmsnorm_fused_quant

## Variant Context
- Input semantic type: Normalization with optional residual add and quantization
- Datatype(s): BF16/FP16 input, FP8/INT8/FP4 quantized output
- Data representation: Row-wise normalization with block-wise quantization (1x128, 1x64, 1x32 groups)
- Target architecture: gfx942 (MI300), gfx950 (MI350)

## Functionality
This kernel performs fused RMSNorm with optional residual addition and quantization. It computes:
1. Optional residual addition: x = input + residual
2. RMSNorm: y = x / sqrt(mean(x²) + epsilon) * weight
3. Optional quantization: output = quantize(y) with per-group scaling

The fusion eliminates intermediate memory accesses between these operations.

## Optimization 1: Vectorized Load with Configurable Chunk Size
- Commit ID: f9ec28ad7
- Optimization type: Memory
- Summary: Implemented vectorized memory loads with configurable chunk sizes (8 or 16 bytes) based on data alignment.

- Detailed explanation:
  The kernel uses template-based vectorized loads that automatically select the optimal load width:
  1. 16-byte loads when `thread_data_size * sizeof(DTYPE_I) % 16 == 0`
  2. 8-byte loads otherwise
  This maximizes memory bandwidth utilization on AMD GPUs which have 128-bit (16-byte) memory interfaces.

- Code excerpt:
    ```cpp
    static constexpr int32_t load_chunk_bytes = sizeof(DTYPE_I) * thread_data_size % 16 == 0 ? 16 : 8;
    static_assert(thread_data_size * sizeof(DTYPE_I) % load_chunk_bytes == 0, 
                  "thread_data_size * sizeof(DTYPE_I) must be a multiple of load_chunk_bytes");
    static constexpr int32_t load_vec_size = load_chunk_bytes / sizeof(DTYPE_I);
    static constexpr int32_t num_load_inst = thread_data_size / load_vec_size;
    
    // Vectorized load
    thread_data_ix2[0] = load_vector_nbytes<DTYPE_I, thread_data_size, load_chunk_bytes, 
                                            load_aux, interleave>(buffer_i, row_offset);
    ```

- Evidence mapping:
  - Configurable chunk size → `load_chunk_bytes` compile-time selection
  - Vectorized load → `load_vector_nbytes` template function
  - Alignment check → `static_assert` for correctness

## Optimization 2: Inline Assembly for Fused Multiply-Accumulate
- Commit ID: f9ec28ad7
- Optimization type: Compute
- Summary: Used inline assembly for fused multiply-accumulate (FMA) operations in the sum-of-squares computation.

- Detailed explanation:
  The kernel uses AMD-specific inline assembly `v_fmac_f32_e32` for computing the sum of squares. This instruction performs `a = a + b * b` in a single cycle, which is more efficient than separate multiply and add operations.

- Code excerpt:
    ```cpp
    float square_sum = 0.0f;
    for(int i = 0; i < thread_data_size; i++)
    {
        asm volatile("v_fmac_f32_e32 %0, %1, %1" : "+v"(square_sum) : "v"(thread_data_float[i]));
        // Equivalent to: square_sum += (thread_data_float[i] * thread_data_float[i]);
    }
    ```

- Evidence mapping:
  - FMA instruction → `v_fmac_f32_e32` inline assembly
  - Single-cycle operation → Fused multiply-accumulate instead of separate ops

## Optimization 3: Packed FP32 Operations for Normalization
- Commit ID: f9ec28ad7
- Optimization type: Compute
- Summary: Used packed FP32 SIMD operations (v_pk_mul_f32) for parallel normalization and weight application.

- Detailed explanation:
  The kernel uses AMD's packed FP32 instructions to process two elements simultaneously:
  1. `v_pk_mul_f32` for multiplying normalized values by the RMS scale
  2. `v_pk_mul_f32` for applying weights to normalized values
  This doubles the throughput for these operations.

- Code excerpt:
    ```cpp
    rcp[0] = rsqrtf(rcp[0] / n + epsilon);
    rcp[1] = rcp[0];
    vec2_f* thread_data_float2 = reinterpret_cast<vec2_f*>(&thread_data_float);
    for(int i = 0; i < thread_data_size / 2; i++)
    {
        asm volatile("v_pk_mul_f32 %0, %1, %2" 
                     : "=v"(thread_data_float2[i]) 
                     : "v"(thread_data_float2[i]), "v"(rcp));
    }
    
    // Weight application with packed multiply
    for(int i = 0; i < thread_data_size / 2; i++)
    {
        // ... weight conversion ...
        asm volatile("v_pk_mul_f32 %0, %1, %2" 
                     : "=v"(thread_data_float2[i]) 
                     : "v"(thread_data_float2[i]), "v"(thread_data_weight_float2));
    }
    ```

- Evidence mapping:
  - Packed multiply → `v_pk_mul_f32` processes 2 FP32 values
  - Replicated scale → `rcp[0] = rcp[1]` for packed operation
  - 2x throughput → Loop processes `thread_data_size / 2` iterations

## Optimization 4: Efficient BF16/FP16 to FP32 Conversion
- Commit ID: f9ec28ad7
- Optimization type: Compute
- Summary: Used specialized inline assembly for efficient BF16/FP16 to FP32 conversion during weight application.

- Detailed explanation:
  The kernel uses different conversion strategies based on input type:
  1. BF16: Bit manipulation with `v_lshlrev_b32` and `v_and_b32` (zero-cost conversion)
  2. FP16: Hardware conversion with `v_cvt_f32_f16` and SDWA (Sub-Dword Addressing)

- Code excerpt:
    ```cpp
    if constexpr(std::is_same_v<DTYPE_I, ck_tile::bf16_t>)
    {
        asm volatile(
            "v_lshlrev_b32_e32 %0, 16 %2\n"
            "v_and_b32_e32 %1 0xffff0000 %2\n"
            : "=v"(thread_data_weight_float2[0]), "=v"(thread_data_weight_float2[1])
            : "v"(thread_data_weight2[i])
        );
    }
    else
    {
        asm volatile(
            "v_cvt_f32_f16_e32 %0 %2\n"
            "v_cvt_f32_f16_sdwa %1 %2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1\n"
            : "=v"(thread_data_weight_float2[0]), "=v"(thread_data_weight_float2[1])
            : "v"(thread_data_weight2[i])
        );
    }
    ```

- Evidence mapping:
  - BF16 conversion → Bit shift and mask (zero mantissa bits)
  - FP16 conversion → Hardware `v_cvt_f32_f16` instruction
  - SDWA for packed access → `src0_sel:WORD_1` for upper 16 bits

## Optimization 5: Fused Max Reduction for Quantization Scale
- Commit ID: f9ec28ad7
- Optimization type: Compute
- Summary: Used `v_max3_f32` instruction for efficient 3-way maximum in quantization scale computation.

- Detailed explanation:
  For quantization, the kernel needs to find the maximum absolute value in each group. Using `v_max3_f32` allows comparing three values in a single instruction, reducing the number of comparison operations by ~33%.

- Code excerpt:
    ```cpp
    float thread_max = 1e-10f;
    if constexpr(thread_data_size % 2 == 0)
    {
        for(int i = 0; i < thread_data_size; i += 2)
        {
            asm volatile("v_max3_f32 %0, %1, %2, %3\n"
                        : "=v"(thread_max)
                        : "v"(thread_max),
                        "v"(fabsf(thread_data_float[i])),
                        "v"(fabsf(thread_data_float[i + 1])));
        }
    }
    ```

- Evidence mapping:
  - 3-way max → `v_max3_f32` instruction
  - Reduced iterations → Loop stride of 2 instead of 1
  - Compile-time optimization → `if constexpr` for even sizes
