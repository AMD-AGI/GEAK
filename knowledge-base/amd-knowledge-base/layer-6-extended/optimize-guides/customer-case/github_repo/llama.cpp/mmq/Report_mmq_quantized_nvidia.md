---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Matrix Multiplication Quantized (MMQ)

## Variant Context
- Input semantic type: Matrix multiplication (GEMM)
- Datatype(s): Quantized weights (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ types)
- Data representation: Block-wise quantized with scales (various block sizes: 32, 64, 256 elements)
- Target architecture: NVIDIA (Volta CC 7.0+, Turing CC 7.5+, Ampere CC 8.0+)

## Functionality
The MMQ kernel performs matrix multiplication where the weight matrix (src0) is stored in a quantized format and the activation matrix (src1) is in FP16/FP32. The kernel dequantizes weights on-the-fly during computation, using shared memory tiling and Tensor Core MMA instructions for high throughput.

Key features:
- Support for 15+ quantization formats
- Stream-K work distribution for load balancing
- Batched operations for MoE (Mixture of Experts)
- Architecture-specific optimizations

---

## Optimization 1: MMA Instructions for Quantized GEMM
- Commit ID: 864a0b67a (initial), refined in subsequent commits
- Optimization type: Compute (Tensor Core utilization)
- Summary: Use Tensor Core MMA instructions for quantized matrix multiplication with on-the-fly dequantization
- Detailed explanation: The kernel loads quantized data into shared memory, dequantizes to INT8/FP16, and uses MMA instructions for the matrix multiply. The dequantization is fused with the data loading to hide latency. Different quantization formats require different dequantization logic but share the same MMA compute path.

- Code excerpt:
    ```cpp
    #ifdef NEW_MMA_AVAILABLE
    // MMA tile types for quantized GEMM
    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_C_I16J8<int> mma_C;
    
    // Load and dequantize Q4_0 data
    x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI4_0) + kqsx + 0] = 
        __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
    x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI4_0) + kqsx + QI4_0] = 
        __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
    
    // MMA computation
    mma_A A;
    A.load(x_tile + ...);
    mma_C C;
    C.mma(A, B);
    ```

- Evidence mapping:
  - "MMA instructions" → `mma_int_A_I16K8`, `mma_int_B_J8K8` types
  - "On-the-fly dequantization" → `__vsubss4` for Q4_0 unpacking
  - "Fused loading" → dequant happens during shared memory store

---

## Optimization 2: Stream-K Work Distribution
- Commit ID: Multiple commits refining stream-k
- Optimization type: Scheduling (load balancing)
- Summary: Implement stream-K algorithm for better GPU utilization with irregular problem sizes
- Detailed explanation: Stream-K distributes work across thread blocks more evenly than traditional tiling. Instead of assigning fixed tiles to blocks, stream-K assigns a stream of K-dimension work that blocks process cooperatively. This improves utilization when the number of tiles doesn't divide evenly by the number of SMs.

- Code excerpt:
    ```cpp
    // Stream-K work distribution
    template<typename T>
    __global__ void mul_mat_q_stream_k(
        const void * __restrict__ x,
        const void * __restrict__ y,
        float * __restrict__ dst,
        const int ne00, const int ne01, const int ne10, const int ne11,
        const int64_t total_tiles_k,
        const int tiles_per_block) {
        
        // Calculate which K tiles this block processes
        const int64_t tile_start = blockIdx.x * tiles_per_block;
        const int64_t tile_end = min(tile_start + tiles_per_block, total_tiles_k);
        
        // Process assigned K tiles
        for (int64_t tile = tile_start; tile < tile_end; ++tile) {
            // Decode tile coordinates
            const int i = tile / ne11;
            const int j = tile % ne11;
            // Process tile...
        }
    }
    
    // Fixup kernel for partial results
    __global__ void stream_k_fixup(...) {
        // Combine partial sums from different blocks
    }
    ```

- Evidence mapping:
  - "Stream-K distribution" → `tiles_per_block` parameter
  - "Cooperative processing" → blocks process overlapping K ranges
  - "Fixup kernel" → `stream_k_fixup` combines partial results

---

## Optimization 3: DP4A Instruction Utilization
- Commit ID: Various commits
- Optimization type: Compute (instruction selection)
- Summary: Use DP4A (4-element dot product) instructions for INT8 quantized computation
- Detailed explanation: DP4A computes a dot product of four INT8 values and accumulates to INT32 in a single instruction. This is ideal for quantized GEMM where weights are stored as 4-bit or 8-bit integers. The kernel packs data appropriately to maximize DP4A throughput.

- Code excerpt:
    ```cpp
    // DP4A-based dot product for Q4_0
    template <int vdr>
    static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
        const int * __restrict__ v, const int * __restrict__ u,
        const float d4, const half2 ds8) {
        
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < vdr; ++i) {
            // DP4A: dot product of 4 INT8 values
            sumi = __dp4a(v[i], u[i], sumi);
        }
        
        // Apply scales
        const float2 ds8f = __half22float2(ds8);
        return d4 * (sumi * ds8f.x - 8.0f * ds8f.y);
    }
    ```

- Evidence mapping:
  - "DP4A instruction" → `__dp4a(v[i], u[i], sumi)` intrinsic
  - "INT8 dot product" → 4 bytes packed in int
  - "Scale application" → `d4 * (sumi * ds8f.x - ...)` for dequantization

---

## Optimization 4: Shared Memory Bank Conflict Avoidance
- Commit ID: Various commits
- Optimization type: Memory (bank conflicts)
- Summary: Pad shared memory tiles to avoid bank conflicts during parallel access
- Detailed explanation: CUDA shared memory is organized in 32 banks. When multiple threads access the same bank, accesses are serialized. The kernel pads tile dimensions to ensure stride patterns avoid conflicts. For MMA tiles, padding of +4 elements ensures 8-byte alignment while avoiding conflicts.

- Code excerpt:
    ```cpp
    // Tile sizes with padding to avoid bank conflicts
    #define MMQ_MMA_TILE_X_K_Q8_0 (2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0 + 4)
    #define MMQ_MMA_TILE_X_K_Q8_1 (2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0 + 4)
    #define MMQ_MMA_TILE_X_K_Q2_K (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K         + 4)
    
    static_assert(MMQ_MMA_TILE_X_K_Q8_0 % 8 == 4, "Wrong padding.");
    static_assert(MMQ_MMA_TILE_X_K_Q8_1 % 8 == 4, "Wrong padding.");
    
    // Padding ensures K % 8 == 4 for optimal MMA access patterns
    ```

- Evidence mapping:
  - "Padding" → `+ 4` in tile size definitions
  - "Bank conflict avoidance" → `% 8 == 4` assertion
  - "MMA alignment" → 8-element alignment for Tensor Core access

---

## Optimization 5: Batched MMQ for MoE
- Commit ID: e1e8e0991
- Optimization type: Compute (batching)
- Summary: Support batched and non-contiguous matrix multiplication for Mixture of Experts models
- Detailed explanation: MoE models route different tokens to different experts, requiring batched GEMM with non-contiguous inputs. This optimization adds support for processing multiple expert matrices in a single kernel launch, with proper handling of the routing indices.

- Code excerpt:
    ```cpp
    // CUDA: batched+noncont MMQ, refactor bs>1 MoE code
    template<typename T>
    __global__ void mul_mat_q_batched(
        const void * __restrict__ x,
        const void * __restrict__ y,
        float * __restrict__ dst,
        const int * __restrict__ ids,  // Expert routing indices
        const int ne00, const int ne01, const int ne10, const int ne11,
        const int ne02,  // Number of experts
        const int64_t nb01, const int64_t nb02,  // Strides
        ...) {
        
        // Get expert index for this token
        const int expert_id = ids[blockIdx.z];
        
        // Offset to correct expert weights
        const void * x_expert = (const char *)x + expert_id * nb02;
        
        // Process with expert-specific weights
        ...
    }
    ```

- Evidence mapping:
  - "Batched operation" → `blockIdx.z` for batch dimension
  - "Expert routing" → `ids[blockIdx.z]` for expert selection
  - "Non-contiguous" → stride parameters `nb01`, `nb02`

---

## Optimization 6: IQ (Integer Quantization) Format Support
- Commit ID: 8e558309d, 69c487f4e
- Optimization type: Compute (format support)
- Summary: Added MMQ support for IQ quantization formats (iq4_nl, iq4_xs, iq2_*, iq3_*)
- Detailed explanation: IQ formats use non-linear quantization with lookup tables for better accuracy at low bit widths. The kernel implements efficient dequantization using shared memory lookup tables and vectorized loads.

- Code excerpt:
    ```cpp
    // CUDA: MMQ support for iq4_nl, iq4_xs
    template <int mmq_y, bool need_check>
    static __device__ __forceinline__ void load_tiles_iq4_nl(
        const char * __restrict__ x, int * __restrict__ x_tile,
        const int kbx0, const int i_max, const int stride) {
        
        // Load IQ4_NL lookup table to shared memory
        __shared__ int8_t iq4nl_table[16];
        if (threadIdx.x < 16 && threadIdx.y == 0) {
            iq4nl_table[threadIdx.x] = kvalues_iq4nl[threadIdx.x];
        }
        __syncthreads();
        
        // Dequantize using lookup table
        const int qs = bxi->qs[kqsx];
        x_qs[...] = iq4nl_table[qs & 0xF];
        x_qs[...] = iq4nl_table[qs >> 4];
    }
    ```

- Evidence mapping:
  - "IQ format support" → `load_tiles_iq4_nl` function
  - "Lookup table" → `iq4nl_table[16]` in shared memory
  - "Non-linear dequant" → table lookup instead of linear scaling
