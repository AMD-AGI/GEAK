# Kernel: Paged Attention

## Variant Context
- Input semantic type: Attention (Query-Key-Value computation with paged KV cache)
- Datatype(s): fp16, bf16, fp32
- Data representation: Paged KV cache with block-based memory management
- Target architecture: Generic CUDA/HIP (sm70+, gfx90a+)

## Functionality
The Paged Attention kernel implements efficient attention computation for LLM inference by:
1. Managing KV cache in fixed-size blocks (pages) for memory efficiency
2. Supporting variable-length sequences with block tables for indirection
3. Computing scaled dot-product attention with optional ALiBi positional encoding
4. Supporting Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

## Optimization 1: Query Vector Shared Memory Optimization
- Commit ID: 79af7e96a
- Optimization type: Memory
- Summary: Move query vectors to shared memory to enable better memory access patterns and reduce redundant global memory loads
- Detailed explanation: 
  The original implementation stored query vectors in registers per thread. This optimization moves the query vectors to shared memory, allowing all threads in a thread group to share the same query data. This reduces redundant global memory loads since the query is loaded once and shared across the thread group, improving memory bandwidth utilization.
- Code excerpt:
    ```cpp
    // Before: Query vectors in registers (per thread)
    Q_vec q_vecs[NUM_VECS_PER_THREAD];
    for (int i = 0; i < NUM_VECS_PER_THREAD; i++) {
      const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
      q_vecs[i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
    }
    
    // After: Query vectors in shared memory (shared across thread group)
    __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
    for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
      const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
      q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
    }
    __syncthreads();
    ```
- Evidence mapping:
  - "Move query vectors to shared memory" → `__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD]`
  - "Reduce redundant loads" → Loop now uses `thread_group_idx` stride to distribute load work across threads
  - "Thread synchronization for shared data" → `__syncthreads()` ensures all threads see the loaded data

## Optimization 2: Multi-Query Attention (MQA) Support
- Commit ID: 96853af5a
- Optimization type: Memory / Compute
- Summary: Add native support for Multi-Query Attention by using head mapping to share KV heads across query heads
- Detailed explanation:
  MQA uses fewer KV heads than query heads, reducing memory bandwidth requirements. This optimization adds a `head_mapping` tensor that maps each query head to its corresponding KV head. The kernel uses this mapping to access the correct KV cache entries, enabling memory savings proportional to the query-to-KV head ratio.
- Code excerpt:
    ```cpp
    // Add head mapping for MQA/GQA support
    const int kv_head_idx = head_mapping[head_idx];
    
    // Use kv_head_idx instead of head_idx for KV cache access
    const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                                    + kv_head_idx * kv_head_stride
                                    + physical_block_offset * x;
    
    const scalar_t* v_ptr = v_cache + physical_block_number * kv_block_stride
                                    + kv_head_idx * kv_head_stride;
    ```
- Evidence mapping:
  - "Head mapping for MQA" → `const int kv_head_idx = head_mapping[head_idx]`
  - "Reduced KV cache memory access" → Using `kv_head_idx` instead of `head_idx` for cache indexing
  - "Flexible stride support" → `kv_block_stride` and `kv_head_stride` parameters for non-contiguous layouts

## Optimization 3: PagedAttention V2 with Partitioning
- Commit ID: 928de4688
- Optimization type: Compute / Scheduling
- Summary: Implement partitioned attention computation for long sequences to improve parallelism and reduce memory pressure
- Detailed explanation:
  PagedAttention V2 divides long sequences into partitions that can be processed in parallel. Each partition computes partial attention results (exp_sums, max_logits, partial outputs), which are then reduced in a second kernel. This enables better GPU utilization for long sequences by:
  1. Increasing parallelism through partition-level parallelization
  2. Reducing shared memory pressure per thread block
  3. Enabling better load balancing across SMs
- Code excerpt:
    ```cpp
    // Grid now includes partition dimension
    // Grid: (num_heads, num_seqs, max_num_partitions)
    
    template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS,
             int PARTITION_SIZE = 0>  // Zero means no partitioning
    __device__ void paged_attention_kernel(
        float* __restrict__ exp_sums,     // [num_seqs, num_heads, max_num_partitions]
        float* __restrict__ max_logits,   // [num_seqs, num_heads, max_num_partitions]
        scalar_t* __restrict__ out,       // [num_seqs, num_heads, max_num_partitions, head_size]
        ...) {
      
      const int partition_idx = blockIdx.z;
      const int max_num_partitions = gridDim.z;
      constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
      
      // Early exit for partitions beyond sequence length
      if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= context_len) {
        return;
      }
      
      // Compute partition boundaries
      const int num_blocks_per_partition = USE_PARTITIONING ? 
          PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;
      const int start_block_idx = USE_PARTITIONING ? 
          partition_idx * num_blocks_per_partition : 0;
      const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition, 
                                     num_context_blocks);
      
      // Store partition results for later reduction
      if (USE_PARTITIONING && thread_idx == 0) {
        float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
                                           + head_idx * max_num_partitions
                                           + partition_idx;
        *max_logits_ptr = qk_max;
        float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions
                                       + head_idx * max_num_partitions
                                       + partition_idx;
        *exp_sums_ptr = exp_sum;
      }
    }
    ```
- Evidence mapping:
  - "Partition-level parallelism" → `const int partition_idx = blockIdx.z` and grid dimension `max_num_partitions`
  - "Reduced memory pressure" → Each partition processes only `PARTITION_SIZE` tokens
  - "Partial results for reduction" → `exp_sums` and `max_logits` arrays store per-partition statistics

## Optimization 4: FP8 KV Cache Support
- Commit ID: 2ff767b51 (ROCm), c83310174 (NVIDIA)
- Optimization type: Memory / Precision
- Summary: Add FP8 (E4M3/E5M2) KV cache support to reduce memory bandwidth and storage requirements
- Detailed explanation:
  FP8 quantization reduces KV cache memory by 2x compared to FP16, enabling longer context lengths and larger batch sizes. The kernel supports both E4M3 (higher precision) and E5M2 (larger dynamic range) formats with per-tensor or per-token scaling factors.
- Code excerpt:
    ```cpp
    // FP8 KV cache data type enumeration
    enum class Fp8KVCacheDataType {
      kAuto = 0,
      kFp8E4M3 = 1,
      kFp8E5M2 = 2,
    };
    
    // Template parameter for FP8 KV cache
    template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
              int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
              int PARTITION_SIZE = 0>
    __device__ void paged_attention_kernel(...,
        const float* k_scale, const float* v_scale, ...) {
      
      // Dequantize FP8 cache values during computation
      // k_scale and v_scale are used to convert FP8 back to compute precision
    }
    ```
- Evidence mapping:
  - "FP8 format support" → `Fp8KVCacheDataType` enum with E4M3 and E5M2 options
  - "Separate cache type" → Template parameter `cache_t` distinct from `scalar_t`
  - "Scaling factors" → `k_scale` and `v_scale` parameters for dequantization

## Optimization 5: Block-Sparse Attention Support
- Commit ID: 8e192ff96
- Optimization type: Compute
- Summary: Add block-sparse attention pattern support for models like Phi-3-Small that use sparse attention
- Detailed explanation:
  Block-sparse attention reduces computation by skipping attention to certain blocks based on a sparsity pattern. This is useful for models with local + strided attention patterns, reducing the O(n²) attention complexity.
- Code excerpt:
    ```cpp
    template <..., bool IS_BLOCK_SPARSE, ...>
    __device__ void paged_attention_kernel(...,
        const int blocksparse_local_blocks,
        const int blocksparse_vert_stride,
        const int blocksparse_block_size,
        const int blocksparse_head_sliding_step) {
      
      // Skip blocks that are not in the sparse attention pattern
      if (IS_BLOCK_SPARSE) {
        // Check if current block should be attended to based on:
        // - Local window (blocksparse_local_blocks)
        // - Vertical stride pattern (blocksparse_vert_stride)
        // - Head-specific sliding (blocksparse_head_sliding_step)
      }
    }
    ```
- Evidence mapping:
  - "Sparse attention template" → `IS_BLOCK_SPARSE` template parameter
  - "Sparsity pattern parameters" → `blocksparse_local_blocks`, `blocksparse_vert_stride`, etc.
  - "Conditional block processing" → Blocks outside sparse pattern are skipped

## Optimization 6: Warp-Level Reduction Optimization
- Commit ID: e4a28e531
- Optimization type: Compute
- Summary: Fix and optimize warp-level reductions for both CUDA and ROCm platforms
- Detailed explanation:
  The block reduction for softmax computation was optimized to correctly handle different warp sizes on CUDA (32) and ROCm (64), ensuring correct parallel reduction across all threads.
- Code excerpt:
    ```cpp
    // Utility function for attention softmax with correct warp handling
    template <int NUM_WARPS>
    inline __device__ float block_sum(float* red_smem, float sum) {
      int warp = threadIdx.x / WARP_SIZE;
      int lane = threadIdx.x % WARP_SIZE;
      
      // Warp-level reduction
      #pragma unroll
      for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        sum += VLLM_SHFL_XOR_SYNC(sum, mask);
      }
      
      // Store warp results to shared memory
      if (lane == 0) {
        red_smem[warp] = sum;
      }
      __syncthreads();
      
      // Final reduction across warps
      if (lane < NUM_WARPS) {
        sum = red_smem[lane];
      }
      #pragma unroll
      for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
        sum += VLLM_SHFL_XOR_SYNC(sum, mask);
      }
      
      return VLLM_SHFL_SYNC(sum, 0);
    }
    ```
- Evidence mapping:
  - "Platform-agnostic warp size" → `WARP_SIZE` macro handles CUDA vs ROCm differences
  - "Two-level reduction" → First within warp using shuffle, then across warps using shared memory
  - "Broadcast final result" → `VLLM_SHFL_SYNC(sum, 0)` broadcasts to all threads
