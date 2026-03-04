---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: ROCm Paged Attention (atrexPA)

## Variant Context
- Input semantic type: Attention (decode phase paged attention)
- Datatype(s): bf16 (bfloat16)
- Data representation: Paged KV cache with block tables
- Target architecture: gfx942 (AMD MI300 series)

## Functionality
This kernel implements paged attention for the decode phase of LLM inference on AMD ROCm GPUs. It consists of two main kernels:
1. **pa_decode_dot_kernel**: Computes the dot product between query and key vectors, producing attention scores
2. **pa_decode_reduce_kernel**: Reduces the partial attention outputs across partitions to produce the final output

The kernel is optimized for low concurrency scenarios (small batch sizes) and supports various head sizes (64, 128) and group sizes (2-16) with configurable partition sizes (256, 512).

## Optimization 1: Triton-based Kernel Generation for ROCm
- Commit ID: a5619d65f
- Optimization type: Compute / Architecture-specific
- Summary: Introduced Triton-generated HSACO kernels specifically optimized for AMD gfx942 architecture
- Detailed explanation: 
  The optimization introduces pre-compiled Triton kernels that are embedded as HSACO (HIP System Architecture Code Object) binaries. This approach:
  1. Eliminates JIT compilation overhead at runtime
  2. Allows architecture-specific optimizations to be baked into the binary
  3. Provides specialized kernel variants for different head sizes (64, 128), group sizes (2-16), and partition sizes (256, 512)
  
- Code excerpt:
    ```cpp
    // From atrexPA.cc - Dispatch macro for different configurations
    #define DISPATCH_HEAD_GRP_PARTITION(head_sz, grp_sz, partition_sz, output_ptr)                                         \
        do {                                                                                                               \
            if (head_sz == 64 && grp_sz == 2 && partition_sz == 256) {                                                     \
                CALL_PA_DECODE_DOT_KERNEL(64, 2, 256, output_ptr);                                                         \
            } else if (head_sz == 64 && grp_sz == 2 && partition_sz == 512) {                                              \
                CALL_PA_DECODE_DOT_KERNEL(64, 2, 512, output_ptr);                                                         \
            } else if (head_sz == 128 && grp_sz == 8 && partition_sz == 256) {                                             \
                CALL_PA_DECODE_DOT_KERNEL(128, 8, 256, output_ptr);                                                        \
            } // ... more configurations
        } while (0)
    ```
- Evidence mapping:
  - Template specialization for different configurations → DISPATCH_HEAD_GRP_PARTITION macro with explicit head_sz, grp_sz, partition_sz combinations
  - Pre-compiled HSACO binaries → `unsigned char _pa_decode_dot_kernel64_2_256_hsaco[13136]` embedded in pa_decode_dot_kernel.cu

## Optimization 2: Two-Phase Attention Computation (Dot + Reduce)
- Commit ID: a5619d65f
- Optimization type: Compute / Memory
- Summary: Split attention computation into dot product and reduction phases for better parallelism
- Detailed explanation:
  The kernel splits the attention computation into two phases:
  1. **Dot Phase**: Each workgroup computes partial attention scores and weighted values for a partition of the KV cache
  2. **Reduce Phase**: Combines partial results using numerically stable softmax reduction
  
  This approach enables:
  - Better GPU occupancy by allowing more parallel workgroups
  - Reduced memory pressure by computing partial results in shared memory
  - Support for arbitrarily long sequences through partitioning

- Code excerpt:
    ```cpp
    // From atrexPA.cc - Two-phase kernel invocation
    // Phase 1: Dot product kernel
    DISPATCH_HEAD_GRP_PARTITION(head_size, grp_size, partition_size, tmp_out_ptr);
    
    // Phase 2: Reduce kernel  
    DISPATCH_REDUCE_KERNEL(head_size, grp_size, num_partitions);
    ```
- Evidence mapping:
  - Two-phase computation → Separate CALL_PA_DECODE_DOT_KERNEL and CALL_PA_DECODE_REDUCE_KERNEL invocations
  - Intermediate storage for partial results → `tmp_out` tensor passed between phases

## Optimization 3: Configurable Partition Size for Memory/Compute Trade-off
- Commit ID: a5619d65f
- Optimization type: Memory / Launch configuration
- Summary: Support for 256 and 512 partition sizes to balance memory usage and parallelism
- Detailed explanation:
  The kernel supports two partition sizes:
  - **256**: Smaller partitions mean more parallel workgroups but higher reduction overhead
  - **512**: Larger partitions reduce reduction overhead but may limit parallelism for short sequences
  
  The choice is made based on sequence length and available GPU resources.

- Code excerpt:
    ```cpp
    // From atrexPA.cc - Partition size selection
    int partition_size = 256;  // or 512 based on configuration
    int num_partitions = DIVIDE_ROUND_UP(max_context_len, partition_size);
    
    // Grid configuration based on partition count
    std::vector<int> grid = {num_partitions, num_heads, batch_size};
    ```
- Evidence mapping:
  - Configurable partition sizes → Both 256 and 512 variants in DISPATCH_HEAD_GRP_PARTITION
  - Dynamic grid sizing → `num_partitions` computed from `max_context_len / partition_size`

## Optimization 4: Group Query Attention (GQA) Support
- Commit ID: a5619d65f
- Optimization type: Compute
- Summary: Native support for grouped query attention with configurable group sizes
- Detailed explanation:
  The kernel natively supports GQA where multiple query heads share the same KV heads. This is implemented through:
  - Specialized kernel variants for group sizes 2-16
  - Efficient memory access patterns that reuse KV cache across query heads in the same group

- Code excerpt:
    ```cpp
    // From atrexPA.cc - GQA configuration
    int grp_size = num_heads / num_kv_heads;  // Number of query heads per KV head
    
    // Dispatch to appropriate kernel variant
    if (head_sz == 128 && grp_sz == 8 && partition_sz == 256) {
        CALL_PA_DECODE_DOT_KERNEL(128, 8, 256, output_ptr);
    }
    ```
- Evidence mapping:
  - GQA support → `grp_sz` parameter in kernel dispatch (values 2-16)
  - Specialized kernels per group size → Separate HSACO binaries for each grp_sz value
