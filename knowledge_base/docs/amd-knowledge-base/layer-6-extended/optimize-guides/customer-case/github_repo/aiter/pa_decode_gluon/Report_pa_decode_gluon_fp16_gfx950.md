# Kernel: pa_decode_gluon

## Variant Context
- Input semantic type: Paged Attention decode (autoregressive inference)
- Datatype(s): FP16/BF16 (query, key, value), FP8 (optional quantized KV cache)
- Data representation: Paged KV cache with block tables
- Target architecture: gfx950 (MI350) with CDNA4 architecture

## Functionality
This kernel implements paged attention for the decode phase of LLM inference. It handles single-token queries against a paged KV cache, supporting features like sliding window attention, sink tokens, and grouped-query attention (GQA). The Gluon implementation uses Triton's experimental Gluon DSL for fine-grained control over AMD GPU resources.

## Optimization 1: Remove Query/Output Transpose Kernels
- Commit ID: fb1c584c7
- Optimization type: Fusion / Memory
- Summary: Eliminated separate transpose kernels by using direct 5D tensor access, reducing kernel launch overhead and memory traffic.

- Detailed explanation:
  The original implementation required separate transpose kernels to convert query and output tensors between different memory layouts:
  1. `transpose_query_gluon_kernel` - Transpose query from [batch, seq, heads, dim] to internal layout
  2. `transpose_output_gluon_kernel` - Transpose output back to original layout
  
  The optimized version directly accesses tensors using 5D indexing with strides (bs, qlen, kv_head, group_size, head_dim), eliminating:
  - Two kernel launches
  - Intermediate tensor allocations
  - Memory bandwidth for reading/writing transposed data

- Code excerpt:
    ```python
    # BEFORE: Separate transpose kernels required
    # transpose_query_gluon_kernel and transpose_output_gluon_kernel
    # Files removed:
    # - transpose_output_gluon_kernel.cpp.jinja
    # - transpose_query_gluon_kernel.cpp.jinja
    # - transpose_query_output_gluon_aot.py
    
    # AFTER: Direct 5D tensor access
    # Replace query strides (seq, head) with (bs, qlen, kv_head, group_size)
    # Add mtp_blocked_query_layout for [seq_len, group_size, head_size] tensor
    # Load query directly with 3D layout and reshape
    ```

- Evidence mapping:
  - Removed transpose kernels → Deleted `transpose_query_output_gluon_aot.py` (653 lines)
  - Removed test file → Deleted `test_transpose_query_output_gluon.py` (867 lines)
  - Direct tensor access → New stride parameters (bs, qlen, kv_head, group_size)

## Optimization 2: MTP Layout Index Conversion
- Commit ID: fb1c584c7
- Optimization type: Memory
- Summary: Implemented proper MTP (Multi-Token Processing) layout index conversion for exp_sums, max_logits, and temporary_output tensors.

- Detailed explanation:
  The optimization adds proper index conversion between MTP layout indices and continuous memory indices. This enables:
  1. Correct access patterns for exp_sums and max_logits in the reduce kernel
  2. Proper temporary_output access with MTP layout
  3. Boundary checking using OUTPUT_SEQ_LEN_POW2 instead of OUTPUT_SEQ_LEN

- Code excerpt:
    ```python
    # Convert MTP layout indices to continuous indices for exp_sums/max_logits access
    # Convert MTP layout indices to continuous indices for temporary_output access
    # Fix OUTPUT_GROUP_SIZE boundary check to use OUTPUT_SEQ_LEN_POW2
    
    # Rename for clarity:
    # qk_row_mask_3d/1d → query_row_mask_3d/1d
    # Add separate pv_row_mask for PV operations with proper layout conversion
    ```

- Evidence mapping:
  - Index conversion → MTP layout to continuous index mapping
  - Boundary fix → `OUTPUT_SEQ_LEN_POW2` for correct masking

## Optimization 3: Triton Version Compatibility for MFMA Instructions
- Commit ID: fb1c584c7
- Optimization type: Compute / Architecture
- Summary: Added dynamic MFMA instruction shape configuration based on Triton version for compatibility with Triton 3.6.0+.

- Detailed explanation:
  The optimization adds version detection and dynamic configuration for AMD MFMA (Matrix Fused Multiply-Add) instructions:
  1. `parse_triton_version()` function for version string parsing
  2. `TRITON_VERSION_GE_3_6_0` flag for version checking
  3. Dynamic `instr_shape` configuration based on Triton version
  4. `MFMA_INSTR_K` based on COMPUTE_TYPE and CDNA_VERSION

- Code excerpt:
    ```python
    # Add parse_triton_version() function to handle version string parsing
    # Add TRITON_VERSION_GE_3_6_0 constexpr flag for version checking
    # Update AMDMFMALayout instr_shape to use dynamic configuration based on Triton version (3.6.0+)
    # Set MFMA_INSTR_K based on COMPUTE_TYPE and CDNA_VERSION
    
    # Version-specific tensor dimension ordering in reduce kernel
    # Skip permute operation for Triton 3.6.0+ as it handles dimensions differently
    ```

- Evidence mapping:
  - Version detection → `parse_triton_version()` function
  - Dynamic MFMA config → `TRITON_VERSION_GE_3_6_0` conditional
  - Architecture-specific → CDNA_VERSION parameter
