# Kernel: Fused MoE (Mixture of Experts)

## Variant Context
- Input semantic type: MoE GEMM (Token routing and expert computation)
- Datatype(s): fp16, bf16
- Data representation: Dense token activations, stacked expert weights
- Target architecture: Generic CUDA/HIP (Triton-based)

## Functionality
The Fused MoE kernel implements efficient Mixture of Experts computation by:
1. Routing tokens to top-k experts using softmax/sigmoid gating
2. Aligning token assignments to block sizes for efficient GEMM
3. Fusing expert GEMM computations with routing weight multiplication
4. Supporting various activation functions (SiLU, GELU, etc.)

## Optimization 1: Initial Fused MoE Implementation with Block Alignment
- Commit ID: 5d60def02
- Optimization type: Compute / Memory
- Summary: Implement fused MoE kernel with CUDA block alignment for efficient expert computation
- Detailed explanation:
  The initial implementation introduces a key optimization: aligning token assignments to CUDA block sizes. This ensures that each expert processes a number of tokens divisible by the block size, enabling efficient tiled matrix multiplication without wasted computation.
- Code excerpt:
    ```python
    def moe_align_block_size(topk_ids: torch.Tensor, block_size: int,
                             num_experts: int):
        """
        Aligns the token distribution across experts to be compatible with 
        block size for matrix multiplication.
        
        This function pads the number of tokens that each expert needs to 
        process so that it is divisible by block_size.
        
        Example:
        Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], 
        block_size = 4, and num_experts = 4:
        - We initially have 12 tokens and 4 experts, each processing 3 tokens
        - As block_size is 4, we pad 1 token for each expert
        - After sorting by expert index, padding tokens (index=12) are ignored
        """
        sorted_ids = torch.empty(
            (topk_ids.numel() + num_experts * (block_size - 1), ),
            dtype=torch.int32, device=topk_ids.device)
        expert_ids = torch.empty((topk_ids.numel() + num_experts, ),
                                 dtype=torch.int32, device=topk_ids.device)
        sorted_ids.fill_(topk_ids.numel())  # Fill with padding value
        ops.moe_align_block_size(topk_ids, num_experts, block_size, 
                                  sorted_ids, expert_ids, num_tokens_post_pad)
        return sorted_ids, expert_ids, num_tokens_post_pad
    ```
- Evidence mapping:
  - "Block alignment" → Padding to make token count divisible by `block_size`
  - "Efficient GEMM" → Sorted token IDs enable coalesced memory access
  - "Expert grouping" → `expert_ids` tensor groups tokens by assigned expert

## Optimization 2: Triton Kernel with L2 Cache Optimization
- Commit ID: cfc15a103
- Optimization type: Memory / Compute
- Summary: Optimize Triton MoE kernel with grouped ordering for L2 cache reuse
- Detailed explanation:
  The Triton kernel uses grouped program ID ordering to promote L2 cache reuse. By processing adjacent blocks together, the kernel maximizes reuse of weight data loaded into L2 cache, reducing memory bandwidth requirements.
- Code excerpt:
    ```python
    @triton.jit
    def fused_moe_kernel(
        a_ptr, b_ptr, c_ptr, topk_weights_ptr,
        sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
        N, K, EM, num_valid_tokens,
        stride_am, stride_ak, stride_be, stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
        compute_type: tl.constexpr,
    ):
        # Map program ids to blocks with grouped ordering for L2 reuse
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        
        # Early exit for padding blocks
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return
    ```
- Evidence mapping:
  - "Grouped ordering" → `GROUP_SIZE_M` groups adjacent M blocks together
  - "L2 cache reuse" → Adjacent blocks share weight data in L2
  - "Early exit" → Skip padding blocks to avoid wasted computation

## Optimization 3: Fused Routing Weight Multiplication
- Commit ID: cfc15a103
- Optimization type: Compute / Fusion
- Summary: Fuse routing weight multiplication into the GEMM epilogue
- Detailed explanation:
  Instead of applying routing weights in a separate kernel, this optimization fuses the multiplication into the GEMM epilogue. This eliminates an extra memory round-trip and reduces kernel launch overhead.
- Code excerpt:
    ```python
    @triton.jit
    def fused_moe_kernel(..., MUL_ROUTED_WEIGHT: tl.constexpr, ...):
        # ... GEMM computation ...
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=token_mask[:, None] & 
                        (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        
        # Fused routing weight multiplication
        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(topk_weights_ptr + offs_token,
                                 mask=token_mask, other=0)
            accumulator = accumulator * moe_weight[:, None]
        
        accumulator = accumulator.to(compute_type)
        tl.store(c_ptrs, accumulator, mask=c_mask)
    ```
- Evidence mapping:
  - "Fused multiplication" → `MUL_ROUTED_WEIGHT` flag enables in-kernel weight application
  - "No extra memory access" → Weights loaded once and applied to accumulator
  - "Compile-time decision" → `tl.constexpr` enables dead code elimination when disabled

## Optimization 4: Hardware-Specific Tuning Configurations
- Commit ID: cfc15a103
- Optimization type: Launch Configuration
- Summary: Add pre-tuned kernel configurations for different GPU architectures
- Detailed explanation:
  Different GPU architectures have different optimal tile sizes and configurations. This optimization adds JSON configuration files with pre-tuned parameters for A100 and H100 GPUs, automatically selecting the best configuration based on problem size.
- Code excerpt:
    ```python
    def get_moe_configs(E: int, N: int, dtype: str) -> Optional[Dict[str, Any]]:
        """Load pre-tuned configurations for the given problem size."""
        device_name = torch.cuda.get_device_name()
        config_file = f"E={E},N={N},device_name={device_name},dtype={dtype}.json"
        config_path = os.path.join(os.path.dirname(__file__), "configs", config_file)
        
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
        return None
    
    # Example config for H100:
    # {
    #   "BLOCK_SIZE_M": 64,
    #   "BLOCK_SIZE_N": 128, 
    #   "BLOCK_SIZE_K": 64,
    #   "GROUP_SIZE_M": 8,
    #   "num_warps": 4,
    #   "num_stages": 3
    # }
    ```
- Evidence mapping:
  - "Architecture-specific" → Config files named with `device_name`
  - "Problem-size tuning" → Configs indexed by `E` (experts) and `N` (hidden dim)
  - "Auto-selection" → `get_moe_configs` automatically loads best config

## Optimization 5: Chunked Processing for Large Inputs
- Commit ID: 12a59959e
- Optimization type: Memory
- Summary: Add chunking mechanism to handle large batch sizes without OOM
- Detailed explanation:
  For very large batch sizes, the intermediate tensors can exceed GPU memory. This optimization processes tokens in chunks, reducing peak memory usage while maintaining correctness.
- Code excerpt:
    ```python
    def fused_moe(hidden_states, w1, w2, gating_output, topk, ...):
        M = hidden_states.shape[0]
        
        # Chunk size to avoid OOM
        CHUNK_SIZE = 64 * 1024  # 64K tokens per chunk
        
        if M > CHUNK_SIZE:
            # Process in chunks
            outputs = []
            for i in range(0, M, CHUNK_SIZE):
                chunk = hidden_states[i:i+CHUNK_SIZE]
                chunk_output = fused_moe_impl(chunk, w1, w2, 
                                               gating_output[i:i+CHUNK_SIZE], 
                                               topk, ...)
                outputs.append(chunk_output)
            return torch.cat(outputs, dim=0)
        else:
            return fused_moe_impl(hidden_states, w1, w2, gating_output, 
                                   topk, ...)
    ```
- Evidence mapping:
  - "Chunked processing" → Loop over `CHUNK_SIZE` segments
  - "Memory bounded" → Each chunk fits in GPU memory
  - "Correctness preserved" → Results concatenated to match non-chunked output

## Optimization 6: TopK Softmax Fusion
- Commit ID: f0d4e1455
- Optimization type: Fusion / Compute
- Summary: Fuse top-k selection with softmax computation in a single CUDA kernel
- Detailed explanation:
  The gating network requires computing softmax over expert scores and selecting top-k experts. This optimization fuses both operations into a single kernel, avoiding intermediate tensor materialization.
- Code excerpt:
    ```cpp
    // Fused topk + softmax kernel
    template <typename T, int EXPERTS, int TOKENS_PER_THREAD>
    __global__ void topk_softmax_kernel(
        T* topk_weights,      // Output: top-k weights
        int* topk_indices,    // Output: top-k expert indices
        const T* gating_output,  // Input: raw gating scores
        int num_tokens, int top_k) {
      
      // Each thread handles TOKENS_PER_THREAD tokens
      // 1. Load gating scores for all experts
      // 2. Find top-k experts using partial sort
      // 3. Compute softmax only over selected experts
      // 4. Store results
      
      float scores[EXPERTS];
      // Load and find max for numerical stability
      float max_score = -FLT_MAX;
      for (int e = 0; e < EXPERTS; e++) {
        scores[e] = gating_output[token_idx * EXPERTS + e];
        max_score = fmaxf(max_score, scores[e]);
      }
      
      // Compute softmax and select top-k simultaneously
      // ... (partial sort + softmax computation)
    }
    ```
- Evidence mapping:
  - "Fused operation" → Single kernel for topk + softmax
  - "No intermediate tensor" → Results written directly to output
  - "Numerical stability" → Max subtraction before exp computation
