# Kernel: ROCm Sampling Kernels (FlashInfer-based)

## Variant Context
- Input semantic type: Token sampling (Top-K, Top-P sampling from probability distribution)
- Datatype(s): fp16, bf16
- Data representation: Probability distribution over vocabulary
- Target architecture: gfx942 (AMD MI300 series)

## Functionality
These kernels implement efficient token sampling operations for LLM inference on AMD ROCm GPUs. They are ported from FlashInfer and include:
- Top-K sampling: Select from top K highest probability tokens
- Top-P (nucleus) sampling: Select from tokens whose cumulative probability exceeds P
- Combined Top-K + Top-P sampling
- Probability renormalization

The kernels use hipcub for efficient parallel primitives like scan and reduce.

## Optimization 1: Increased Vector Size for AMD Architecture (16B → 64B)
- Commit ID: c3842c8bb
- Optimization type: Memory
- Summary: Increased vector load/store size from 16 bytes to 64 bytes for better memory bandwidth utilization on AMD GPUs
- Detailed explanation:
  AMD MI300 GPUs have wider memory interfaces compared to NVIDIA GPUs. The optimization increases the vector size used for memory operations from 16 bytes to 64 bytes:
  - For fp16: 32 elements per vector operation (vs 8 before)
  - For bf16: 32 elements per vector operation (vs 8 before)
  
  This better utilizes the memory bandwidth of AMD's HBM3 memory subsystem.

- Code excerpt:
    ```cpp
    // Before
    // const uint32_t vec_size = std::gcd(16 / sizeof(T), d);
    
    // After
    #define VEC_BYTES 64
    const uint32_t vec_size = std::gcd(VEC_BYTES / sizeof(T), d);
    ```
- Evidence mapping:
  - Vector size constant → `#define VEC_BYTES 64` (was hardcoded 16)
  - Applied to all sampling functions → TopKSamplingFromProb, TopPSamplingFromProb, TopKTopPSamplingFromProb, TopPRenormProb, TopKRenormProb

## Optimization 2: Removed Unused Dispatch Macro
- Commit ID: c3842c8bb
- Optimization type: Code cleanup / Compile time
- Summary: Removed unused DISPATCH_SOFTMAX_CACHE_INPUT macro to simplify code
- Detailed explanation:
  The `DISPATCH_SOFTMAX_CACHE_INPUT` macro was defined but not used in the sampling kernels. Removing it:
  - Reduces code complexity
  - Eliminates potential confusion
  - Slightly reduces compile time

- Code excerpt:
    ```cpp
    // Removed:
    // #define DISPATCH_SOFTMAX_CACHE_INPUT(cache_input, CACHE_INPUT, ...) \
    //   if (cache_input) {                                                \
    //     constexpr bool CACHE_INPUT = true;                              \
    //     __VA_ARGS__                                                     \
    //   } else {                                                          \
    //     constexpr bool CACHE_INPUT = false;                             \
    //     __VA_ARGS__                                                     \
    //   }
    ```
- Evidence mapping:
  - Macro removal → DISPATCH_SOFTMAX_CACHE_INPUT deleted from kernel.cuh

## Optimization 3: FlashInfer Sampling Kernel Port for AMD
- Commit ID: 7d282629c
- Optimization type: Compute / Architecture-specific
- Summary: Ported FlashInfer's efficient sampling kernels to AMD ROCm
- Detailed explanation:
  This commit introduces a complete port of FlashInfer's sampling kernels to AMD ROCm, including:
  - Top-K sampling with parallel prefix sum
  - Top-P (nucleus) sampling with cumulative probability computation
  - Combined Top-K + Top-P sampling
  - Efficient use of hipcub for parallel primitives
  
  The implementation uses warp-level operations and block-level reductions optimized for AMD's wavefront size (64 threads).

- Code excerpt:
    ```cpp
    // From kernel.cuh - Block configuration for AMD
    constexpr BlockScanAlgorithm SCAN_ALGO = BLOCK_SCAN_WARP_SCANS;
    constexpr BlockReduceAlgorithm REDUCE_ALGO = BLOCK_REDUCE_WARP_REDUCTIONS;
    
    // Compute capacity dispatch for different AMD GPUs
    auto compute_capacity = GetCudaComputeCapability();
    DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
        // Launch kernel with appropriate thread count
    });
    ```
- Evidence mapping:
  - hipcub usage → `BlockScanAlgorithm`, `BlockReduceAlgorithm` from hipcub
  - AMD-specific dispatch → `GetCudaComputeCapability()` and `DISPATCH_COMPUTE_CAP_NUM_THREADS`

## Optimization 4: Deterministic Sampling with Philox RNG
- Commit ID: 7d282629c
- Optimization type: Compute
- Summary: Support for deterministic sampling using Philox random number generator
- Detailed explanation:
  The kernels support deterministic sampling through:
  - Philox RNG with configurable seed and offset
  - Per-batch random state management
  - Reproducible results across runs with same seed
  
  This is important for debugging and reproducibility in production.

- Code excerpt:
    ```cpp
    hipError_t TopKSamplingFromProb(T* probs, IdType* output, IdType* indices, 
                                     T* top_k_val_arr, T* uniform_samples,
                                     uint32_t batch_size, uint32_t top_k_val, uint32_t d,
                                     bool deterministic, 
                                     uint64_t* philox_seed,    // RNG seed
                                     uint64_t* philox_offset,  // RNG offset
                                     hipStream_t stream = 0);
    ```
- Evidence mapping:
  - Deterministic flag → `bool deterministic` parameter
  - Philox RNG state → `uint64_t* philox_seed`, `uint64_t* philox_offset` parameters

## Optimization 5: Warp Size Fix for AMD (64 vs 32)
- Commit ID: 41ace8512
- Optimization type: Compute / Architecture-specific
- Summary: Fixed warp size handling for AMD GPUs (wavefront size = 64)
- Detailed explanation:
  AMD GPUs use a wavefront size of 64 threads, unlike NVIDIA's warp size of 32. This commit fixes:
  - Warp-level reduction operations
  - Shared memory sizing for warp-level operations
  - Thread indexing within warps
  
  Incorrect warp size assumptions can lead to incorrect results or crashes.

- Code excerpt:
    ```cpp
    // AMD wavefront size is 64, not 32
    constexpr int WARP_SIZE = 64;  // For AMD GPUs
    
    // Warp-level operations must account for this
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    ```
- Evidence mapping:
  - Warp size constant → Correct WARP_SIZE for AMD architecture
  - Thread indexing → Proper warp_id and lane_id calculations
