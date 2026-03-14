# Kernel: ROCm Flash Multi-Head Attention (FMHA)

## Variant Context
- Input semantic type: Attention (prefill phase multi-head attention)
- Datatype(s): bf16 (bfloat16), fp16, fp8
- Data representation: BSHD (Batch, Sequence, Head, Dimension) layout
- Target architecture: gfx942 (AMD MI300 series)

## Functionality
This kernel implements Flash Multi-Head Attention for the prefill phase on AMD ROCm GPUs. It wraps the AMD AIter (AI Accelerator Iterator) library's optimized attention kernels, providing:
- Memory-efficient attention computation using tiling
- Support for various attention patterns (causal, bidirectional)
- Multi-head and grouped-query attention support
- Optional features like ALiBi positional encoding and softmax capping

## Optimization 1: Migration to New AIter MHA API
- Commit ID: 80eeede7f
- Optimization type: Compute / Architecture-specific
- Summary: Migrated from legacy fmha_fwd to new aiter::mha_fwd API for improved performance
- Detailed explanation:
  The optimization migrates from the older `fmha_fwd` function to the new `aiter::mha_fwd` API. The new API:
  - Provides better kernel selection based on problem characteristics
  - Supports additional optimization paths for specific configurations
  - Offers improved performance through updated Composable Kernel implementations

- Code excerpt:
    ```cpp
    // From rocmFmhaWrapper.cc - Before optimization
    // auto fmha_traits = fmha_fwd_traits{hdim_q, hdim_v, data_type, ...};
    // float run_time = fmha_fwd(fmha_traits, fmha_args, stream_config);
    
    // After optimization
    float run_time;
    run_time = aiter::mha_fwd(
        fmha_args, stream_config, data_type, mode == mode_enum::group, 
        mask.type, bias.type, lse, false);
    ```
- Evidence mapping:
  - API migration → Changed from `fmha_fwd(fmha_traits, fmha_args, ...)` to `aiter::mha_fwd(fmha_args, ...)`
  - Simplified interface → Traits struct replaced with direct parameters

## Optimization 2: Specialized Path for MLA (Multi-head Latent Attention)
- Commit ID: 80eeede7f
- Optimization type: Compute
- Summary: Added optimized path for MLA with bf16 and 128 head dimension
- Detailed explanation:
  For Multi-head Latent Attention (MLA) used in models like DeepSeek, a specialized optimization path is enabled when:
  - Data type is bf16
  - Head dimension is 128
  - Mask type is "b" (bidirectional/batch)
  
  This specialized path uses assembly-optimized kernels for maximum performance.

- Code excerpt:
    ```cpp
    // From rocmFmhaWrapper.cc - MLA optimization
    float run_time;
    if (data_type == "bf16" && size_per_head_ == 128 && msk_str == "b")
        run_time = aiter::mha_fwd(
            fmha_args, stream_config, data_type, mode == mode_enum::group, 
            mask.type, bias.type, lse, true);  // true = use optimized path
    else
        run_time = aiter::mha_fwd(
            fmha_args, stream_config, data_type, mode == mode_enum::group, 
            mask.type, bias.type, lse, false);
    ```
- Evidence mapping:
  - Condition check → `if (data_type == "bf16" && size_per_head_ == 128 && msk_str == "b")`
  - Optimized path flag → Last parameter `true` enables specialized kernel

## Optimization 3: Header Cleanup and Dependency Reduction
- Commit ID: 80eeede7f
- Optimization type: Compile time / Maintenance
- Summary: Removed unused headers and simplified dependencies
- Detailed explanation:
  The optimization removes unused header dependencies:
  - Removed `mask.hpp` - mask handling moved to runtime parameters
  - Removed `bias.hpp` - bias handling simplified
  - Changed from `fmha_fwd.hpp` to `mha_fwd.h` - new API header
  
  This reduces compile time and simplifies the codebase.

- Code excerpt:
    ```cpp
    // Before
    #include "fmha_fwd.hpp"
    #include "ck_tile/host.hpp"
    #include "mask.hpp"
    #include "utils.hpp"
    #include "bias.hpp"
    
    // After
    #include "mha_fwd.h"
    #include "ck_tile/host.hpp"
    #include "utils.hpp"
    ```
- Evidence mapping:
  - Removed headers → `mask.hpp` and `bias.hpp` removed
  - New header → `mha_fwd.h` replaces `fmha_fwd.hpp`

## Optimization 4: Traits-Free Kernel Dispatch
- Commit ID: 80eeede7f
- Optimization type: Compute / Flexibility
- Summary: Removed static traits struct in favor of runtime parameter dispatch
- Detailed explanation:
  The old API required constructing a `fmha_fwd_traits` struct with compile-time characteristics. The new API accepts these as runtime parameters, enabling:
  - More flexible kernel selection at runtime
  - Reduced template instantiation overhead
  - Easier integration with dynamic configurations

- Code excerpt:
    ```cpp
    // Before - Static traits struct
    auto fmha_traits = fmha_fwd_traits{hdim_q,
                                       hdim_v,
                                       data_type,
                                       mode == mode_enum::group,
                                       is_v_rowmajor,
                                       has_logits_soft_cap,
                                       mask.type,
                                       bias.type,
                                       lse,
                                       p_drop > 0.0f,
                                       squant};
    float run_time = fmha_fwd(fmha_traits, fmha_args, stream_config);
    
    // After - Runtime parameters
    run_time = aiter::mha_fwd(
        fmha_args, stream_config, 
        data_type,                    // Runtime parameter
        mode == mode_enum::group,     // Runtime parameter
        mask.type,                    // Runtime parameter
        bias.type,                    // Runtime parameter
        lse,                          // Runtime parameter
        use_optimized_path);          // Runtime parameter
    ```
- Evidence mapping:
  - Traits struct commented out → `// auto fmha_traits = fmha_fwd_traits{...}`
  - Runtime dispatch → Parameters passed directly to `aiter::mha_fwd`
