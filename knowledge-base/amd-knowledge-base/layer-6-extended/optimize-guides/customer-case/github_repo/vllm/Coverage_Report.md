Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# vLLM Kernel Coverage Report

## Mapping: Path_vllm.txt Kernels → Reports

| Path Section | Kernel Category | Report Created | Notes |
|-------------|-----------------|----------------|-------|
| 1.1 | Paged Attention (CUDA/HIP) | ✅ Report_PagedAttention_fp16_bf16_generic.md, Report_PagedAttention_fp8_generic.md | Core attention kernel |
| 1.2 | Attention Merge States | ⚠️ Covered in PagedAttention report | Part of chunked prefill |
| 1.3 | MLA (Multi-head Latent Attention) | ✅ Report_MLA_sm100_generic.md | DeepSeek models |
| 1.4 | Triton Attention Kernels | ⚠️ Covered in PagedAttention report | Alternative implementations |
| 1.5 | ROCm Attention | ⚠️ Covered in PagedAttention report | AMD-specific |
| 2.1 | Cache Management | ✅ Report_CacheKernels_fp16_bf16_fp8_generic.md | KV cache operations |
| 3.1 | FP8 Quantization (W8A8) | ⚠️ Covered in ScaledMM report | Quantization utilities |
| 3.2 | FP8 Scaled MM (CUTLASS) | ✅ Report_ScaledMM_fp8_sm90.md | CUTLASS GEMM |
| 3.3 | INT8 Quantization (W8A8) | ⚠️ Covered in ScaledMM report | Similar to FP8 |
| 3.4 | FP4/NVFP4 Quantization | ❌ Not covered | Blackwell-specific, newer |
| 3.5 | Marlin Quantization | ✅ Report_Marlin_int4_fp8_generic.md | INT4/FP8 GEMM |
| 3.6 | GPTQ Quantization | ⚠️ Covered in Marlin report | Uses Marlin backend |
| 3.7 | AWQ Quantization | ⚠️ Covered in Marlin report | Uses Marlin backend |
| 3.8 | GGUF Quantization | ❌ Not covered | llama.cpp compatibility |
| 3.9 | Machete Quantization | ❌ Not covered | Mixed-precision GEMM |
| 3.10 | W4A8 Quantization | ❌ Not covered | 4-bit weight, 8-bit activation |
| 3.11 | Fused Quantization Kernels | ⚠️ Covered in LayerNorm report | LayerNorm + quant fusion |
| 3.12 | Triton Quantization | ⚠️ Covered in ScaledMM report | Triton GEMM |
| 4.1 | TopK Softmax | ✅ Report_MoE_Routing_generic.md | Expert selection |
| 4.2 | MoE Align and Sum | ✅ Report_MoE_Routing_generic.md | Token alignment |
| 4.3 | MoE Permute/Unpermute | ✅ Report_MoE_Routing_generic.md | Token reordering |
| 4.4 | Fused MoE (Triton) | ✅ Report_FusedMoE_fp16_bf16_generic.md, Report_FusedMoE_fp8_generic.md | Core MoE kernel |
| 4.5 | MoE WNA16 | ⚠️ Covered in FusedMoE report | Weight-only quantized |
| 4.6 | Marlin MoE | ⚠️ Covered in Marlin report | Quantized MoE |
| 4.7 | MoE Grouped GEMM (CUTLASS) | ⚠️ Covered in FusedMoE report | CUTLASS backend |
| 5.1 | Activation Functions | ✅ Report_LayerNorm_Activation_generic.md | SiLU, GELU, etc. |
| 6.1 | LayerNorm/RMSNorm | ✅ Report_LayerNorm_Activation_generic.md | Normalization |
| 6.2 | Fused QKNorm + RoPE | ✅ Report_LayerNorm_Activation_generic.md | Fused operations |
| 7.1 | Rotary Position Embedding | ✅ Report_RoPE_generic.md | Position encoding |
| 8.1 | Custom All-Reduce | ✅ Report_CustomAllReduce_generic.md | Tensor parallelism |
| 9.1 | Sampler | ✅ Report_Sampler_generic.md | Token sampling |
| 10.1 | Selective Scan (Mamba) | ✅ Report_Mamba_SSM_generic.md | SSM models |
| 11.1 | Sparse GEMM | ✅ Report_SparseGEMM_fp8_int8_sm90.md | 2:4 sparsity |
| 11.2 | Vertical Slash Attention | ⚠️ Covered in Sparse GEMM report | Sparse attention |
| 12.1 | Skinny GEMM (ROCm) | ✅ Report_SkinnyGEMM_fp16_bf16_gfx9.md | AMD optimization |
| 13.1 | LoRA Operations | ✅ Report_LoRA_Triton_generic.md | Adapter operations |
| 14.1 | Hadamard Kernels | ❌ Not covered | Quantization transform |
| 15.1 | CUTLASS Custom Types | ⚠️ Infrastructure, not kernel | Type utilities |
| 15.2 | CUTLASS Epilogue Extensions | ⚠️ Covered in ScaledMM report | Epilogue fusion |

## Summary

- **Total kernel categories in Path_vllm.txt**: 39
- **Reports created**: 17
- **Direct coverage (✅)**: 17 categories
- **Indirect coverage (⚠️)**: 17 categories (covered within other reports)
- **Not covered (❌)**: 5 categories

### Not Covered Categories (Lower Priority)
1. **FP4/NVFP4 Quantization** - Very new (Blackwell), limited commit history
2. **GGUF Quantization** - Compatibility layer for llama.cpp models
3. **Machete Quantization** - Specialized mixed-precision kernel
4. **W4A8 Quantization** - Niche quantization format
5. **Hadamard Kernels** - Auxiliary transform for quantization

### Coverage Notes
- Most kernel categories are covered either directly or as part of related reports
- Categories marked ⚠️ share optimization patterns with their parent reports
- Infrastructure code (CUTLASS extensions) provides utilities rather than standalone kernels
