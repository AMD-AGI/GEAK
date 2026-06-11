# SOTA matrix — operator × backend

AUTO-GENERATED from per-card frontmatter (`index/_gen_registry.py`). Each cell links to the SOTA card.
Legend: 🟢 sota · 🟡 competitive · 🧪 experimental · 🟤 legacy · ⚪ na · `·` no card.

Coverage: **54 operators**, **225 backend cards**.

## GEMM
| operator | triton | flydsl | hip | ck | asm | tilelang | gluon | hipkittens | rocwmma | aiter | hipblaslt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [dense_gemm](../operators/dense_gemm/overview.md) | [🟡](../operators/dense_gemm/backends/triton.md) | [🟢](../operators/dense_gemm/backends/flydsl.md) | · | [🟡](../operators/dense_gemm/backends/ck.md) | [🟢](../operators/dense_gemm/backends/asm.md) | [🟡](../operators/dense_gemm/backends/tilelang.md) | [🟢](../operators/dense_gemm/backends/gluon.md) | [🟢](../operators/dense_gemm/backends/hipkittens.md) | [🟤](../operators/dense_gemm/backends/rocwmma.md) | [🟢](../operators/dense_gemm/backends/aiter.md) | [🟢](../operators/dense_gemm/backends/hipblaslt.md) |
| [batched_gemm](../operators/batched_gemm/overview.md) | [🟡](../operators/batched_gemm/backends/triton.md) | · | [🟤](../operators/batched_gemm/backends/hip.md) | [🟡](../operators/batched_gemm/backends/ck.md) | [🟡](../operators/batched_gemm/backends/asm.md) | · | · | · | · | [🟢](../operators/batched_gemm/backends/aiter.md) | [🟢](../operators/batched_gemm/backends/hipblaslt.md) |
| [grouped_gemm_moe](../operators/grouped_gemm_moe/overview.md) | [🟡](../operators/grouped_gemm_moe/backends/triton.md) | [🟢](../operators/grouped_gemm_moe/backends/flydsl.md) | [🟡](../operators/grouped_gemm_moe/backends/hip.md) | [🟡](../operators/grouped_gemm_moe/backends/ck.md) | · | [🧪](../operators/grouped_gemm_moe/backends/tilelang.md) | · | · | · | [🟢](../operators/grouped_gemm_moe/backends/aiter.md) | · |
| [splitk_streamk_gemm](../operators/splitk_streamk_gemm/overview.md) | [🟢](../operators/splitk_streamk_gemm/backends/triton.md) | [🟡](../operators/splitk_streamk_gemm/backends/flydsl.md) | [🟡](../operators/splitk_streamk_gemm/backends/hip.md) | [🟡](../operators/splitk_streamk_gemm/backends/ck.md) | [🟡](../operators/splitk_streamk_gemm/backends/asm.md) | · | · | · | · | · | [🟡](../operators/splitk_streamk_gemm/backends/hipblaslt.md) |
| [scaled_quant_gemm](../operators/scaled_quant_gemm/overview.md) | [🟢](../operators/scaled_quant_gemm/backends/triton.md) | [🟢](../operators/scaled_quant_gemm/backends/flydsl.md) | [🟡](../operators/scaled_quant_gemm/backends/hip.md) | [🟡](../operators/scaled_quant_gemm/backends/ck.md) | [🟡](../operators/scaled_quant_gemm/backends/asm.md) | · | [🟢](../operators/scaled_quant_gemm/backends/gluon.md) | [🟢](../operators/scaled_quant_gemm/backends/hipkittens.md) | · | [🟢](../operators/scaled_quant_gemm/backends/aiter.md) | [🟡](../operators/scaled_quant_gemm/backends/hipblaslt.md) |
| [gemm_epilogue_fused](../operators/gemm_epilogue_fused/overview.md) | [🟡](../operators/gemm_epilogue_fused/backends/triton.md) | [🟡](../operators/gemm_epilogue_fused/backends/flydsl.md) | [🟤](../operators/gemm_epilogue_fused/backends/hip.md) | [🟢](../operators/gemm_epilogue_fused/backends/ck.md) | · | · | · | · | · | [🟢](../operators/gemm_epilogue_fused/backends/aiter.md) | [🟡](../operators/gemm_epilogue_fused/backends/hipblaslt.md) |
| [skinny_gemv_decode](../operators/skinny_gemv_decode/overview.md) | [🟡](../operators/skinny_gemv_decode/backends/triton.md) | [🟡](../operators/skinny_gemv_decode/backends/flydsl.md) | [🟡](../operators/skinny_gemv_decode/backends/hip.md) | · | [🟡](../operators/skinny_gemv_decode/backends/asm.md) | · | · | · | · | [🟢](../operators/skinny_gemv_decode/backends/aiter.md) | · |

## Attention
| operator | triton | flydsl | hip | ck | asm | tilelang | hipkittens | aiter | fa_rocm | sglang_kernels | vllm_kernels |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [attention_prefill_fmha](../operators/attention_prefill_fmha/overview.md) | [🟡](../operators/attention_prefill_fmha/backends/triton.md) | · | · | [🟢](../operators/attention_prefill_fmha/backends/ck.md) | [🟢](../operators/attention_prefill_fmha/backends/asm.md) | [🟡](../operators/attention_prefill_fmha/backends/tilelang.md) | [🟢](../operators/attention_prefill_fmha/backends/hipkittens.md) | [🟢](../operators/attention_prefill_fmha/backends/aiter.md) | [🟡](../operators/attention_prefill_fmha/backends/fa_rocm.md) | · | · |
| [attention_decode_paged](../operators/attention_decode_paged/overview.md) | [🟡](../operators/attention_decode_paged/backends/triton.md) | · | [🟢](../operators/attention_decode_paged/backends/hip.md) | [🟡](../operators/attention_decode_paged/backends/ck.md) | · | · | · | [🟢](../operators/attention_decode_paged/backends/aiter.md) | [🟡](../operators/attention_decode_paged/backends/fa_rocm.md) | · | [🟢](../operators/attention_decode_paged/backends/vllm_kernels.md) |
| [mla_attention](../operators/mla_attention/overview.md) | [🟡](../operators/mla_attention/backends/triton.md) | · | [🟡](../operators/mla_attention/backends/hip.md) | [🟡](../operators/mla_attention/backends/ck.md) | · | · | · | [🟢](../operators/mla_attention/backends/aiter.md) | [⚪](../operators/mla_attention/backends/fa_rocm.md) | · | · |
| [gqa_mqa_attention](../operators/gqa_mqa_attention/overview.md) | [🟡](../operators/gqa_mqa_attention/backends/triton.md) | · | · | [🟡](../operators/gqa_mqa_attention/backends/ck.md) | · | · | [🟢](../operators/gqa_mqa_attention/backends/hipkittens.md) | [🟢](../operators/gqa_mqa_attention/backends/aiter.md) | [🟡](../operators/gqa_mqa_attention/backends/fa_rocm.md) | · | · |
| [sliding_window_attention](../operators/sliding_window_attention/overview.md) | [🟡](../operators/sliding_window_attention/backends/triton.md) | · | · | [🟢](../operators/sliding_window_attention/backends/ck.md) | · | · | · | [🟡](../operators/sliding_window_attention/backends/aiter.md) | [🟡](../operators/sliding_window_attention/backends/fa_rocm.md) | · | · |
| [chunked_prefill](../operators/chunked_prefill/overview.md) | [🟢](../operators/chunked_prefill/backends/triton.md) | · | · | · | · | · | · | [🟢](../operators/chunked_prefill/backends/aiter.md) | · | [🟢](../operators/chunked_prefill/backends/sglang_kernels.md) | [🟢](../operators/chunked_prefill/backends/vllm_kernels.md) |
| [sparse_attention_nsa](../operators/sparse_attention_nsa/overview.md) | [🟢](../operators/sparse_attention_nsa/backends/triton.md) | · | [🧪](../operators/sparse_attention_nsa/backends/hip.md) | · | · | [🧪](../operators/sparse_attention_nsa/backends/tilelang.md) | · | · | · | · | · |
| [linear_attention_gated_delta](../operators/linear_attention_gated_delta/overview.md) | [🟢](../operators/linear_attention_gated_delta/backends/triton.md) | [🧪](../operators/linear_attention_gated_delta/backends/flydsl.md) | [🧪](../operators/linear_attention_gated_delta/backends/hip.md) | · | · | [🧪](../operators/linear_attention_gated_delta/backends/tilelang.md) | · | · | · | · | · |
| [context_parallel_attention](../operators/context_parallel_attention/overview.md) | [🟡](../operators/context_parallel_attention/backends/triton.md) | · | · | · | · | · | · | [🟡](../operators/context_parallel_attention/backends/aiter.md) | · | [🧪](../operators/context_parallel_attention/backends/sglang_kernels.md) | · |
| [speculative_decode_verify](../operators/speculative_decode_verify/overview.md) | [🟡](../operators/speculative_decode_verify/backends/triton.md) | · | · | · | · | · | · | [🟡](../operators/speculative_decode_verify/backends/aiter.md) | · | [🟢](../operators/speculative_decode_verify/backends/sglang_kernels.md) | [🟢](../operators/speculative_decode_verify/backends/vllm_kernels.md) |

## Norm / Act / Pos
| operator | triton | flydsl | hip | ck | asm | tilelang | hipkittens | aiter | miopen | vllm_kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| [rmsnorm](../operators/rmsnorm/overview.md) | [🟢](../operators/rmsnorm/backends/triton.md) | [🟡](../operators/rmsnorm/backends/flydsl.md) | [🟢](../operators/rmsnorm/backends/hip.md) | · | · | · | [🟢](../operators/rmsnorm/backends/hipkittens.md) | [🟢](../operators/rmsnorm/backends/aiter.md) | · | [🟢](../operators/rmsnorm/backends/vllm_kernels.md) |
| [layernorm](../operators/layernorm/overview.md) | [🟢](../operators/layernorm/backends/triton.md) | [🧪](../operators/layernorm/backends/flydsl.md) | [🟢](../operators/layernorm/backends/hip.md) | · | · | · | · | [🟢](../operators/layernorm/backends/aiter.md) | [🟡](../operators/layernorm/backends/miopen.md) | [🟢](../operators/layernorm/backends/vllm_kernels.md) |
| [softmax](../operators/softmax/overview.md) | [🟢](../operators/softmax/backends/triton.md) | [🧪](../operators/softmax/backends/flydsl.md) | [🟢](../operators/softmax/backends/hip.md) | · | · | · | · | [🟢](../operators/softmax/backends/aiter.md) | · | · |
| [act_and_mul_silu_gelu](../operators/act_and_mul_silu_gelu/overview.md) | [🟢](../operators/act_and_mul_silu_gelu/backends/triton.md) | [🟢](../operators/act_and_mul_silu_gelu/backends/flydsl.md) | [🟢](../operators/act_and_mul_silu_gelu/backends/hip.md) | · | · | · | · | [🟢](../operators/act_and_mul_silu_gelu/backends/aiter.md) | · | [🟢](../operators/act_and_mul_silu_gelu/backends/vllm_kernels.md) |
| [fused_add_rmsnorm](../operators/fused_add_rmsnorm/overview.md) | [🟢](../operators/fused_add_rmsnorm/backends/triton.md) | · | [🟢](../operators/fused_add_rmsnorm/backends/hip.md) | · | · | · | · | [🟢](../operators/fused_add_rmsnorm/backends/aiter.md) | · | [🟢](../operators/fused_add_rmsnorm/backends/vllm_kernels.md) |
| [fused_norm_quant](../operators/fused_norm_quant/overview.md) | [🟢](../operators/fused_norm_quant/backends/triton.md) | · | [🟢](../operators/fused_norm_quant/backends/hip.md) | · | · | · | · | [🟢](../operators/fused_norm_quant/backends/aiter.md) | · | · |
| [rope](../operators/rope/overview.md) | [🟢](../operators/rope/backends/triton.md) | · | [🟢](../operators/rope/backends/hip.md) | · | · | · | [🟡](../operators/rope/backends/hipkittens.md) | [🟢](../operators/rope/backends/aiter.md) | · | [🟢](../operators/rope/backends/vllm_kernels.md) |
| [mrope](../operators/mrope/overview.md) | [🟢](../operators/mrope/backends/triton.md) | · | [🟡](../operators/mrope/backends/hip.md) | · | · | · | · | [🟢](../operators/mrope/backends/aiter.md) | · | · |
| [alibi](../operators/alibi/overview.md) | [🟢](../operators/alibi/backends/triton.md) | · | [🟢](../operators/alibi/backends/hip.md) | · | · | · | · | · | · | · |

## MoE
| operator | triton | flydsl | hip | ck | asm | tilelang | aiter | mori | vllm_kernels |
|---|---|---|---|---|---|---|---|---|---|
| [moe_routing_topk](../operators/moe_routing_topk/overview.md) | [🟡](../operators/moe_routing_topk/backends/triton.md) | · | [🟢](../operators/moe_routing_topk/backends/hip.md) | · | · | · | [🟢](../operators/moe_routing_topk/backends/aiter.md) | · | [🟢](../operators/moe_routing_topk/backends/vllm_kernels.md) |
| [moe_dispatch_combine](../operators/moe_dispatch_combine/overview.md) | [🧪](../operators/moe_dispatch_combine/backends/triton.md) | · | [🟢](../operators/moe_dispatch_combine/backends/hip.md) | · | · | · | [🟢](../operators/moe_dispatch_combine/backends/aiter.md) | [🟢](../operators/moe_dispatch_combine/backends/mori.md) | · |
| [fused_moe_grouped_gemm](../operators/fused_moe_grouped_gemm/overview.md) | [🟡](../operators/fused_moe_grouped_gemm/backends/triton.md) | [🟢](../operators/fused_moe_grouped_gemm/backends/flydsl.md) | [🟢](../operators/fused_moe_grouped_gemm/backends/hip.md) | [🟢](../operators/fused_moe_grouped_gemm/backends/ck.md) | · | · | [🟢](../operators/fused_moe_grouped_gemm/backends/aiter.md) | · | · |
| [shared_expert_fusion](../operators/shared_expert_fusion/overview.md) | [🟡](../operators/shared_expert_fusion/backends/triton.md) | · | [🟢](../operators/shared_expert_fusion/backends/hip.md) | · | · | · | [🟢](../operators/shared_expert_fusion/backends/aiter.md) | · | · |

## Collectives
| operator | triton | flydsl | hip | ck | asm | tilelang | aiter | rccl | vllm_kernels |
|---|---|---|---|---|---|---|---|---|---|
| [allreduce](../operators/allreduce/overview.md) | · | · | [🟢](../operators/allreduce/backends/hip.md) | · | · | · | · | [🟢](../operators/allreduce/backends/rccl.md) | [🟢](../operators/allreduce/backends/vllm_kernels.md) |
| [allgather](../operators/allgather/overview.md) | · | · | [🟡](../operators/allgather/backends/hip.md) | · | · | · | · | [🟢](../operators/allgather/backends/rccl.md) | · |
| [reduce_scatter](../operators/reduce_scatter/overview.md) | · | · | [🟡](../operators/reduce_scatter/backends/hip.md) | · | · | · | · | [🟢](../operators/reduce_scatter/backends/rccl.md) | · |
| [fused_allreduce_rmsnorm](../operators/fused_allreduce_rmsnorm/overview.md) | · | · | [🟢](../operators/fused_allreduce_rmsnorm/backends/hip.md) | · | · | · | [🟢](../operators/fused_allreduce_rmsnorm/backends/aiter.md) | [🟡](../operators/fused_allreduce_rmsnorm/backends/rccl.md) | · |

## Quantization
| operator | triton | flydsl | hip | ck | asm | tilelang | gluon | aiter | vllm_kernels |
|---|---|---|---|---|---|---|---|---|---|
| [quant_dequant_fp8](../operators/quant_dequant_fp8/overview.md) | [🟡](../operators/quant_dequant_fp8/backends/triton.md) | · | [🟢](../operators/quant_dequant_fp8/backends/hip.md) | · | [🟡](../operators/quant_dequant_fp8/backends/asm.md) | · | · | [🟢](../operators/quant_dequant_fp8/backends/aiter.md) | [🟢](../operators/quant_dequant_fp8/backends/vllm_kernels.md) |
| [quant_int8](../operators/quant_int8/overview.md) | [🟡](../operators/quant_int8/backends/triton.md) | · | [🟢](../operators/quant_int8/backends/hip.md) | · | · | · | · | [🟢](../operators/quant_int8/backends/aiter.md) | [🟢](../operators/quant_int8/backends/vllm_kernels.md) |
| [quant_fp4_mxfp](../operators/quant_fp4_mxfp/overview.md) | [🟢](../operators/quant_fp4_mxfp/backends/triton.md) | · | [🟢](../operators/quant_fp4_mxfp/backends/hip.md) | [🟢](../operators/quant_fp4_mxfp/backends/ck.md) | · | · | [🟢](../operators/quant_fp4_mxfp/backends/gluon.md) | [🟢](../operators/quant_fp4_mxfp/backends/aiter.md) | · |
| [kv_cache_quant](../operators/kv_cache_quant/overview.md) | [🟡](../operators/kv_cache_quant/backends/triton.md) | · | [🟢](../operators/kv_cache_quant/backends/hip.md) | · | · | · | · | [🟢](../operators/kv_cache_quant/backends/aiter.md) | [🟢](../operators/kv_cache_quant/backends/vllm_kernels.md) |

## Embedding / Sampling
| operator | triton | flydsl | hip | ck | asm | tilelang | aiter | vllm_kernels |
|---|---|---|---|---|---|---|---|---|
| [embedding](../operators/embedding/overview.md) | [🟡](../operators/embedding/backends/triton.md) | · | [🟡](../operators/embedding/backends/hip.md) | · | · | · | · | [🟢](../operators/embedding/backends/vllm_kernels.md) |
| [lm_head_logits](../operators/lm_head_logits/overview.md) | [🟡](../operators/lm_head_logits/backends/triton.md) | · | [🟡](../operators/lm_head_logits/backends/hip.md) | · | · | · | [🟢](../operators/lm_head_logits/backends/aiter.md) | [🟢](../operators/lm_head_logits/backends/vllm_kernels.md) |
| [sampling_topk_topp](../operators/sampling_topk_topp/overview.md) | [🟡](../operators/sampling_topk_topp/backends/triton.md) | · | [🟢](../operators/sampling_topk_topp/backends/hip.md) | · | · | · | [🟢](../operators/sampling_topk_topp/backends/aiter.md) | [🟢](../operators/sampling_topk_topp/backends/vllm_kernels.md) |

## Elementwise / Reduction
| operator | triton | flydsl | hip | ck | asm | tilelang | pytorch_inductor |
|---|---|---|---|---|---|---|---|
| [elementwise](../operators/elementwise/overview.md) | [🟢](../operators/elementwise/backends/triton.md) | · | [🟢](../operators/elementwise/backends/hip.md) | · | · | · | [🟢](../operators/elementwise/backends/pytorch_inductor.md) |
| [reduction](../operators/reduction/overview.md) | [🟢](../operators/reduction/backends/triton.md) | [🧪](../operators/reduction/backends/flydsl.md) | [🟢](../operators/reduction/backends/hip.md) | [🟢](../operators/reduction/backends/ck.md) | · | · | · |
| [cumsum_scan](../operators/cumsum_scan/overview.md) | [🟢](../operators/cumsum_scan/backends/triton.md) | · | [🟢](../operators/cumsum_scan/backends/hip.md) | · | · | · | · |
| [argmax_topk](../operators/argmax_topk/overview.md) | [🟢](../operators/argmax_topk/backends/triton.md) | · | [🟢](../operators/argmax_topk/backends/hip.md) | · | · | · | · |
| [cast_fill_copy](../operators/cast_fill_copy/overview.md) | [🟢](../operators/cast_fill_copy/backends/triton.md) | · | [🟢](../operators/cast_fill_copy/backends/hip.md) | · | · | · | [🟢](../operators/cast_fill_copy/backends/pytorch_inductor.md) |

## Convolution
| operator | triton | flydsl | hip | ck | asm | tilelang | aiter | miopen |
|---|---|---|---|---|---|---|---|---|
| [causal_conv1d](../operators/causal_conv1d/overview.md) | [🟢](../operators/causal_conv1d/backends/triton.md) | · | [🟢](../operators/causal_conv1d/backends/hip.md) | · | · | · | [🟢](../operators/causal_conv1d/backends/aiter.md) | · |
| [depthwise_conv](../operators/depthwise_conv/overview.md) | [🟡](../operators/depthwise_conv/backends/triton.md) | · | [🟡](../operators/depthwise_conv/backends/hip.md) | · | · | · | · | [🟢](../operators/depthwise_conv/backends/miopen.md) |
| [conv2d](../operators/conv2d/overview.md) | · | · | [🟡](../operators/conv2d/backends/hip.md) | [🟢](../operators/conv2d/backends/ck.md) | · | · | · | [🟢](../operators/conv2d/backends/miopen.md) |

## Data movement
| operator | triton | flydsl | hip | ck | asm | tilelang | aiter | mori | vllm_kernels |
|---|---|---|---|---|---|---|---|---|---|
| [transpose](../operators/transpose/overview.md) | [🟡](../operators/transpose/backends/triton.md) | · | [🟢](../operators/transpose/backends/hip.md) | · | · | · | · | · | · |
| [gather_scatter](../operators/gather_scatter/overview.md) | [🟢](../operators/gather_scatter/backends/triton.md) | · | [🟢](../operators/gather_scatter/backends/hip.md) | · | · | · | [🟢](../operators/gather_scatter/backends/aiter.md) | · | · |
| [all_to_all_dispatch_combine](../operators/all_to_all_dispatch_combine/overview.md) | · | · | [🟡](../operators/all_to_all_dispatch_combine/backends/hip.md) | · | · | · | [🟢](../operators/all_to_all_dispatch_combine/backends/aiter.md) | [🟢](../operators/all_to_all_dispatch_combine/backends/mori.md) | · |
| [paged_kv_copy](../operators/paged_kv_copy/overview.md) | [🟡](../operators/paged_kv_copy/backends/triton.md) | · | [🟢](../operators/paged_kv_copy/backends/hip.md) | · | · | · | [🟢](../operators/paged_kv_copy/backends/aiter.md) | · | [🟢](../operators/paged_kv_copy/backends/vllm_kernels.md) |
| [layout_shuffle](../operators/layout_shuffle/overview.md) | [🟡](../operators/layout_shuffle/backends/triton.md) | [🟡](../operators/layout_shuffle/backends/flydsl.md) | [🟡](../operators/layout_shuffle/backends/hip.md) | · | · | · | [🟢](../operators/layout_shuffle/backends/aiter.md) | · | · |

