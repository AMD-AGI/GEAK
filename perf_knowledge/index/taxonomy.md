# Taxonomy — controlled vocabularies

These ids are authoritative. Use them verbatim in frontmatter and in `sota_registry.yaml`.

## Hardware generations (`gen`)
| id | arch | products |
|---|---|---|
| `gfx906` | (GCN/pre-CDNA, ref) | MI50/60 |
| `gfx908` | CDNA1 | MI100 |
| `gfx90a` | CDNA2 | MI210, MI250, MI250X |
| `gfx942` | CDNA3 | MI300A, MI300X, MI325X |
| `gfx950` | CDNA4 | MI350X, MI355X |
| `gfx1151` | RDNA 3.5 (APU) | Ryzen AI MAX+ 395 / Radeon 8060S ("Strix Halo") |

> `gfx1151` is **not CDNA**: wave32, 2 SIMDs/CU, WMMA instead of MFMA, no AGPRs, LPDDR5X UMA with a
> 32 MB last-level cache, and none of the CDNA4 block-scaled FP4/FP6 stack. Same family: `gfx1150`,
> `gfx1152`.
>
> A `gen=gfx1151` query is **no longer empty**, but read what it means. `gfx1151` is listed only on
> backends whose **source is portable** to RDNA (`triton`, `hip`, `vllm_kernels`, `pytorch_inductor`,
> `rocwmma`, plus `aiter` on `dense_gemm` where we measured it live) — it says *this can be built and
> run here*, never *this was measured here*. The vendor-asset backends (`ck`, `hipblaslt`, `asm`,
> `flydsl`, `hipkittens`, `gluon`, `tilelang`, `mori`, `rccl`, …) are deliberately **not** listed for
> gfx1151: on the Strix box `gradlib`, `ckProfiler` and `hipblaslt-bench` are all measured absent and
> aiter ships 1 gfx1151 asset file against 54 for gfx942. Two operator groups are also withheld —
> anything fp8/FP4-scaled (no fp8 matrix instruction on RDNA, so the candidate runs emulated) and the
> multi-GPU collectives (single-device APU). Per-card detail lives in each card's
> "On gfx1151" section; the hardware baseline is `hardware/rdna35_gfx1151/`.

## dtypes (`dtype`)
`fp32` · `tf32`(N/A on CDNA4, removed) · `bf16` · `fp16` ·
`fp8_e4m3_fnuz` · `fp8_e5m2_fnuz` (CDNA3 FNUZ) · `fp8_e4m3` · `fp8_e5m2` (OCP, CDNA4) ·
`fp6_e2m3` · `fp6_e3m2` · `fp4_e2m1` · `mxfp8` · `mxfp6` · `mxfp4` (block-scaled, E8M0 scale) ·
`int8` · `int4`

## Regimes (`regime`)
`prefill` (large-M GEMM, long-seq attn) · `decode` (skinny-M/batch GEMM, paged attn) ·
`training` (fwd+bwd) · `both`

## Operators (`operator`) — the Cartesian rows (~50)
**GEMM family**: `dense_gemm` · `batched_gemm` · `grouped_gemm_moe` · `splitk_streamk_gemm` ·
`scaled_quant_gemm` · `gemm_epilogue_fused` · `skinny_gemv_decode`
**Attention**: `attention_prefill_fmha` · `attention_decode_paged` · `mla_attention` ·
`gqa_mqa_attention` · `sliding_window_attention` · `sparse_attention_nsa` ·
`linear_attention_gated_delta` · `chunked_prefill` · `context_parallel_attention` ·
`speculative_decode_verify`
**Norm/Act**: `rmsnorm` · `layernorm` · `softmax` · `act_and_mul_silu_gelu` · `fused_add_rmsnorm` ·
`fused_norm_quant`
**Positional**: `rope` · `mrope` · `alibi`
**Embedding/sampling**: `embedding` · `lm_head_logits` · `sampling_topk_topp`
**Elementwise/reduction/scan**: `elementwise` · `reduction` · `cumsum_scan` · `argmax_topk` ·
`cast_fill_copy`
**Conv**: `causal_conv1d` · `depthwise_conv` · `conv2d`
**Data movement**: `transpose` · `gather_scatter` · `all_to_all_dispatch_combine` · `paged_kv_copy` ·
`layout_shuffle`
**Quant ops**: `quant_dequant_fp8` · `quant_int8` · `quant_fp4_mxfp` · `kv_cache_quant`
**MoE**: `moe_routing_topk` · `moe_dispatch_combine` · `fused_moe_grouped_gemm` ·
`shared_expert_fusion`
**Collectives**: `allreduce` · `allgather` · `reduce_scatter` · `fused_allreduce_rmsnorm`

## Backends (`backend`) — the Cartesian columns
**Core authoring languages (priority; every op gets a card or `na`)**:
`triton` · `flydsl` · `hip` · `ck` (ck_tile + classic) · `asm` (mfma/raw asm) · `tilelang`
**Other authoring languages**: `gluon` (Triton's low-level dialect; CDNA4 scaled-MFMA/MXFP4) · `rocwmma` · `hipkittens` · `mojo` · `cutlass_port`
**Library / auto backends (select-an-impl)**:
`aiter` · `hipblaslt` · `rocblas` · `ck_lib` · `miopen` · `pytorch_inductor` · `mori` · `rccl` ·
`fa_rocm` (FlashAttention-ROCm)
**Explicitly N/A on AMD** (record as `na` with reason): `flashinfer` (NVIDIA-only), `cutlass` (native),
`cudnn`, `transformer_engine`.

## `status` values
`sota` (current best for that cell) · `competitive` (close, situational) · `experimental` ·
`legacy` (superseded) · `na` (not applicable — give reason).

## Sources
- gfx ↔ product mapping & dtypes: AMD CDNA3/CDNA4 ISA guides; CDNA4 whitepaper
  (https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf).
- FNUZ fp8 on CDNA3 vs OCP fp8/MXFP on CDNA4: Matrix Core blog
  (https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html).
- FlashInfer NVIDIA-only: SGLang on MI300X fallback note (research 2026-06).
