# Taxonomy — controlled vocabularies

These ids are authoritative. Use them verbatim in frontmatter and in `sota_registry.yaml`.

## Hardware generations (`gen`) + SKUs (`sku`)
| id | arch | skus (products) |
|---|---|---|
| `gfx906` | (GCN/pre-CDNA, ref) | `mi50` `mi60` |
| `gfx908` | CDNA1 | `mi100` |
| `gfx90a` | CDNA2 | `mi210` `mi250` `mi250x` |
| `gfx942` | CDNA3 | `mi300a` `mi300x` `mi325x` |
| `gfx950` | CDNA4 | `mi350x` `mi355x` |
| `gfx1250` | CDNA5 | `mi450` |

`gfx1250`/CDNA5/MI450 source: `expert_skills/skills/gluon_authoring/references/gluon/atoms-reference.md:50`
(the only in-repo attestation). No verified peaks yet — roofline `peaks.md` carries it as `status: unverified`.

The **`sku`** axis (frontmatter `skus: [...]`) is used ONLY where SKUs of one gen differ materially — e.g.
MI350X vs MI355X clock/TDP feed a different roofline denominator. Leave `skus:` empty for SKU-agnostic facts.
Machine-readable mirror of this table + all vocabularies below: [`_kb_vocab.py`](_kb_vocab.py) (single source of truth).

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

## kernel_class (`kernel_class`) — the ONE unifying kernel-type axis
Two-level dotted ids. This is the single axis that joins the ~50 operator ids, the `learned/` hand-written
groups, and kernel_workflow's free-text `op_kind`. Use exactly these:

**GEMM**: `gemm.dense` `gemm.batched` `gemm.grouped_moe` `gemm.splitk_streamk` `gemm.scaled_quant`
`gemm.epilogue_fused` `gemm.skinny_decode`
**Attention**: `attn.prefill` `attn.decode_paged` `attn.mla` `attn.gqa_mqa` `attn.sparse` `attn.linear`
`attn.spec_decode`
**Other**: `norm_act` · `positional` · `quant` · `moe.routing` `moe.dispatch` · `collective` ·
`data_movement` · `elementwise_reduction` · `conv` · `embedding_sampling` · `method`

### operator → kernel_class (full map; `_backfill_kb.py` auto-fills from this)
| operator | kernel_class | | operator | kernel_class |
|---|---|---|---|---|
| dense_gemm | gemm.dense | | rope / mrope / alibi | positional |
| batched_gemm | gemm.batched | | embedding / lm_head_logits / sampling_topk_topp | embedding_sampling |
| grouped_gemm_moe | gemm.grouped_moe | | elementwise / reduction / cumsum_scan / argmax_topk / cast_fill_copy | elementwise_reduction |
| splitk_streamk_gemm | gemm.splitk_streamk | | causal_conv1d / depthwise_conv / conv2d | conv |
| scaled_quant_gemm | gemm.scaled_quant | | transpose / gather_scatter / all_to_all_dispatch_combine / paged_kv_copy / layout_shuffle | data_movement |
| gemm_epilogue_fused | gemm.epilogue_fused | | quant_dequant_fp8 / quant_int8 / quant_fp4_mxfp / kv_cache_quant | quant |
| skinny_gemv_decode | gemm.skinny_decode | | moe_routing_topk | moe.routing |
| attention_prefill_fmha / chunked_prefill / context_parallel_attention | attn.prefill | | moe_dispatch_combine / shared_expert_fusion | moe.dispatch |
| attention_decode_paged | attn.decode_paged | | fused_moe_grouped_gemm | gemm.grouped_moe |
| mla_attention | attn.mla | | allreduce / allgather / reduce_scatter / fused_allreduce_rmsnorm | collective |
| gqa_mqa_attention | attn.gqa_mqa | | rmsnorm / layernorm / softmax / act_and_mul_silu_gelu / fused_add_rmsnorm / fused_norm_quant | norm_act |
| sliding_window_attention / sparse_attention_nsa | attn.sparse | | linear_attention_gated_delta | attn.linear |
| speculative_decode_verify | attn.spec_decode | | | |

## lever (`levers`) — the optimization-MEANS vocabulary
`tile.static-shape` `tile.autotune-db` `tile.splitk` `tile.streamk` ·
`fusion.epilogue` `fusion.prologue` `fusion.norm-quant` ·
`backend.swap` `config.per-shape-tune` `env.flag` ·
`layout.swizzle` `layout.xcd-l2` · `pipeline.async-lds` `pipeline.software-pipeline` ·
`occupancy.vgpr` `vectorize.coalesce` · `dtype.microscale` `dtype.downcast` ·
`graph.cudagraph-safe` `host.launch-overhead` `host.kernel-merge`

## cost (`cost`) — the "simple → complex" construction-cost ladder
| id | meaning | typical |
|---|---|---|
| `L0` | env var / flag, zero code change, minutes | `env.flag`, `--attention-backend triton` |
| `L1` | config / autotune, no source change, hours | `config.per-shape-tune`, aiter tuning DB |
| `L2` | backend swap / off-the-shelf lib, integration seam | `backend.swap`, Triton→CK |
| `L3` | rewrite kernel / DSL port, days | `tile.static-shape` rewrite, Gluon/FlyDSL port |

`kb_resolve.py` outputs in ASCENDING cost order (try the cheap lever first). Plus `risk`:
`parity-safe` · `numerics-affecting` · `integration-risky`.

## bound_type (`bound_type`) — the roofline routing key
`hbm_bw` · `mfma_compute` · `lds_bank` · `l2_locality` · `launch_overhead` · `occupancy` · `sync` · `host_bound`

## lifecycle (`lifecycle`) — the in/at/out-of-store state machine
`candidate` (written, not yet on-box reproduced) · `active` (measured-verified) ·
`stale` (upstream/stack drift — still surfaced, down-weighted, marked ⚠) ·
`archived` (refuted / superseded — still queryable, never deleted).
Layer tag (`layer`): `reference` (perf_knowledge) · `learned` (curated cards) · `artifact` (kb_artifacts, code-carrying).

## `status` values
`sota` (current best for that cell) · `competitive` (close, situational) · `experimental` ·
`legacy` (superseded) · `na` (not applicable — give reason).

## Sources
- gfx ↔ product mapping & dtypes: AMD CDNA3/CDNA4 ISA guides; CDNA4 whitepaper
  (https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf).
- FNUZ fp8 on CDNA3 vs OCP fp8/MXFP on CDNA4: Matrix Core blog
  (https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html).
- FlashInfer NVIDIA-only: SGLang on MI300X fallback note (research 2026-06).
