# Attention decode: native numerics facts

Paged decode attention, GQA / MQA attention and sliding-window attention with sinks, as the three vLLM ROCm
attention backends compute them: `ROCM_ATTN`, `ROCM_AITER_UNIFIED_ATTN` and `TRITON_ATTN_DIFFKV`. Not here: MLA
and sparse attention (`mla_and_sparse_attention.md`), the RoPE and cache-write kernels (`rope_and_kv_write.md`),
SGLang (`sglang_backends.md`), how to derive a tolerance or run a flip test (`parity_method.md`).

**Versions.** Line numbers are valid at vLLM commit `f09d5e30c`, aiter tag `v0.1.26` (rows marked
`@285579b6` were read at that aiter commit) and, for SGLang, commit `22e1cb6d98`; in any other version
locate the code by symbol.

**Status.** `capture` = verified on a model capture (a captured decode step of a served model was reproduced
with this fact; "2 models" when two served models agree). `probe` = verified by a real-op probe (the installed
native op was run on synthetic inputs and compared with an emulation of this fact, usually bit for bit).
`source only` = read from source, never run against the native op: treat it as a suspect and confirm it with
the flip test in `parity_method.md` before relying on it.

**Model-package tags.** `<PKG_POSTNORM_MOE>` = the package under `vllm/models/` of a post-norm MoE model with
fused norm / routing kernels (the one containing `common/ops/fused_rmsnorm.py`); resolve a tag by listing
`vllm/models/` in the checkout.

## Which backend is running

| Backend / provider | Selected when | Where to read it |
|---|---|---|
| `ROCM_ATTN` (`RocmAttentionImpl`) | vLLM's backend selection, or `--attention-backend` | server log: `Using ROCM_ATTN backend out of potential backends` |
| - its HIP custom paged-attention kernel | all of: CDNA arch, no sliding window, bf16 / fp16 query, head size 64 or 128, block size 16 or 32, GQA ratio 1 to 16, max sequence length up to 128K, no sinks, power-of-two block size, packed cache layout (ATT-03) | `use_rocm_custom_paged_attention` in `vllm/platforms/rocm.py`; this kernel's arithmetic was not read |
| - its Triton fallback `kernel_paged_attention_2d` | any of those conditions false (a window, sinks, block size 64, a padded layout ...) | `chunked_prefill_paged_decode` in `vllm/v1/attention/ops/chunked_prefill_paged_decode.py` |
| `ROCM_AITER_UNIFIED_ATTN` (`RocmAiterUnifiedAttentionImpl`) | vLLM's backend selection, or `--attention-backend` | server log: `Using ROCM_AITER_UNIFIED_ATTN backend out of potential backends` and `Using aiter unified attention for RocmAiterUnifiedAttentionImpl` |
| - its single-pass 2D kernel / split-K 3D kernel | chosen per call on gfx942 (ATT-04); gfx950 and gfx1250 dispatch other (Gluon) kernels that were not read | `use_2d_kernel` in aiter `aiter/ops/triton/attention/unified_attention.py` |
| `TRITON_ATTN_DIFFKV` (`TritonAttentionDiffKVImpl`; K and V head sizes may differ; the backend MiMo-V2.6-Flash runs on) | vLLM's backend selection, or `--attention-backend` | the attention-backend line of the server log, or the class of `layer.impl` |
| - its split-K 3D kernel / single-pass 2D kernel | chosen per call from the batch (ATT-01, ATT-02) | `unified_attention_diffkv` in `vllm/v1/attention/ops/triton_unified_attention_diffkv.py` |

Read these deployment facts before applying any row below:

| Deployment fact | Where to read it | Rows it selects |
|---|---|---|
| hardware generation | device arch (`rocminfo`, or `torch.cuda.get_device_properties().gcnArchName`) | fp8 flavour: e4m3fnuz on gfx942, OCP e4m3 on gfx950 (ATT-10) |
| KV cache dtype | `cache_config.cache_dtype`: `auto`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | ATT-10, ATT-11, ATT-14, ATT-35 |
| fp8 query | `layer.query_quant is not None` on the live layer (a model may set it to None after init) | ATT-11, ATT-18, ATT-20, ATT-21 |
| decode width | rows per step as the kernel sees them: under a full CUDA graph the padded capture size, not the live request count | ATT-01 |
| CUDA-graph mode and capture sizes | `compilation_config.cudagraph_mode`, `compilation_config.cudagraph_capture_sizes` (in the configured order) | ATT-02 |
| batch-invariant mode | `VLLM_BATCH_INVARIANT` in the serving process environment | ATT-01 |
| block size | `cache_config.block_size` | ATT-03, ATT-21, ATT-22 |
| sliding window | `layer.impl.sliding_window`, stored as `(w - 1, 0)` for a window of w keys | ATT-26 to ATT-28 |
| sinks and their dtype | `layer.impl.sinks` (None, or one logit per query head); dtype read off that tensor | ATT-29 to ATT-32 |
| scales | `layer.impl.scale`, `layer._q_scale`, `layer._k_scale`, `layer._v_scale` | ATT-13 to ATT-16 |
| cache and block-table strides | `kv_cache.stride()` of the tensor `forward()` receives; `attn_metadata.block_table.stride(0)` | ATT-06 to ATT-09 |
| who writes the cache | `VLLM_ROCM_USE_AITER`; `compilation_config.pass_config.fuse_rope_kvcache`, `fuse_qk_norm_rope_kvcache`, `rope_kvcache_fusion_max_token_num` | ATT-33, ATT-34 |
| gate compile scope | engine compilation mode, the model class decorator, `torch._inductor.config.emulate_precision_casts` | ATT-39, ATT-40, ATT-42 |

## Kernel path inside the backend

| id | Fact | Applies to | Status | Effect if wrong | Source |
|---|---|---|---|---|---|
| ATT-01 | The split-K ("3D") kernel runs iff the batch is decode-only (one query token per request), the number of rows is at most `seq_threshold_3D` (ATT-02), the per-segment scratch buffers exist and `VLLM_BATCH_INVARIANT` is off; otherwise the single-pass ("2D") kernel runs. The rule has no window term. (Checked on width-1 decode; the 2D side, wide batches, was not run.) | vLLM · `TRITON_ATTN_DIFFKV` · gfx942 | capture (1 model) | the wrong path means the wrong tile and segment count: see ATT-20, ATT-23 | `vllm/v1/attention/ops/triton_unified_attention_diffkv.py:465-480` (`unified_attention_diffkv`), `:47` |
| ATT-02 | `seq_threshold_3D` = 128 // (kv heads on this rank); when decode CUDA graphs are on (mode FULL, FULL_DECODE_ONLY or FULL_AND_PIECEWISE) it is replaced by the capture size nearest to that value (a tie goes to the first size in the configured order). | vLLM · `TRITON_ATTN_DIFFKV` (inherited from the Triton backend's metadata builder) | source only | as ATT-01 | `vllm/v1/attention/backends/triton_attn.py:58-59`, `:121-149` (`TritonAttentionMetadataBuilder.__init__`) |
| ATT-03 | Decode runs the HIP custom paged-attention kernel iff the platform check passes (CDNA arch, no sliding window, bf16 / fp16 query, head size 64 or 128, block size 16 or 32, GQA ratio 1 to 16, max sequence length up to 128K, no sinks) and the block size is a power of two and the cache layout is packed (the K page stride equals the page size); otherwise the Triton `kernel_paged_attention_2d`. Every `ROCM_ATTN` row below describes the Triton fallback only. | vLLM · `ROCM_ATTN` | source only | not quantified (two different kernels) | `vllm/platforms/rocm.py:458-480` (`use_rocm_custom_paged_attention`); `vllm/v1/attention/ops/chunked_prefill_paged_decode.py:24-37`, `:387-410` |
| ATT-04 | On gfx942 a decode batch takes the single-pass 2D kernel iff the layer has a sliding window, or the longest context is at most 512 keys, or the 2D program count exceeds 4 times the CU count; otherwise the split-K 3D kernel, whose segment count is chosen per call from the longest context. Without a window, tile and segment count are therefore not constants of the deployment. | vLLM · `ROCM_AITER_UNIFIED_ATTN` · gfx942 | source only | not quantified | aiter: `aiter/ops/triton/attention/unified_attention.py:174-201` (`use_2d_kernel`), `:379-408`; aiter: `aiter/ops/triton/utils/unified_attention_utils.py:162-208` (`compute_segment_params`) |
| ATT-05 | This vLLM commit imports `aiter.ops.triton.unified_attention`; aiter 0.1.26 ships the function as `aiter.ops.triton.attention.unified_attention` and the old module path does not import. Every row for this backend was read from the 0.1.26 module and holds only if the deployment's aiter matches it; the backend itself was never run. | vLLM · `ROCM_AITER_UNIFIED_ATTN` | source only | not quantified | `vllm/v1/attention/backends/rocm_aiter_unified_attn.py:184-191`; aiter: `aiter/ops/triton/attention/unified_attention.py:204-231` (`unified_attention`) |

## KV cache layout and dtype

| id | Fact | Applies to | Status | Effect if wrong | Source |
|---|---|---|---|---|---|
| ATT-06 | Packed layout: the cache tensor is logical [blocks, kv_heads, block_size, qk + v]. K is the first `head_size_qk` elements and V the next `head_size_v` elements of the same row: one tensor, V offset = `head_size_qk` elements. | vLLM · `TRITON_ATTN_DIFFKV`; `ROCM_AITER_UNIFIED_ATTN` (same split, after a transpose) | capture (1 model) on `TRITON_ATTN_DIFFKV` (MiMo-V2.6-Flash: the captured pages read back in this layout and a reference built on it matched the captured attention output); source only on `ROCM_AITER_UNIFIED_ATTN` | not quantified (a wrong layout reads unrelated values) | `vllm/v1/attention/backends/triton_attn_diffkv.py:165-193` (`TritonAttentionDiffKVImpl.forward`); `vllm/v1/attention/backends/rocm_aiter_unified_attn.py:205-212` (`_split_kv_cache`), `:117-136` |
| ATT-07 | A page can be larger than kv_heads * block_size * (qk + v) elements (a padded page size) and the physical dimension order is the allocator's: read the page, head and position strides off `kv_cache.stride()`, never derive them from the shape. | vLLM · `TRITON_ATTN_DIFFKV` (any backend with a padded cache spec) | source only (no padded page was captured) | not quantified | `vllm/v1/kv_cache_interface.py:586-600` |
| ATT-08 | K-half / V-half layout: the cache has two head slots, logical [num_blocks, 2, block_size, num_kv_heads * head_size], transposed into a K half and a V half. `kv_cache.stride(0)` is the page stride; `kv_cache.stride(1)` is the K-to-V distance: `num_blocks` pages under the LHBNC layout (all K pages, then all V pages), half a page under LBHNC. (Shape and strides confirmed on the live cache tensor of a Qwen3 deployment; no attention output of this backend was compared.) | vLLM · `ROCM_ATTN` | capture (1 model) | not quantified | `vllm/v1/attention/backends/rocm_attn.py:225-248`, `:420-424`; `vllm/v1/kv_cache_layout.py:11-28`; `vllm/v1/kv_cache_interface.py:563-590` |
| ATT-09 | Inside a page the K half is viewed as [kv_heads, head_size / x, block_size, x] with x = 16 / element size in bytes (8 for bf16, 16 for fp8) and the V half as [kv_heads, head_size, block_size]. Element offsets inside the half: K dim d of position p of kv head g at `g*head_size*block_size + (d // x)*block_size*x + p*x + d % x`; V dim d at `g*head_size*block_size + d*block_size + p`. | vLLM · `ROCM_ATTN` | source only | not quantified | `vllm/v1/attention/ops/paged_attn.py:17-29` (`PagedAttention.split_kv_cache`); `vllm/v1/attention/ops/chunked_prefill_paged_decode.py:165-191`; `csrc/cache_kernels.cu:283-317` |
| ATT-10 | An `fp8` / `fp8_e4m3` cache is viewed as the platform fp8 dtype before the attention: e4m3fnuz on gfx942, OCP e4m3 on gfx950 (`fp8_e5m2` as float8_e5m2 on `ROCM_ATTN`); any other setting keeps the cache in the model dtype. | vLLM · all three backends | capture (1 model) on `TRITON_ATTN_DIFFKV` (a gfx942 capture decoded as e4m3fnuz); source only on the other two backends and for gfx950 | not quantified | `vllm/v1/attention/backends/triton_attn_diffkv.py:189-191`; `vllm/v1/attention/backends/rocm_attn.py:426-428`; `vllm/v1/attention/backends/rocm_aiter_unified_attn.py:290-293` |

## Query quantisation and logit scale

| id | Fact | Applies to | Status | Effect if wrong | Source |
|---|---|---|---|---|---|
| ATT-11 | With an fp8 KV cache the attention layer statically quantises the query to the platform fp8 before the kernel (operation: ATT-13) and passes `q_descale`. Not when the backend declines (on ROCm: gfx90a and gfx11 decline) or when the model sets `layer.query_quant` to None after init. | vLLM · `TRITON_ATTN_DIFFKV`; `ROCM_AITER_UNIFIED_ATTN` (declines on gfx11 only) · fp8 KV cache | capture (1 model) on `TRITON_ATTN_DIFFKV` (a reference had to round its query to fp8 to match the captured steps; flipping it moved the error by percents); source only on `ROCM_AITER_UNIFIED_ATTN` | percents of the output when flipped; shows on short contexts and a freshly filled cache, averaged away on long contexts | `vllm/model_executor/layers/attention/attention.py:470-488`, `:511-526` (`Attention.__init__` / `forward`); `vllm/v1/attention/backends/triton_attn.py:553-558`; `vllm/v1/attention/backends/rocm_aiter_unified_attn.py:192-193` |
| ATT-12 | This backend keeps `supports_quant_query_input = False`: the query is never quantised, reaches the kernel in the model dtype, and the logit scale is the plain softmax scale. | vLLM · `ROCM_ATTN` | source only | as ATT-11 when an fp8 query is assumed | `vllm/v1/attention/backend.py:842`; `vllm/v1/attention/backends/rocm_attn.py:444-468` |
| ATT-13 | The static per-tensor query quant is `q.to(fp32) * (fp32 reciprocal of q_scale)`, clamped to the fp8 range, then converted (`QuantFP8.forward_native`); with aiter on ROCm, `QuantFP8.forward_hip` runs aiter's static per-tensor quant, whose kernel multiplies by the hardware reciprocal of the scale (`__builtin_amdgcn_rcpf`). At `q_scale` 1 every path is the plain convert; at another scale the two reciprocals can differ by one ulp (never checked: the captured model has `q_scale` 1). | vLLM · `TRITON_ATTN_DIFFKV`, `ROCM_AITER_UNIFIED_ATTN` · fp8 query | source only | not quantified | `vllm/model_executor/layers/quantization/input_quant_fp8.py:136-160`, `:215-221`; `vllm/_aiter_ops.py:1385-1396`; aiter: `csrc/kernels/quant_kernels.cu:761-768` |
| ATT-14 | Logit scale = softmax scale * `q_descale` (fp8 query only) * `k_descale` (fp8 cache only), multiplied in fp32. The accumulator is multiplied by `v_descale` in the epilogue (on the split-K path: per segment, before the merge). The captured model has unit scales; a non-unit scale is unverified. | vLLM · `TRITON_ATTN_DIFFKV` · fp8 cache | source only | not quantified | `vllm/v1/attention/ops/triton_unified_attention_diffkv.py:169-173`, `:283-284` (`kernel_unified_attention_diffkv`); `vllm/v1/attention/backends/triton_attn_diffkv.py:203-207` |
| ATT-15 | Base-2 softmax: scale = softmax scale * log2(e) * `q_descale` * `k_descale` in fp32 and `P = exp2(S - m)`: the same function as `exp(s - m)` up to about one fp32 ulp. The output is `acc * (v_descale / L)`: a multiply by a reciprocal, not a division. | vLLM · `ROCM_AITER_UNIFIED_ATTN` · gfx942 2D kernel | source only | inside a tolerance: the numerator-rounding term (ATT-18) already covers one fp32 ulp | aiter: `aiter/ops/triton/_triton_kernels/attention/unified_attention.py:149-151`, `:291-302`, `:530-536` (`kernel_unified_attention_2d`); `vllm/v1/attention/backends/rocm_aiter_unified_attn.py:314-322` |
| ATT-16 | An fp8 cache is dequantised up front: `K = (K_fp8.to(fp32) * k_scale).to(query dtype)`, the same for V; bf16 K / V are used as loaded; `S = scale * dot(Q, K)` accumulates in fp32. The cache scale is not folded into the logit scale, and K / V are rounded to the query dtype after scaling (identical to a plain fp8 read only at unit scales). | vLLM · `ROCM_ATTN` Triton fallback | source only | not quantified | `vllm/v1/attention/ops/chunked_prefill_paged_decode.py:228-243` (`kernel_paged_attention_2d`) |
| ATT-17 | K and V tiles are cast to the query's dtype right after the load, so both dots run in the query's dtype: fp8 K / V stay fp8 under an fp8 query and become bf16 (exactly) under a bf16 query. | vLLM · `TRITON_ATTN_DIFFKV`, `ROCM_AITER_UNIFIED_ATTN` | source only | not quantified | `vllm/v1/attention/ops/triton_unified_attention_diffkv.py:228-240`; aiter: `aiter/ops/triton/_triton_kernels/attention/unified_attention.py:432`, `:453` |

## Probability rounding, tile size and segments

An exact-softmax kernel is more accurate than these backends and still misses a tolerance calibrated on them.
The rows below are the reason; they matter most where no averaging hides them (a short context, a freshly
filled cache), so a check on long contexts only does not cover them.

| id | Fact | Applies to | Status | Effect if wrong | Source |
|---|---|---|---|---|---|
| ATT-18 | The value dot uses the tile's probabilities rounded to the query's dtype, `dot(P.to(V.dtype), V)` (fp8 under an fp8 query, bf16 otherwise), while the running denominator L sums the unrounded fp32 P. `P = exp(S - m)` with m the running max including the current tile, so the tile length decides where each probability lands on the format's grid. | vLLM · `TRITON_ATTN_DIFFKV` | capture (1 model) with fp8 P; bf16 P was not checked on a model | numerator and denominator disagree by up to one ulp of the P format (about 6% for e4m3) per term | `vllm/v1/attention/ops/triton_unified_attention_diffkv.py:270-280`; `vllm/v1/attention/ops/triton_attention_helpers.py:461-484` (`softmax_step`) |
| ATT-19 | Same numerator rounding in the other two backends: `dot(P.to(V.dtype), V)` with L summing fp32 P. On the `ROCM_ATTN` fallback the query stays in the model dtype (ATT-12) and the output is `acc / (L + 1e-10)`; L is at least 1 there, so the 1e-10 is below an fp32 ulp. | vLLM · `ROCM_AITER_UNIFIED_ATTN` 2D kernel; `ROCM_ATTN` Triton fallback | source only | as ATT-18 | aiter: `aiter/ops/triton/_triton_kernels/attention/unified_attention.py:505-528`; `vllm/v1/attention/ops/chunked_prefill_paged_decode.py:254-283` |
| ATT-20 | Tile = 32 keys on the single-pass (2D) path; on the split-K (3D) path 16 for a 2-byte query (bf16 / fp16) and 32 for an fp8 query. | vLLM · `TRITON_ATTN_DIFFKV` | capture (1 model) for the fp8-query split-K tile of 32; the bf16-query tile of 16 was not checked | a wrong tile changes the running max each key is normalised on: about one ulp of the P format on most outputs once the context exceeds one tile | `vllm/v1/attention/ops/triton_unified_attention_diffkv.py:496-500` (`unified_attention_diffkv`) |
| ATT-21 | Tile = max(MIN, min(MAX, next power of two of the block size)) with MIN / MAX from the arch's tuned table: for decode on gfx942 MAX is 64 for every head size, MIN is 32 with an fp8 query and fp8 cache and 1 otherwise. Net: 16 / 32 / 64 for a block size up to 16 / up to 32 / larger, raised to at least 32 by an fp8 query. | vLLM · `ROCM_AITER_UNIFIED_ATTN` · gfx942 2D kernel | source only | as ATT-20 | aiter: `aiter/ops/triton/utils/unified_attention_utils.py:151-159` (`compute_tile_params`); aiter: `aiter/ops/triton/configs/gfx942/triton/attention/unified_attention/DEFAULT.json`; aiter: `aiter/ops/triton/attention/unified_attention.py:562-575` |
| ATT-22 | Tile = min(block size, 128) for a power-of-two block size, else 32; keys are addressed through `block_table[pos // block_size]`. | vLLM · `ROCM_ATTN` Triton fallback | source only | as ATT-20 | `vllm/v1/attention/ops/chunked_prefill_paged_decode.py:456-464`, `:148-163` |
| ATT-23 | The split-K path runs 16 segments (64 only on one non-ROCm device class); the single-pass path is one segment. A segment covers `ceil(seq_len / (16 * tile))` tiles of the absolute-position grid and has its own running max, denominator (starting at 1.0) and accumulator; a segment that starts at or past `seq_len` does not run. | vLLM · `TRITON_ATTN_DIFFKV` | capture (1 model): on width-1 decode an unsplit reference was bit-exact only on steps whose context fits one segment (at those lengths one 32-key tile); a 16-segment emulation was bit-exact on every one of 50 fresh steps of a full-attention layer | past one segment the split changes most output elements by about one ulp of the P format; a segment boundary one tile off stays inside a stage tolerance but shows at a layer-level check on long contexts | `vllm/v1/attention/backends/triton_attn.py:151-162`, `:181-182`; `vllm/v1/attention/ops/triton_unified_attention_diffkv.py:136-139` |
| ATT-24 | Segment merge: overall max = the largest segment running max (absent segments count as -inf); every segment's denominator and accumulator is multiplied by `exp(m_seg - overall max)`; denominators and accumulators are summed in segment order; one division at the end (0 when the summed denominator is 0). | vLLM · `TRITON_ATTN_DIFFKV` split-K | source only (exercised indirectly by the bit-exact 16-segment emulation of ATT-23) | a merge without the rescale, or a dropped segment, is a gross error; a different fp32 summation order stays inside the accumulation term of a tolerance | `vllm/v1/attention/ops/triton_unified_attention_diffkv.py:331-407` (`kernel_reduce_segments_diffkv`) |
| ATT-25 | One program per (sequence, kv head) walks the whole context with one running max: no split-K, one segment. | vLLM · `ROCM_ATTN` Triton fallback | source only | as ATT-23 if a split is assumed | `vllm/v1/attention/ops/chunked_prefill_paged_decode.py:91-153`, `:480-485` |

A candidate kernel may spread a kv head's segments over parallel tasks without changing any row above, as long as
each task holds whole backend segments on absolute positions, walks them with the same per-segment running maxima,
and the owner merges the tasks' (max, denominator, accumulator) exactly as ATT-24 does. Only the fp32 summation
order then differs from the backend. Changing the tile or the segment boundaries is a different computation.

## Sliding window and sinks

| id | Fact | Applies to | Status | Effect if wrong | Source |
|---|---|---|---|---|---|
| ATT-26 | A model window of w keys is stored as `sliding_window = (w - 1, 0)` and reaches the kernel as w: the last w keys including the current token are kept. | vLLM · `TRITON_ATTN_DIFFKV`; `ROCM_ATTN` and `ROCM_AITER_UNIFIED_ATTN` pass the same value | capture (1 model) on `TRITON_ATTN_DIFFKV` (windows of w - 1, w and w + 1 keys were compared against captured steps; w fit); source only on the other two | off by one key on every context longer than the window (not quantified) | `vllm/v1/attention/ops/triton_unified_attention_diffkv.py:456`, `:273-279`; `vllm/v1/attention/backends/rocm_attn.py:281-286`, `:460-462`; aiter: `aiter/ops/triton/attention/unified_attention.py:248` |
| ATT-27 | The window masks, it does not re-index. The tile loop starts at the tile that holds the first kept key (`first_kept // tile`), tiles stay on the absolute-position grid (a window that starts mid-tile keeps that tile's grid), and split-K segments divide the full `seq_len`, not the window. | vLLM · `TRITON_ATTN_DIFFKV`; `ROCM_AITER_UNIFIED_ATTN` 2D kernel | source only (a reference that tiled the kept window from its own index 0 measured a max relative error of 0.11 to 0.20 on the windowed layers of MiMo-V2.6-Flash against 0.007 on its full-attention layers; the absolute-grid reference was not re-measured there) | tiling the kept window from index 0 puts every probability on another grid: roughly doubles the error | `vllm/v1/attention/ops/triton_attention_helpers.py:145-270` (`compute_tile_loop_bounds`); `vllm/v1/attention/ops/triton_unified_attention_diffkv.py:191-205`; aiter: `aiter/ops/triton/_triton_kernels/attention/unified_attention.py:271-290`, `:479-484` |
| ATT-28 | Keys with `(seq_len - 1 - p) >= w` are set to -10000 (not -inf) on the absolute-position tiles; no tile is pruned, and `exp(-10000 - m)` is exactly 0 in fp32, so masked tiles drop out exactly. | vLLM · `ROCM_ATTN` Triton fallback (the HIP custom kernel never runs with a window) | source only | not quantified | `vllm/v1/attention/ops/chunked_prefill_paged_decode.py:246-249` |
| ATT-29 | A per-head sink logit seeds the running max (M = sink) with the denominator starting at L = 1, which is the sink's own `exp(0)` term. On the split-K path only segment 0 loads the sink; every other segment starts at M = -inf and its initial L = 1 is zeroed by the first rescale `exp(-inf - m)`. Without sinks M = -inf in every segment. A sink belongs to exactly one segment. | vLLM · `TRITON_ATTN_DIFFKV` | capture (1 model): dropping the sinks gave a max relative error of 0.92 on the windowed layers of MiMo-V2.6-Flash | sink dropped or read in the wrong format: 2.2x to 38x tolerance | `vllm/v1/attention/ops/triton_attention_helpers.py:113-142` (`init_softmax_M`); `vllm/v1/attention/ops/triton_unified_attention_diffkv.py:176-180` |
| ATT-30 | Same seeding on base-2 logits: M = sink * log2(e) with L = 1; without sinks M = -inf and L = 1 (zeroed by the first rescale). | vLLM · `ROCM_AITER_UNIFIED_ATTN` 2D kernel | source only | as ATT-29 | aiter: `aiter/ops/triton/_triton_kernels/attention/unified_attention.py:218-232`; `vllm/v1/attention/backends/rocm_aiter_unified_attn.py:95-97` |
| ATT-31 | With sinks M starts at the sink and L at 1; without sinks M = -inf and L = 0 (not 1). The backend reports that it does not support sinks, so a layer with sinks normally selects another backend. | vLLM · `ROCM_ATTN` Triton fallback | source only | as ATT-29 | `vllm/v1/attention/ops/chunked_prefill_paged_decode.py:126-135`; `vllm/v1/attention/backends/rocm_attn.py:179-184` |
| ATT-32 | The sink tensor the kernel reads is the model's sink parameter exactly as built and loaded (vLLM's `make_attention_sink` builds it in the default dtype: bf16 for a bf16 model). The kernel loads one element per query head and widens it to fp32 (exact for bf16 and fp32); nothing converts it on the host. Read the dtype and the attribute off the live layer (`layer.impl.sinks`); do not assume fp32 or a parameter name. | vLLM · all three backends | capture (1 model) on `TRITON_ATTN_DIFFKV` (MiMo-V2.6-Flash keeps one bf16 sink per head; a kernel that loads it as bf16 and widens it matches the captured backend); source only on the other two | reading 4 bytes per head from a bf16 sink, or running without the sink because a default attribute name found nothing: 2.2x to 38x tolerance | `vllm/model_executor/model_loader/weight_utils.py:1584-1592`; `vllm/v1/attention/backends/triton_attn.py:544-546`; `vllm/v1/attention/ops/triton_attention_helpers.py:134-141` |

Two consequences for a kernel that re-implements the split-K path with a window:

- A segment with no kept key must drop out. Without a sink, a segment that lies entirely before the window ends
  with M = -inf; vLLM stores -inf for it and the merge treats it as absent. A merge that evaluates
  `exp(-inf - (-inf))` returns NaN for every output (windowed layer, no sinks, context longer than the window
  plus one segment).
- A masked position must not be addressed. A tile's tail past `seq_len - 1`, and with a window its head before the
  first kept key, is masked out of the softmax, but its block-table entry may not exist (a compact table ends at
  the row's last block, and padding entries are not pages). Clamp the addressed position into
  [first kept key, `seq_len - 1`]; an unclamped kernel reads past the table.

## Current-token KV write seen from the attention call

| id | Fact | Applies to | Status | Effect if wrong | Source |
|---|---|---|---|---|---|
| ATT-33 | All three backends declare `forward_includes_kv_cache_update = False`: `Attention.forward` writes the current token's K / V into the cache (`impl.do_kv_cache_update` at the layer's slot mapping) immediately before the attention kernel, which then reads that token back from the cache as its last key. A kernel that replaces the whole attention call owns this write. `TRITON_ATTN_DIFFKV` never takes the fused RoPE + cache-write path. | vLLM · all three backends | source only | a kernel that folds the write must attend the current token exactly as the cache stores it (ATT-35); attending a stale slot instead is 2x to 170x tolerance | `vllm/model_executor/layers/attention/attention.py:544-583`, `:719-751`; `vllm/v1/attention/backends/triton_attn.py:348`; `vllm/v1/attention/backends/rocm_attn.py:196` |
| ATT-34 | With `VLLM_ROCM_USE_AITER` and the compile pass `fuse_rope_kvcache` (or `fuse_qk_norm_rope_kvcache`, for models with a q / k norm) on, and for compile ranges up to `rope_kvcache_fusion_max_token_num` tokens (default 256), the fused RoPE + cache-write kernel writes the cache and the attention call only reads it. | vLLM · `ROCM_ATTN`, `ROCM_AITER_UNIFIED_ATTN` | source only | decides which kernel owns the write (ATT-33) | `vllm/v1/attention/backends/rocm_attn.py:528-638`; `vllm/compilation/passes/fusion/rope_kvcache_fusion.py:421-461`; `vllm/config/compilation.py:148-156` |
| ATT-35 | The write copies a bf16 k / v into a bf16 cache; into an fp8 cache it stores `fp8(fp32(x) / scale)` with the fp32 `k_scale` / `v_scale`: a division, then the conversion on store (Triton store cast on `TRITON_ATTN_DIFFKV`; the HIP fp8 convert in the native write of the two ROCm backends). The scale is part of the stored value; the logits get `k_scale` back through the logit scale. The rounding of the fp32-to-fp8 conversion (round to nearest even with saturation is assumed) was not verified. Write kernels in detail: `rope_and_kv_write.md`. | vLLM · all three backends · fp8 cache | source only | scale handling flipped: 11x to 57x tolerance (negative control on a synthetic shape with `k_scale` 0.3); attending the unrounded bf16 k / v instead of the stored value: one fp8 ulp of that key | `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:486-560` (`reshape_and_cache_kernel_flash_diffkv`); `csrc/cache_kernels.cu:252-333`; `csrc/quantization/w8a8/fp8/amd/quant_utils.cuh:313-319` |
| ATT-36 | A token's slot is `block_table[pos // B] * B + pos % B` (int64). Rows past the live batch under a full CUDA graph get slot -1 and `seq_len` 0: the write skips a negative slot, and the native attention leaves such a row's output unwritten (uninitialised memory). Exclude padded rows from any comparison. | vLLM · `TRITON_ATTN_DIFFKV` (the writes of the two ROCm backends skip negative slots too) | source only | false mismatches on padded rows | `vllm/v1/worker/block_table.py:85-87`; `vllm/v1/worker/gpu_model_runner.py:2143-2152`, `:3864-3872`; `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:508-518` |

## Output gate after the attention

For models whose layer multiplies the attention output by a sigmoid gate (optionally behind a per-head norm). The
gate is model code, not part of any attention backend, so these rows hold for all three backends.

| id | Fact | Applies to | Status | Effect if wrong | Source |
|---|---|---|---|---|---|
| ATT-37 | `Attention.forward` allocates its output in the query's dtype (bf16 for a bf16 model): whatever follows it (a gate, an output norm) starts from a bf16-rounded attention output. | vLLM · all attention backends | source only | 0.02x to 0.2x tolerance: invisible at stage level | `vllm/model_executor/layers/attention/attention.py:499-539` (`Attention.forward`) |
| ATT-38 | The output gate takes one of three forms: `attn * sigmoid(gate)` with the gate taken from the second half of each head of the q projection; `sigmoid(og) * attn` with `og` the last q-sized columns of a fused [q, k, v, gate] projection; or, when the layer also has an output norm, a weightless per-head gated RMS norm `rms_norm_gated(attn, og, None, None, 'sigmoid', eps)` with the model config's `rms_norm_eps`. No gated model was captured. | vLLM · a hybrid model with gated attention layers; a post-norm MoE model with an output gate | source only | gate, norm or eps missing or wrong: 2.7x to 400x tolerance; eps alone: 2.7x to 55x (it decides heads whose attention output is small, as on a long context) | `vllm/model_executor/models/qwen3_next.py:458-469`; `vllm/models/<PKG_POSTNORM_MOE>/amd/model.py:653-695`; `vllm/models/<PKG_POSTNORM_MOE>/common/layers.py:339-400` |
| ATT-39 | Eager torch evaluates `attn * sigmoid(gate)` on bf16 tensors with two roundings: `bf16(fp32(attn) * fp32(bf16(sigmoid(gate))))`. | PyTorch eager · bf16 · gfx942 (and CPU) | probe (equal to the two-rounding formula on every one of 131072 elements; the one-rounding formula differs on about a quarter of them by one bf16 ulp) | 0.02x to 0.2x tolerance: invisible at stage level, so choose it from the compile scope (ATT-42), not from a passing test | torch 2.13: eager bf16 elementwise semantics (no source line) |
| ATT-40 | Under `torch.compile` (Inductor) the same expression is one fused fp32 kernel rounded once at the store, unless `emulate_precision_casts` is set. | PyTorch Inductor · bf16 · gfx942 | probe (equal to the one-rounding formula on every element; the vLLM piecewise graph around the attention op itself was not captured) | as ATT-39 | torch 2.13: `torch/_inductor/config.py` (`emulate_precision_casts`) |
| ATT-41 | The gated RMS norm loads x and the gate as fp32 and stores `y = x * (1 / sqrt(sum(x^2) / D + eps)) * sigmoid(gate)` once in x's dtype, per row of D = head dim (one row per head; the gate is read through its row stride). The sigmoid is not rounded, compiled or not. | vLLM Triton `rms_norm_gated` · bf16 · gfx942 | probe (equal to the one-rounding fp32 formula except 2 to 3 of 131072 elements by one ulp; about a quarter of the elements differ from the two-rounding formula) | norm or eps wrong: 2.7x to 400x tolerance | `vllm/third_party/flash_linear_attention/ops/kda.py:156-247`, `:317-405`, `:407-453` |
| ATT-42 | Whether the gate runs eagerly is decided by the model class's compile scope, not only by the engine's compilation mode: a class decorated `@ignore_torch_compile` runs eagerly even under a compiling engine; one decorated `@support_torch_compile` is compiled. | vLLM · gated-attention models | source only | selects ATT-39 or ATT-40 | `vllm/models/<PKG_POSTNORM_MOE>/amd/model.py:882-884`; `vllm/model_executor/models/qwen3_next.py:632-633` |

## What one served model showed

MiMo-V2.6-Flash on `TRITON_ATTN_DIFFKV`, fp8 KV cache, gfx942, TP 4, width-1 decode steps captured at the
attention boundary:

- With an fp8 query, fp8 P and tile 32, an unsplit reference was bit-exact only on the steps whose context fits
  one segment; adding the 16-segment split made it bit-exact on all 50 fresh steps of a full-attention layer
  (ATT-11, ATT-18, ATT-20, ATT-23 together).
- Dropping the sinks: max relative error 0.92 on the windowed layers (ATT-29).
- Tiling the kept window from its own index 0: max relative error 0.11 to 0.20 on the windowed layers against
  0.007 on the full-attention layers (ATT-27, whose own row is still source only).
- The fp8 query rounding is applied identically by the backend and by a faithful candidate, so it cancels: the
  stage tolerance derived without it is 0.13; counting it as an error source gives 0.26, about twice as loose
  (method: `parity_method.md`).

## Not covered by any check

- `TRITON_ATTN_DIFFKV`: a bf16 query (tile 16, P rounded to bf16); the single-pass 2D path (wide batches,
  batch-invariant mode); non-unit q / k / v scales; a padded page; speculative-verify batches (several query
  tokens per request, with their own tiling); ALiBi, logit soft-cap and chunked lookback.
- `ROCM_ATTN`: the HIP custom paged-attention kernel (its arithmetic was not read at all); the Triton fallback
  was read but no captured step of this backend was compared; an fp8 cache with non-unit k / v scales; an
  `fp8_e5m2` cache; the interleaved V cache layout set by the fused q / k-norm + RoPE + cache kernel.
- `ROCM_AITER_UNIFIED_ATTN`: never run (ATT-05); the split-K 3D path and its per-call segment count; the gfx950 /
  gfx1250 kernels and the shuffled K / V cache layout; non-unit scales.
- All backends: gfx950 (OCP e4m3) for every row; the rounding mode and saturation of the fp32-to-fp8 conversion
  in the cache write; any gated model on a capture; rows with more than one query token.
- A model that multiplies v (or k) by a constant between the projection and the attention call does so before
  the cache write; that multiply and its bf16 rounding belong to the producer of v (see `rope_and_kv_write.md`).
