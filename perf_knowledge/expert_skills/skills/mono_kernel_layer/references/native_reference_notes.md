# Native reference notes: how vLLM on ROCm computes a decode layer

Facts that a float64 reference implementation of a layer's stages needs and that cost the most time to
rediscover. Each MiMo fact was checked against a frozen capture of the layer: the number after it is the
reference's `max_rel_err` against the captured boundary (MiMo-V2.6-Flash-MOPD, TP4, gfx942, vLLM
0.32.1rc1). The DeepSeek-V4.1 sections say for themselves what they were checked against. Start from the
layer's source file and from the backends vLLM logged when the layer was captured (the server log names the
MoE backend and the expert class), and read a backend's source only for what is not here. For the lever
that applies these facts to a kernel, see [numerics.md](numerics.md).

Conventions on this page:
- `topk_route`, `moe`, `dequant_mxfp4`, `deinterleave_gate_up`, `gather_paged_kv`, `decode_attention`,
  `silu_and_mul`, `allreduce` and the like name the reference helpers the findings were obtained with; they
  are written as pseudo-code for the operation and are not part of this repository.
- `W1-r2` / `W1-r3` are run labels of the DeepSeek-V4.1 captures (gfx942, TP4) the checks were made on; the
  capture artifacts are likewise not part of this repository.
- `extra.kv_window`, `extra["caches"]`, `layer_weights`, `written`, `tie_tokens`,
  `stage_boundary_calls_per_step`, `compare.form` and `compare.router.tie_rel` are keys of the capture
  records, reports and comparison settings of the source repository's layer tasks. They are kept so that
  an item can be found in such a capture or meta; an ordinary GEAK task's `meta.json` does not carry them.

## Dense fp8 linears (`activation_scheme: dynamic`, block 128x128)

- The reference linear `fp8_block_linear(x, weight, weight_scale_inv)` quantizes the bf16 input to fp8 per
  128-element group (`act_quant_fp8`) before the GEMM. Without it the MiMo MLP measured 0.036 instead of
  0.018.
- On gfx942 the loaded format is `float8_e4m3fnuz` (max 240), not the checkpoint's e4m3fn.

## Paged KV cache (fp8 kv_cache_dtype, `TRITON_ATTN_DIFFKV` and the full-attention backend)

- The capture of a paged-KV layer records the window of the cache as `extra.kv_window[0]`: `pages`
  [n_blocks, Hk, block_size, k_dim + v_dim] uint8 holding fp8 (K then V in the last axis), plus
  `first_block` and `seq_len`; `gather_paged_kv` reads it. A sliding-window layer's capture keeps only
  the window's blocks (`first_block > 0`).
- The page row of the current token already holds the native write (captured after the step); it matches
  the reference's own fp8 K/V to 1 fp8 ulp. Use the reference's own row to test the write.
- `k_scale` / `v_scale` are in `extra` (1.0 for MiMo).

## Sliding window and sinks (`TRITON_ATTN_DIFFKV`, `unified_attention_diffkv`)

- Window w is passed as `(w - 1, 0)`: the last w keys, the current token included. That is
  `decode_attention(window=w)`.
- Sinks (`self_attn.attention_sink_bias`, per-rank [Hq]) start the online softmax at M = sink, L = 1:
  `decode_attention(sinks=...)`. Dropping them gives 0.92 on MiMo.
- MiMo's SWA layers are sink-dominated (sink 0.6-1.1, all logits far below), so the attention output is
  not renormalized and the `self_attn` boundary measures 0.02-0.10 against an otherwise exact reference,
  while the full-attention layers measure 0.008. Checked and ruled out: qkv and RoPE (page rows match),
  page layout, window edge (w-1, w, w+1), sink scale, q/k/v/P rounding to bf16 or fp8 (they move the
  error by < 0.006). The cause is not found; the chained bound (~0.3) passes it. The layer-level
  comparison (`compare.form` decoder_residual) is on the layer's increment, so this error is not diluted
  by the residual stream and shows in the measured error. Note it with the reference; do not
  widen a bound for it.

## mxfp4 MoE experts (`moe_backend=aiter_triton_mxfp4_bf16`: `AiterW4A16ExpertsMonolithic`)

- Weights are in `layer_weights` under `deep`:
  `mlp.experts.routed_experts|w13_weight.storage.data` [E, H/2, 2I] uint8,
  `...|quant_method.moe_quant_config._w1.scale.b_mx_scale.storage.data` [E, H/32, 2I] uint8 (e8m0),
  `...|w2_weight.storage.data` [E, I/2, H], `..._w2.scale.b_mx_scale.storage.data` [E, I/32, H].
  This is the transposed view vLLM's `_swizzle_mxfp4` makes (`StridedLayout(-2)`; scales unswizzled on
  gfx942 and on gfx950 at TP > 2). `dequant_mxfp4(data, scale, e)` -> `[N, K]`.
- w13 rows are interleaved `g0, u0, g1, u1, ...` for the a16w4 kernel: `deinterleave_gate_up` ->
  `[gate | up]`, then `silu_and_mul`.
- Activations stay bf16 (w4a16: no activation quant); the activated intermediate is stored in bf16
  between the two GEMMs (`moe(..., act_round="bf16")`). Every rank holds all E experts with I/TP columns;
  the `mlp` boundary is after the TP all-reduce.
- Pass `moe` callables (`lambda e: ...dequant_mxfp4(w, s, e)`) so only routed experts are dequantized; all
  256 in float64 are ~1 GiB per rank.
- Router (MiMo, `RoutingMethodType.DeepSeekV3` with `n_group=1`): `mlp.gate.weight` bf16 on a bf16 input,
  fp32 logits; `topk_route(logits, 8, scoring="sigmoid", bias=e_score_correction_bias)` (the bias picks,
  the un-biased sigmoid weighs, renormalized: `norm_topk_prob`), no routed scaling, no shared experts. MoE
  `mlp` boundary: 0.0055-0.0066 on every captured step.
- Near ties: on full_attn_moe 5 of 14 captured steps have a token whose 8th/9th biased score differ by
  < 1e-3 (relative); one bf16 ulp on the router input moves that gap by up to 4e-3, so the reference may
  pick another expert there. The layer-level comparison (`LAYER`) does not compare such tokens on the MoE
  output (`compare.router.tie_rel`: 0.0082 on full_attn_moe, 0.0008 on swa_moe) and reports them as
  `tie_tokens`; the unittest compares them against the native output for every routing the tie allows.
  Route on the reference's own logits; do not copy the native top-k.

## DeepSeek-V4.1 MoE (router `scoring_func: sqrtsoftplus`, experts `swiglu_limit`)

Checked, by tests in the source repository, against DeepSeek's `inference/model.py` (`Gate`, `Expert`) and
vLLM's router / activation code; not yet against a DSv4.1 capture.

- Router: `ffn.gate.weight` [384, H], fp32 logits. Config `scoring_func` sqrtsoftplus, `topk_method`
  noaux_tc, `norm_topk_prob`, `routed_scaling_factor` 1.5, top `num_experts_per_tok` 6 (DSpark draft
  layers: `dspark_num_experts_per_tok` 3 of `dspark_n_routed_experts` 128):
  `topk_route(logits, 6, scoring="sqrtsoftplus", bias=e_score_correction_bias, routed_scale=1.5)`. Scores
  are sqrt(softplus(logits)); the bias (checkpoint `ffn.gate.bias`: mean 9.4-57.6 over the backbone
  layers, 15.6 on layers.30 and 57.6 on layers.39, std 0.02-0.15; 0 on the draft layers) only picks, the
  un-biased scores weigh, renormalized, x1.5. On the deployed ROCm backend every token routes with this
  bias, image placeholder ids included: `ffn.gate.bias_vl` takes no part in routing there (next item).
  Scored as sigmoid, 64 of 64 test tokens route to another expert set.
- `ffn.gate.bias_vl` [384] fp32, the vision checkpoint's second routing bias (vLLM's CUDA path and its
  router-based backends pick an image token's experts with it), takes no part in routing on the deployed
  backend (`--moe-backend aiter_triton_mxfp4_bf16`, `VLLM_ROCM_USE_AITER_MOE=1`, gfx942): every token
  routes with `e_score_correction_bias`, the image placeholder ids 129264-129268
  (`IMAGE_SENTINEL_BASE_ID`, `models/deepseek_v41/common/mm_preprocess.py:51`) included. The layer only
  checks that it is set: the MoE forward (`models/deepseek_v4/amd/model.py:787-804`, which DSv4.1's MoE
  inherits) then wants `input_ids` and raises without them (`:792-793`). Checked on the installed vLLM
  0.32.1rc1.dev602+g5d35ce393 (paths under `vllm/`; `fused_moe/` is `model_executor/layers/fused_moe/`)
  and the W1-r3 captures (TP4, every kind):
  - every capture's server.log names the MoE backend and the expert class: line 161 is `[mxfp4.py:609]
    Using 'AITER_TRITON_MXFP4_BF16' Mxfp4 MoE backend.`, and one of lines 236-246 (the line differs per
    kind) is `[mxfp4.py:2304] Using AiterW4A16ExpertsMonolithic` (logged by
    `fused_moe/oracle/mxfp4.py:609` and `:2303-2304`);
  - a monolithic expert class routes inside its own kernel: `fused_moe/runner/moe_runner.py:620-627`
    hands it the router logits and `input_ids`, and only the modular path calls `router.select_experts`
    (`:628-635`). `Mxfp4MoEMethod.apply_monolithic`
    (`model_executor/layers/quantization/mxfp4.py:1114-1139`; DSv4.1's quant config picks that method,
    `models/deepseek_v41/quant_config.py:261`) passes `e_score_correction_bias`, `input_ids` and
    `hash_indices_table`, no `bias_vl`; `AiterW4A16ExpertsMonolithic.apply`
    (`fused_moe/experts/aiter_mxfp4_w4a16_moe.py:409-465`) routes in
    `aiter_triton_kernel_w4a16_moe_forward` with `aiter_routing(bias=e_score_correction_bias)`
    (`:178-194`) and reads `input_ids` only with a hash table (`:161-177`), which DSv4.1 has none of
    (`models/deepseek_v41/amd/model.py:116`);
  - `bias_vl` picks experts only where a router picks them, behind a modular expert class (no
    `Monolithic` in its name), in `fused_topk_bias` (`fused_moe/router/fused_topk_bias_router.py:141-336`):
    by `dsv4_topk` on CUDA only (`fused_moe/router/dsv4_topk.py`: `can_use_dsv4_topk` wants CUDA, `:23`;
    the kernel is defined only under `is_cuda()`, `:38`; its image branch, `:63-75`), else for
    sqrtsoftplus scoring by `vllm_topk_softplus_sqrt` (`fused_topk_bias_router.py:74-102`, the op
    `topk_hash_softplus_sqrt` at `:87-100`), or on the torch path (`fused_topk_bias_router.py:321-336`).
    The AITER mega-MoE routing reads it as well (`models/deepseek_v4/amd/model.py:830-847`,
    `--moe-backend aiter_mega_moe`), but DSv4.1 never takes that path: its MoE sets
    `use_mega_moe = False` (`models/deepseek_v41/amd/model.py:142`).
    `models/deepseek_v41/amd/model.py:132-141` only registers `bias_vl` and attaches it to the router;
  - on the GPU (W1-r3 cr0_swa_engram, the unittest's native leg: the deployed layer and backend), the
    control `image_tokens_text_bias` (`bias_vl` holding `e_score_correction_bias` during the forward, on
    eager cases whose input_id is 129264) changed nothing on 4 ranks in any check, and the reverse control
    `image_tokens_bias_vl` (an image placeholder token routed with `bias_vl`) failed eager at 0.55-0.64.

  So a reference or a candidate routes every token with `e_score_correction_bias` on this backend; one that
  routes the image placeholder ids with `bias_vl` (the CUDA path's way) is wrong here, and only cases
  whose `input_ids` hold such an id can show it (no captured decode step does). Under another
  `--moe-backend` or platform, read the expert class the capture's server.log names first: a class
  without `Monolithic` routes through `router.select_experts`, which reads `bias_vl` for those ids.
- Experts, routed and shared: SwiGLU clamped at config `swiglu_limit` 10.0, up to [-10, 10] and gate <=
  10, then silu(gate) * up (DeepSeek `Expert`; vLLM fused MoE `swiglu_limit`, shared expert
  `SiluAndMulWithClamp`). Routed: `moe(..., swiglu_limit=10.0)`; shared: `silu_and_mul(h,
  swiglu_limit=10.0)`. Within +-10 the clamp changes nothing, so a capture that stays inside cannot show a
  missing one: pass it anyway.
- `topk_route` / `moe` raise on a `scoring` / `act` they do not know (they used to fall back to sigmoid /
  gelu_tanh).

## DeepSeek-V4.1 decoder layer boundaries (mHC)

From vLLM's `DeepseekV4DecoderLayer.forward` (vllm/models/deepseek_v41/amd/model.py:295-455) and the W1-r2
captures (gfx942, TP4: the same calls on every rank and each of the 14 steps).

- `mhc_pre_delayed` is called twice a step (`stage_boundary_calls_per_step` `{"min": 2, "max": 2}`):
  `call=0` at the attention entry, `call=1` at the FFN entry. It returns
  `(residual, post_mix, res_mix, x, pre_mix)`. Given `sublayer_out` it first applies the previous
  sublayer's post, and `residual` is the stream after it: the FFN entry always (the attention's post); the
  attention entry too (the previous layer's FFN post), except on layer 0 (no sublayer before it) and on an
  Engram layer.
- `mhc_post` is called only by an Engram layer (W1-r2 layers 1 and 14), once a step at the attention
  entry, before `engram` and `mhc_pre_delayed`. Every other layer never calls it, so the capture does not
  list it.
- `attn_norm` / `ffn_norm` run only where the seam's RMSNorm is not fused into `mhc_pre_delayed`
  (`fuse_seam_norm`: AITER's fused delayed seam, gfx950 only; layer 0's attention entry never fuses it):
  once a step each on gfx942. Read the calls of the task at hand from its own capture
  (`stage_boundary_calls_per_step` in a source-repository layer task), not from this list.

## DeepSeek-V4.1 compressor latent (`attn.compressor`)

From vLLM's code (vllm/models/deepseek_v41/compressor.py, common/ops/fused_compress_quant_cache.py); the
rows it writes were checked against the kernel run by the Triton interpreter on random batches
(source-repository validation issue ISSUE-020), the values not yet against a DSv4.1 capture.

- `forward(kv_score, positions)` returns a bf16 latent [T, 512] allocated with `torch.empty`. A row holds
  a value only where a group of `compress_ratio` tokens closes: ratio 2 at an odd position whose request
  has a ring block (its state-ring slot >= 0), ratio 1 where the main cache's slot is >= 0. The other rows
  are whatever was in memory (W1 cr2: inf / NaN at the even positions, different per rank). The capture
  records `written` and NaN there; the reference check compares only the written rows. Its readers (the
  compressed-cache insert, the indexer's K store) read only those rows too.
- Ratio 2, row t: per dim, the softmax over the pair's two score halves weighs their kv halves (kv_score
  fp32 = `[kv | score]`), then `pooled * rsqrt(mean(pooled^2) + eps) * norm.weight`, rounded to bf16. The
  pair's first row is row t-1 of kv_score, except for the first token of a request in the step
  (single-token decode): then it is the ring row of position t-1
  (`extra["caches"]["<attn>.compressor.state_cache"]`, slot = block * capacity + pos % capacity,
  capacity 8 without speculative tokens). Ratio 1: the same norm of the token's own row, no gate.

## TP all-reduce

- `QUICK_REDUCE` is first in vLLM's dispatch, but with bf16 at TP4 its smallest quantized message is 16
  MiB (INT4), so decode (one token, 8 KiB) goes through a full-precision path. The reference all-reduce is
  the exact sum (`allreduce`).

## Sources

- External code cited by file and line (vLLM 0.32.1rc1.dev602+g5d35ce393; DeepSeek `inference/model.py`):
  not part of this repository.
