"""rank_model.py of task `dsv41_mono_target_layer` (UT contract family `mono_layer`): the
model-specific half of the rank harness. WORKED EXAMPLE for references/mono-layer-ut.md
(DeepSeek-V4.1-Flash, vLLM --monokernel, MI355X gfx950, TP2).

Provenance: 2026-11-13 deployed path, non-index layers, GPU-verified (vllm 0.33.0rc1.dev910, MI355X
TP2: every correctness-mode gate incl. replay + interleave, 6/6 negative controls; rank_chain
timing). 2026-11-16 (written without a GPU; GPU-verified 2026-11-17, every gate incl. the native
floor -- references/mono_models.json oracles_verified): the index-source and fold-0 layer types, the
`split` and `native` oracle paths, near-tie routing alignment. Sources read: the installed vLLM
= vllm-project/vllm 4fab5a07 (vllm/models/deepseek_v41/amd/{mono/runner.py, mono/layer.py,
mono_decode.py, model.py}) and vLLM's own numerics test of these kernels,
tests/models/test_dsv41_mono_numerics.py (`_check_layers`), whose op chain the native path runs.

Entry under test: `vllm.models.deepseek_v41.amd.mono.runner:DSV41MonoLayer.forward` (runner.py:351).
Live call site (installed vllm, same bytes as vllm 4fab5a07):
  vllm/models/deepseek_v41/amd/mono_decode.py:650 `MonoDecodeLayer.__call__` (one call a backbone
  decoder layer, from DeepseekV4DecoderLayer.forward on a decode-only step of 1..48 rows, eager
  or a FULL graph) -> `runner.forward(self.weights(layer), x, residual, post_mix, res_mix,
  pre_mix, positions, swa.slot_mapping, _record_view(attn.swa_cache_layer.kv_cache,
  swa.block_size), swa.decode_swa_indices, swa.decode_swa_lens, swa.token_to_req_indices,
  fold=fold, topk_indices=attn.topk_indices_buffer, comp_cache=_record_view(
  attn._compressed_kv_cache(), comp.block_size // attn.compress_ratio),
  comp_block_table=comp.block_table, [select=...])` (mono_decode.py:726). `fold` is False on the
  first layer and after an Engram write (served layers 0, 1, 14); `select` (vLLM's compressor +
  indexer, writing the top-512 into topk_indices between K1 and K2) only on an index source
  (served 2, 8, 14, 20, 24, 28, 32, 36); ratio-0 layers (0, 1) pass no compressed cache.
  runner.forward (runner.py:351) makes two device launches, K1
  (aiter::dsv41_mono_k1_s<M>_tp2_r<ratio>_tl0_f<fold>_i<index>) and K2
  (aiter::dsv41_mono_k2_s<M>_tp2_r<ratio>_tl0_i<index>). The runner is the process's one
  `_mono_runner` (mono_decode.py:80): DSV41MonoLayer(tp, rank, get_tp_group().cpu_group, device),
  shared by every layer; every TP rank makes the same calls in the same order; no end-of-step hook.

A call() step is LAUNCHES = 5 UT layers, one of each served layer type (LAYER_TYPES):
  0  r0 fold 0         served layer 1 (Engram: the residual arrives settled, pre-mix carried in)
  1  r2 fold 1         served layer 3  (and 4..19 but 8 / 14)
  2  r1 fold 1         served layer 21 (and 22..39 but the index sources)
  3  r2 fold 1 index   served layer 8  (and 2)
  4  r1 fold 1 index   served layer 24 (and 20, 28, 32, 36)
so every expected K1 / K2 build of the target runs except K1 r2 f0 i1 (served layer 14 alone:
fold 0 AND an index source; both branches run here separately; meta launch_waivers). An index
layer's `select` stands in for vLLM's indexer: it copies a precomputed top-512 into the step's
top-k buffer, which holds -1 (no compressed key) until then, so a K2 that read the table before
select ran attends over the window only and fails the oracles. rank_chain's chain: layer 0 the
r0 fold-0 type (call()'s layer 0, untimed: rank_chain subtracts step(1)), then CHAIN_PATTERN
(3 r2 : 3 r1 : 1 r2-index : 1 r1-index, the served mix of the fold-1 layers within a few %).

Paths (PATHS[0] the deployed one; oracle legs run on the baseline overlay):
  deployed  runner.forward as served.
  split     the same deployed call (the baseline's K1 + K2). DSV41MonoLayer has no per-stage mode
            (the FFN launch is K2's FFN half after VLLM's attention, a different attention; vLLM's
            non-mono chain runs other kernels, so neither meets the family's 1/128 split bound),
            so -- as for the Muse Spark draft -- split is the baseline's own code: the candidate is
            judged against the baseline on every case and every gated output, and the baseline's
            floor is trivially met. The baseline itself rests on `native`.
  native    vLLM's own non-mono decode chain for the layer, op for op as vLLM's numerics test
            (tests/models/test_dsv41_mono_numerics.py `_check_layers`) runs it: the mHC seams
            (vllm.model_executor.layers.mhc.MHCPreDelayedOp, aiter's fused delayed seam), mxfp8
            quantize + rocm_mxfp8_block32_gemm (wq_a | wkv), fused_q_kv_rmsnorm, wq_b, vLLM's
            fused q-rope / kv-rope-quant-insert (writes the step's fp8_ds_mla records: kv_rows),
            rocm_aiter_ops.triton_sparse_mla_fwd over the SWA records and the top-k compressed
            records with the attention sink, rocm_inverse_rope_mxfp8_rows, the grouped wo_a bmm,
            wo_b; the attention all-reduce as fp32 sums of the ranks' bf16 partials over the
            gloo group, rounded once (vLLM's exact decode all-reduce at <= 48 rows); the FFN
            seam; vLLM's DeepseekV4MoE (vllm.models.deepseek_v41.amd.model, AITER A8W4 MXFP4
            backend + MXFP8 shared expert). The MoE is built at TP 1 on this rank's TP shard
            (config moe_intermediate_size = 2304 / WORLD, vLLM's model-parallel state of TP size
            1 a rank over the harness's gloo world), its loaded tensors replaced by this rank's
            deployed ones after vLLM's own post-load processing (same shapes, asserted), and its
            output -- this rank's partial, as a TP-WORLD module computes it -- summed over the
            ranks like the attention. Each rank computes its own TP shard: no weight is
            gathered. Teacher-forced per launch (the Muse W8 rule): launch i first runs the
            deployed launch on the deployed path's state before it, the native result is computed
            from that same state, and the deployed outputs carry to the next launch, so an error
            is measured once and never compounds. Near-tie routing: the runner exposes no ids,
            so the kernel's routing is recomputed with vLLM's rule (sqrt(softplus(x W^T)) + bias,
            top-6) on the FFN input K2 actually used (its scratch `normed` region) and on the
            native one; a token whose top-6 differs only by picks within TIE_TOL of the native
            boundary takes the kernel's picks (forced through e_score_correction_bias + BIG on
            those experts for that token's row alone: selection only, the weights stay vLLM's
            unbiased scores); any other difference stays and shows as error. Counts in facts().
            Bounds: the family's native SQNR floors, PROVISIONAL -- calibrate at the first GPU
            floor run (record the baseline's sqnr against native) as mono_layer.json bounds_doc
            prescribes; never loosen them to pass.

Rules (references/mono-layer-ut.md section 2) and how they are met here:
  * one process a rank, the subject's real peer buffer: DSV41MonoLayer over the harness's gloo
    group (its PeerBuffer handshake over UncachedIpcHeap), built as _mono_runner builds it;
  * weights: TP shards (wq_b, wo_a, wo_b, attn_sink, the routed and shared experts) from a rank
    seed; replicated tensors (mHC seams, norms, wqkv, router) from a seed common to every rank;
    norms 1 + 0.1 N(0,1), asserted not all ones and pairwise different (q_norm != kv_norm on
    their common dims too);
  * index dtypes as served (positions / slot_mapping int64, swa_indices / swa_lens /
    token_to_req / topk_indices / comp_block_table int32), page tables over shuffled pools: the
    SWA pages and the compressed-cache pages are two different tensors behind two different
    shuffled maps (asserted non-identity and different);
  * kv_targets / writes: launch i writes layer i's SWA records at slot_mapping (K1's insert; the
    native insert rewrites the same slots); the compressed caches are read only (pool 1, past
    the SWA caches: every changed byte is kv_stray). The fp8_ds_mla page cannot be indexed by
    slot on dim 0 (a page holds its 576-B data rows, then their 8-B scale words), so kv_bytes /
    kv_decode (mono_harness hooks) name a record's 584 bytes and decode them;
  * the stream is the mHC residual [rows, 4, 5120] bf16 (STREAM_SHAPE) and the seam mixes travel
    with it as STREAM_AUX = post [4] | comb [4, 4] | pre [4] (f32): launch() takes the previous
    layer's (out, residual, post, comb, pre) and returns this layer's.

Constants and their sources:
  * dims: config.json text_config (hidden 5120, hc_mult 4, 64 heads, head_dim 512, o_groups 8,
    384 experts top-6, moe_intermediate 2304, sliding_window 128, index_topk 512, compress_ratios,
    index_source_layer_ids, engram_layer_ids) and the subject's own modules (attention.plan.Dims,
    stages.dims.Dims): read at build, asserted;
  * cache layout: the profile evidence of the served image's predecessor
    (/shared_nfs/yueliu14/geak_profile_dsv4flash_ci121144693935_1102, captures of
    DSV41MonoLayer.forward): SWA cache [blocks, 32, 584] and compressed cache [blocks, 128 /
    ratio, 584], both with the deployed 210240-B page stride; comp_block_table [requests, 8192];
    serving args --block-size 128, --max-num-seqs 8, DSpark num_speculative_tokens 5 (6 rows a
    request);
  * RoPE: vLLM's own builder (common/rope.py build_deepseek_v4_rope) over config.json, per
    compress ratio, as the served AttnWeights.cos_sin (rotary_emb.cos_sin_cache).
"""

import os

import torch

WEIGHT_SEED = 20261113
ROWS_PER_REQ = 6            # 1 + DSpark num_speculative_tokens (serving.json --speculative-config)
SWA_BLOCK = 32              # the SWA cache's records a page (profile captures: [*, 32, 584])
CACHE_BLOCK = 128           # --block-size: a compressed page holds CACHE_BLOCK / ratio records
PAGE_STRIDE = 210240        # the deployed physical page stride of both caches (profile captures)
COMP_BT_COLS = 8192         # comp_block_table columns as served ([requests, 8192])
RECORD, DATA, NOPE = 584, 576, 448
MAX_REQS = 8                # --max-num-seqs
# the largest context a case may have (sizes the rope table): above every case of the 2026-11-13
# campaign's manifest (up to 475426 keys a request, tp2-ep1-c32/c8 observed_workload.json), below
# the served --max-model-len 1048576 (the campaign's UT lane raised it from 262144)
MAX_CONTEXT = 524288
# the compressed pool holds the pages of the widest step: 8 requests x 262144 keys (the c32 rows-48
# cases are <= 192222 keys a request; the longer contexts come with <= 2 requests)
POOL_CONTEXT = 262144
SWA_POOL_PAGES = 256        # >> 8 requests x 6 window pages, shuffled
COMP_POOL_PAGES = MAX_REQS * (POOL_CONTEXT // CACHE_BLOCK + 1) + 1
SWA, COMP = 0, 1            # contract axis `pools`: the SWA cache (written) / compressed (read only)
MODEL_CONFIG_DIR = "/shared_nfs/models/DeepSeek-V4.1-Flash"   # config.json (no weights read)

# (compress ratio, fold, index source, the served layer the type stands for)
R0_F0, R2, R1, R2_IDX, R1_IDX = (0, False, False, 1), (2, True, False, 3), (1, True, False, 21), \
    (2, True, True, 8), (1, True, True, 24)
LAYER_TYPES = (R0_F0, R2, R1, R2_IDX, R1_IDX)          # call()'s step, one launch each
CHAIN_PATTERN = (R2, R1, R2, R1, R2_IDX, R2, R1, R1_IDX)  # rank_chain's layers 1, 2, ...
TOPK = 6
TIE_TOL = 5e-3              # native: a top-6 pick within this of the boundary may flip (provisional)
FORCE_BIAS = 1.0e4          # native: added to e_score_correction_bias to force a token's picks


def layer_type(layer, n_layers=None):
    """UT layer `layer`'s (ratio, fold, index, served layer): call()'s step (n_layers None) is
    LAYER_TYPES; a chain's layer 0 is call()'s layer 0, then CHAIN_PATTERN."""
    if n_layers is None:
        return LAYER_TYPES[layer]
    return LAYER_TYPES[0] if layer == 0 else CHAIN_PATTERN[(layer - 1) % len(CHAIN_PATTERN)]


def _mono():
    from vllm.models.deepseek_v41.amd.mono import layer as L
    from vllm.models.deepseek_v41.amd.mono import runner as R
    from vllm.models.deepseek_v41.amd.mono.attention import plan as AP
    from vllm.models.deepseek_v41.amd.mono.stages import dims as MD
    from vllm.models.deepseek_v41.amd.mono.stages import seam as SEAM
    return R, L, AP, MD, SEAM


def _config_dir():
    return os.environ.get("DSV41_MODEL_CONFIG_DIR") or MODEL_CONFIG_DIR


_ROPE = {}


def _cos_sin(dev, ratio):
    """The layer's rope table as served (AttnWeights.cos_sin = attn.rotary_emb.cos_sin_cache):
    vLLM's build_deepseek_v4_rope over the checkpoint's config.json, fp32 [positions, 64]."""
    key = (str(dev), ratio)
    if key not in _ROPE:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
        from vllm.transformers_utils.config import get_config

        cfg = get_config(_config_dir(), trust_remote_code=True)
        cfg = getattr(cfg, "text_config", cfg)
        with set_current_vllm_config(VllmConfig()):
            rope = build_deepseek_v4_rope(cfg, head_dim=512, rope_head_dim=64,
                                          max_position_embeddings=cfg.max_position_embeddings,
                                          compress_ratio=ratio)
        cs = rope.cos_sin_cache[:MAX_CONTEXT + 64].to(device=dev, dtype=torch.float32).contiguous()
        assert cs.shape[-1] == 64, cs.shape
        _ROPE[key] = cs
    return _ROPE[key]


def _gen(dev, *key):
    s = WEIGHT_SEED
    for k in key:
        s = s * 1000003 + int(k)
    return torch.Generator(device=dev).manual_seed(s % (1 << 62))


def _mx8(n, k, g, dev):
    """A random MXFP8 weight [n, k] e4m3 and its E8M0 32 x 32 block scales (vLLM's loaded layout)."""
    w = torch.randn(n, k, device=dev, generator=g).clamp(-6, 6).to(torch.float8_e4m3fn)
    s = torch.randint(118, 123, (n // 32, k // 32), device=dev, generator=g, dtype=torch.uint8)
    return w, s


def make_weights(dev, rank, tp, layer, ltype, cos_sin, experts=384):
    """UT layer `layer`'s MonoLayerWeights on this rank (runner.py:111): replicated tensors from
    gs (seed common to the ranks), TP shards from gr (rank seed). MXFP4 experts as random FP4
    bytes with E8M0 scales 118..121 (in aiter's shuffle orders the bytes are as random). The
    draws depend on the layer index only, never on its type, so two models with the same first
    layers hold the same weights for them."""
    R, L, AP, MD, SEAM = _mono()
    d, md = AP.Dims(tp), MD.Dims(tp)
    H, E = AP.HIDDEN, experts
    bf, u8 = torch.bfloat16, torch.uint8
    gs = _gen(dev, 1, layer)
    gr = _gen(dev, 2, layer, rank)

    def rn(shape, g, scale=1.0):
        return torch.randn(shape, device=dev, generator=g) * scale

    def code(shape, lo, hi, g):
        return torch.randint(lo, hi, shape, device=dev, generator=g, dtype=u8)

    kw = {}
    for seam in ("attn", "ffn"):
        kw["hc_%s_fn" % seam] = rn((24, 4 * H), gs, 0.01).contiguous()
        kw["hc_%s_scale" % seam] = (rn((3,), gs, 0.5).abs() + 0.5).contiguous()
        kw["hc_%s_base" % seam] = rn((24,), gs, 0.1).contiguous()
        kw["%s_norm" % seam] = (1 + rn((H,), gs, 0.1)).to(bf)
    kw["gate_w"] = rn((E, H), gs, 0.02).to(bf)
    kw["bias"] = rn((E,), gs, 0.02)
    inter = md.inter
    kw["w13"] = code((E, 2 * inter, H // 2), 0, 256, gr).view(torch.float4_e2m1fn_x2)
    kw["w13_s"] = code((E * 2 * inter, H // 32), 118, 122, gr)
    kw["w2"] = code((E, H, inter // 2), 0, 256, gr).view(torch.float4_e2m1fn_x2)
    kw["w2_s"] = code((E * H, md.down_scale_cols), 118, 122, gr)
    kw["sgu"], kw["sgu_s"] = _mx8(2 * md.sh_inter, H, gr, dev)
    kw["sw2"], kw["sw2_s"] = _mx8(H, md.sh_inter, gr, dev)
    att = {}
    att["wqkv"], att["wqkv_scale"] = _mx8(AP.QKV_ROWS, H, gs, dev)          # replicated: [wq_a; wkv]
    att["wq_b"], att["wq_b_scale"] = _mx8(d.q_rows, AP.Q_RANK, gr, dev)     # this rank's heads
    att["wo_a"], att["wo_a_scale"] = _mx8(d.o_rows, d.group_k, gr, dev)     # this rank's groups
    att["wo_b"], att["wo_b_scale"] = _mx8(H, d.o_rows, gr, dev)             # this rank's columns
    att["q_norm"] = (1 + rn((AP.Q_RANK,), gs, 0.1)).to(bf)
    att["kv_norm"] = (1 + rn((AP.KV_DIM,), gs, 0.1)).to(bf)
    att["attn_sink"] = rn((d.heads,), gr)
    attn = R.AttnWeights(layer_id=ltype[3], cos_sin=cos_sin, ratio=ltype[0], **att)
    attn.check(d)
    w = R.MonoLayerWeights(attn=attn, **kw)
    w.check(tp, E)
    norms = [w.attn_norm, w.ffn_norm, attn.q_norm, attn.kv_norm]
    assert all(not bool((n == 1).all()) for n in norms), "a norm weight is all ones"
    for a in range(len(norms)):
        for b in range(a + 1, len(norms)):
            n = min(norms[a].numel(), norms[b].numel())
            assert not bool(torch.equal(norms[a][:n], norms[b][:n])), "two norm weights are equal"
    return w


def _fill_records(page_bytes, entries, g):
    """Random fp8_ds_mla records in every page of `page_bytes` [pages, PAGE_STRIDE] u8 (as vLLM's
    insert lays them out: a page's `entries` data rows of 576 B -- 448 e4m3 nope dims, 64 bf16
    rope dims -- then their 8-B scale words, 7 UE8M0 codes and a zero). Values ~ N(0, 1) after
    the scale, like a normed latent. Chunked to bound the temporaries."""
    pages = page_bytes.shape[0]
    dev = page_bytes.device
    for p0 in range(0, pages, 256):
        p1 = min(pages, p0 + 256)
        blk = page_bytes[p0:p1]
        n = p1 - p0
        data = blk[:, :entries * DATA].view(n, entries, DATA)
        nope = (torch.randn(n, entries, NOPE, device=dev, generator=g) * 96).clamp(-448, 448)
        data[..., :NOPE].copy_(nope.to(torch.float8_e4m3fn).view(torch.uint8))
        rope = torch.randn(n, entries, 64, device=dev, generator=g).to(torch.bfloat16)
        data[..., NOPE:].copy_(rope.view(torch.uint8).view(n, entries, 128))
        sc = blk[:, entries * DATA:entries * DATA + entries * 8].view(n, entries, 8)
        sc.zero_()
        sc[..., :7].copy_(torch.randint(119, 122, (n, entries, 7), device=dev, generator=g, dtype=torch.uint8))
        del nope, rope


def make_caches(dev, layer, ratio):
    """UT layer `layer`'s SWA cache and (ratio > 0) compressed cache, flat [pages, PAGE_STRIDE] u8
    each, filled with valid random records from a seed common to the ranks (the MLA latent is one
    KV head, replicated over TP: every rank writes the same records) and per layer, so two models
    with the same first layers hold the same caches whatever their layer count."""
    g = _gen(dev, 3, layer)
    swa = torch.zeros(SWA_POOL_PAGES, PAGE_STRIDE, dtype=torch.uint8, device=dev)
    _fill_records(swa, SWA_BLOCK, g)
    if not ratio:
        return swa, None
    comp = torch.zeros(COMP_POOL_PAGES, PAGE_STRIDE, dtype=torch.uint8, device=dev)
    _fill_records(comp, CACHE_BLOCK // ratio, g)
    return swa, comp


def record_view(flat, block):
    """mono_decode._record_view: [pages, block, 584] over the page stride."""
    return torch.as_strided(flat, (flat.shape[0], block, RECORD), (flat.stride(0), RECORD, 1))


def _random_moe(moe, g):
    """Valid placeholder values in the MoE's loaded (pre-processing) tensors, as vLLM's numerics
    test `_random_moe` fills them, so vLLM's post-load processing runs on well-formed bytes; the
    processed tensors are then replaced by the UT layer's (VllmNative.load_moe)."""
    u8 = torch.uint8
    for lin in (moe.shared_experts.gate_up_proj, moe.shared_experts.down_proj):
        n = lin.weight.shape[0]
        w = torch.randn(lin.weight.shape, generator=g, device=g.device)
        lin.weight.data.copy_(w.clamp(-6, 6).to(lin.weight.dtype))
        sc = lin.weight_scale.data.view(u8)
        rows = -(-n // 32) if sc.shape[0] == n else sc.shape[0]
        blocks = torch.randint(118, 123, (rows, sc.shape[1]), generator=g, device=g.device, dtype=u8)
        sc.copy_(blocks.repeat_interleave(32, 0)[:n] if sc.shape[0] == n else blocks)
    e = moe.experts.routed_experts
    for name, lo, hi in (("weight", 0, 256), ("weight_scale", 118, 122)):
        for w in ("w13", "w2"):
            t = getattr(e, "%s_%s" % (w, name)).data.view(u8)
            t.copy_(torch.randint(lo, hi, t.shape, generator=g, device=g.device, dtype=u8))
    for t in (moe.gate.weight, moe.gate.e_score_correction_bias):
        t.data.copy_(torch.randn(t.shape, generator=g, device=g.device) * 0.02)


_NATIVE = {}


def vllm_native(world, dev, experts, topk, moe_layer):
    """The process's VllmNative for this MoE (one a process: vLLM registers a module's prefix in
    its forward context once, and its model-parallel state is process-wide; a harness that
    builds a second RankModel in the process reuses it)."""
    key = (world, str(dev), experts, topk, moe_layer)
    if key not in _NATIVE:
        _NATIVE[key] = VllmNative(world, dev, experts, topk, moe_layer)
    return _NATIVE[key]


class VllmNative:
    """vLLM's non-mono decode chain of one layer on this rank (the native path; see the module
    docstring). Built lazily at the first native launch, on every rank in lockstep (vLLM's
    model-parallel groups are created collectively over the harness's gloo world)."""

    def __init__(self, world, dev, experts, topk, moe_layer):
        import torch.distributed as dist
        from vllm._aiter_ops import rocm_aiter_ops

        self.world, self.dev, self.experts, self.topk = world, dev, experts, topk
        # the checkpoint's MXFP4 experts run on AITER's MoE (serving.json env); its flags are read
        # at import, so refresh them (vLLM's numerics test does the same)
        os.environ["VLLM_ROCM_USE_AITER"] = "1"
        os.environ["VLLM_ROCM_USE_AITER_MOE"] = "1"
        if os.environ.get("VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4", "0").strip().lower() not in ("", "0", "false"):
            raise RuntimeError("native: VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4 is set; the served MoE is A8W4")
        rocm_aiter_ops.refresh_env_variables()
        from vllm.config import set_current_vllm_config
        from vllm.distributed import parallel_state as ps
        from vllm.engine.arg_utils import EngineArgs
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
        from vllm.model_executor.layers.mhc import MHCPreDelayedOp
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        from vllm.models.deepseek_v41.amd.model import DeepseekV4MoE
        from vllm.utils.torch_utils import set_default_torch_dtype
        from vllm.v1.worker.workspace import init_workspace_manager

        _, _, _, MD, _ = _mono()
        vc = EngineArgs(model=_config_dir(), tensor_parallel_size=1, moe_backend="aiter", load_format="dummy",
                        max_model_len=65536, language_model_only=True,
                        attention_config={"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": True}
                        ).create_engine_config()
        # this rank's TP shard of the experts: a TP-1 module of moe_intermediate_size / WORLD
        shard = MD.MOE_INTER // world
        hf = vc.model_config.hf_config
        for c in {id(x): x for x in (hf, getattr(hf, "text_config", None)) if x is not None}.values():
            if hasattr(c, "moe_intermediate_size"):
                c.moe_intermediate_size = shard
        cfg = getattr(hf, "text_config", hf)
        self.eps, self.hc_eps, self.iters = float(cfg.rms_norm_eps), float(cfg.hc_eps), int(cfg.hc_sinkhorn_iters)
        with set_current_vllm_config(vc):
            if not ps.model_parallel_is_initialized():
                # TP groups of one rank each over the harness's (already initialized) gloo world
                ps.init_distributed_environment(world_size=dist.get_world_size(), rank=dist.get_rank(),
                                                local_rank=torch.cuda.current_device(), backend="gloo")
                ps.ensure_model_parallel_initialized(1, 1, backend="gloo")
            if ps.get_tensor_model_parallel_world_size() != 1:
                raise RuntimeError("native: vLLM's TP group is not of size 1")
        init_workspace_manager(dev)
        with set_current_vllm_config(vc), set_default_torch_dtype(torch.bfloat16), dev:
            moe = DeepseekV4MoE(vc, prefix="model.layers.%d.ffn" % moe_layer)
            self.mhc = MHCPreDelayedOp()
        if moe.shared_experts is None or moe.n_routed_experts != experts or moe.n_activated_experts != topk:
            raise RuntimeError("native: DeepseekV4MoE is not the served MoE (shared %s, %d / %d)"
                               % (moe.shared_experts is not None, moe.n_routed_experts, moe.n_activated_experts))
        _random_moe(moe, torch.Generator(device=dev).manual_seed(WEIGHT_SEED))
        process_weights_after_loading(moe, vc.model_config, dev)
        e = moe.experts.routed_experts
        if getattr(e.quant_method, "mxfp4_backend", None) != Mxfp4MoeBackend.AITER_MXFP4_BF16:
            raise RuntimeError("native: MoE backend %s, the served one is AITER_MXFP4_BF16"
                               % getattr(e.quant_method, "mxfp4_backend", None))
        if getattr(moe.shared_experts, "_gateup_scale", None) is not None:
            raise RuntimeError("native: the shared gate_up was B-preshuffled at load; the served layer reads it row-major")
        self.vc, self.mod, self.loaded = vc, moe, None
        self.ties = []

    def begin(self, model):
        """A RankModel starts using this chain: its layers' tensors are not loaded yet."""
        self.loaded = None

    # ------------------------------------------------------------------ the MoE's tensors
    def load_moe(self, i, w):
        """Replace the MoE's processed tensors by UT layer i's deployed ones (the tensors the mono
        runner reads are vLLM's processed ones: mono_decode.MonoDecodeLayer.weights)."""
        if self.loaded == (id(w), i):
            return
        moe = self.mod
        e, sh = moe.experts.routed_experts, moe.shared_experts
        pairs = (("gate.weight", moe.gate.weight, w.gate_w), ("bias", moe.gate.e_score_correction_bias, w.bias),
                 ("w13", e.w13_weight, w.w13), ("w13_s", e.w13_weight_scale, w.w13_s),
                 ("w2", e.w2_weight, w.w2), ("w2_s", e.w2_weight_scale, w.w2_s),
                 ("sgu", sh.gate_up_proj.weight, w.sgu), ("sgu_s", sh.gate_up_proj.weight_scale, w.sgu_s),
                 ("sw2", sh.down_proj.weight, w.sw2), ("sw2_s", sh.down_proj.weight_scale, w.sw2_s))
        with torch.no_grad():
            for name, param, ours in pairs:
                dst = param.data
                if dst.element_size() == 1:
                    dst, src = dst.view(torch.uint8), ours.view(torch.uint8)
                else:
                    src = ours
                if tuple(dst.shape) != tuple(src.shape) or dst.dtype != src.dtype:
                    raise RuntimeError("native: vLLM's processed %s is %s %s, the deployed one %s %s"
                                       % (name, tuple(dst.shape), dst.dtype, tuple(src.shape), src.dtype))
                dst.copy_(src)
        self.loaded = (id(w), i)

    def _moe_call(self, x):
        from vllm.forward_context import set_forward_context

        with set_forward_context(None, self.vc):
            ids = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)   # text tokens
            # cloned: the MoE may hand back a buffer of its own that the next call reuses
            return self.mod(x.clone(), ids).clone()

    def _scores(self, x):
        """vLLM's routing scores (DeepseekV4MoE: sqrtsoftplus, e_score_correction_bias for selection)."""
        logits = x.float() @ self.mod.gate.weight.float().T
        return torch.nn.functional.softplus(logits).sqrt()

    def moe(self, i, xf, normed_k):
        """This rank's MoE partial [rows, 5120] bf16 of the native FFN input xf, near-tie routing
        aligned with the kernel's (module docstring)."""
        bias = self.mod.gate.e_score_correction_bias
        s_ref = self._scores(xf) + bias.float()
        s_k = self._scores(normed_k) + bias.float()
        ids_ref = s_ref.topk(self.topk, dim=1).indices
        ids_k = s_k.topk(self.topk, dim=1).indices
        out = self._moe_call(xf)
        adopted, beyond = [], 0
        for t in range(xf.shape[0]):
            ref, got = set(ids_ref[t].tolist()), set(ids_k[t].tolist())
            if ref == got:
                continue
            bound = s_ref[t, ids_ref[t]].min().item()
            worst = s_ref[t, sorted(got - ref)].min().item()
            if worst >= bound - TIE_TOL:
                adopted.append(t)
            else:
                beyond += 1
        if adopted:
            saved = bias.data.clone()
            try:
                for t in adopted:
                    bias.data.copy_(saved)
                    bias.data[ids_k[t]] += FORCE_BIAS
                    out[t:t + 1] = self._moe_call(xf[t:t + 1].contiguous())
            finally:
                bias.data.copy_(saved)
        self.ties.append({"launch": i, "tokens": int(xf.shape[0]), "near_tie_adopted": len(adopted),
                          "beyond_tie": beyond})
        return out

    # ------------------------------------------------------------------ the attention half
    def seam(self, w, sub, residual, x, post, comb, pre):
        """vLLM's delayed mHC seam (DeepseekV4DecoderLayer.forward: mhc_pre with the carried
        pre-mix, the owed post folded in when x is given, the following norm fused) ->
        (residual, post, comb, layer input, next pre-mix)."""
        fold = x is not None
        return self.mhc(residual, getattr(w, "hc_%s_fn" % sub), getattr(w, "hc_%s_scale" % sub),
                        getattr(w, "hc_%s_base" % sub), self.eps, self.hc_eps, self.hc_eps, 2.0, self.iters,
                        pre_mix=pre, sublayer_out=x if fold else None, post_layer_mix=post if fold else None,
                        comb_res_mix=comb if fold else None, norm_weight=getattr(w, "%s_norm" % sub),
                        norm_eps=self.eps)

    def attention(self, a, xa, m, swa_view, comp_view, topk, comp_entries):
        """This rank's unreduced wo_b output [rows, 5120] bf16, and the step's records inserted
        into swa_view (vLLM's fused insert)."""
        import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import rocm_mxfp8_block32_gemm
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
        from vllm.models.common.ops import fused_q_kv_rmsnorm
        from vllm.models.deepseek_v41.amd.rocm import compute_global_topk_ragged_indices_and_indptr
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
            build_ragged_indices_from_dense, rocm_inverse_rope_mxfp8_rows)

        _, _, AP, _, _ = _mono()
        d = AP.Dims(self.world)
        bf = torch.bfloat16
        M, H, G = xa.shape[0], d.heads, d.groups
        x8, xs = mxfp8_e4m3_quantize(xa)
        qr_kv = rocm_mxfp8_block32_gemm(x8, xs, a.wqkv, a.wqkv_scale, bf)
        qr, kv = fused_q_kv_rmsnorm(qr_kv[:, :AP.Q_RANK], qr_kv[:, AP.Q_RANK:], a.q_norm, a.kv_norm, self.eps)
        q8, qs = mxfp8_e4m3_quantize(qr)
        q = rocm_mxfp8_block32_gemm(q8, qs, a.wq_b, a.wq_b_scale, bf).view(M, H, AP.HEAD_DIM)
        swa2d = swa_view.view(swa_view.shape[0], -1)
        q_roped = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q, kv, swa2d, m["slot_mapping"], m["positions"], a.cos_sin, H, self.eps, SWA_BLOCK,
            False, False, True, False)
        swa_ind, swa_ptr = build_ragged_indices_from_dense(m["swa_indices"].reshape(M, -1), m["swa_lens"])
        top_ind = top_ptr = None
        if comp_view is not None:
            valid = torch.ones(M, dtype=torch.bool, device=xa.device)
            top_ind, top_ptr, _ = compute_global_topk_ragged_indices_and_indptr(
                topk, m["token_to_req"], m["comp_block_table"], comp_entries, valid)
        attn = torch.empty(M, H, AP.HEAD_DIM, dtype=bf, device=xa.device)
        rocm_aiter_ops.triton_sparse_mla_fwd(q_roped, swa_view, attn, AP.HEAD_DIM ** -0.5, swa_ptr, swa_ind,
                                             attn_sink=a.attn_sink, extra_kv_buffer=comp_view,
                                             extra_kv_indptr=top_ptr, extra_kv_indices=top_ind)
        a8 = torch.empty(M, H * AP.HEAD_DIM, dtype=torch.float8_e4m3fn, device=xa.device)
        a_s = torch.empty(M, H * AP.HEAD_DIM // 32, dtype=torch.uint8, device=xa.device)
        rocm_inverse_rope_mxfp8_rows(attn, m["positions"], a.cos_sin, AP.ROPE, a8, a_s)
        z = torch.ops.vllm.rocm_dsv41_mxfp8_wo_a_bmm(a8, a_s, a.wo_a, a.wo_a_scale, G, AP.O_RANK)
        z8, zs = mxfp8_e4m3_quantize(z)
        return rocm_mxfp8_block32_gemm(z8, zs, a.wo_b, a.wo_b_scale, bf).contiguous()


class RankModel:
    # ---- frozen by the scaffold from run/ut_contract.json; change only with the contract
    WORLD = 2                  # ranks (const axis world)
    LAUNCHES = 5               # layer launches a call() step makes (const axis launches)
    POOLS = 2                  # KV pools (const axis pools): 0 SWA cache (written), 1 compressed (read only)
    HIDDEN = 5120
    TOP_K = 6                  # routing ids per token (layer_kind routed_moe), None for dense
    KV_WIDTH = 512             # elements of one token's record, decoded to float32 (kv_decode)
    STREAM_DTYPE = torch.bfloat16
    INPUT_DTYPE = torch.bfloat16
    STREAM_SHAPE = (4, 5120)        # mhc: [rows, hc, hidden] (contract stream_layout)
    STREAM_AUX = (24, torch.float32)  # mhc: post (hc) | comb (hc x hc) | pre (hc) a row; launch() takes and returns it
    # ---- paths: PATHS[0] is the deployed one call() runs; every other entry is an oracle path
    # (cases.ORACLES[name] runs the contract oracle whose `path` names it)
    PATHS = ('deployed', 'split', 'native')
    PACKAGE = 'vllm.models.deepseek_v41.amd.mono'   # the whole-package bind: its bytes key the JIT cache
    JIT_CACHE_ENV = "FLYDSL_RUNTIME_CACHE_DIR"   # the JIT's cache-dir variable; None if it has none
    # never poisoned: ready flags / plain-word bitmasks (x8rdy is atomic_or'ed), the MoE's queue and
    # arrival counters, the tuned shared-down tile ranks (layer.scratch_layout region names)
    KEEP_REGIONS = r"rdy|ugq|dq|dpb|flag|trace"
    NOTES = [
        "launches = 5 layer types: r0 fold 0 (served 1), r2 (3), r1 (21), r2 index (8), r1 index (24); "
        "K1 r2 f0 i1 (served layer 14 only) is waived",
        "index layers: select copies a precomputed top-512 into the step's top-k buffer (vLLM's indexer stands in)",
        "split = the deployed call on the baseline overlay (no per-stage mode in DSV41MonoLayer)",
        "native = vLLM's non-mono chain (mHC seams, ROCm attention ops, all-reduce over gloo, DeepseekV4MoE at TP 1 "
        "on this rank's shard), teacher-forced per launch, near-tie routing aligned; bounds provisional",
        "routing ids are not exposed by DSV41MonoLayer (they live in scratch mailbox pairs): ids = not_produced",
        "no device deadline / error words: a wait that never completes is caught only by the gloo timeout",
        "no compile-only mode: the first launch of a (width, ratio, fold, index) JIT-compiles on every rank",
        "stream_aux (the first layer's incoming mixes) is uniform [0, 1), not a Sinkhorn mix; every later layer takes the kernel's own",
        "compressed caches: random valid records, one a layer; topk_indices: min((pos + 1) // ratio, 512) distinct "
        "random rows a token, the rest -1",
        "RoPE tables: vLLM build_deepseek_v4_rope over the checkpoint config.json, per compress ratio",
    ]

    @staticmethod
    def page_width(context):
        """comp_block_table columns a request of `context` keys needs (CACHE_BLOCK tokens a page)."""
        return (int(context) + CACHE_BLOCK - 1) // CACHE_BLOCK + 1

    def __init__(self, rank, dev, group, n_layers=None):
        """call() step (n_layers None): the LAYER_TYPES layers; rank_chain: n_layers layers, layer 0
        call()'s layer 0 then CHAIN_PATTERN. The runner is built as mono_decode._mono_runner builds
        it (one a rank, shared by every layer): DSV41MonoLayer(tp, rank, cpu_group, device)."""
        R, L, AP, MD, SEAM = _mono()
        assert (AP.HIDDEN, L.MAX_TOKENS, AP.WINDOW, AP.TOPK) == (self.HIDDEN, 48, 128, 512)
        assert (R.HC, self.STREAM_SHAPE[0], self.STREAM_AUX[0]) == (4, 4, 4 * 6)
        assert (R.EXPERTS, R.TOPK) == (384, TOPK)
        self.rank, self.dev, self.group = rank, dev, group
        self.n = len(LAYER_TYPES) if n_layers is None else int(n_layers)
        self.types = [layer_type(j, n_layers) for j in range(self.n)]
        self.path = self.PATHS[0]
        self.runner = R.DSV41MonoLayer(self.WORLD, rank, group, dev)
        self.weights = [make_weights(dev, rank, self.WORLD, j, t, _cos_sin(dev, t[0])) for j, t in enumerate(self.types)]
        swa, comp = zip(*[make_caches(dev, j, t[0]) for j, t in enumerate(self.types)])
        # caches: the n SWA caches (pool 0, layer l's K1 writes cache l), then the compressed
        # caches of the ratio > 0 layers (pool 1, read only)
        self.caches = [(t,) for t in swa] + [(t,) for t in comp if t is not None]
        self.swa_views = [record_view(t, SWA_BLOCK) for t in swa]
        self.comp_views = [record_view(t, CACHE_BLOCK // self.types[j][0]) if t is not None else None
                           for j, t in enumerate(comp)]
        self._built = set()
        self._native = None
        self._teacher = None

    # ------------------------------------------------------------------ layers and paths
    def pool_of(self, layer):
        return SWA if layer < self.n else COMP

    def writes(self, path, i):
        """Launch i writes UT layer i's SWA records on every path (native: the teacher's K1, then
        vLLM's insert, into the same slots)."""
        return (i,)

    def set_path(self, path):
        if path not in self.PATHS:
            raise ValueError("unknown path %r" % (path,))
        self.path = path

    # ------------------------------------------------------------------ a step's metadata
    def step_meta(self, rows, context, width=None):
        """A decode step of rows / 6 requests (1 + 5 DSpark tokens each), request r with
        max(7, context - 37 r) keys including the step's 6 rows; as vLLM's builders lay it out:
        positions, the SWA slot of each token over a shuffled SWA pool, each token's causal
        128-key window (oldest first, -1 padded) and its length, token -> request, the shuffled
        compressed-cache block table, per compress ratio a top-512 of the token's compressed rows
        (topk_r*: the earlier index source's, which a non-source layer reads) and another one
        (sel_r*: what an index source's select writes), and topk_buf (all -1: the buffer select
        fills). The same on every rank (the TP ranks share one batch)."""
        if rows % ROWS_PER_REQ or not 1 <= rows <= 48:
            raise ValueError("rows %d: decode steps are requests x %d rows, at most 48" % (rows, ROWS_PER_REQ))
        Rq = rows // ROWS_PER_REQ
        lens = [max(ROWS_PER_REQ + 1, int(context) - 37 * r) for r in range(Rq)]
        if max(lens) > MAX_CONTEXT:
            raise ValueError("context %d > MAX_CONTEXT %d (the rope table's size)" % (max(lens), MAX_CONTEXT))
        if sum(self.page_width(L) for L in lens) > COMP_POOL_PAGES - 1:
            raise ValueError("%d requests x %d keys exceed the compressed pool (%d pages)"
                             % (Rq, int(context), COMP_POOL_PAGES - 1))
        g = torch.Generator(device="cpu").manual_seed(7919 * rows + int(context))
        swa_perm = (torch.randperm(SWA_POOL_PAGES - 1, generator=g) + 1).tolist()
        comp_perm = (torch.randperm(COMP_POOL_PAGES - 1, generator=g) + 1).tolist()
        pos, t2r, slots = [], [], []
        swa_idx = torch.full((rows, 1, 128), -1, dtype=torch.int32)
        swa_len = torch.zeros(rows, dtype=torch.int32)
        pages = {}

        def slot(r, p):
            key = (r, p // SWA_BLOCK)
            if key not in pages:
                pages[key] = swa_perm.pop()
            return pages[key] * SWA_BLOCK + p % SWA_BLOCK

        for r in range(Rq):
            for j in range(ROWS_PER_REQ):
                p = lens[r] - ROWS_PER_REQ + j
                pos.append(p)
                t2r.append(r)
                slots.append(slot(r, p))
        for i, p in enumerate(pos):
            win = range(max(p - 127, 0), p + 1)
            swa_idx[i, 0, :len(win)] = torch.tensor([slot(t2r[i], q) for q in win], dtype=torch.int32)
            swa_len[i] = len(win)
        # as served: [requests, 8192] (wider only if a width asks for it)
        cols = max(COMP_BT_COLS, width or 0, max(self.page_width(L) for L in lens))
        comp_bt = torch.zeros(Rq, cols, dtype=torch.int32)
        for r in range(Rq):
            n = self.page_width(lens[r])
            comp_bt[r, :n] = torch.tensor([comp_perm.pop() for _ in range(n)], dtype=torch.int32)
        m = dict(positions=torch.tensor(pos, dtype=torch.int64),
                 slot_mapping=torch.tensor(slots, dtype=torch.int64),
                 swa_indices=swa_idx, swa_lens=swa_len,
                 token_to_req=torch.tensor(t2r, dtype=torch.int32),
                 comp_block_table=comp_bt,
                 topk_buf=torch.full((rows, 512), -1, dtype=torch.int32))
        for ratio in (1, 2):
            for name in ("topk_r%d" % ratio, "sel_r%d" % ratio):
                topk = torch.full((rows, 512), -1, dtype=torch.int32)
                for i, p in enumerate(pos):
                    cand = (p + 1) // ratio
                    n = min(cand, 512)
                    topk[i, :n] = torch.randperm(cand, generator=g)[:n].to(torch.int32)
                m[name] = topk
        # DESIGN 1.5: shuffled, non-identity maps; the SWA and compressed tables differ
        assert bool((m["slot_mapping"] != m["positions"]).any()), "identity SWA slot map"
        assert sorted(set(pages.values())) != list(range(1, len(pages) + 1)), "SWA pages not shuffled"
        n0 = self.page_width(lens[0])
        assert not torch.equal(comp_bt[0, :n0].long(), comp_bt[0, 0].long() + torch.arange(n0)), "compressed pages not shuffled"
        m = {k: v.to(self.dev) for k, v in m.items()}
        assert m["comp_block_table"].data_ptr() != m["swa_indices"].data_ptr()
        return m

    def kv_targets(self, m):
        """Pool 0: the step's SWA slots (K1 writes each token's record there); pool 1: none."""
        return (m["slot_mapping"], m["slot_mapping"][:0])

    def kv_bytes(self, layer, t, idx):
        """A slot's 584 record bytes in the flat SWA cache [pages, PAGE_STRIDE]: its 576-B data
        row at page * stride + off * 576, then its 8-B scale word at page * stride + 32 * 576 +
        off * 8 (front.py's insert, mono_decode._record_view). int64 [rows, 584], graph-safe."""
        stride = t.stride(0)
        page, off = idx // SWA_BLOCK, idx % SWA_BLOCK
        d0 = page * stride + off * DATA
        s0 = page * stride + SWA_BLOCK * DATA + off * 8
        ar_d = torch.arange(DATA, device=idx.device, dtype=torch.int64)
        ar_s = torch.arange(8, device=idx.device, dtype=torch.int64)
        return torch.cat([d0[:, None] + ar_d, s0[:, None] + ar_s], dim=1)

    def kv_decode(self, layer, raw):
        """A record's 512 values: dims 0..447 e4m3 x 2^(UE8M0 - 127) a 64-dim block, dims
        448..511 bf16 (the rope dims). The 8th scale byte is written 0 (front.py: w1's top
        byte); anything else marks the row NaN. A poisoned (0x7F) e4m3 byte decodes as NaN."""
        n = raw.shape[0]
        nope = raw[:, :NOPE].contiguous().view(torch.float8_e4m3fn).float().view(n, 7, 64)
        sc = raw[:, DATA:DATA + 7].to(torch.int32)
        scale = torch.exp2((sc - 127).float()).masked_fill(sc == 255, float("nan"))   # E8M0 0xFF = NaN
        vals = torch.cat([(nope * scale[..., None]).view(n, NOPE),
                          raw[:, NOPE:DATA].contiguous().view(torch.bfloat16).float()], dim=1)
        bad = raw[:, DATA + 7] != 0
        return torch.where(bad[:, None], torch.full_like(vals, float("nan")), vals)

    def bind(self, m):
        """Allocate the step width's scratch (a width's first step is eager, as in vLLM) for every
        ratio of the step's layers, so the harness can lay it out and poison it; the kernels
        JIT-compile at their first launch (no compile-only mode). Returns the step metadata."""
        rows = int(m["positions"].numel())
        if not torch.cuda.is_current_stream_capturing():
            for t in self.types:
                key = (rows, t[0])
                if key not in self._built:
                    self.runner.scratch(rows, self.runner.tuned(rows, t[0]))
                    self._built.add(key)
        return m

    # ------------------------------------------------------------------ launches
    @staticmethod
    def _mixes(rows, aux):
        post = aux[:, :4].reshape(rows, 4, 1).contiguous()
        comb = aux[:, 4:20].reshape(rows, 4, 4).contiguous()
        pre = aux[:, 20:24].contiguous()
        return post, comb, pre

    def _deployed(self, i, rows, m, stream, x, aux):
        """mono_decode.py:726's call for layer i: keywords and argument types as served (x bf16
        [rows, 5120], residual bf16 [rows, 4, 5120], post [rows, 4, 1] / comb [rows, 4, 4] / pre
        [rows, 4] f32 contiguous, outs not passed). A fold-0 layer (the Engram layer) takes the
        settled residual and the carried pre-mix; post / comb are passed but unread. An index
        source passes select, which writes the top-k K2 reads. Returns (residual, out, mixes)."""
        w = self.weights[i]
        ratio, fold, index, _ = self.types[i]
        post, comb, pre = self._mixes(rows, aux)
        kw = {}
        if ratio:
            kw.update(topk_indices=m["topk_buf"] if index else m["topk_r%d" % ratio],
                      comp_cache=self.comp_views[i], comp_block_table=m["comp_block_table"])
        if index:
            buf, sel = m["topk_buf"], m["sel_r%d" % ratio]
            kw["select"] = lambda normed, qr: buf.copy_(sel)
        out, res, post_o, comb_o, pre_o = self.runner.forward(
            w, x, stream, post, comb, pre, m["positions"], m["slot_mapping"], self.swa_views[i],
            m["swa_indices"], m["swa_lens"], m["token_to_req"], fold=fold, **kw)
        mix = torch.cat([post_o.reshape(rows, 4), comb_o.reshape(rows, 16), pre_o.reshape(rows, 4)], dim=1)
        return res, out, mix

    def launch(self, i, rows, m, fb, stream, x, aux):
        if self.path == "native":
            return self._native_launch(i, rows, m, stream, x, aux)
        return self._deployed(i, rows, m, stream, x, aux)

    def end_step(self):
        """Nothing: the kernels move their epoch themselves (runner.py:12, no host step hook)."""

    # ------------------------------------------------------------------ the native path
    def _all_reduce(self, t):
        """fp32 sum of the ranks' bf16 partials through the gloo group, rounded once to bf16."""
        import torch.distributed as dist

        h = t.float().cpu().contiguous()
        dist.all_reduce(h, op=dist.ReduceOp.SUM, group=self.group)
        return h.to(self.dev).to(torch.bfloat16)

    def _kernel_ffn_input(self, i, rows):
        """The FFN input K2 just used: its scratch `normed` region (layer.scratch_layout), as
        vLLM's numerics test reads it."""
        R, L, AP, MD, SEAM = _mono()
        r = self.runner
        tuned = bool(r.tuned(rows, self.types[i][0]))
        off = L.scratch_layout(rows, r.tp, r.experts, r.topk, r.ug_groups, tuned)["normed"][0]
        return r.scratch(rows, tuned)[off:off + rows * self.HIDDEN * 2].view(torch.bfloat16).view(rows, self.HIDDEN).clone()

    def _native_launch(self, i, rows, m, stream, x, aux):
        """Teacher-forced (module docstring): the deployed launch i on the deployed state first,
        then vLLM's chain from that same state; returns the native (residual, out, mixes)."""
        if self._native is None:
            self._native = vllm_native(self.WORLD, self.dev, 384, TOPK, moe_layer=3)
            self._native.begin(self)
        V = self._native
        if i == 0:
            self._teacher = (stream.detach().clone(), x.detach().clone(), aux.detach().clone())
        t_res, t_x, t_aux = self._teacher
        d_res, d_out, d_mix = self._deployed(i, rows, m, t_res.clone(), t_x.clone(), t_aux.clone())
        normed_k = self._kernel_ffn_input(i, rows)
        self._teacher = (d_res.detach().clone(), d_out.detach().clone(), d_mix.detach().clone())
        w = self.weights[i]
        ratio, fold, index, _ = self.types[i]
        post, comb, pre = self._mixes(rows, t_aux)
        with torch.no_grad():
            if fold:
                res, post_a, comb_a, xa, attn_pre = V.seam(w, "attn", t_res, t_x, post, comb, pre)
            else:   # the Engram layer: the residual arrives settled, the pre-mix carried in
                res, post_a, comb_a, xa, attn_pre = V.seam(w, "attn", t_res, None, None, None, pre)
            topk = (m["sel_r%d" % ratio] if index else m["topk_r%d" % ratio]) if ratio else None
            part = V.attention(w.attn, xa, m, self.swa_views[i], self.comp_views[i], topk,
                               CACHE_BLOCK // ratio if ratio else 0)
            att = self._all_reduce(part)
            res2, post2, comb2, xf, ffn_pre = V.seam(w, "ffn", res, att, post_a, comb_a, attn_pre)
            V.load_moe(i, w)
            out = self._all_reduce(V.moe(i, xf, normed_k))
        mix = torch.cat([post2.reshape(rows, 4).float(), comb2.reshape(rows, 16).float(),
                         ffn_pre.reshape(rows, 4).float()], dim=1)
        return res2.contiguous(), out.contiguous(), mix.contiguous()

    def buffers(self):
        """produced: the runner's K1 -> K2 hand-off buffers (NaN before each launch: a K2 that
        reads one K1 did not write reads NaN). No ids buffer (DSV41MonoLayer keeps its routing in
        scratch mailbox pairs); no buffer is read across launches (the stream is passed)."""
        r = self.runner
        return {"ids": None,
                "produced": {n: getattr(r, n, None) for n in ("q", "res_mid", "post_a", "comb_a", "pre_a")},
                "inputs": {}, "diag": {}}

    def shared_layouts(self, rows):
        """The width's scratches (one a tuned flag, layer.scratch_layout) and the symmetric peer
        buffer (two parities of the attention then the MoE partials: layer.peer_half_bytes,
        seam.attn_peer_bytes). Read from the subject's modules; a candidate may move them."""
        r = self.runner
        specs, kept = [], []
        try:
            R, L, AP, MD, SEAM = _mono()
            for tuned in sorted({bool(r.tuned(rows, t[0])) for t in self.types}):
                specs.append(("scratch_s%d%s" % (rows, "_tuned" if tuned else ""), r._scratch.get((rows, tuned)),
                              (lambda t: lambda: L.scratch_layout(rows, r.tp, r.experts, r.topk, r.ug_groups, t))(tuned)))
            half = L.peer_half_bytes(r.tp)
            attn = SEAM.attn_peer_bytes(L.MAX_TOKENS, r.tp)
            specs.append(("peer", getattr(getattr(r, "peer", None), "bytes", None),
                          lambda: {"attn_p0": (0, attn), "ffn_p0": (attn, half - attn),
                                   "attn_p1": (half, attn), "ffn_p1": (half + attn, half - attn)}))
        except Exception as e:  # noqa: BLE001
            kept.append({"region": "scratch, peer", "why": "layout unavailable: %r" % e})
        kept.append({"region": "epoch", "why": "the epoch word, the CTA marks and the MoE's counters"})
        return specs, kept

    # ------------------------------------------------------------------ health
    def device_errors(self):
        """None: the DSv4.1 kernels have no deadline and no error words (a stuck wait is caught
        by the harness's gloo timeout only)."""
        return None

    def facts(self):
        r = self.runner
        return {"layers": self.n, "types": [list(t) for t in self.types],
                "swa_pages": SWA_POOL_PAGES, "comp_pages": COMP_POOL_PAGES, "page_stride": PAGE_STRIDE,
                "swa_block": SWA_BLOCK, "tp": r.tp, "kernels": [str(k) for k in r._kernels],
                "native_ties": list(self._native.ties) if self._native is not None else None}

    def close(self):
        """Release the peer buffer (IPC handles)."""
        close = getattr(getattr(self.runner, "peer", None), "close", None)
        if close:
            close()
