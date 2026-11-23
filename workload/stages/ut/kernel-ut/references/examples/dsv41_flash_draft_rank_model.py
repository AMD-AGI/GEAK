"""rank_model.py of task `dsv41_mono_draft_layer` (UT contract family `mono_layer`): the
model-specific half of the rank harness for the DeepSeek-V4.1-Flash DSpark DRAFT layers (vLLM
--monokernel, MI355X gfx950, TP2). WORKED EXAMPLE for references/mono-layer-ut.md.

Provenance (2026-11-16): written WITHOUT a GPU from the target example
(references/examples/dsv41_flash_target_rank_model.py, whose deployed path is GPU-verified) and
the installed vLLM = vllm-project/vllm 4fab5a07 (vllm/models/deepseek_v41/amd/{dspark.py,
mono_decode.py, mono/runner.py}, vllm/v1/attention/backends/mla/{sparse_swa.py,
compressor_utils.py}); GPU-verified 2026-11-17 (references/mono_models.json oracles_verified).
The helpers above RankModel (weights, records, VllmNative) are the target example's, verbatim.

Entry under test: `vllm.models.deepseek_v41.amd.mono.runner:DSV41MonoLayer.forward`, on the
DSpark draft's own runner. Live call site: the draft model DSparkDeepseekV4Model.forward
(vllm/models/deepseek_v41/amd/dspark.py:188) broadcasts the input embeddings to the hc copies
([T, 4, 5120]) and runs its num_nextn_predict_layers = 3 DeepseekV4DecoderLayer's (served layers
40, 41, 42: compress ratio 0, no indexer) in order; each one's DeepseekV4DecoderLayer.forward ->
mono_decode.py:650 `MonoDecodeLayer.__call__` (draft=True, created by MonoDecodeLayer._create:
experts / topk = config dspark_n_routed_experts / dspark_num_experts_per_tok = 128 / 3,
swa_width = get_dspark_swa_index_width(sliding_window 128, num_speculative_tokens 5) = 192) ->
`_mono_runner(x.device, self)` = the process's one draft runner DSV41MonoLayer(tp, rank,
get_tp_group().cpu_group, device, experts=128, topk=3, ug_groups=DRAFT_UG_GROUPS (1),
swa_width=192) (mono_decode.py:80), shared by the three draft layers -> `runner.forward(
self.weights(layer), x, residual, post_mix, res_mix, pre_mix, positions, swa.slot_mapping,
_record_view(attn.swa_cache_layer.kv_cache, swa.block_size), swa.decode_swa_indices,
swa.decode_swa_lens, swa.token_to_req_indices, fold=fold)` (mono_decode.py:726; ratio 0: no
compressed cache, no top-k, no select). The first draft layer (residual None, x 3-D) takes
`residual, x = x, x[:, 0].contiguous()`, pre_mix = runner.first_pre_mix(rows) (stream 0) and
post_mix = res_mix = pre_mix (unread without the fold), fold=False; the next two fold=True on
the previous layer's outputs. Device launches: K1 aiter::dsv41_mono_k1_s<M>_tp2_r0_tl0_f<0|1>_i0
_e128_k3_g1_w192, K2 aiter::dsv41_mono_k2_s<M>_tp2_r0_tl0_i0_e128_k3_g1_w192. Every TP rank makes
the same calls in the same order; no end-of-step hook.

A step: rows = requests x DSpark block (num_speculative_tokens = dspark_block_size = 5; widths 5,
10, 20, 40 observed, at most 8 requests). The block is non-causal: vLLM's DSpark SWA metadata
(sparse_swa.py ComputeDSparkNoncausalSWAIndicesKernel) gives every token of a request the same
keys, the 128 context keys before the block plus the whole block (its own 5 rows, future ones
included: K1 inserts all of them before K2 attends), -1 padded to the width 192. The context's
records are random valid fp8_ds_mla records (served: precompute_and_store_context_kv from the
target's hidden states).

call() = LAUNCHES = 3 UT layers = the draft's three layers: 0 fold 0 (served 40), 1 and 2 fold 1
(41, 42): every expected draft build runs. rank_chain: layer 0 fold 0 (call()'s layer 0,
untimed: rank_chain subtracts step(1)), then fold-1 layers (the served 2 of 3).

Paths: deployed / split (the same deployed call on the baseline overlay: no per-stage mode) /
native = vLLM's non-mono chain exactly as in the target example (VllmNative): the first layer's
seam as vLLM's draft runs it (pre_mix None = stream 0, no fold), no compressed keys, the
DSpark DeepseekV4MoE (prefix layer 40: 128 experts, top-3) at TP 1 on this rank's shard,
teacher-forced per launch, near-tie routing aligned (top-3). Bounds: the family's native SQNR
floors, PROVISIONAL until the first GPU floor run.

Rules (references/mono-layer-ut.md section 2): as the target example (TP shards from a rank
seed, replicated tensors from a common seed, norms asserted distinct, index dtypes as served,
the SWA pages behind a shuffled non-identity map; launch i writes UT layer i's block records
through kv_bytes / kv_decode; one pool: the draft layers read no compressed cache).
"""

import os

import torch

WEIGHT_SEED = 20261116
DRAFT_BLOCK = 5             # rows a request: DSpark num_speculative_tokens = config dspark_block_size
MAX_ROWS = 40               # 8 requests (--max-num-seqs) x 5
SWA_WIDTH = 192             # get_dspark_swa_index_width(128, 5): the draft runner's swa_width (the w192 builds)
WINDOW = 128                # config sliding_window: the context keys before the block
SWA_BLOCK = 32              # the SWA cache's records a page (attention.py DeepseekV4SWACache block_size=32)
CACHE_BLOCK = 128           # --block-size (unused by the ratio-0 draft layers; VllmNative's signature)
PAGE_STRIDE = 210240        # the deployed physical page stride (profile captures; one page size a KV group)
RECORD, DATA, NOPE = 584, 576, 448
MAX_CONTEXT = 524288        # rope table rows: positions are the target's (cases up to 475426 keys a request)
SWA_POOL_PAGES = 256        # >> 8 requests x 6 pages (133 keys a request), shuffled
COMP_POOL_PAGES = 1         # no compressed cache (make_caches is not asked for one)
SWA = 0                     # contract axis `pools` = 1: the layer's SWA cache
MODEL_CONFIG_DIR = "/shared_nfs/models/DeepSeek-V4.1-Flash"   # config.json (no weights read)
EXPERTS, TOPK = 128, 3      # config dspark_n_routed_experts / dspark_num_experts_per_tok

# (compress ratio, fold, index source, the served layer the type stands for)
F0, F1A, F1B = (0, False, False, 40), (0, True, False, 41), (0, True, False, 42)
LAYER_TYPES = (F0, F1A, F1B)
TIE_TOL = 5e-3              # native: a top-3 pick within this of the boundary may flip (provisional)
FORCE_BIAS = 1.0e4          # native: added to e_score_correction_bias to force a token's picks


def layer_type(layer, n_layers=None):
    """UT layer `layer`'s (ratio, fold, index, served layer): call()'s step is the draft's three
    layers; a chain's layer 0 is call()'s layer 0, then fold-1 layers (served 41, 42, 41, ...)."""
    if n_layers is None:
        return LAYER_TYPES[layer]
    return F0 if layer == 0 else (F1A, F1B)[(layer - 1) % 2]


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
    LAUNCHES = 3               # layer launches a call() step makes (const axis launches): the draft's 3 layers
    POOLS = 1                  # KV pools (const axis pools): 0 the layer's SWA cache
    HIDDEN = 5120
    TOP_K = 3                  # routing ids per token (layer_kind routed_moe): dspark_num_experts_per_tok
    KV_WIDTH = 512             # elements of one token's record, decoded to float32 (kv_decode)
    STREAM_DTYPE = torch.bfloat16
    INPUT_DTYPE = torch.bfloat16
    STREAM_SHAPE = (4, 5120)        # mhc: [rows, hc, hidden] (contract stream_layout)
    STREAM_AUX = (24, torch.float32)  # mhc: post (hc) | comb (hc x hc) | pre (hc) a row; launch() takes and returns it
    # ---- paths: PATHS[0] is the deployed one call() runs; every other entry is an oracle path
    PATHS = ('deployed', 'split', 'native')
    PACKAGE = 'vllm.models.deepseek_v41.amd.mono'   # the whole-package bind: its bytes key the JIT cache
    JIT_CACHE_ENV = "FLYDSL_RUNTIME_CACHE_DIR"   # the JIT's cache-dir variable; None if it has none
    KEEP_REGIONS = r"rdy|ugq|dq|dpb|flag|trace"
    NOTES = [
        "the DSpark draft's 3 layers (served 40, 41, 42; compress ratio 0): layer 0 fold 0 (the broadcast "
        "embedding: residual = the stream, x = its stream 0, pre-mix = stream 0), layers 1, 2 fold 1",
        "the stream entering the step is random per hc copy (served: the embedding's 4 equal copies); the "
        "first layer reads stream 0 as its input on every path",
        "non-causal block: every token of a request attends to the 128 context keys before the block plus the "
        "whole block (sparse_swa.py DSpark indices), -1 padded to 192",
        "split = the deployed call on the baseline overlay (no per-stage mode in DSV41MonoLayer)",
        "native = vLLM's non-mono chain (mHC seams, ROCm attention ops, all-reduce over gloo, the DSpark "
        "DeepseekV4MoE 128 / 3 at TP 1 on this rank's shard), teacher-forced, near-tie routing aligned; bounds provisional",
        "routing ids are not exposed by DSV41MonoLayer: ids = not_produced",
        "no device deadline / error words: a wait that never completes is caught only by the gloo timeout",
        "context records: random valid fp8_ds_mla records (served: written by precompute_and_store_context_kv)",
    ]

    @staticmethod
    def page_width(context):
        """SWA page-table columns a request needs: the draft reads only its window, so a request
        maps at most ceil((WINDOW + DRAFT_BLOCK) / SWA_BLOCK) + 1 pages (never a context's)."""
        return (WINDOW + DRAFT_BLOCK + SWA_BLOCK - 1) // SWA_BLOCK + 1

    def __init__(self, rank, dev, group, n_layers=None):
        """call() step (n_layers None): the draft's 3 layers; rank_chain: n_layers layers, layer 0
        fold 0 then fold 1. The runner is built as mono_decode._mono_runner builds the draft's:
        DSV41MonoLayer(tp, rank, cpu_group, device, experts=128, topk=3,
        ug_groups=DRAFT_UG_GROUPS, swa_width=get_dspark_swa_index_width(128, 5))."""
        R, L, AP, MD, SEAM = _mono()
        from vllm.models.deepseek_v41.amd.mono_decode import DRAFT_UG_GROUPS
        from vllm.v1.attention.backends.mla.compressor_utils import get_dspark_swa_index_width

        assert (AP.HIDDEN, L.MAX_TOKENS, AP.WINDOW) == (self.HIDDEN, 48, WINDOW)
        assert (R.HC, self.STREAM_SHAPE[0], self.STREAM_AUX[0]) == (4, 4, 4 * 6)
        assert get_dspark_swa_index_width(WINDOW, DRAFT_BLOCK) == SWA_WIDTH, "the draft's SWA width moved"
        assert MAX_ROWS <= L.MAX_TOKENS
        self.rank, self.dev, self.group = rank, dev, group
        self.n = len(LAYER_TYPES) if n_layers is None else int(n_layers)
        self.types = [layer_type(j, n_layers) for j in range(self.n)]
        self.path = self.PATHS[0]
        self.runner = R.DSV41MonoLayer(self.WORLD, rank, group, dev, experts=EXPERTS, topk=TOPK,
                                       ug_groups=DRAFT_UG_GROUPS, swa_width=SWA_WIDTH)
        self.weights = [make_weights(dev, rank, self.WORLD, j, t, _cos_sin(dev, t[0]), experts=EXPERTS)
                        for j, t in enumerate(self.types)]
        swa = [make_caches(dev, j, t[0])[0] for j, t in enumerate(self.types)]
        self.caches = [(t,) for t in swa]          # pool 0 only: launch l writes cache l
        self.swa_views = [record_view(t, SWA_BLOCK) for t in swa]
        self._built = set()
        self._native = None
        self._teacher = None

    # ------------------------------------------------------------------ layers and paths
    def pool_of(self, layer):
        return SWA

    def writes(self, path, i):
        """Launch i writes UT layer i's block records on every path (native: the teacher's K1,
        then vLLM's insert, into the same slots)."""
        return (i,)

    def set_path(self, path):
        if path not in self.PATHS:
            raise ValueError("unknown path %r" % (path,))
        self.path = path

    # ------------------------------------------------------------------ a step's metadata
    def step_meta(self, rows, context, width=None):
        """A draft step of rows / 5 requests: request r has max(6, context - 37 r) keys including
        its 5-token block (positions prefix .. prefix + 4, prefix = keys - 5); the block's slots
        over a shuffled SWA pool; every token of request r the same non-causal keys -- the slots
        of positions max(prefix - 128, 0) .. keys - 1, -1 padded to 192 -- and their count;
        token -> request. The same on every rank."""
        if rows % DRAFT_BLOCK or not 1 <= rows <= MAX_ROWS:
            raise ValueError("rows %d: draft steps are requests x %d rows, at most %d" % (rows, DRAFT_BLOCK, MAX_ROWS))
        Rq = rows // DRAFT_BLOCK
        lens = [max(DRAFT_BLOCK + 1, int(context) - 37 * r) for r in range(Rq)]
        if max(lens) > MAX_CONTEXT:
            raise ValueError("context %d > MAX_CONTEXT %d (the rope table's size)" % (max(lens), MAX_CONTEXT))
        g = torch.Generator(device="cpu").manual_seed(7919 * rows + int(context) + 13)
        swa_perm = (torch.randperm(SWA_POOL_PAGES - 1, generator=g) + 1).tolist()
        pos, t2r, slots = [], [], []
        swa_idx = torch.full((rows, 1, SWA_WIDTH), -1, dtype=torch.int32)
        swa_len = torch.zeros(rows, dtype=torch.int32)
        pages = {}

        def slot(r, p):
            key = (r, p // SWA_BLOCK)
            if key not in pages:
                pages[key] = swa_perm.pop()
            return pages[key] * SWA_BLOCK + p % SWA_BLOCK

        for r in range(Rq):
            prefix = lens[r] - DRAFT_BLOCK
            keys = [slot(r, q) for q in range(max(prefix - WINDOW, 0), lens[r])]
            for j in range(DRAFT_BLOCK):
                i = r * DRAFT_BLOCK + j
                pos.append(prefix + j)
                t2r.append(r)
                slots.append(slot(r, prefix + j))
                swa_idx[i, 0, :len(keys)] = torch.tensor(keys, dtype=torch.int32)
                swa_len[i] = len(keys)
        m = dict(positions=torch.tensor(pos, dtype=torch.int64),
                 slot_mapping=torch.tensor(slots, dtype=torch.int64),
                 swa_indices=swa_idx, swa_lens=swa_len,
                 token_to_req=torch.tensor(t2r, dtype=torch.int32))
        # DESIGN 1.5: a shuffled, non-identity slot map
        assert bool((m["slot_mapping"] != m["positions"]).any()), "identity SWA slot map"
        assert sorted(set(pages.values())) != list(range(1, len(pages) + 1)), "SWA pages not shuffled"
        return {k: v.to(self.dev) for k, v in m.items()}

    def kv_targets(self, m):
        """Pool 0: the block's SWA slots (K1 writes each token's record there)."""
        return (m["slot_mapping"],)

    def kv_bytes(self, layer, t, idx):
        """A slot's 584 record bytes in the flat SWA cache [pages, PAGE_STRIDE] (as the target
        example: data row at page * stride + off * 576, scale word after the page's 32 rows)."""
        stride = t.stride(0)
        page, off = idx // SWA_BLOCK, idx % SWA_BLOCK
        d0 = page * stride + off * DATA
        s0 = page * stride + SWA_BLOCK * DATA + off * 8
        ar_d = torch.arange(DATA, device=idx.device, dtype=torch.int64)
        ar_s = torch.arange(8, device=idx.device, dtype=torch.int64)
        return torch.cat([d0[:, None] + ar_d, s0[:, None] + ar_s], dim=1)

    def kv_decode(self, layer, raw):
        """A record's 512 values (the target example's decode: 448 e4m3 x UE8M0 a 64-dim block,
        64 bf16 rope dims; a non-zero pad byte or a poisoned byte marks the row NaN)."""
        n = raw.shape[0]
        nope = raw[:, :NOPE].contiguous().view(torch.float8_e4m3fn).float().view(n, 7, 64)
        sc = raw[:, DATA:DATA + 7].to(torch.int32)
        scale = torch.exp2((sc - 127).float()).masked_fill(sc == 255, float("nan"))
        vals = torch.cat([(nope * scale[..., None]).view(n, NOPE),
                          raw[:, NOPE:DATA].contiguous().view(torch.bfloat16).float()], dim=1)
        bad = raw[:, DATA + 7] != 0
        return torch.where(bad[:, None], torch.full_like(vals, float("nan")), vals)

    def bind(self, m):
        """Allocate the step width's scratch (eager first step), as the target example."""
        rows = int(m["positions"].numel())
        if not torch.cuda.is_current_stream_capturing() and rows not in self._built:
            self.runner.scratch(rows, self.runner.tuned(rows, 0))
            self._built.add(rows)
        return m

    # ------------------------------------------------------------------ launches
    @staticmethod
    def _mixes(rows, aux):
        post = aux[:, :4].reshape(rows, 4, 1).contiguous()
        comb = aux[:, 4:20].reshape(rows, 4, 4).contiguous()
        pre = aux[:, 20:24].contiguous()
        return post, comb, pre

    def _deployed(self, i, rows, m, stream, x, aux):
        """mono_decode.py:726's call for draft layer i. Layer 0 (fold 0) as mono_decode makes it
        for the first draft layer: residual = the stream, x = its stream 0, pre_mix = post_mix =
        res_mix = runner.first_pre_mix(rows); the harness's attention input and mixes are unused
        there. Returns (residual, out, mixes)."""
        w = self.weights[i]
        _, fold, _, _ = self.types[i]
        if fold:
            post, comb, pre = self._mixes(rows, aux)
        else:
            x = stream[:, 0].contiguous()
            post = comb = pre = self.runner.first_pre_mix(rows)
        out, res, post_o, comb_o, pre_o = self.runner.forward(
            w, x, stream, post, comb, pre, m["positions"], m["slot_mapping"], self.swa_views[i],
            m["swa_indices"], m["swa_lens"], m["token_to_req"], fold=fold)
        mix = torch.cat([post_o.reshape(rows, 4), comb_o.reshape(rows, 16), pre_o.reshape(rows, 4)], dim=1)
        return res, out, mix

    def launch(self, i, rows, m, fb, stream, x, aux):
        if self.path == "native":
            return self._native_launch(i, rows, m, stream, x, aux)
        return self._deployed(i, rows, m, stream, x, aux)

    def end_step(self):
        """Nothing: the kernels move their epoch themselves."""

    # ------------------------------------------------------------------ the native path
    def _all_reduce(self, t):
        """fp32 sum of the ranks' bf16 partials through the gloo group, rounded once to bf16."""
        import torch.distributed as dist

        h = t.float().cpu().contiguous()
        dist.all_reduce(h, op=dist.ReduceOp.SUM, group=self.group)
        return h.to(self.dev).to(torch.bfloat16)

    def _kernel_ffn_input(self, i, rows):
        """The FFN input K2 just used: its scratch `normed` region (layer.scratch_layout)."""
        R, L, AP, MD, SEAM = _mono()
        r = self.runner
        tuned = bool(r.tuned(rows, 0))
        off = L.scratch_layout(rows, r.tp, r.experts, r.topk, r.ug_groups, tuned)["normed"][0]
        return r.scratch(rows, tuned)[off:off + rows * self.HIDDEN * 2].view(torch.bfloat16).view(rows, self.HIDDEN).clone()

    def _native_launch(self, i, rows, m, stream, x, aux):
        """Teacher-forced: the deployed launch i on the deployed state first, then vLLM's chain
        from that same state (the first layer's seam as vLLM's draft runs it: pre_mix None, i.e.
        stream 0, no fold); returns the native (residual, out, mixes)."""
        if self._native is None:
            self._native = vllm_native(self.WORLD, self.dev, EXPERTS, TOPK, moe_layer=40)
            self._native.begin(self)
        V = self._native
        if i == 0:
            self._teacher = (stream.detach().clone(), x.detach().clone(), aux.detach().clone())
        t_res, t_x, t_aux = self._teacher
        d_res, d_out, d_mix = self._deployed(i, rows, m, t_res.clone(), t_x.clone(), t_aux.clone())
        normed_k = self._kernel_ffn_input(i, rows)
        self._teacher = (d_res.detach().clone(), d_out.detach().clone(), d_mix.detach().clone())
        w = self.weights[i]
        _, fold, _, _ = self.types[i]
        with torch.no_grad():
            if fold:
                post, comb, pre = self._mixes(rows, t_aux)
                res, post_a, comb_a, xa, attn_pre = V.seam(w, "attn", t_res, t_x, post, comb, pre)
            else:
                res, post_a, comb_a, xa, attn_pre = V.seam(w, "attn", t_res, None, None, None, None)
            part = V.attention(w.attn, xa, m, self.swa_views[i], None, None, 0)
            att = self._all_reduce(part)
            res2, post2, comb2, xf, ffn_pre = V.seam(w, "ffn", res, att, post_a, comb_a, attn_pre)
            V.load_moe(i, w)
            out = self._all_reduce(V.moe(i, xf, normed_k))
        mix = torch.cat([post2.reshape(rows, 4).float(), comb2.reshape(rows, 16).float(),
                         ffn_pre.reshape(rows, 4).float()], dim=1)
        return res2.contiguous(), out.contiguous(), mix.contiguous()

    def buffers(self):
        """produced: the runner's K1 -> K2 hand-off buffers (NaN before each launch). No ids buffer."""
        r = self.runner
        return {"ids": None,
                "produced": {n: getattr(r, n, None) for n in ("q", "res_mid", "post_a", "comb_a", "pre_a")},
                "inputs": {}, "diag": {}}

    def shared_layouts(self, rows):
        """The width's scratch (layer.scratch_layout of the draft runner's experts / topk /
        ug_groups) and the symmetric peer buffer, read from the subject's modules."""
        r = self.runner
        specs, kept = [], []
        try:
            R, L, AP, MD, SEAM = _mono()
            tuned = bool(r.tuned(rows, 0))
            specs.append(("scratch_s%d%s" % (rows, "_tuned" if tuned else ""), r._scratch.get((rows, tuned)),
                          lambda: L.scratch_layout(rows, r.tp, r.experts, r.topk, r.ug_groups, tuned)))
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
        """None: no deadline and no error words (a stuck wait is caught by the gloo timeout)."""
        return None

    def facts(self):
        r = self.runner
        return {"layers": self.n, "types": [list(t) for t in self.types], "swa_pages": SWA_POOL_PAGES,
                "page_stride": PAGE_STRIDE, "swa_block": SWA_BLOCK, "swa_width": SWA_WIDTH, "tp": r.tp,
                "experts": r.experts, "topk": r.topk, "kernels": [str(k) for k in r._kernels],
                "native_ties": list(self._native.ties) if self._native is not None else None}

    def close(self):
        """Release the peer buffer (IPC handles)."""
        close = getattr(getattr(self.runner, "peer", None), "close", None)
        if close:
            close()
