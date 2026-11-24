"""rank_model.py of task `muse_mono_draft_layer` (UT contract family `mono_layer`): the model-specific
half of the rank harness for the Muse Spark DFlash2 DRAFT layer at TP8 (layer_kind dense).
WORKED EXAMPLE for references/mono-layer-ut.md (a model whose layer is NOT one launch and whose
deployed path has no merged launch).

Provenance (2026-11-12): written by a fresh UT-lane agent in the end-to-end pipeline test, from
the scaffold and the target example; native path replaced by the implementer with SGLang's own
DFlash modules as DESIGN 2 prescribes (W10). GPU-verified on the served image (1106, sgl_amd_mono
ef38b0e): every gate of the contract including the native oracle floor (runs/gpu/W10b).

Entry under test: `sgl_amd_mono.muse_spark.draft:_Draft.forward`.

Live call site (sgl_amd_mono ef38b0e as installed in the image, = the task's kernel_src copy):
  sglang/srt/monokernels/runtime.py:349, inside `_install_forward.forward` (def at :329: the launcher
  frame of every d* kernel in evidence/source_map.json), calls
  `built.forward(Plan(key=key), *args, **kwargs)`, and `built` is the `_Draft` that
  draft.py::_build (:501) returns. `_Draft.forward(plan, input_ids, positions, forward_batch,
  input_embeds)` (draft.py:393) runs, for every draft layer i (draft.py:409-431, STAGE 2):
      first = residual is None
      h_in, r_in = hidden, (hidden if first else residual)
      attn_out, residual = runner.launch_attn(i, rows, h_in.contiguous(), r_in.contiguous(), first,
                                              positions, forward_batch, stream, trace=check)   # :413
      hidden, residual = runner.launch(i, rows, attn_out.contiguous(), residual.contiguous(), stream)  # :429
  then runner.end_step() (:432) and the final norm m.norm(hidden, residual) (:433, SGLang's RMSNorm,
  not a monokernel launch). DraftRunner.launch_attn (draft.py:264) = D-pre, D-attn, D-post;
  DraftRunner.launch (:312) = D-MLP; end_step (:325) bumps the step epoch. Every TP rank runs the
  same sequence (5 layers, 20 launches a step: the census' launches_per_instance 20).

How a harness launch maps onto it. The harness drives ONE layer per RankModel.launch(i) (it poisons
and captures between layers). The draft's call() must go through `_Draft.forward`, which loops over
every layer itself, so launch(i) runs the subject's own `_Draft.forward` with a runner proxy
(`_OneLayer`): forward's calls for layer i reach the real DraftRunner (or the native runner),
every other layer's calls are host-only pass-throughs, and forward's own end_step is deferred to
RankModel.end_step (the harness calls it once after the step's launches, as forward does). The
proxy injects (x, stream) as the (hidden, residual) layer i receives, so the call layer i gets is
exactly forward's: first=True with h_in = r_in = input_embeds for i = 0 (the stream: the draft's
input embeddings ARE its residual stream), first=False with (hidden, residual) = (x, stream) for
i >= 1, the runner's own out / res_out views between launches.

  stream  <-> the draft's `residual` (bf16): layer_out[i] = res_out after layer i
  x       <-> the draft's `hidden` (bf16; D-MLP's out); unused by launch 0 (first=True), exactly as
              the deployed layer 0 has no separate attention input; attn_out = the last D-MLP out

Paths:
  deployed  STAGE 2 through _Draft.forward (the only kernel path the server runs; STAGE is read
            from SGLANG_AMD_MONO_DRAFT_STAGE at import, draft.py:93, and must be 2 here);
  split     the same per-stage launches. The draft has no merged whole-layer launch in this
            package version: its deployed path IS the per-stage path, so the split oracle (on the
            baseline overlay) is the baseline's own deployed code -- the candidate is judged
            against the baseline, and the baseline's oracle floor is trivially met;
  native    SGLang's own DFlash decoder layer (sglang.srt.models.dflash.DFlashDecoderLayer) through
            sgl_amd_mono's check-mode references _Draft._attention_reference / _reference
            (draft.py:437 / :451), built on every rank at TP 1 with the full weights (the eight
            ranks' shards regenerated from their seeds) under an SGLang runtime context of
            tp_size 1 (override_server_args + override_permanently(tp_group=None)); the attention
            backend is _NativeAttention (prefix K / V of all KV heads gathered over gloo). Driven
            through the same `_Draft.forward` (the proxy's target is _SGLangNativeRunner) and
            teacher-forced per layer (the deployed layer runs first on the deployed state). The
            draft's correctness rests on this oracle: split is the baseline's own code. Measured
            vs the deployed kernels (r8 @ 4226): stream_out delta 58 dB, layer_out 40-53,
            attn_out 41, kv_rows 36 (one fp8 code).

Inputs: the draft's compact (windowed) KV cache (--speculative-draft-window-size 2048): a
request's draft prefix is min(context - 8, 2048) keys (draft_attn.py MAX_PARTS: "a window of
2048 + 8 keys"), D-attn runs with window 0 over seq_lens + 8 keys through req_to_token (the page
id of a 16-key page read from its first key's slot). One pool (contract axis pools = 1), fp8 e4m3
[slots, 1, 128] at unit scale, addressed through req_to_token directly (no full-to-SWA map: there
is nothing to make non-identity). Index tensors as served: positions / out_cache_loc / seq_lens /
req_pool_indices int64, req_to_token int32 over a shuffled page pool and shuffled request rows.
"""

import os
import types

import torch

WEIGHT_SEED = 20261112
POSITIONS = 1 << 20          # rotary table rows: the served contexts (target positions) reach 566k
ROPE_THETA = 10000.0         # the draft's rope_theta is not in the evidence; sgl-amd-mono's dhalf test uses 10000
Q = 8                        # DRAFT_TOKENS: rows a request (sgl_amd_mono.muse_spark.DRAFT_TOKENS)
WINDOW = 2048                # --speculative-draft-window-size: the compact cache's prefix keys
PAGE = 16
POOL_PAGES = 2048            # compact-pool pages a layer (32768 slots); two requests use <= 2 x 129
REQ_ROWS = 64                # req_to_token rows (request pool)
# page 0 and the last page are never handed out: a slot-shift control (slot + 1) stays inside the
# cache, and a control that reads the int64 slots as int32 writes slot 0, which no row owns
FIRST_PAGE, LAST_PAGE = 1, POOL_PAGES - 2


def _mods():
    from sgl_amd_mono.muse_spark import draft
    from sgl_amd_mono.muse_spark.gfx950 import draft_attn as DA
    from sgl_amd_mono.muse_spark.gfx950 import draft_attn_half as DH
    from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D
    return draft, DA, DH, D


def _cos_sin(dev):
    """The rotary table as D-pre reads it: fp32 [positions, 128], cos (64) then sin (64), neox."""
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, 128, 2, dtype=torch.float32, device=dev) / 128))
    out = torch.empty(POSITIONS, 128, dtype=torch.float32, device=dev)
    step = 1 << 18
    for s in range(0, POSITIONS, step):
        f = torch.arange(s, s + step, dtype=torch.float32, device=dev)[:, None] * inv[None, :]
        out[s:s + step, :64] = f.cos()
        out[s:s + step, 64:] = f.sin()
    return out


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def make_layer(dev, rank, layer, cos_sin):
    """One draft layer with the attribute paths draft.py reads (_attn_views :114, _mlp_views :102,
    launch_attn's self_attn.attn). Replicated tensors (norms, the convolutions' kernel projections
    and bases) from a seed common to every rank; TP-sharded ones (qkv rows: this rank's 8 q heads,
    its KV head; o_proj columns; gate / up rows; down columns) from a rank seed. Every norm weight
    is drawn separately (1 + s N(0, 1)): none is all ones, no two are equal, q_norm != k_norm (a
    kernel that normalized k with q's weight must be caught). Scales follow sgl-amd-mono's
    test/kernels draft tests (conv tap 0 ~ 1, tap 1 ~ 0); w_o / w_down are sized so that each
    sublayer's contribution to the residual is O(0.5-1) of the N(0, 1) stream (a contribution far
    below the stream's 1/128 tolerance would hide every bug in it)."""
    _, DA, DH, D = _mods()
    bf = torch.bfloat16
    gs = torch.Generator(device=dev).manual_seed(WEIGHT_SEED + layer)
    gr = torch.Generator(device=dev).manual_seed(WEIGHT_SEED + 1000 + layer + 100 * (rank + 1))

    def rn(shape, g, scale=1.0):
        return torch.randn(shape, device=dev, generator=g) * scale

    def base(g):
        b = rn((2, D.TAPS, D.H), g, 0.05)
        b[:, 0] += 1.0
        return b.to(bf).contiguous()

    w_in = (1.0 + 0.1 * rn(D.H, gs)).to(bf)
    w_kpa = rn((D.COEF_ROWS, D.H), gs, 0.01).to(bf)
    base_a = base(gs)
    q_norm = (1.0 + 0.25 * rn(DH.HEAD_DIM, gs)).to(bf)
    k_norm = (1.0 + 0.25 * rn(DH.HEAD_DIM, gs)).to(bf)
    w_post = (1.0 + 0.1 * rn(D.H, gs)).to(bf)
    w_kp = rn((D.COEF_ROWS, D.H), gs, 0.01).to(bf)
    base_m = base(gs)
    w_qkv = rn((DH.QKV_ROWS, D.H), gr, 0.02).to(bf)
    w_o = rn((D.H, DH.Q_K), gr, 0.03).to(bf)
    w_gu = rn((D.GU_ROWS, D.H), gr, 0.02).to(bf)
    w_down = rn((D.H, D.I), gr, 0.005).to(bf)
    norms = [w_in, w_post, q_norm, k_norm]
    assert all(not bool((n == 1).all()) for n in norms), "a norm weight is all ones"
    assert not torch.equal(w_in, w_post) and not torch.equal(q_norm, k_norm), "two norm weights are equal"
    attn = _ns(layer_id=layer, scaling=DA.HEAD_DIM ** -0.5, k_scale=None, v_scale=None)
    return _ns(
        input_layernorm=_ns(weight=w_in),
        post_attention_layernorm=_ns(weight=w_post),
        attention_conv=_ns(kernel_projection=_ns(weight=w_kpa), base_kernel=base_a),
        mlp_conv=_ns(kernel_projection=_ns(weight=w_kp), base_kernel=base_m),
        self_attn=_ns(qkv_proj=_ns(weight=w_qkv), q_norm=_ns(weight=q_norm), k_norm=_ns(weight=k_norm),
                      rotary_emb=_ns(cos_sin_cache=cos_sin), o_proj=_ns(weight=w_o), attn=attn),
        mlp=_ns(gate_up_proj=_ns(weight=w_gu), down_proj=_ns(weight=w_down)),
    )


def make_caches(dev, rank, n_layers):
    """Seeded fp8 caches [slots, 1 KV head, 128] a layer (this rank's KV head: a rank seed; K at
    std 2 so that the prefix's scores spread (std ~2 against the normed q) and the attention is not
    near-uniform over 2056 keys, where its output and any bug in it would vanish), drawn
    in layer order from one generator, so two models with the same first layers (rank_chain's
    parity check) hold the same caches for them."""
    _, DA, _, _ = _mods()
    g = torch.Generator(device=dev).manual_seed(WEIGHT_SEED + 17 * (rank + 1))
    out = []
    for _ in range(n_layers):
        shape = (POOL_PAGES * PAGE, DA.LOCAL_KV_HEADS, DA.HEAD_DIM)
        k = (torch.randn(shape, device=dev, generator=g) * 2).to(torch.float8_e4m3fn)
        v = torch.randn(shape, device=dev, generator=g).to(torch.float8_e4m3fn)
        out.append((k, v))
    return out


class _KVPool:
    """token_to_kv_pool: get_kv_buffer(layer_id) -> (k, v) (draft.py _check_cache :250, launch_attn :280)."""

    def __init__(self, caches):
        self.caches = caches

    def get_kv_buffer(self, layer_id):
        return self.caches[layer_id]


class _Norm:
    """model.norm: forward's final `hidden, _ = m.norm(hidden, residual)` (draft.py:433) is SGLang's
    RMSNorm, not a monokernel launch; the harness reads layer outputs, so it is a pass-through."""

    def __call__(self, hidden, residual=None):
        return (hidden, residual) if residual is not None else hidden


class _OneLayer:
    """The runner _Draft.forward drives, as one layer's launch: forward calls launch_attn / launch for
    every layer; only layer `want` reaches `target` (the DraftRunner, or the native runner). Layers
    before it return `inp` = (x, stream) from launch(), so layer `want` receives forward's own
    (hidden, residual) arguments built from the harness's operands; layers after it pass their
    arguments through. end_step is deferred to RankModel.end_step. Any other runner attribute
    forward asks for is a changed interface this proxy does not know: fail loudly."""

    def __init__(self):
        self.target, self.want, self.inp, self.out = None, None, None, None

    def launch_attn(self, i, rows, hidden, residual, first, positions, forward_batch, stream, trace=False):
        if i == self.want:
            return self.target.launch_attn(i, rows, hidden, residual, first, positions, forward_batch, stream, trace=trace)
        return hidden, residual

    def launch(self, i, rows, attn_out, residual, stream):
        if i == self.want:
            self.out = self.target.launch(i, rows, attn_out, residual, stream)
            return self.out
        if i < self.want:
            return self.inp
        return attn_out, residual

    def end_step(self):
        pass

    def __getattr__(self, name):
        raise AttributeError("rank_model._OneLayer: _Draft.forward asked the runner for %r; the UT's per-layer proxy "
                             "drives launch_attn / launch / end_step only (the draft's runner interface changed)" % name)


def _grouped_conv(x, delta, base, D):
    """sglang.srt.models.dflash._grouped_conv as sgl-amd-mono's test/kernels write it (bf16)."""
    blocks = x.unflatten(-1, (D.GROUPS, D.GROUP))
    coefficients = base.view(1, D.TAPS, D.GROUPS, D.GROUP) + delta.unsqueeze(-1)
    out = coefficients[:, 0] * blocks
    position = torch.arange(x.shape[0], device=x.device) & (D.BLOCK - 1)
    for tap in range(1, D.TAPS):
        shifted = torch.nn.functional.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        shifted = torch.where((position >= tap).view(-1, 1, 1), shifted, torch.zeros_like(shifted))
        out = out + coefficients[:, tap] * shifted
    return out.flatten(-2)


def _rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(torch.bfloat16)


class _NativeRunner:
    """The native path's runner: DraftRunner.launch_attn / launch in torch (SGLang's DFlash math;
    see the module docstring), the TP sums over the harness's gloo group. Eager only."""

    def __init__(self, layers, backend, group, dev):
        self.layers, self.backend, self.group, self.dev = layers, backend, group, dev

    def _all_reduce(self, t):
        import torch.distributed as dist

        acc = t.float().cpu()
        dist.all_reduce(acc, group=self.group)
        return acc.to(self.dev).to(torch.bfloat16)

    def launch_attn(self, i, rows, hidden, residual, first, positions, forward_batch, stream, trace=False):
        _, DA, DH, D = _mods()
        bf = torch.bfloat16
        L = self.layers[i]
        sa = L.self_attn
        S = rows
        hidden, residual = hidden[:S], residual[:S]
        res2 = hidden if first else (residual.float() + hidden.float()).to(bf)
        h = _rms(res2, L.input_layernorm.weight, D.EPS)
        coef = (h @ L.attention_conv.kernel_projection.weight.T).reshape(S, 2, D.TAPS, D.GROUPS)
        base = L.attention_conv.base_kernel
        hc = _grouped_conv(h, coef[:, 0], base[0], D)
        qkv = hc @ sa.qkv_proj.weight.T
        q, k, v = qkv[:, :DH.Q_K], qkv[:, DH.Q_K:DH.Q_K + DH.HEAD_DIM], qkv[:, DH.Q_K + DH.HEAD_DIM:]
        cs = sa.rotary_emb.cos_sin_cache[positions[:S].long()].float()
        cos, sin = cs[:, None, :64], cs[:, None, 64:]

        def norm_rope(t, wn):
            heads = t.reshape(S, -1, DH.HEAD_DIM).float()
            n = (heads * torch.rsqrt(heads.pow(2).mean(-1, keepdim=True) + D.EPS) * wn.float()).to(bf).float()
            x1, x2 = n[..., :64], n[..., 64:]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(bf).reshape(S, -1)

        q, k = norm_rope(q, sa.q_norm.weight), norm_rope(k, sa.k_norm.weight)
        kc, vc = self.backend.token_to_kv_pool.get_kv_buffer(sa.attn.layer_id)
        slots = forward_batch.out_cache_loc[:S].long()
        # fp8 stores / loads through uint8 views (index_put / index on float8 are not on every torch build)
        kc.view(torch.uint8)[slots, 0] = k.float().clamp(-448.0, 448.0).to(kc.dtype).view(torch.uint8)
        vc.view(torch.uint8)[slots, 0] = v.float().clamp(-448.0, 448.0).to(vc.dtype).view(torch.uint8)
        table = self.backend.req_to_token_pool.req_to_token
        o = torch.empty(S, DH.Q_K, device=self.dev, dtype=bf)
        lens = forward_batch.seq_lens
        rpi = forward_batch.req_pool_indices
        for r in range(S // Q):
            keys = int(lens[r]) + Q            # the compact prefix, then the block (encoder-only: all 8 see all 8)
            sl = table[int(rpi[r]), :keys].long()
            K = kc.view(torch.uint8)[sl, 0].view(kc.dtype).float()
            V = vc.view(torch.uint8)[sl, 0].view(vc.dtype).float()
            qh = q[r * Q:(r + 1) * Q].float().reshape(Q, DA.GQA, DA.HEAD_DIM)
            s = float(sa.attn.scaling) * torch.einsum("thd,kd->thk", qh, K)
            p = torch.softmax(s, dim=-1).to(bf).float()
            o[r * Q:(r + 1) * Q] = torch.einsum("thk,kd->thd", p, V).reshape(Q, DH.Q_K).to(bf)
        d = self._all_reduce(o @ sa.o_proj.weight.T)
        attn_out = _grouped_conv(d, coef[:, 1], base[1], D)
        return attn_out.contiguous(), res2.contiguous()

    def launch(self, i, rows, attn_out, residual, stream):
        _, _, _, D = _mods()
        bf = torch.bfloat16
        L = self.layers[i]
        S = rows
        res3 = (residual[:S].float() + attn_out[:S].float()).to(bf)
        h = _rms(res3, L.post_attention_layernorm.weight, D.EPS)
        coef = (h @ L.mlp_conv.kernel_projection.weight.T).reshape(S, 2, D.TAPS, D.GROUPS)
        base = L.mlp_conv.base_kernel
        hc = _grouped_conv(h, coef[:, 0], base[0], D)
        gu = hc @ L.mlp.gate_up_proj.weight.T
        g, u = gu[:, :D.I], gu[:, D.I:]
        a = (torch.nn.functional.silu(g.float()).to(bf).float() * u.float()).to(bf)
        d = self._all_reduce(a @ L.mlp.down_proj.weight.T)
        out = _grouped_conv(d, coef[:, 1], base[1], D)
        return out.contiguous(), res3.contiguous()



# ----------------------------------------------------------------------------- native (W10, DESIGN 2)
# The draft's native oracle is SGLang's own DFlash decoder layer (sglang.srt.models.dflash), driven
# through sgl_amd_mono's check-mode references _Draft._attention_reference / _reference
# (draft.py:437 / :451): input_layernorm, attention_conv, self_attn (qkv_proj, q / k norm, rope,
# RadixAttention, o_proj), post_attention_layernorm, mlp_conv, mlp -- code the monokernel's author
# did not write. Built on every rank at TP 1 with the FULL weights (the eight ranks' shards
# regenerated from their seeds and concatenated as SGLang's TP sharding lays them out), under an
# SGLang runtime context of tp_size 1 (no tensor-parallel group: no SGLang collective runs). The
# one piece SGLang cannot supply outside a server is the attention backend: _NativeAttention below
# (the compact cache's prefix keys of all eight KV heads gathered over the harness's gloo group,
# then the block's own K / V; every block token sees the whole block; P rounded to bf16).
_SG_CTX = {}


def _sglang_context():
    """Publish a dummy-boundary ServerArgs (tp_size 1) and an explicit `tp_group = None`, once a
    process (sglang's own unit-test mechanism: RuntimeContext.override_server_args)."""
    if _SG_CTX:
        return _SG_CTX
    from sglang.srt.runtime_context import get_context, get_parallel

    ov = get_context().override_server_args(speculative_algorithm="DFLASH", speculative_num_draft_tokens=Q,
                                            speculative_draft_attention_backend="triton")
    ov.install()
    get_parallel().override_permanently(tp_group=None)
    if int(get_parallel().tp_size) != 1:
        raise RuntimeError("native: the SGLang runtime context is not tp_size 1")
    _SG_CTX["override"] = ov
    return _SG_CTX


def _sglang_config():
    from transformers import PretrainedConfig

    draft, _, _, _ = _mods()
    return PretrainedConfig(**draft.DRAFT_CONFIG, vocab_size=202048, model_type="qwen3", rope_theta=ROPE_THETA,
                            max_position_embeddings=POSITIONS, dflash_config=dict(draft.DFLASH_CONFIG),
                            architectures=["DFlash2DraftModel"], hidden_act="silu", attention_bias=False)


def _full_weights(dev, layer, cos_sin, world):
    """SGLang's unsharded parameters of draft layer `layer`: every rank's shards (make_layer, from
    their seeds) concatenated in rank order the way SGLang's TP linears shard them -- qkv_proj rows
    [q heads | k head | v head] a rank -> [all q | all k | all v]; o_proj / down_proj columns;
    gate_up rows [gate | up] a rank -> [all gate | all up]; norms and the convolutions replicated."""
    _, DA, DH, D = _mods()
    sh = [make_layer(dev, r, layer, cos_sin) for r in range(world)]
    a = sh[0]
    qk, hd = DH.Q_K, DH.HEAD_DIM
    qkv = [s.self_attn.qkv_proj.weight for s in sh]
    gu = [s.mlp.gate_up_proj.weight for s in sh]
    half = gu[0].shape[0] // 2
    w = {
        "input_layernorm.weight": a.input_layernorm.weight,
        "post_attention_layernorm.weight": a.post_attention_layernorm.weight,
        "self_attn.qkv_proj.weight": torch.cat([t[:qk] for t in qkv] + [t[qk:qk + hd] for t in qkv]
                                               + [t[qk + hd:qk + 2 * hd] for t in qkv]),
        "self_attn.o_proj.weight": torch.cat([s.self_attn.o_proj.weight for s in sh], dim=1),
        "self_attn.q_norm.weight": a.self_attn.q_norm.weight,
        "self_attn.k_norm.weight": a.self_attn.k_norm.weight,
        "mlp.gate_up_proj.weight": torch.cat([t[:half] for t in gu] + [t[half:] for t in gu]),
        "mlp.down_proj.weight": torch.cat([s.mlp.down_proj.weight for s in sh], dim=1),
        "attention_conv.base_kernel": a.attention_conv.base_kernel,
        "attention_conv.kernel_projection.weight": a.attention_conv.kernel_projection.weight,
        "mlp_conv.base_kernel": a.mlp_conv.base_kernel,
        "mlp_conv.kernel_projection.weight": a.mlp_conv.kernel_projection.weight,
    }
    for r in range(1, world):   # the replicated tensors really are the same on every rank
        assert torch.equal(sh[r].input_layernorm.weight, a.input_layernorm.weight)
        assert torch.equal(sh[r].mlp_conv.base_kernel, a.mlp_conv.base_kernel)
    return w


class _NativeAttention:
    """The attention backend SGLang's RadixAttention calls (forward_context.get_attn_backend()):
    stores this rank's KV head of the block's K / V (fp8, unit scale) at out_cache_loc -- what the
    UT's kv_rows read --, gathers every rank's KV head of the request's compact prefix over the gloo
    group, and attends: keys = the prefix and the whole block (the draft block is encoder-only), P
    rounded to bf16 (the draft kernels' convention, test/kernels test_muse_spark_dattn)."""

    def __init__(self, model):
        self.m = model

    def forward(self, q, k, v, layer, fb, save_kv_cache=True, **kw):
        import torch.distributed as dist

        m = self.m
        bf, f8 = torch.bfloat16, torch.float8_e4m3fn
        H, KH, D = layer.tp_q_head_num, layer.tp_k_head_num, layer.qk_head_dim
        S = q.shape[0]
        k8 = k.reshape(S, KH, D).float().clamp(-448.0, 448.0).to(f8)
        v8 = v.reshape(S, KH, D).float().clamp(-448.0, 448.0).to(f8)
        kc, vc = m.caches[layer.layer_id]
        slots = fb.out_cache_loc[:S].long()
        if save_kv_cache:
            kc.view(torch.uint8)[slots, 0] = k8[:, m.rank].view(torch.uint8)
            vc.view(torch.uint8)[slots, 0] = v8[:, m.rank].view(torch.uint8)
        table = m.backend.req_to_token_pool.req_to_token
        out = torch.empty(S, H * D, device=q.device, dtype=bf)
        g = H // KH
        for r in range(S // Q):
            prefix, rp = int(fb.seq_lens[r]), int(fb.req_pool_indices[r])
            sl = table[rp, :prefix].long()
            mine = torch.stack([kc.view(torch.uint8)[sl, 0], vc.view(torch.uint8)[sl, 0]]).cpu().contiguous()
            every = [torch.empty_like(mine) for _ in range(m.WORLD)]
            dist.all_gather(every, mine, group=m.group)
            pk = torch.stack([e[0] for e in every], dim=1).to(q.device).view(f8).float()   # [prefix, KH, D]
            pv = torch.stack([e[1] for e in every], dim=1).to(q.device).view(f8).float()
            K = torch.cat([pk, k8[r * Q:(r + 1) * Q].float()])
            V = torch.cat([pv, v8[r * Q:(r + 1) * Q].float()])
            qq = q[r * Q:(r + 1) * Q].float().reshape(Q, KH, g, D)
            sc = float(layer.scaling) * torch.einsum("tkgd,nkd->tkgn", qq, K)
            pr = torch.softmax(sc, dim=-1).to(bf).float()
            out[r * Q:(r + 1) * Q] = torch.einsum("tkgn,nkd->tkgd", pr, V).reshape(Q, H * D).to(bf)
        return out


class _Mode:
    def is_extend(self):
        return False

    def is_target_verify(self):
        return True

    def is_decode(self):
        return False

    def is_idle(self):
        return False

    def is_draft_extend(self):
        return False


class _SGLangNativeRunner:
    """The native path's runner (launch_attn / launch, as DraftRunner): each layer through
    _Draft._attention_reference / _reference on SGLang's DFlashDecoderLayer (see above)."""

    def __init__(self, model):
        self.m, self.sg = model, {}
        self.backend = _NativeAttention(model)

    def _layer(self, i):
        if i not in self.sg:
            from sglang.srt.models import dflash as SD
            from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

            _sglang_context()
            cfg = _sglang_config()
            dc = parse_dflash_draft_config(draft_hf_config=cfg)
            bs = dc.resolve_block_size(default=16)

            def conv():
                return SD.DFlashGroupedConv(int(cfg.hidden_size), bs, dc.conv_kernel_size, dc.conv_group_size)

            L = SD.DFlashDecoderLayer(cfg, i, attention_conv=conv(), mlp_conv=conv()).to(self.m.dev).to(torch.bfloat16)
            w = _full_weights(self.m.dev, i, self.m.cos_sin, self.m.WORLD)
            params = dict(L.named_parameters())
            if set(params) != set(w):
                raise RuntimeError("native: SGLang DFlashDecoderLayer parameters %s != the UT's %s"
                                   % (sorted(set(params) ^ set(w)), ""))
            with torch.no_grad():
                for n, t in w.items():
                    if tuple(params[n].shape) != tuple(t.shape):
                        raise RuntimeError("native: %s shape %s != %s" % (n, tuple(params[n].shape), tuple(t.shape)))
                    params[n].copy_(t)
            del w
            L.eval()
            self.sg[i] = L
        return self.sg[i]

    def _fb(self, fb, S):
        return _ns(out_cache_loc=fb.out_cache_loc[:S], seq_lens=fb.seq_lens, req_pool_indices=fb.req_pool_indices,
                   positions=fb.positions, forward_mode=_Mode(), mha_return_lse=False, batch_size=int(fb.seq_lens.numel()))

    def launch_attn(self, i, rows, hidden, residual, first, positions, forward_batch, stream, trace=False):
        from sglang.srt.model_executor.forward_context import ForwardContext, set_forward_context

        draft, _, _, _ = _mods()
        L = self._layer(i)
        S = rows
        set_forward_context(ForwardContext(attn_backend=self.backend))
        with torch.no_grad():
            ao, res = draft._Draft._attention_reference(L, hidden[:S].contiguous(), None if first else residual[:S].contiguous(),
                                                        positions[:S], self._fb(forward_batch, S))
        return ao.contiguous(), res.contiguous()

    def launch(self, i, rows, attn_out, residual, stream):
        draft, _, _, _ = _mods()
        L = self._layer(i)
        with torch.no_grad():
            out, res = draft._Draft._reference(L, attn_out[:rows].contiguous(), residual[:rows].contiguous())
        return out.contiguous(), res.contiguous()

    def end_step(self):
        pass

class RankModel:
    # ---- frozen by the scaffold from run/ut_contract.json; change only with the contract
    WORLD = 8                  # ranks (const axis world)
    LAUNCHES = 5               # layer launches a call() step makes (const axis launches)
    POOLS = 1                  # KV pools (const axis pools): index 0 full, 1 sliding window
    HIDDEN = 8192
    TOP_K = None                  # routing ids per token (layer_kind routed_moe), None for dense
    KV_WIDTH = 128             # elements of one token's K (or V) row, decoded to float32
    STREAM_DTYPE = torch.bfloat16
    INPUT_DTYPE = torch.bfloat16
    # ---- paths: PATHS[0] is the deployed one call() runs; every other entry is an oracle path
    # (cases.ORACLES[name] runs the contract oracle whose `path` names it)
    PATHS = ('deployed', 'split', 'native')
    PACKAGE = 'sgl_amd_mono'         # the whole-package bind: its bytes key the JIT cache
    JIT_CACHE_ENV = "FLYDSL_RUNTIME_CACHE_DIR"   # the JIT's cache-dir variable; None if it has none
    KEEP_REGIONS = r"flag|trace|ugq"           # scratch regions never poisoned (hand-off flags ...)
    NOTES = [
        "LAUNCHES=5 = the draft's 5 layers (DRAFT_CONFIG num_hidden_layers); one harness launch = one draft layer "
        "= 4 device launches (D-pre, D-attn, D-post, D-MLP), not one",
        "launch i runs the subject's _Draft.forward through a per-layer runner proxy (rank_model._OneLayer); "
        "forward's final m.norm (SGLang RMSNorm) is a pass-through",
        "launch 0 is the deployed first layer (first=True): x (attn_input) is unused, the stream is the input embeddings",
        "split == deployed: the draft has no merged whole-layer launch, so the split oracle is the baseline's own code",
        "one pool: the compact windowed draft cache through req_to_token, no full-to-SWA map",
        "draft prefix = min(context - 8, 2048) (--speculative-draft-window-size 2048): every served case attends 2056 keys",
        "DraftRunner._backend (SGLang's forward-context attention backend) is replaced on the instance by the UT's backend",
        "launch 0 copies res_out into the harness's stream tensor (rank_chain parity reads the stream in place); "
        "one D2D copy a step, outside rank_chain's (length - short) difference",
        "rope_theta 10000 (not in the evidence; the table's values do not change the kernel's work)",
    ]

    @staticmethod
    def page_width(context):
        """req_to_token columns a request of `context` (target) keys needs: its compact draft keys
        (min(context - 8, 2048) + 8), padded to a page."""
        keys = min(max(Q + 1, int(context)) - Q, WINDOW) + Q
        return (keys + PAGE - 1) // PAGE * PAGE

    def __init__(self, rank, dev, group, n_layers=None):
        draft, DA, DH, D = _mods()
        from sgl_amd_mono.muse_spark import DRAFT_TOKENS, TP
        from sgl_amd_mono.muse_spark.gfx950.config import WIDTHS

        if not (draft.USE_KERNELS and draft.STAGE == 2 and not draft.SKIP_LAYERS):
            raise RuntimeError("the draft must run its stage-2 kernels (USE_KERNELS %s, STAGE %s, SKIP %s): unset "
                               "SGLANG_AMD_MONO_DRAFT_KERNELS / _STAGE / _SKIP" % (draft.USE_KERNELS, draft.STAGE,
                                                                                    draft.SKIP_LAYERS))
        assert (DRAFT_TOKENS, TP, D.H, DA.LOCAL_KV_HEADS * DA.HEAD_DIM) == (Q, self.WORLD, self.HIDDEN, self.KV_WIDTH)
        assert draft.DRAFT_CONFIG["num_hidden_layers"] == self.LAUNCHES
        self.rank, self.dev, self.group = rank, dev, group
        n = self.LAUNCHES if n_layers is None else int(n_layers)
        cos_sin = _cos_sin(dev)
        self.layers = [make_layer(dev, rank, i, cos_sin) for i in range(n)]
        for i, layer in enumerate(self.layers):
            why = draft._attn_view_reasons(i, draft._attn_views(layer)) + draft._mlp_view_reasons(i, draft._mlp_views(layer))
            if why:
                raise RuntimeError("draft layer views: %s" % why)
        self.caches = make_caches(dev, rank, n)
        self.backend = _ns(token_to_kv_pool=_KVPool(self.caches),
                           req_to_token_pool=_ns(req_to_token=torch.zeros(REQ_ROWS, PAGE, dtype=torch.int32, device=dev)))
        # as draft.py::_build (:509-514): the runner over the layers, the wait deadline, every width built
        self.runner = draft.DraftRunner(list(self.layers), rank, self.WORLD, group, dev)
        self.runner.set_deadline(60.0)
        # SGLang's forward context is not here: the runner's backend lookup (draft.py:239) answers the UT's backend
        self.runner._backend = lambda: self.backend
        widths = list(WIDTHS)
        start = len(widths) * rank // self.WORLD
        for rows in widths[start:] + widths[:start]:
            self.runner.precompile(rows)
        self.cos_sin = cos_sin
        self.native = _SGLangNativeRunner(self)     # W10: SGLang's own DFlash modules (DESIGN 2)
        self._teacher = None
        self.proxy = _OneLayer()
        self.proxy.target = self.runner
        self.path = self.PATHS[0]
        self.draft = draft._Draft(_ns(layers=self.layers, norm=_Norm()), None, self.proxy)

    # ------------------------------------------------------------------ layers and paths
    def pool_of(self, layer):
        return 0

    def writes(self, path, i):
        return (i,)

    def set_path(self, path):
        if path not in self.PATHS:
            raise ValueError(path)
        self.path = path
        self.proxy.target = self.native if path == "native" else self.runner
        if path == "native":
            # build every SGLang layer now: set_path runs before the harness's barrier, so no rank
            # builds (seconds: eight ranks' shards, SGLang init) while its peers spin in an in-kernel wait
            for i in range(len(self.layers)):
                self.native._layer(i)

    # ------------------------------------------------------------------ a step's metadata
    def step_meta(self, rows, context, width=None):
        """The step's compact-cache tables, the same on every rank (the TP ranks share one batch).
        Requests = rows / 8; request r stands for a target request of context - 37 r keys: its
        draft prefix is min(that - 8, 2048) keys, its block's 8 keys follow; its pages are distinct
        random pages of a shuffled pool, its req_to_token row a random row; the block's slots are
        out_cache_loc; positions are the target positions of the block."""
        R = rows // Q
        full = [max(Q + 1, int(context) - 37 * r) for r in range(R)]
        prefix = [min(L - Q, WINDOW) for L in full]
        g = torch.Generator(device="cpu").manual_seed(7919 * rows + int(context))
        perm = torch.randperm(LAST_PAGE - FIRST_PAGE + 1, generator=g) + FIRST_PAGE
        rpi = torch.randperm(REQ_ROWS, generator=g)[:R]
        cols = max(int(width or 0), self.page_width(context))
        table = torch.zeros(REQ_ROWS, cols, dtype=torch.int32)
        slots, at = [], 0
        for r in range(R):
            keys = prefix[r] + Q
            npg = (keys + PAGE - 1) // PAGE
            pages = perm[at:at + npg]
            at += npg
            sl = (pages[:, None] * PAGE + torch.arange(PAGE)[None, :]).reshape(-1)[:keys]
            table[int(rpi[r]), :keys] = sl.to(torch.int32)
            slots += sl[prefix[r]:keys].tolist()
        m = dict(
            table=table,
            seq_lens=torch.tensor(prefix, dtype=torch.int64),
            out_cache_loc=torch.tensor(slots, dtype=torch.int64),
            positions=torch.tensor([full[r] - Q + j for r in range(R) for j in range(Q)], dtype=torch.int64),
            req_pool_indices=rpi.to(torch.int64),
        )
        return {k: v.to(self.dev) for k, v in m.items()}

    def kv_targets(self, m):
        return (m["out_cache_loc"],)

    def bind(self, m):
        """The backend's req_to_token is the step's table (launch_attn reads it from the backend at
        every launch, draft.py:282); the forward batch carries the rest."""
        self.backend.req_to_token_pool.req_to_token = m["table"]
        return _ns(out_cache_loc=m["out_cache_loc"], req_pool_indices=m["req_pool_indices"], seq_lens=m["seq_lens"],
                   positions=m["positions"])

    # ------------------------------------------------------------------ launches
    def _forward_layer(self, i, rows, m, fb, stream, x, target):
        p = self.proxy
        p.target = target
        p.want, p.inp, p.out = i, (x, stream), None
        self.draft.forward(_ns(key=_ns(rows=rows)), None, m["positions"], fb, input_embeds=stream)
        if p.out is None:
            raise RuntimeError("_Draft.forward made no launch for layer %d" % i)
        return p.out

    def launch(self, i, rows, m, fb, stream, x):
        """Draft layer i through the subject's _Draft.forward (draft.py:393), see the module docstring.
        native (W10): teacher-forced as the target's -- the deployed layer i runs first on the deployed
        path's state before it, the SGLang-module layer i from that same state, so one layer's error is
        measured once and never compounds through the next four."""
        if self.path == "native":
            if i == 0:
                self._teacher = (x.detach().clone(), stream.detach().clone())
            tx, ts = self._teacher
            d_out, d_res = self._forward_layer(i, rows, m, fb, ts.clone(), tx.clone(), self.runner)
            self._teacher = (d_out.detach().clone(), d_res.detach().clone())
            out, res_out = self._forward_layer(i, rows, m, fb, ts.clone(), tx.clone(), self.native)
            stream.copy_(res_out)
            return stream, out
        p = self.proxy
        p.want, p.inp, p.out = i, (x, stream), None
        plan = _ns(key=_ns(rows=rows))
        self.draft.forward(plan, None, m["positions"], fb, input_embeds=stream)
        if p.out is None:
            raise RuntimeError("_Draft.forward made no launch for layer %d" % i)
        out, res_out = p.out
        if stream.data_ptr() != res_out.data_ptr():
            stream.copy_(res_out)
        return res_out, out

    def end_step(self):
        self.runner.end_step()

    def buffers(self):
        """produced: what a layer's launches write and only that layer reads (NaN before every launch):
        D-pre's res' / q / coefficients, D-attn's o, D-post's attn_out. inputs: D-MLP's out / res_out,
        which the next layer's D-pre reads (NaN before launch 0 only). No ids (dense)."""
        r = self.runner
        get = lambda n: getattr(r, n, None)  # noqa: E731
        return {"ids": None,
                "produced": {n: get(n) for n in ("res_pre", "q", "o", "coef", "attn_out")},
                "inputs": {n: get(n) for n in ("out", "res_out")},
                "diag": {"o": get("o"), "attn_out": get("attn_out")}}

    def shared_layouts(self, rows):
        """(specs, kept): every scratch and peer region of the four launches, from the subject's modules
        (D-attn's at this width: its partial records move with the row count)."""
        r = self.runner
        specs, kept = [], []
        try:
            _, DA, DH, D = _mods()
            npes = r.npes
            specs += [("scratch_dmlp", getattr(r, "scratch", None), D.dm_scratch_layout),
                      ("scratch_dpre", getattr(r, "scratch_pre", None), DH.dp_scratch_layout),
                      ("scratch_dattn", getattr(r, "scratch_attn", None), lambda: DA.dattn_scratch_layout(rows, Q)),
                      ("scratch_dpost", getattr(r, "scratch_post", None), DH.do_scratch_layout),
                      ("peers_dmlp", getattr(getattr(r, "peers", None), "bytes", None), lambda: D.dm_sym_layout(npes)),
                      ("peers_dpost", getattr(getattr(r, "peers_post", None), "bytes", None),
                       lambda: DH.do_sym_layout(npes))]
        except Exception as e:  # noqa: BLE001 - a candidate may move the layouts: report, do not crash
            kept.append({"region": "draft scratch / peers", "why": "layout unavailable: %r" % e})
        return specs, kept

    # ------------------------------------------------------------------ health
    def device_errors(self):
        from sgl_amd_mono.common import guard

        ep = self.runner.epoch
        if bool(ep[guard.ERR_DEADLINE:guard.ERR_FAULT + 1].any()):
            return "epoch words %s" % ep.tolist()
        return None

    def facts(self):
        return {"pool_pages": POOL_PAGES, "page": PAGE, "req_rows": REQ_ROWS, "window": WINDOW,
                "rope_theta": ROPE_THETA, "layers": len(self.layers),
                "stage_env": os.environ.get("SGLANG_AMD_MONO_DRAFT_STAGE"),
                "kernels": [str(k) for k in list(getattr(self.runner, "_pre", {})) + list(getattr(self.runner, "_launch", {}))]}

    def close(self):
        for name in ("peers", "peers_post"):
            close = getattr(getattr(self.runner, name, None), "close", None)
            if close:
                close()
