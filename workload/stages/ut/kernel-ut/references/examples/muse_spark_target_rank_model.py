"""rank_model.py of task `muse_mono_target_layer` (UT contract family `mono_layer`): the model-specific half of
the rank harness. WORKED EXAMPLE for references/mono-layer-ut.md.

Provenance (2026-11-12): written by a fresh UT-lane agent in the end-to-end pipeline test
(task split -> scaffold_ut -> FILL), from the scaffold and the earlier
example; native path made teacher-forced with near-tie routing alignment by the implementer (W8).
GPU-verified on the served Muse Spark TP8/EP8 image (1106, sgl_amd_mono ef38b0e): every gate of the
contract including the native oracle floor (runs/gpu/W8b), negative controls, replay + interleave,
rank_chain timing.

rank_harness.py (generic, frozen, never edited) owns the process topology, the command loop,
poisoning, cache snapshot / restore / stray counting, graph replay, the interleave sequence,
rank_chain timing and the JIT cache key. It imports this file and asks RankModel for everything
that depends on the model. Structure follows references/examples/muse_spark_target_rank_model.py
(verified on GPU); every assumption it makes was re-checked against the installed package
(ef38b0e, byte-identical to kernel_src/sgl_amd_mono) and the deployed caller, see the notes
below. Constants are this model's (sgl_amd_mono/muse_spark/gfx950/config.py) or this task's
cases (run/ut_cases.json), not copied numbers.

Entry under test: `sgl_amd_mono.muse_spark.gfx950.runner:MuseMonoRunner.launch` (runner.py:504).
Live call site (installed sglang 0.5.21+gc722c0a1 + sgl_amd_mono ef38b0e; the profile's frames,
evidence/shapes/*.json, show the same chain):
  sglang/srt/monokernels/runtime.py:330 `forward` (decorated at :329, the line the profile's frame reports) (installed by _install_forward as model.forward
  for every served ShapeKey) -> `built.forward(Plan(key=key), *args, **kwargs)` (:349)
  -> sgl_amd_mono/muse_spark/spec.py:194 `_Built.forward` -> :207 `self._backbone(...)`
  -> spec.py:210 `_Built._backbone`: stream = input_embeds.to(residual_dtype) (fp32), then for
     i in range(62): `stream, attn_input = r.launch(i=i, rows=rows, q_len=q_len,
     positions=positions, forward_batch=forward_batch, stream=stream, attn_input=attn_input,
     pre_norm_dtype=pre_norm_dtype)` (:229, keywords only; pre_norm_dtype = bf16 only for the
     last layer, None otherwise), and after the loop `r.end_step()` (:239). Every TP rank makes
     the same calls in the same order. rows = plan.key.rows (8 or 16 here: C1 / C2 verify),
     q_len = plan.key.tokens_per_request = 8 (DFLASH, 8 draft tokens).
  Served layers 0 and 61 hold bf16 experts and take the module fallback (runner.py:508-514,
  run/server_calls.json: every recorded call is i = 0 or 61, line 509); layers 1..60 are MXFP4:
  layer 1's launch runs its K1 standalone then the merged launch with layer 2's K1 (n1), layers
  2..59 the merged launch only (n1), layer 60 the merged launch without a next K1 (n0, layer 61 is
  bf16). That is what one call() step reproduces with two MXFP4 layers and a trailing bf16 one.

Rules (references/mono-layer-ut.md section 2) and how they are met here:
  * one process a rank and the subject's real peer buffer: MuseMonoRunner over `group` (its
    PeerBuffer handshake), TPX experts and the sharded pooler as served (env defaults);
  * weights: TP shards (qkvg rows, o_proj columns, TP-expert slices) from a rank seed, replicated
    tensors from a seed common to every rank; every weighted norm 1 + 0.1 N(0,1), asserted not
    all ones and pairwise different. The q / k head norms are weightless in this model
    (weights.LayerViews carries no q / k gain; K1 normalizes each head and scales q by QK_SCALE);
  * index dtypes as served (positions int64: run/server_calls.json; out_cache_loc / seq_lens
    int64; page tables int32 contiguous, runner.py:491), page tables over a shuffled pool; the
    SWA pool is a different, smaller tensor set behind a non-identity page map (asserted);
  * kv_targets / writes name exactly the slots a launch writes.

Paths (PATHS[0] the deployed one):
  deployed  runner.use_layer True: launch(0) = layer 0's K1 + the merged launch with layer 1's K1
            (writes layers 0 and 1); launch(1) = the merged launch without a next K1 (writes nothing);
  split     runner.use_layer False (runner.py:183, instance attribute): K1 / K-attn / K2 / K3 a
            layer, launch(i) writes layer i;
  native    each launch's math by the per-stage torch references of sgl-amd-mono test/kernels
            (test_muse_spark_k1.py:47 `_reference`, test_muse_spark_kattn.py:35, test_muse_spark_k2.py:61,
            test_muse_spark_k3.py:64 `_mx_dequant` / :88 `_reference`) on this rank's shards, the
            o_proj partials and the EP experts' finalize partials combined with
            torch.distributed.all_reduce over `group` (gloo, CPU tensors). Teacher-forced (W8): launch
            i first runs the deployed launch on the deployed path's state before it and computes its
            native result from that same state, so one layer's error never compounds into the next
            (without it a layer-0 near-tie routing flip took layer 1's attention to 15 dB); where the
            kernel's top-8 differs from the reference's only within TIE_TOL of the boundary the
            kernel's picks are taken (sgl-amd-mono test_muse_spark_k2.py `_check_routing`'s rule).
            Poisons and writes like the deployed path (layers 0 and 1 before launch 0); each native
            K1 overwrites its layer's rows with its own values.
            Bounds: the family's native SQNR floors, calibrated on this example (mono_layer.json
            bounds_doc): baseline vs native stream_out ~27 dB, layer_out ~22, attn_out ~24.
"""

import types

import torch

WEIGHT_SEED = 20261112
# rotary rows: the served contexts reach 566410 keys (run/ut_cases.json DECODE_1_bucket of c10)
POSITIONS = 1 << 20
# full-pool pages a layer: the largest step is one 566410-key request (35401 pages) or two
# 177184-key ones (22148); the interleave draws from the same contexts
POOL_PAGES = 80000
# SWA-pool pages: smaller than the full pool (as SGLang's --swa-full-tokens-ratio 0.3 pool),
# room for MAX_TOKENS / 8 = 8 requests' windows (~146 pages each)
SWA_POOL_PAGES = 6151
SWA_SLACK = 256              # keys mapped before a window's first key (one K-attn chunk, layout.ATTN_CHUNK)
Q_LEN = 8                    # muse_spark.DRAFT_TOKENS; spec.py: q_len = speculative_num_draft_tokens
FULL, SWA = 0, 1             # contract axis `pools`
DEADLINE_S = 60.0            # in-kernel wait deadline (serving: guards.deadline_s = min(MAX, watchdog / 2))
USE_LAYER = {"deployed": True, "split": False}   # runner.use_layer per kernel path
# which layers' caches launch i writes, per path (the poisoning targets)
# native: each launch first runs the deployed launch as its teacher (W8), so layer 1's rows are
# written by deployed launch 0 before native launch 1 overwrites them with its own K1: poison both
# layers before launch 0, exactly as on the deployed path
WRITES = {"deployed": ((0, 1), ()), "split": ((0,), (1,)), "native": ((0, 1), ())}
TIE_TOL = 1e-3   # sgl-amd-mono test_muse_spark_k2.py:33: a top-8 pick within this of the boundary may flip
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def is_swa(layer):
    """UT layer j stands for served layer j + 1. Served layer_types: full attention iff
    i % 4 == 1 (sgl-amd-mono test/unit/test_muse_spark_spec.py:46, 16 full / 46 sliding of 62;
    corroborated by run/server_calls.json: the served layer 0 runs `_swa_verify_partials`, the
    served layer 61 (61 % 4 == 1) does not). So UT layer j is full iff j % 4 == 0: layer 0 full,
    layer 1 sliding, and rank_chain's 10 layers keep the served 1-in-4 share of full attention."""
    return layer % 4 != 0


def pool_of(layer):
    return SWA if is_swa(layer) else FULL


def _cfg():
    from sgl_amd_mono.muse_spark.gfx950 import config
    return config


def _cos_sin(dev):
    """fp32 [POSITIONS, HEAD_DIM] (cos | sin), the layout weights._attention_reasons checks."""
    HEAD_DIM = _cfg().HEAD_DIM
    inv = 1.0 / (500000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=dev) / HEAD_DIM))
    freqs = torch.arange(POSITIONS, dtype=torch.float32, device=dev)[:, None] * inv[None, :]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def _expert_group(dev, layer, e0):
    """EP rank e0 / 32's whole experts of `layer` in the Quark layout (w13, s13, w2, s2), from a
    seed every rank shares: make_views cuts the TP-expert slices from them and the native path
    recomputes them for its own EP group."""
    c = _cfg()
    ge = torch.Generator(device=dev).manual_seed(WEIGHT_SEED + 1000 * (layer + 1) + e0)
    w13 = torch.randint(0, 256, (c.LOCAL_EXPERTS, 2 * c.INTER, c.MOE_HIDDEN // 2), device=dev, dtype=torch.uint8, generator=ge)
    s13 = torch.randint(118, 128, (c.LOCAL_EXPERTS, 2 * c.INTER, c.MOE_HIDDEN // 32), device=dev, dtype=torch.uint8, generator=ge)
    w2 = torch.randint(0, 256, (c.LOCAL_EXPERTS, c.MOE_HIDDEN, c.INTER // 2), device=dev, dtype=torch.uint8, generator=ge)
    s2 = torch.randint(118, 128, (c.LOCAL_EXPERTS, c.MOE_HIDDEN, c.INTER // 32), device=dev, dtype=torch.uint8, generator=ge)
    return w13, s13, w2, s2


def make_views(dev, rank, world, layer, cos_sin):
    """One MXFP4 layer's LayerViews (weights.py:388) with deployed-layout weights: replicated
    tensors (router hi / lo, bias, norms, residual gates, pooler, unpooler) from a seed common to
    every rank; TP-sharded ones (qkvg rows [QKVG_ROWS, HIDDEN], o_proj columns [HIDDEN, Q_K]) from
    a rank seed; and this rank's TP-expert slices of the whole 256-expert table in global expert
    order, cut as weights.tp_expert_copies (weights.py:423) lays them out (gate rows
    d RS .. and up rows INTER + d RS .. of w13 / s13, the d-th CS columns of w2 / s2). Set as
    runner.tpx_w before bind, so bind never builds them over SGLang's device group.
    alpha / beta in [0.5, 1] / [0, 0.5] (a norm-preserving gate's range)."""
    c = _cfg()
    from sgl_amd_mono.muse_spark.weights import LayerViews

    bf = torch.bfloat16
    gs = torch.Generator(device=dev).manual_seed(WEIGHT_SEED + layer)
    gr = torch.Generator(device=dev).manual_seed(WEIGHT_SEED + layer + 100 * (rank + 1))

    def rn(shape, g, scale=1.0):
        return torch.randn(shape, device=dev, generator=g) * scale

    def ru(shape, g):
        return torch.rand(shape, device=dev, generator=g)

    router = rn((c.HIDDEN, c.EXPERTS), gs, 0.02)
    hi = router.T.to(bf).contiguous()
    lo = (router.T - hi.float()).to(bf).contiguous()
    RS, CS = c.INTER // world, c.MOE_HIDDEN // world
    w13s, s13s, w2s, s2s = [], [], [], []
    for e0 in range(0, c.EXPERTS, c.LOCAL_EXPERTS):
        w13, s13, w2, s2 = _expert_group(dev, layer, e0)

        def rows13(t):
            return torch.cat([t[:, rank * RS:(rank + 1) * RS], t[:, c.INTER + rank * RS:c.INTER + (rank + 1) * RS]], dim=1)

        w13s.append(rows13(w13)); s13s.append(rows13(s13))
        w2s.append(w2[:, rank * CS:(rank + 1) * CS]); s2s.append(s2[:, rank * CS:(rank + 1) * CS])
        del w13, s13, w2, s2
    tpx = tuple(torch.cat(x).contiguous() for x in (w13s, s13s, w2s, s2s))
    del w13s, s13s, w2s, s2s
    view = LayerViews(
        layer_id=layer, kind="mxfp4", is_swa=is_swa(layer), use_rope=True, cos_sin=cos_sin,
        w_qkvg=rn((c.QKVG_ROWS, c.HIDDEN), gr, 0.02).to(bf),
        w_o=rn((c.HIDDEN, c.Q_K), gr, 0.02).to(bf),
        alpha=ru(c.HIDDEN, gs) * 0.5 + 0.5, beta=ru(c.HIDDEN, gs) * 0.5,
        pre_w=(1.0 + 0.1 * rn(c.HIDDEN, gs)).to(bf),
        gate_hi=hi, gate_lo=lo, bias=rn(c.EXPERTS, gs, 0.01),
        w_pool=rn((c.MOE_HIDDEN, c.HIDDEN), gs, 0.02).to(bf),
        xn_w=(1.0 + 0.1 * rn(c.MOE_HIDDEN, gs)).to(bf),
        w13=None, s13=None, w2=None, s2=None,
        pen_w=(1.0 + 0.1 * rn(c.MOE_HIDDEN, gs)).to(bf),
        w_unpool=rn((c.HIDDEN, c.MOE_HIDDEN), gs, 0.02).to(bf),
        pfw=(1.0 + 0.1 * rn(c.HIDDEN, gs)).to(bf),
        alpha2=ru(c.HIDDEN, gs) * 0.5 + 0.5, beta2=ru(c.HIDDEN, gs) * 0.5,
        npw=(1.0 + 0.1 * rn(c.HIDDEN, gs)).to(bf),
    )
    norms = [view.pre_w, view.xn_w, view.pen_w, view.pfw, view.npw]
    assert all(not bool((n == 1).all()) for n in norms), "a norm weight is all ones"
    for a in range(len(norms)):
        for b in range(a + 1, len(norms)):
            if norms[a].shape == norms[b].shape:
                assert not bool(torch.equal(norms[a], norms[b])), "two norm weights are equal"
    return view, tpx


def _module(layer, rank, swa):
    """The module surface the runner reads (runner.SGLANG_READS: self_attn.attn.layer_id /
    sliding_window_size / scaling / k_scale / v_scale, mlp.local_expert_start). The served
    sliding layers' window is config.sliding_window - 1 = WINDOW (rocm_muse_spark/model.py:318)."""
    c = _cfg()
    attn = types.SimpleNamespace(layer_id=layer, sliding_window_size=c.WINDOW if swa else -1, scaling=c.SOFTMAX_SCALE,
                                 k_scale=None, v_scale=None)
    return types.SimpleNamespace(self_attn=types.SimpleNamespace(attn=attn),
                                 mlp=types.SimpleNamespace(local_expert_start=rank * c.LOCAL_EXPERTS))


def make_caches(dev, rank, n_layers):
    """Seeded FP8 e4m3 caches [slots, LOCAL_KV_HEADS, HEAD_DIM] (the shape cache_reasons checks),
    layer i in the pool pool_of(i) names (full: POOL_PAGES pages, SWA: SWA_POOL_PAGES). Drawn in
    order from one rank seed, so two models with the same first layers (rank_chain's parity
    check) hold the same caches for them."""
    c = _cfg()
    g = torch.Generator(device=dev).manual_seed(WEIGHT_SEED + 17 * (rank + 1))
    caches = []
    for i in range(n_layers):
        slots = (SWA_POOL_PAGES if pool_of(i) == SWA else POOL_PAGES) * c.PAGE
        k = (torch.randn(slots, c.LOCAL_KV_HEADS, c.HEAD_DIM, device=dev, generator=g) * 2).to(torch.float8_e4m3fn)
        v = torch.randn(slots, c.LOCAL_KV_HEADS, c.HEAD_DIM, device=dev, generator=g).to(torch.float8_e4m3fn)
        caches.append((k, v))
    return caches


class _Pool:
    """The SWAKVPool surface the runner binds (runner.SGLANG_READS: layers_mapping,
    get_kv_buffer, full_to_swa_index_mapping). The full-to-SWA slot table is int64 and
    contiguous; its contents are the step's, written in place (the runner keeps the tensor)."""

    def __init__(self, caches, n_layers, dev):
        self.caches = caches
        self.layers_mapping = {i: (i, pool_of(i) == SWA) for i in range(n_layers)}
        self.full_to_swa_index_mapping = torch.zeros(POOL_PAGES * _cfg().PAGE, device=dev, dtype=torch.int64)

    def get_kv_buffer(self, layer_id):
        return self.caches[layer_id]


class _Mode:
    """forward_batch.forward_mode: a DFLASH verify step (runner._attn_args: extra = q_len)."""

    def is_target_verify(self):
        return True


# ----------------------------------------------------------------------------- native references
def _bf(t):
    return t.to(torch.bfloat16).float()


def _rms(x, eps, w=None):
    """RMSNorm in fp32, the weight after the scale, one bf16 rounding (test_muse_spark_k2.py:52)."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if w is not None:
        y = y * w.float()
    return y.to(torch.bfloat16)


def _mx_dequant(x):
    """x (fp32 [..., N]) through aiter's `_mxfp4_quant_op` and back (test_muse_spark_k3.py:64)."""
    shape = x.shape
    xb = x.float().reshape(*shape[:-1], -1, 32)
    amax = xb.abs().amax(-1, keepdim=True)
    bits = (amax.view(torch.int32) + 0x200000) & -0x800000
    unb = (((bits >> 23) & 0xFF) - 129).clamp(min=-127)
    qx = xb * torch.ldexp(torch.ones_like(amax), -unb)
    a = qx.abs()
    ab = a.view(torch.int32)
    den = ((a + 4194304.0).view(torch.int32) - (149 << 23)) & 0xFF
    nrm = ((ab + ((-126 << 23) + 0x1FFFFF) + ((ab >> 22) & 1)) >> 22) & 0xFF
    code = torch.where(a >= 6.0, torch.full_like(den, 7), torch.where(a < 1.0, den, nrm))
    mag = torch.tensor(E2M1, device=x.device)[code.long()]
    val = torch.where(qx < 0, -mag, mag)
    return (val * torch.ldexp(torch.ones_like(amax), unb)).reshape(shape)


def _fp4_decode(packed, scales):
    """Quark MXFP4 [rows, k / 2] u8 (low nibble the even k) + E8M0 [rows, k / 32] -> fp32
    [rows, k] (test_muse_spark_k3.py:49 `_fp4_weight`)."""
    rows, half = packed.shape
    k = 2 * half
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(rows, k)
    mag = torch.tensor(E2M1, device=packed.device)[(codes & 7).long()]
    val = torch.where(codes & 8 != 0, -mag, mag)
    scale = torch.ldexp(torch.ones(1, device=packed.device), scales.to(torch.int32) - 127)
    return (val.reshape(rows, k // 32, 32) * scale[..., None]).reshape(rows, k)


class RankModel:
    # ---- frozen by the scaffold from run/ut_contract.json; change only with the contract
    WORLD = 8                  # ranks (const axis world)
    LAUNCHES = 2               # layer launches a call() step makes (const axis launches)
    POOLS = 2                  # KV pools (const axis pools): index 0 full, 1 sliding window
    HIDDEN = 8192
    TOP_K = 8                  # routing ids per token (layer_kind routed_moe), None for dense
    KV_WIDTH = 128             # elements of one token's K (or V) row, decoded to float32
    STREAM_DTYPE = torch.float32
    INPUT_DTYPE = torch.bfloat16
    # ---- paths: PATHS[0] is the deployed one call() runs; every other entry is an oracle path
    # (cases.ORACLES[name] runs the contract oracle whose `path` names it)
    PATHS = ('deployed', 'split', 'native')
    PACKAGE = 'sgl_amd_mono'         # the whole-package bind: its bytes key the JIT cache
    JIT_CACHE_ENV = "FLYDSL_RUNTIME_CACHE_DIR"   # the JIT's cache-dir variable; None if it has none
    KEEP_REGIONS = r"flag|trace|ugq"           # scratch regions never poisoned (hand-off flags ...)
    NOTES = [
        "q / k head norms are weightless in this model (weights.LayerViews has no q / k gain): nothing to vary",
        "r8 is one request: multi-request page-table indexing is covered by the r16 cases only",
        "req_to_token is not read by the runner (runner.SGLANG_READS) and is not built",
        "every UT layer uses RoPE (use_rope=1); the served NoPE schedule (config layer_rope_theta) was not "
        "available to the lane, so a NoPE full-attention layer's K1 branch (use_rope=0) is not exercised",
        "UT layer j = served layer j + 1: full attention iff j % 4 == 0 (served i % 4 == 1)",
        "native path: torch references of sgl-amd-mono test/kernels per stage, o_proj and EP finalize "
        "partials combined with dist.all_reduce over the gloo group; experts by EP group, not TPX slices",
    ]

    @staticmethod
    def page_width(context):
        """Page-table columns a request of `context` keys needs (page size config.PAGE = 16)."""
        return (int(context) + 15) // 16

    def __init__(self, rank, dev, group, n_layers=None):
        """call() step (n_layers None): two MXFP4 layers (0 full, 1 sliding) and a trailing bf16
        one (as served layer 61 after layer 60), so launch(1) is the n0 merged launch. rank_chain
        (n_layers N): N MXFP4 layers, every timed launch an n1 merged launch with a next layer.
        The runner is built as spec._build (spec.py:385) builds it: MuseMonoRunner(views, modules,
        boundaries, rank, npes, cpu_group, device), then set_deadline; runner.tpx_w is set
        before bind (runner.py:407: bind would build them over SGLang's device group)."""
        from sgl_amd_mono.muse_spark.gfx950.runner import MuseMonoRunner

        c = _cfg()
        assert (c.HIDDEN, c.TOP_K, c.LOCAL_KV_HEADS * c.HEAD_DIM) == (self.HIDDEN, self.TOP_K, self.KV_WIDTH)
        assert c.PAGE == 16 and Q_LEN in (8,), (c.PAGE, Q_LEN)
        self.rank, self.dev, self.group = rank, dev, group
        self.path = self.PATHS[0]
        self.cos_sin = _cos_sin(dev)
        views, tpx = [], []
        n_mx = 2 if n_layers is None else n_layers
        for layer in range(n_mx):
            v, t = make_views(dev, rank, self.WORLD, layer, self.cos_sin)
            views.append(v)
            tpx.append(t)
        if n_layers is None:
            # layer 2 has bf16 experts (as the model's last layer): layer 1 then ends its merged
            # launch without a next K1 (runner._next_k1). It is never launched.
            views.append(views[1].__class__(**{**{f: getattr(views[1], f) for f in views[1].__struct_fields__},
                                               "layer_id": 2, "kind": "bf16", "is_swa": is_swa(2)}))
            tpx.append(None)
        self.views = views
        n = len(views)
        modules = [_module(i, rank, pool_of(i) == SWA) for i in range(n)]
        self.runner = MuseMonoRunner(views=views, modules=modules, boundaries=[None] * n, rank=rank, npes=self.WORLD,
                                     cpu_group=group, device=dev)
        if not hasattr(self.runner, "use_layer"):
            raise RuntimeError("MuseMonoRunner has no use_layer: the split oracle path cannot be selected")
        if not (self.runner.tpx and self.runner.pshard and not self.runner.fp8_dense):
            # the served build: SGLANG_AMD_MONO_MUSE_TPX=1, _PSHARD=1, _FP8_DENSE=0 (defaults, serving.json sets none)
            raise RuntimeError("runner tpx=%s pshard=%s fp8_dense=%s, the server ran tpx / pshard / bf16 dense"
                               % (self.runner.tpx, self.runner.pshard, self.runner.fp8_dense))
        self.runner.set_deadline(DEADLINE_S)
        self.runner.tpx_w = tpx
        self.caches = make_caches(dev, rank, n)
        self.pool = _Pool(self.caches, n, dev)
        self.runner.bind(types.SimpleNamespace(forward_metadata=None), self.pool)
        self._compiled = set()
        self.tie_stats = []      # native path: near-tie routing adoptions a launch (W8)
        self._teacher = None

    # ------------------------------------------------------------------ layers and paths
    def pool_of(self, layer):
        return pool_of(layer)

    def writes(self, path, i):
        return WRITES[path][i]

    def set_path(self, path):
        if path not in self.PATHS:
            raise ValueError("unknown path %r" % (path,))
        self.path = path
        self.runner.use_layer = USE_LAYER.get(path, True)

    # ------------------------------------------------------------------ a step's metadata
    def step_meta(self, rows, context, width=None):
        """The case's page table (a shuffled full pool, distinct pages a request), its SWA map and
        SWA page table, lengths, slots and positions -- the same on every rank, as the TP ranks
        share one batch. Requests = rows / Q_LEN; request r has max(Q_LEN + 1, context - 37 r)
        keys including the step's Q_LEN rows (the census context is a request's keys). seq_lens
        are the keys before the step (the runner adds q_len on a verify step)."""
        WINDOW, PAGE = _cfg().WINDOW, _cfg().PAGE
        if rows % Q_LEN:
            raise ValueError("rows %d is not a multiple of q_len %d (the monokernel path needs it)" % (rows, Q_LEN))
        R = rows // Q_LEN
        lens = [max(Q_LEN + 1, int(context) - 37 * r) for r in range(R)]
        g = torch.Generator(device="cpu").manual_seed(7919 * rows + int(context))
        perm = torch.randperm(POOL_PAGES - 1, generator=g) + 1          # full page 0 unused
        swa_perm = torch.randperm(SWA_POOL_PAGES - 1, generator=g) + 1   # SWA page 0 is the dummy
        blocks = max((L + PAGE - 1) // PAGE for L in lens)
        width = max(width or 0, blocks)
        table = torch.zeros(R, width, dtype=torch.int32)
        swa_table = torch.zeros(R, width, dtype=torch.int32)
        page_map = torch.zeros(POOL_PAGES, dtype=torch.int64)
        slots, at, sat = [], 0, 0
        for r in range(R):
            n = (lens[r] + PAGE - 1) // PAGE
            if at + n > POOL_PAGES - 1:
                raise RuntimeError("full pool too small: %d pages" % (at + n))
            table[r, :n] = perm[at:at + n].to(torch.int32)
            at += n
            first = max(0, lens[r] - Q_LEN - WINDOW - SWA_SLACK) // PAGE
            k = n - first
            if sat + k > SWA_POOL_PAGES - 1:
                raise RuntimeError(f"SWA pool too small: {sat + k} in-window pages")
            sp = swa_perm[sat:sat + k]
            sat += k
            page_map[table[r, first:n].to(torch.int64)] = sp
            swa_table[r, first:n] = sp.to(torch.int32)
            for j in range(Q_LEN):
                p = lens[r] - Q_LEN + j
                slots.append(int(table[r, p // PAGE]) * PAGE + p % PAGE)
        swa_map = (page_map[:, None] * PAGE + torch.arange(PAGE, dtype=torch.int64)[None, :]).reshape(-1)
        # DESIGN 1.5: the SWA table is not the full one and the map is not the identity
        assert not torch.equal(swa_table, table), "swa_page_table equals kv_indices"
        assert bool((swa_map != torch.arange(swa_map.numel(), dtype=torch.int64)).any()), "identity SWA map"
        m = dict(
            table=table, swa_table=swa_table, swa_map=swa_map,
            seq_lens=torch.tensor([L - Q_LEN for L in lens], dtype=torch.int64),
            out_cache_loc=torch.tensor(slots, dtype=torch.int64),
            positions=torch.tensor([lens[r] - Q_LEN + j for r in range(R) for j in range(Q_LEN)], dtype=torch.int64),
            req_pool_indices=torch.randperm(4096, generator=g)[:R].to(torch.int64),
        )
        m = {k: v.to(self.dev) for k, v in m.items()}
        assert m["table"].data_ptr() != m["swa_table"].data_ptr()
        return m

    def kv_targets(self, m):
        """The step's slots in each pool (graph-safe: device ops on the meta tensors)."""
        return (m["out_cache_loc"], m["swa_map"][m["out_cache_loc"]])

    def bind(self, m):
        """Point the runner's backend at the step's tables (backend.forward_metadata.kv_indices /
        swa_page_table, runner.py:490), write the step's SWA map in place, and -- as spec._build
        does for every width before serving (runner.precompile) -- build the path's launches of
        this width here, before the harness's barrier, so no rank compiles while its peers spin
        in an in-kernel wait. Returns the forward batch launch() passes."""
        self.runner.backend.forward_metadata = types.SimpleNamespace(kv_indices=m["table"], swa_page_table=m["swa_table"])
        self.pool.full_to_swa_index_mapping.copy_(m["swa_map"])
        rows = int(m["out_cache_loc"].numel())
        key = (rows, bool(self.runner.use_layer))
        if key not in self._compiled and not torch.cuda.is_current_stream_capturing():
            self.runner.precompile(rows, Q_LEN)
            self._compiled.add(key)
        return types.SimpleNamespace(out_cache_loc=m["out_cache_loc"], seq_lens=m["seq_lens"], positions=m["positions"],
                                     req_pool_indices=m["req_pool_indices"], forward_mode=_Mode())

    # ------------------------------------------------------------------ launches
    def launch(self, i, rows, m, fb, stream, x):
        """spec.py:229's call, keywords and argument types as served (stream fp32 [rows, HIDDEN]
        contiguous, attn_input bf16, positions int64, pre_norm_dtype None for an MXFP4 layer)."""
        if self.path == "native":
            return self._native_launch(i, rows, m, fb, stream, x)
        return self.runner.launch(i=i, rows=rows, q_len=Q_LEN, positions=m["positions"], forward_batch=fb,
                                  stream=stream, attn_input=x, pre_norm_dtype=None)

    def end_step(self):
        """spec.py:239: r.end_step() (the epoch bump), after every path's launches."""
        self.runner.end_step()

    def buffers(self):
        """ids: the routing ids (-7 before each launch); produced: float buffers each launch
        writes (NaN before each launch); inputs: float buffers a launch reads from the previous
        launch (NaN before launch 0 only -- the merged path's launch 1 reads the q / k / v / g
        launch 0's next K1 left, the split path's K1 reads attn_in); diag: report-only."""
        r = self.runner
        q, k, v, g = (getattr(r, n, None) for n in ("q", "k", "v", "g"))
        return {"ids": getattr(r, "ids", None),
                "produced": {n: getattr(r, n, None) for n in ("o", "scores", "xin")},
                "inputs": {n: getattr(r, n, None) for n in ("q", "k", "v", "g", "attn_in")},
                "diag": {"o": getattr(r, "o", None),
                         "qkvg": (q, k, v, g) if all(t is not None for t in (q, k, v, g)) else None,
                         "scores": getattr(r, "scores", None), "xin": getattr(r, "xin", None)}}

    def shared_layouts(self, rows):
        """(specs, kept): [(name, buffer, layout_fn -> {region: (offset, bytes)})] read from the
        subject's own modules (layout.py, stages/post_attn.py, stages/ffn.py; a candidate may move
        them: report, do not crash). The peer buffer's guard flags follow sym_bytes(npes) and stay
        outside the layout (unpoisoned)."""
        r = self.runner
        specs, kept = [], []
        try:
            from sgl_amd_mono.muse_spark.gfx950.layout import k1_scratch_layout, kattn_scratch_layout
            specs += [("scratch_k1", getattr(r, "scratch_k1", None), k1_scratch_layout),
                      ("scratch_attn", getattr(r, "scratch_attn", None), lambda: kattn_scratch_layout(rows, Q_LEN))]
        except Exception as e:  # noqa: BLE001
            kept.append({"region": "scratch_k1, scratch_attn", "why": "layout unavailable: %r" % e})
        try:
            from sgl_amd_mono.muse_spark.gfx950.stages.post_attn import k2_scratch_layout, sym_layout
            specs += [("scratch_k2", getattr(r, "scratch_k2", None), k2_scratch_layout)]
            peers = getattr(getattr(r, "peers", None), "bytes", None)
            specs += [("peers", peers, lambda: sym_layout(r.npes))]
        except Exception as e:  # noqa: BLE001
            kept.append({"region": "scratch_k2, peers", "why": "layout unavailable: %r" % e})
        try:
            from sgl_amd_mono.muse_spark.gfx950.stages.ffn import k3_scratch_layout
            specs += [("scratch_k3", getattr(r, "scratch_k3", None), k3_scratch_layout)]
        except Exception as e:  # noqa: BLE001
            kept.append({"region": "scratch_k3", "why": "layout unavailable: %r" % e})
        return specs, kept

    # ------------------------------------------------------------------ the native path
    def _all_reduce(self, t):
        """fp32 sum over the ranks through the gloo group (CPU tensors), back on this device."""
        import torch.distributed as dist

        h = t.float().cpu().contiguous()
        dist.all_reduce(h, op=dist.ReduceOp.SUM, group=self.group)
        return h.to(self.dev)

    def _native_k1(self, L, x, m):
        """test_muse_spark_k1.py:47 `_reference`: q | k | v | gate projection rounded once, the
        weightless per-head norm, q x QK_SCALE rounded, interleaved RoPE with bf16 cos / sin and
        each product rounded; k / v to the layer's cache as fp8 (clamp 448)."""
        c = _cfg()
        rows = x.shape[0]
        y = (x.float() @ L.w_qkvg.float().T).to(torch.bfloat16)
        q = y[:, :c.K_OFF].reshape(rows, c.LOCAL_Q_HEADS, c.HEAD_DIM)
        k = y[:, c.K_OFF:c.V_OFF].reshape(rows, c.LOCAL_KV_HEADS, c.HEAD_DIM)
        v = y[:, c.V_OFF:c.G_OFF].reshape(rows, c.LOCAL_KV_HEADS, c.HEAD_DIM)
        g = y[:, c.G_OFF:]
        pos = m["positions"]

        def norm(t):
            tf = t.float()
            return _bf(tf * torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + c.RMS_EPS))

        def rotate(value):
            if not L.use_rope:
                return value.to(torch.bfloat16)
            cs = self.cos_sin[pos]
            d = torch.arange(c.HEAD_DIM, device=value.device)
            cos = _bf(cs[:, None, d // 2])
            sin = _bf(cs[:, None, c.HEAD_DIM // 2 + d // 2])
            other = value.reshape(rows, value.shape[1], c.HEAD_DIM // 2, 2).flip(-1).reshape(rows, value.shape[1], c.HEAD_DIM)
            rotated = torch.where(d % 2 == 0, -other, other)
            return (_bf(value * cos) + _bf(rotated * sin)).to(torch.bfloat16)

        q = rotate(_bf(norm(q) * c.QK_SCALE))
        k = rotate(norm(k))
        return q.reshape(rows, -1), k, v.to(torch.bfloat16), g

    def _native_attn(self, i, q, m):
        """test_muse_spark_kattn.py:35 `_reference`: per request and KV head, causal (and on a
        sliding layer windowed) softmax over the keys its page table maps, K / V fp32 from the
        fp8 cache, P rounded to bf16; o bf16."""
        c = _cfg()
        sw = is_swa(i)
        k_cache, v_cache = self.caches[i]
        table = m["swa_table"] if sw else m["table"]
        window = c.WINDOW if sw else 0
        scale = float(c.SOFTMAX_SCALE)
        rows = q.shape[0]
        R = rows // Q_LEN
        GQA = c.LOCAL_Q_HEADS // c.LOCAL_KV_HEADS
        out = torch.zeros(rows, c.Q_K, device=q.device)
        dev = q.device
        for r in range(R):
            keys = int(m["seq_lens"][r]) + Q_LEN
            ctx = keys - Q_LEN
            lo = max(0, ctx - window + 1) if window else 0
            kidx = torch.arange(lo, keys, device=dev)
            slots = table[r].long()[kidx // c.PAGE] * c.PAGE + kidx % c.PAGE
            K = k_cache[slots].float()          # [n, KV heads, HEAD_DIM]
            V = v_cache[slots].float()
            qpos = torch.arange(Q_LEN, device=dev)
            ok = kidx[None, :] < (ctx + qpos[:, None] + 1)
            if window:
                ok &= (ctx + qpos[:, None] - kidx[None, :]) < window
            qr = q[r * Q_LEN:(r + 1) * Q_LEN].float().reshape(Q_LEN, c.LOCAL_KV_HEADS, GQA, c.HEAD_DIM)
            for h in range(c.LOCAL_KV_HEADS):
                s = scale * torch.einsum("qgd,nd->qgn", qr[:, h], K[:, h])
                s = torch.where(ok[:, None, :], s, torch.full_like(s, float("-inf")))
                p = torch.softmax(s, dim=-1).to(torch.bfloat16).float()
                o = torch.einsum("qgn,nd->qgd", p, V[:, h])
                out[r * Q_LEN:(r + 1) * Q_LEN, h * GQA * c.HEAD_DIM:(h + 1) * GQA * c.HEAD_DIM] = o.reshape(Q_LEN, -1)
        return out.to(torch.bfloat16)

    def _native_experts(self, i, xn, ids, scores):
        """test_muse_spark_k3.py:88 `_reference`'s experts and finalize for this rank's EP group
        (experts rank * 32 ..): MXFP4 input, gate / up on the E2M1 weights in fp32, bf16(silu(gate)
        up) re-quantized, the down GEMM rounded, v = bf16(y pen_w), score / rms(v) weighted;
        returns the fp32 partial [rows, MOE_HIDDEN] (the all-reduce sums the ranks')."""
        c = _cfg()
        L = self.views[i]
        e0 = self.rank * c.LOCAL_EXPERTS
        w13, s13, w2, s2 = _expert_group(self.dev, i, e0)
        rows = xn.shape[0]
        out = torch.zeros(rows, c.MOE_HIDDEN, device=self.dev)
        xq = _mx_dequant(xn.float())
        for e in sorted(set(int(v) for v in ids.flatten().tolist())):
            le = e - e0
            if not 0 <= le < c.LOCAL_EXPERTS:
                continue
            tok, slot = (ids == e).nonzero(as_tuple=True)
            dec13 = _fp4_decode(w13[le], s13[le])                     # [2 INTER, MOE_HIDDEN]
            gu = xq[tok] @ dec13.T
            del dec13
            gate, up = gu[:, :c.INTER], gu[:, c.INTER:]
            mid = _mx_dequant(_bf(gate * torch.sigmoid(gate) * up))
            dec2 = _fp4_decode(w2[le], s2[le])                        # [MOE_HIDDEN, INTER]
            y = _bf(mid @ dec2.T)
            del dec2
            v = _bf(y * L.pen_w.float())
            sc = scores[tok, slot].float() / torch.sqrt(v.pow(2).mean(-1) + c.POST_EPS)
            out.index_add_(0, tok, sc[:, None] * v)
        del w13, s13, w2, s2
        return out

    def _align_ties(self, ids, biased, ids_k):
        """sgl-amd-mono test_muse_spark_k2.py:104 `_check_routing`'s rule, applied instead of
        asserted: where the kernel's top-8 differs from this reference's only by picks within
        TIE_TOL of the boundary (a near-tie the kernel's 24-bit routing key or GEMM order may flip),
        take the kernel's picks; any other difference keeps the reference's (and shows as an
        error). Returns (ids, tokens adopted, tokens that differ beyond a tie)."""
        out = ids.clone()
        adopted = beyond = 0
        for t in range(ids.shape[0]):
            ref, got = set(ids[t].tolist()), set(ids_k[t].tolist())
            if ref == got:
                continue
            extra = sorted(got - ref)
            bound = biased[t, ids[t]].min().item()
            worst = biased[t, extra].min().item()
            if worst >= bound - TIE_TOL and len(got) == len(ref):
                out[t] = ids_k[t]
                adopted += 1
            else:
                beyond += 1
        return out, adopted, beyond

    def _native_launch(self, i, rows, m, fb, stream, x):
        """W8: launch i teacher-forced. The deployed launch i runs first on the state the deployed
        path has before it (its inputs, the routing ids it picks); this launch's native result is
        then computed from that same state, so an error of one layer is measured once and never
        compounds into the next (a near-tie routing flip in layer 0 otherwise reaches layer 1's
        attention input). The returned stream / attention input are the native ones; the deployed
        state carries to the next launch in self._teacher."""
        if i == 0:
            self._teacher = (stream.detach().clone(), x.detach().clone())
        t_stream, t_x = self._teacher
        d_stream, d_x = self.runner.launch(i=i, rows=rows, q_len=Q_LEN, positions=m["positions"], forward_batch=fb,
                                           stream=t_stream.clone(), attn_input=t_x.clone(), pre_norm_dtype=None)
        ids_k = self.runner.ids[:rows].long().clone()
        self._teacher = (d_stream.detach().clone(), d_x.detach().clone())
        return self._native_math(i, rows, m, stream, t_stream, t_x, ids_k)

    def _native_math(self, i, rows, m, stream, t_stream, x, ids_k):
        """Layer i without monokernel code: K1 (and the cache write), the attention, K2's math
        (test_muse_spark_k2.py:61, the o_proj all-reduce as dist.all_reduce of the ranks' bf16
        partials in fp32, rounded once), K3's (test_muse_spark_k3.py:88 with the EP finalize
        partials summed over the ranks, rounded once, the unpooler and the FFN boundary, not the
        last layer). The stream is updated in place; returns (stream, next attention input)."""
        c = _cfg()
        L = self.views[i]
        bf = torch.bfloat16
        with torch.no_grad():
            q, k, v, g = self._native_k1(L, x, m)
            slots = m["out_cache_loc"]
            if is_swa(i):
                slots = m["swa_map"][slots]
            kc, vc = self.caches[i]
            # through uint8 views (as the harness poisons them): index_put on a float8 tensor is not
            # guaranteed on every torch build
            for cache, val in ((kc, k), (vc, v)):
                q8 = val.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                cache.view(torch.uint8).index_copy_(0, slots, q8.view(torch.uint8))
            o = self._native_attn(i, q, m)
            # K2: norm-gate, o_proj partial, the all-reduce, the attention boundary
            of = o.float().reshape(rows, -1, c.HEAD_DIM)
            rstd = torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + c.RMS_EPS)
            og = (of * rstd * torch.sigmoid(g.float().reshape(rows, -1, c.HEAD_DIM))).to(bf)
            part = (og.reshape(rows, -1).float() @ L.w_o.float().T).to(bf)
            ao = self._all_reduce(part.float()).to(bf)
            nb = _rms(ao, c.POST_EPS).float()
            st = t_stream.float()
            new = L.alpha * st + L.beta * nb
            xf = _rms(new, c.RMS_EPS, L.pre_w)
            logits = xf.float() @ L.gate_hi.float().T + xf.float() @ L.gate_lo.float().T
            sc = torch.sigmoid(logits * c.ROUTER_MULT)
            biased = sc + L.bias
            ids = torch.argsort(biased, dim=-1, descending=True, stable=True)[:, :c.TOP_K]
            ids, adopted, beyond = self._align_ties(ids, biased, ids_k)
            self.tie_stats.append({"launch": i, "tokens": rows, "near_tie_adopted": adopted, "beyond_tie": beyond})
            picks = sc.gather(1, ids)
            scores = picks / (picks.sum(-1, keepdim=True) + c.ROUTE_EPS) * c.ROUTE_SCALE
            pooled = (xf.float() @ L.w_pool.float().T).to(bf)
            xn = _rms(pooled, c.RMS_EPS, L.xn_w)
            # K3: experts (EP group), finalize all-reduce, unpooler, FFN boundary
            fin = self._all_reduce(self._native_experts(i, xn, ids, scores)).to(bf)
            branch = _bf(fin.float() @ L.w_unpool.float().T)
            post_input = _bf(branch * L.pfw.float())
            nb2 = _bf(post_input * torch.rsqrt(post_input.pow(2).mean(-1, keepdim=True) + c.POST_EPS))
            new2 = L.alpha2 * new + L.beta2 * nb2
            attn_out = (new2 * torch.rsqrt(new2.pow(2).mean(-1, keepdim=True) + c.RMS_EPS) * L.npw.float()).to(bf)
            stream.copy_(new2)
            r = self.runner
            r.ids[:rows].copy_(ids.to(torch.int32))
            r.scores[:rows].copy_(scores)
            r.xin[:rows].copy_(xn)
            r.o[:rows].copy_(o)
            r.attn_in[:rows].copy_(attn_out)
        return stream, r.attn_in[:rows]

    # ------------------------------------------------------------------ health
    def device_errors(self):
        """The guard block's error words (guard.ERR_DEADLINE .. ERR_FAULT) of the epoch buffer."""
        from sgl_amd_mono.common import guard

        ep = self.runner.epoch
        if bool(ep[guard.ERR_DEADLINE:guard.ERR_FAULT + 1].any()):
            return "epoch words %s" % ep.tolist()
        return None

    def facts(self):
        """Pool sizes and other facts for run/harness_report.jsonl."""
        r = self.runner
        return {"full_pages": POOL_PAGES, "swa_pages": SWA_POOL_PAGES, "swa_slack_keys": SWA_SLACK,
                "layers": len(self.views), "swa_layers": [i for i in range(len(self.views)) if is_swa(i)],
                "tpx": r.tpx, "pshard": r.pshard, "fp8_dense": r.fp8_dense, "deadline_s": DEADLINE_S,
                "kernels": [str(k) for k in list(getattr(r, "_layer", {})) + list(getattr(r, "_k1", {}))]}

    def close(self):
        """Release the peer buffer (IPC handles)."""
        close = getattr(getattr(self.runner, "peers", None), "close", None)
        if close:
            close()
