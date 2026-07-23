"""Live-path FlyDSL integration for vLLM int4 W4A16 fused-MoE (memory-neutral).

Replaces the Triton `fused_moe_kernel_gptq_awq` path with FlyDSL's
`compile_moe_gemm2` for the no-zp W4A16 grouped MoE GEMM.

Two-part design:
  1) LOAD TIME -- `convert_layer_inplace(layer)` (called from the WNA16 MoE
     method's process_weights_after_loading): converts each expert weight from
     vLLM's packed int4 layout ([E,N,K//2] uint8, low-nibble=even-k) into FlyDSL's
     shuffled packed-int4 layout, and the scales ([E,N,G] bf16) into FlyDSL's
     (E,G//2,N,2) bf16 layout.  The converted buffers have the SAME byte size as
     the originals, so they REPLACE the param storage in place -- memory-neutral.
     IMPORTANT: BOTH the weight AND the scale param must be re-homed
     (layer.w*_weight_packed.data AND layer.w*_weight_scale.data). Re-homing only
     the weight leaves the original scale storage alive on the layer while the
     converted scale is also cached -> scales DUPLICATED -> ~+14.5 GiB -> KV OOM.
     Conversion is chunked over experts so the transient stays ~1 GB.
     Per-tensor metadata is cached keyed by the new data_ptr.

  2) RUN TIME -- `flydsl_fused_experts_impl(...)` (called from fused_experts_impl
     when VLLM_USE_FLYDSL_MOE=1): runs gemm1(gate_up, no routed weight) ->
     silu_and_mul -> gemm2(down, routed weight) -> moe_sum, using the precomputed
     buffers.  The compiled executable is cached per (tensor, stage, M).

NOTE: once weights are converted in place, the Triton path can no longer read
them, so there is NO fallback for converted layers -- correctness is gated by the
offline validator (validate_shim_offline.py) and in-server GSM8K.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Make FlyDSL (`flydsl`, `kernels.moe_gemm_2stage`) importable. Self-contained: NO hardcoded
# machine-specific path. Set FLYDSL_ROOT to your FlyDSL checkout (kernels + build-fly bindings from
# the SAME tree); if unset, we rely on flydsl already being importable (pip-installed / on PYTHONPATH).
FLYDSL_ROOT = os.environ.get("FLYDSL_ROOT", "")
if FLYDSL_ROOT:
    FLY_PKG = os.path.join(FLYDSL_ROOT, os.environ.get("FLY_BUILD", "build-fly"), "python_packages")
    for _p in (FLYDSL_ROOT, FLY_PKG):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)

import torch  # noqa: E402

try:
    import triton  # noqa: E402
    import triton.language as tl  # noqa: E402
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

_flyc = None
_compile_moe_gemm2 = None
_compile_moe_gemm1 = None
_build_routing_buffers = None
_shuffle_weight = None
_shuffle_scale = None
_pack_int4 = None

# meta keyed by converted-tensor data_ptr:
#   dict(role, N, K, E, G, gs, scale=<flat bf16>, inter, hidden)
_WCACHE = {}
# compiled exe keyed by (data_ptr, stage, M, top_k)
_ECACHE = {}
# per-device cached zero-size fp32 scale sentinel (W4A16 -> no activation scale).
# Reused across every gemm call so we do not dispatch a fresh 0-elem alloc per stage.
_EMPTY_SCALE = {}


def _empty_scale(dev):
    t = _EMPTY_SCALE.get(dev)
    if t is None:
        t = torch.empty((0,), device=dev, dtype=torch.float32)
        _EMPTY_SCALE[dev] = t
    return t


def _lazy_init():
    global _flyc, _compile_moe_gemm2, _compile_moe_gemm1, _build_routing_buffers
    global _shuffle_weight, _shuffle_scale, _pack_int4
    if _flyc is not None:
        return
    import flydsl.compiler as flyc
    from kernels.moe_gemm_2stage import compile_moe_gemm2, compile_moe_gemm1
    from tests.kernels.test_moe_gemm import (
        build_routing_buffers, _pack_shuffled_int8_to_packed_int4_no_perm,
    )
    from tests.utils import shuffle_weight, shuffle_scale_for_int4
    _flyc = flyc
    _compile_moe_gemm2 = compile_moe_gemm2
    _compile_moe_gemm1 = compile_moe_gemm1
    _build_routing_buffers = build_routing_buffers
    _shuffle_weight = shuffle_weight
    _shuffle_scale = shuffle_scale_for_int4
    _pack_int4 = _pack_shuffled_int8_to_packed_int4_no_perm


# ---- weight/scale conversion (load-time) -------------------------------------
def _convert_weight_flat(w_uint8, N, K, chunk=32):
    """vLLM [E,N,K//2] uint8 -> FlyDSL flat packed-int4 (int8 view), chunked.

    Same total byte count as the input.
    """
    E = w_uint8.shape[0]
    parts = []
    for e0 in range(0, E, chunk):
        sub = w_uint8[e0:e0 + chunk]                       # [c, N, K//2]
        c = sub.shape[0]
        low = (sub & 0xF).to(torch.int16)
        high = ((sub >> 4) & 0xF).to(torch.int16)
        w = torch.stack([low, high], dim=-1).view(c, N, K)  # uint4 [0,15]
        w_signed = (w - 8).to(torch.int8)                   # fold zp=8 -> [-8,7]
        w_shuf = _shuffle_weight(w_signed).reshape(c * N, K)
        parts.append(_pack_int4(w_shuf).view(-1))
        del low, high, w, w_signed, w_shuf
    return torch.cat(parts).contiguous()


def _convert_scale_flat(w_scale):
    """vLLM scale [E,N,G] bf16 -> FlyDSL bf16 flat ((E,G//2,N,2) layout).

    shuffle_scale_for_int4 branches on dtype: bf16 input -> (E,G//2,N,2) packing
    that the scale_is_bf16=True kernel expects. Must stay bf16 throughout.
    """
    s = w_scale.to(torch.bfloat16).permute(0, 2, 1).contiguous()  # [E,G,N] bf16
    s = _shuffle_scale(s, group_size=32)                          # -> (E,G//2,N,2)
    return s.contiguous().view(-1)


def _register(tensor, role, N, K, E, G, scale_flat, inter, hidden):
    _WCACHE[tensor.data_ptr()] = dict(
        role=role, N=N, K=K, E=E, G=G, gs=K // G, scale=scale_flat,
        inter=inter, hidden=hidden,
    )


def convert_layer_inplace(layer):
    """Convert w13/w2 packed weights + scales to FlyDSL layout, in place.

    Frees the original buffers (memory-neutral). Idempotent per layer.
    """
    _lazy_init()
    if getattr(layer, "_flydsl_converted", False):
        return
    w13 = layer.w13_weight_packed.data       # [E, 2*inter, hidden//2] uint8
    w2 = layer.w2_weight_packed.data         # [E, hidden, inter//2] uint8
    s13 = layer.w13_weight_scale.data        # [E, 2*inter, hidden//gs] bf16
    s2 = layer.w2_weight_scale.data          # [E, hidden, inter//gs] bf16

    E, N1, hidden_half = w13.shape
    hidden = hidden_half * 2
    inter = N1 // 2
    G1 = s13.shape[2]
    Eh, Nh, inter_half = w2.shape
    G2 = s2.shape[2]

    w13_flat = _convert_weight_flat(w13, N1, hidden)
    s13_flat = _convert_scale_flat(s13)
    layer.w13_weight_packed.data = w13_flat   # original [E,N1,hidden//2] freed
    # CRITICAL: re-home the SCALE param too, not just the weight. The runtime caches
    # s13_flat in _WCACHE; if we leave layer.w13_weight_scale pointing at the original
    # [E,N,G] bf16 storage, the scale is DUPLICATED (orig on the layer + flat in cache)
    # ~= +246 MiB/layer x num_layers ~= +14.5 GiB -> collapses the KV pool -> OOM at
    # determine_available_memory. Pointing the param at the cached flat buffer makes
    # convert ACTUALLY memory-neutral (verified 2026-06-26: convert alloc delta 0 GiB
    # vs +14.5 GiB before; Available KV 5.96 -> 20.13 GiB; starts at mem 0.9 + 262144).
    layer.w13_weight_scale.data = s13_flat
    _register(layer.w13_weight_packed, "w13", N1, hidden, E, G1, s13_flat, inter, hidden)
    del w13, s13
    torch.cuda.empty_cache()

    w2_flat = _convert_weight_flat(w2, hidden, inter)
    s2_flat = _convert_scale_flat(s2)
    layer.w2_weight_packed.data = w2_flat
    layer.w2_weight_scale.data = s2_flat   # re-home scale param too (see w13 note above)
    _register(layer.w2_weight_packed, "w2", hidden, inter, E, G2, s2_flat, inter, hidden)
    del w2, s2
    torch.cuda.empty_cache()

    layer._flydsl_converted = True


# ---- runtime -----------------------------------------------------------------
def _pick_tiles(M, N, K):
    tile_m = 16 if M <= 64 else 32
    tile_n = 256
    while N % tile_n != 0:
        tile_n //= 2
    tile_k = 128
    while (K % tile_k != 0) or (tile_k % 32 != 0) or ((tile_m * tile_k * 2) % 256 != 0):
        tile_k -= 32
        if tile_k < 32:
            tile_k = 32
            break
    return tile_m, tile_n, tile_k


def _get_exe1(wptr, M, hidden, N1, inter, E, top_k):
    """Stage-1 (gate_up) executable: compile_moe_gemm1 with COMPACT-input
    in-kernel sorted-row gather (reads A[m] via sorted_ids//top_k -> no
    repeat_interleave transient). Output is the expanded [M*top_k, 2*inter]."""
    key = (wptr, 1, M, top_k)
    ent = _ECACHE.get(key)
    if ent is not None:
        return ent
    # Keep the proven _pick_tiles heuristic; call it with the same (N1, hidden)
    # the previous stage-1 path used so tile_m/tile_n/tile_k are unchanged.
    tile_m, tile_n, tile_k = _pick_tiles(M, N1, hidden)
    exe = _compile_moe_gemm1(
        model_dim=hidden, inter_dim=inter, experts=E, topk=top_k,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        doweight_stage1=False, in_dtype="int4_bf16", group_size=32,
        out_dtype="bf16", scale_is_bf16=True, use_cshuffle_epilog=False,
    )
    ent = dict(exe=exe, tile_m=tile_m, compiled=None)
    _ECACHE[key] = ent
    return ent


def _get_exe2(wptr, M, hidden, inter, E, top_k):
    """Stage-2 (down) executable: compile_moe_gemm2 with accumulate=True
    (in-kernel atomic top-k reduce) so the output is written DIRECTLY as
    [M, hidden] -- no expanded [M*top_k, hidden] buffer, no host moe_sum."""
    key = (wptr, 2, M, top_k)
    ent = _ECACHE.get(key)
    if ent is not None:
        return ent
    tile_m, tile_n, tile_k = _pick_tiles(M, hidden, inter)
    exe = _compile_moe_gemm2(
        model_dim=hidden, inter_dim=inter, experts=E, topk=top_k,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        doweight_stage2=True, in_dtype="int4_bf16", group_size=32,
        out_dtype="bf16", accumulate=True, scale_is_bf16=True,
    )
    ent = dict(exe=exe, tile_m=tile_m, compiled=None)
    _ECACHE[key] = ent
    return ent


# "torch" is the portable moe-sorting fallback (aiter's ops.shuffle sort is not
# importable in this FlyDSL checkout -> HAS_AITER False). It mirrors aiter's
# moe_sorting semantics and is what the FlyDSL 2-stage tests use on this box.
_SORT_MODE = os.environ.get("FLYDSL_MOE_SORT_MODE", "torch")

_VLLM_ALIGN = None


def _get_vllm_align():
    """vLLM's fused GPU moe_align_block_size -- the fast, cudagraph-capture-safe
    routing sort. The portable torch fallback (moe_sorting_torch_native) runs a
    384-iteration host loop (~40-60 ms here) that would dominate every bucket;
    aiter's GPU sort is unimportable in this checkout (FlyDSL ABI mismatch), so
    we build the FlyDSL routing layout from vLLM's align kernel instead."""
    global _VLLM_ALIGN
    if _VLLM_ALIGN is None:
        try:
            from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
                moe_align_block_size,
            )
            _VLLM_ALIGN = moe_align_block_size
        except Exception:
            _VLLM_ALIGN = False
    return _VLLM_ALIGN


if _HAS_TRITON:
    @triton.jit
    def _routing_post_kernel(
        flat_ptr,       # int32 [S]  vLLM sorted flat token indices (padding == numel)
        twf_ptr,        # fp32  [M*tk] flattened topk_weights
        fused_ptr,      # int32 [S]  OUT: (slot<<24)|token
        sw_ptr,         # fp32  [S]  OUT: sorted routed weight
        S, numel, tk, M,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < S
        flat = tl.load(flat_ptr + offs, mask=mask, other=numel)
        valid = flat < numel
        flat_c = tl.where(valid, flat, 0)
        tok = tl.where(valid, flat_c // tk, M)          # padding token == M (>= M -> skipped by kernel)
        slot = tl.where(valid, flat_c % tk, tk)         # padding slot  == tk
        fused = (slot << 24) | tok
        w = tl.load(twf_ptr + flat_c, mask=(mask & valid), other=0.0)
        w = tl.where(valid, w, 0.0)
        tl.store(fused_ptr + offs, fused, mask=mask)
        tl.store(sw_ptr + offs, w, mask=mask)


def _routing_post_fused(flat, twf, numel, tk, M):
    """Collapse the ~13-op torch post-processing of vLLM's moe_align output
    (flat->token/slot decode, (slot<<24)|token fused encode, sorted-weight gather,
    zero-fill of padding) into ONE capture-safe GPU kernel.  Returns
    (fused_sorted_ids int32 [S], sorted_weight fp32 [S]).  No .item()/host sync,
    fixed padded layout -> cudagraph-capture-safe."""
    S = flat.numel()
    fused = torch.empty(S, dtype=torch.int32, device=flat.device)
    sw = torch.empty(S, dtype=torch.float32, device=flat.device)
    BLOCK = 256
    grid = ((S + BLOCK - 1) // BLOCK,)
    _routing_post_kernel[grid](flat, twf, fused, sw, S, numel, tk, M, BLOCK=BLOCK)
    return fused, sw


def _build_routing(tk_ids, tk_w, E, N, tile_m):
    """Routing buffers. blocks/sorted_* depend only on routing+tile_m (not N),
    so the result is reused across both stages of a layer. tk_ids/tk_w are the
    REAL [M, top_k] tensors; the kernel decodes token=(sorted_id & 0xFFFFFF) and
    slot=(sorted_id >> 24) to gather compact A[token] / scatter out[token]."""
    align = _get_vllm_align()
    if align and _HAS_TRITON:
        M, tk = tk_ids.shape
        numel = M * tk
        # vLLM: sorted_tok holds flat indices into [M*tk] (padding == numel);
        # flat r -> token = r // tk, slot = r % tk. The ~13-op torch re-encode to
        # FlyDSL's fused (slot<<24)|token layout + sorted-weight gather is FUSED
        # into one GPU kernel (_routing_post_fused) so the decode routing floor is
        # intrinsic (memo-independent) and stays cudagraph-capture-safe.
        sorted_tok, expert_ids, num_pad = align(tk_ids, tile_m, E)
        twf = tk_w.reshape(-1)
        fused, sw = _routing_post_fused(sorted_tok, twf, numel, tk, M)
        num_valid = num_pad.reshape(-1)[:1].to(torch.int32)
        return dict(sorted_ids=fused,
                    sorted_expert_ids=expert_ids,
                    sw_1d=sw,
                    num_valid_ids=num_valid, blocks=int(expert_ids.numel()))

    routing = _build_routing_buffers(
        topk_ids=tk_ids, topk_weights=tk_w, experts=E,
        model_dim=N, tile_m=tile_m, moe_sort_mode=_SORT_MODE,
    )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, ssz, blocks = routing
    return dict(sorted_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
                sw_1d=sorted_weights.contiguous().view(-1),
                num_valid_ids=num_valid_ids, blocks=int(blocks))


def _grouped_gemm1(a, w_packed, scale_flat, N1, hidden, inter, E, exe_ent, rt, M, top_k):
    """Stage-1 gate_up + FUSED silu_and_mul: compact A[M,hidden] -> act[M*top_k, inter].

    compile_moe_gemm1 gathers the compact activation row A[sorted_id//top_k]
    in-kernel (so no [M*top_k, hidden] pre-gather ever materialises) AND fuses the
    SiLU(gate)*up epilogue -- the output is already the [M*top_k, inter] activation,
    so NO separate host silu_and_mul pass is needed.
    """
    flyc = _flyc
    DEV = a.device
    a_scale_1d = _empty_scale(DEV)
    out = torch.empty(M * top_k, inter, device=DEV, dtype=torch.bfloat16)

    def args(o):
        # gemm1 launch order: (o, x, w, sx, sw, sorted_ids, expert_ids,
        #   sorted_weights, num_valid_ids, tokens=M, inter_dim, model_dim, blocks, stream)
        return (o.view(-1), a.reshape(-1), w_packed, a_scale_1d, scale_flat,
                rt["sorted_ids"], rt["sorted_expert_ids"], rt["sw_1d"],
                rt["num_valid_ids"], M, inter, hidden, rt["blocks"],
                torch.cuda.current_stream())

    cexe = exe_ent["compiled"]
    if cexe is None:
        cexe = flyc.compile(exe_ent["exe"], *args(out))
        exe_ent["compiled"] = cexe
    cexe(*args(out))
    return out


def _grouped_gemm2(act, w_packed, scale_flat, hidden, inter, E, exe_ent, rt, M, top_k):
    """Stage-2 down + in-kernel top-k reduce (accumulate=True).

    act[M*top_k, inter] -> out[M, hidden] directly. The kernel atomic-adds each
    routed row into out[sorted_id//top_k] with the routed weight folded in
    (doweight_stage2=True), so the peak allocation NO LONGER scales with top_k
    and the host .sum(1) is gone.
    """
    flyc = _flyc
    DEV = act.device
    a_scale_1d = _empty_scale(DEV)
    # atomic accumulate -> must start zeroed. Allocate with empty + a SINGLE zero_()
    # (torch.zeros would memset here and the compile branch used to zero AGAIN -> two
    # memsets/call; one memset is enough and saves a dispatch on the tiny decode path).
    out = torch.empty(M, hidden, device=DEV, dtype=torch.bfloat16)

    def args(o):
        # gemm2 atomic launch order: (o, x, w, sx, sw, sorted_ids, expert_ids,
        #   sorted_weights, num_valid_ids, tokens=M, model_dim=hidden, inter_dim=inter, blocks, stream)
        return (o.view(-1), act.reshape(-1), w_packed, a_scale_1d, scale_flat,
                rt["sorted_ids"], rt["sorted_expert_ids"], rt["sw_1d"],
                rt["num_valid_ids"], M, hidden, inter, rt["blocks"],
                torch.cuda.current_stream())

    cexe = exe_ent["compiled"]
    if cexe is None:
        cexe = flyc.compile(exe_ent["exe"], *args(out))
        exe_ent["compiled"] = cexe
    out.zero_()
    cexe(*args(out))
    return out


def _ensure_compact(w1, w2, w1_scale, w2_scale, E):
    """Convert+cache raw vLLM-packed int4 weights (compact-operand harness path).

    In the isolated unittest the candidate receives raw packed uint8 weights
    each call (convert_layer_inplace is a live-server load-time hook that is NOT
    invoked here), so convert once keyed by the ORIGINAL data_ptr and cache the
    FlyDSL-layout weight + scale. Reuses the exact same _convert_* helpers as
    convert_layer_inplace (which stays unchanged) -> identical numeric layout.
    """
    k1 = w1.data_ptr()
    m1 = _WCACHE.get(k1)
    if m1 is not None and "wflat" in m1:
        return m1, _WCACHE[w2.data_ptr()]

    E1, N1, hidden_half = w1.shape          # w1: [E, 2*inter, hidden//2]
    hidden = hidden_half * 2
    inter = N1 // 2
    G1 = w1_scale.shape[2]
    w1_flat = _convert_weight_flat(w1, N1, hidden)
    s1_flat = _convert_scale_flat(w1_scale)
    m1 = dict(role="w13", N=N1, K=hidden, E=E, G=G1, gs=hidden // G1,
              scale=s1_flat, inter=inter, hidden=hidden, wflat=w1_flat)
    _WCACHE[k1] = m1

    Eh, Nh, inter_half = w2.shape           # w2: [E, hidden, inter//2]
    G2 = w2_scale.shape[2]
    w2_flat = _convert_weight_flat(w2, hidden, inter)
    s2_flat = _convert_scale_flat(w2_scale)
    m2 = dict(role="w2", N=hidden, K=inter, E=E, G=G2, gs=inter // G2,
              scale=s2_flat, inter=inter, hidden=hidden, wflat=w2_flat)
    _WCACHE[w2.data_ptr()] = m2
    return m1, m2


def _run_fused(a, w1flat, w2flat, m1, m2, tk_ids, tk_w, E, top_k):
    """Shared fused MoE flow: compact gemm1 -> silu_and_mul -> gemm2 top-k reduce.

    Stage-2 output is [M, hidden] (memory-contract compliant); no [M*top_k, *]
    activation/pre-gather transient survives past the kernels.
    """
    M, hidden = a.shape
    N1 = m1["N"]           # 2*inter
    inter = N1 // 2

    exe1 = _get_exe1(m1_ptr(m1, w1flat), M, hidden, N1, inter, E, top_k)
    exe2 = _get_exe2(m1_ptr(m2, w2flat), M, hidden, inter, E, top_k)
    # tile_m is identical for both stages (depends only on M); build routing once.
    rt = _build_routing(tk_ids, tk_w, E, N1, exe1["tile_m"])

    # stage 1: gate_up + FUSED silu_and_mul -- COMPACT input, in-kernel gather.
    # Output is already the [M*top_k, inter] activation (SiLU(gate)*up folded in the epilogue).
    act = _grouped_gemm1(a, w1flat, m1["scale"], N1, hidden, inter, E, exe1, rt, M, top_k)

    # stage 2: down (routed weight applied) -- in-kernel top-k reduce -> [M, hidden]
    out = _grouped_gemm2(act, w2flat, m2["scale"], hidden, inter, E, exe2, rt, M, top_k)
    return out


def m1_ptr(meta, w):
    """Cache key for the executable: prefer the converted-weight data_ptr so the
    compiled exe is stable across calls that reuse the same cached weight."""
    return w.data_ptr()


def flydsl_fused_experts_impl(
    hidden_states, w1=None, w2=None, topk_weights=None, topk_ids=None, inplace=False,
    activation="silu", apply_router_weight_on_input=False,
    global_num_experts=-1, expert_map=None,
    w1_scale=None, w2_scale=None, **_ignored,
):
    """Fused int4-W4A16 MoE via FlyDSL. Accepts EITHER the compact-operand dict
    (isolated unittest harness: cand(inp)) OR the vLLM positional signature
    (live server, weights already converted in place)."""
    _lazy_init()

    # ---- compact-operand dict path (isolated harness) ----
    if isinstance(hidden_states, dict):
        inp = hidden_states
        A = inp["A"]
        w1 = inp["w1"]; w2 = inp["w2"]
        w1_scale = inp["w1_scale"]; w2_scale = inp["w2_scale"]
        topk_ids = inp["topk_ids"]; topk_weights = inp["topk_weights"]
        E = int(inp["E"])
        top_k = int(inp["topk"])
        m1, m2 = _ensure_compact(w1, w2, w1_scale, w2_scale, E)
        a = A.to(torch.bfloat16)
        tk_ids = topk_ids.to(torch.int32).contiguous()
        tk_w = topk_weights.float().contiguous()
        out = _run_fused(a, m1["wflat"], m2["wflat"], m1, m2, tk_ids, tk_w, E, top_k)
        return out.to(A.dtype)

    # ---- vLLM positional path (weights converted at load via convert_layer_inplace) ----
    m1 = _WCACHE.get(w1.data_ptr())
    m2 = _WCACHE.get(w2.data_ptr())
    if m1 is None or m2 is None:
        raise RuntimeError("FlyDSL: weights not converted (cache miss)")
    E = m1["E"]
    top_k = topk_ids.shape[1]
    a = hidden_states.to(torch.bfloat16)
    tk_ids = topk_ids.to(torch.int32).contiguous()
    tk_w = topk_weights.float().contiguous()
    out = _run_fused(a, w1, w2, m1, m2, tk_ids, tk_w, E, top_k)
    out = out.to(hidden_states.dtype)
    if inplace:
        hidden_states.copy_(out)
        return hidden_states
    return out
