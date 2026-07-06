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

_flyc = None
_compile_moe_gemm2 = None
_build_routing_buffers = None
_shuffle_weight = None
_shuffle_scale = None
_pack_int4 = None

# meta keyed by converted-tensor data_ptr:
#   dict(role, N, K, E, G, gs, scale=<flat bf16>, inter, hidden)
_WCACHE = {}
# compiled exe keyed by (data_ptr, stage, M, top_k)
_ECACHE = {}


def _lazy_init():
    global _flyc, _compile_moe_gemm2, _build_routing_buffers
    global _shuffle_weight, _shuffle_scale, _pack_int4
    if _flyc is not None:
        return
    import flydsl.compiler as flyc
    from kernels.moe_gemm_2stage import compile_moe_gemm2
    from tests.kernels.test_moe_gemm import (
        build_routing_buffers, _pack_shuffled_int8_to_packed_int4_no_perm,
    )
    from tests.utils import shuffle_weight, shuffle_scale_for_int4
    _flyc = flyc
    _compile_moe_gemm2 = compile_moe_gemm2
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


def _get_exe(wptr, stage, M, N, K, E, top_k):
    key = (wptr, stage, M, top_k)
    ent = _ECACHE.get(key)
    if ent is not None:
        return ent
    tile_m, tile_n, tile_k = _pick_tiles(M, N, K)
    exe = _compile_moe_gemm2(
        model_dim=N, inter_dim=K, experts=E, topk=1,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        doweight_stage2=(stage == 2), in_dtype="int4_bf16", group_size=32,
        out_dtype="bf16", accumulate=False, scale_is_bf16=True,
    )
    ent = dict(exe=exe, tile_m=tile_m, compiled=None)
    _ECACHE[key] = ent
    return ent


_SORT_MODE = os.environ.get("FLYDSL_MOE_SORT_MODE", "aiter")


def _build_routing(tk_ids, tk_w, E, N, tile_m):
    """Routing buffers. blocks/sorted_* depend only on routing+tile_m (not N),
    so the result is reused across both stages of a layer."""
    routing = _build_routing_buffers(
        topk_ids=tk_ids, topk_weights=tk_w, experts=E,
        model_dim=N, tile_m=tile_m, moe_sort_mode=_SORT_MODE,
    )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, ssz, blocks = routing
    return dict(sorted_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
                sw_1d=sorted_weights.contiguous().view(-1),
                num_valid_ids=num_valid_ids, blocks=int(blocks))


def _grouped_gemm(a2, w_packed, scale_flat, N, K, E, exe_ent, rt):
    flyc = _flyc
    kt = a2.shape[0]
    DEV = a2.device
    a2_scale_1d = torch.empty((0,), device=DEV, dtype=torch.float32)
    out = torch.zeros(kt, N, device=DEV, dtype=torch.bfloat16)

    def args(o):
        return (o.view(-1), a2.reshape(-1), w_packed, a2_scale_1d, scale_flat,
                rt["sorted_ids"], rt["sorted_expert_ids"], rt["sw_1d"],
                rt["num_valid_ids"], kt, N, K, rt["blocks"],
                torch.cuda.current_stream())

    cexe = exe_ent["compiled"]
    if cexe is None:
        cexe = flyc.compile(exe_ent["exe"], *args(out))
        exe_ent["compiled"] = cexe
    out.zero_()
    cexe(*args(out))
    return out


def flydsl_fused_experts_impl(
    hidden_states, w1, w2, topk_weights, topk_ids, inplace,
    activation="silu", apply_router_weight_on_input=False,
    global_num_experts=-1, expert_map=None,
    w1_scale=None, w2_scale=None, **_ignored,
):
    _lazy_init()
    m1 = _WCACHE.get(w1.data_ptr())
    m2 = _WCACHE.get(w2.data_ptr())
    if m1 is None or m2 is None:
        raise RuntimeError("FlyDSL: weights not converted (cache miss)")

    M, hidden = hidden_states.shape
    E = m1["E"]
    N1 = m1["N"]          # 2*inter (stage1 output)
    inter = N1 // 2
    top_k = topk_ids.shape[1]
    a = hidden_states.to(torch.bfloat16)

    tk_ids = topk_ids.reshape(-1, 1).contiguous().to(torch.int32)
    tk_w = topk_weights.reshape(-1, 1).float().contiguous()

    exe1 = _get_exe(w1.data_ptr(), 1, M, N1, hidden, E, top_k)
    exe2 = _get_exe(w2.data_ptr(), 2, M, hidden, inter, E, top_k)
    # tile_m is identical for both stages (same M); build routing once and reuse.
    rt = _build_routing(tk_ids, tk_w, E, N1, exe1["tile_m"])

    # stage 1: gate_up (no routed weight)
    a2_1 = a.repeat_interleave(top_k, dim=0).contiguous()
    out1 = _grouped_gemm(a2_1, w1, m1["scale"], N1, hidden, E, exe1, rt)

    # silu_and_mul (bf16)
    gate, up = out1[:, :inter], out1[:, inter:]
    act = (torch.nn.functional.silu(gate) * up).contiguous()

    # stage 2: down (routed weight applied)
    out2 = _grouped_gemm(act, w2, m2["scale"], hidden, inter, E, exe2, rt)

    out = out2.view(M, top_k, hidden).sum(dim=1).to(hidden_states.dtype)
    if inplace:
        hidden_states.copy_(out)
        return hidden_states
    return out
