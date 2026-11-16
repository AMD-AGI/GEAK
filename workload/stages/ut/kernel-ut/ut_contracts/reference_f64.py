#!/usr/bin/env python3
"""Float64 references for the UT contract families.

Each `<name>(L, sym, rows=None)` reads the LOGICAL operands `L` (float64 for floating inputs,
int64 for indices; a quantized operand's logical value is its dequantized float64 tensor, scales
applied and layout undone, as the task's quantizer hook returns it) and returns
`(outputs, magnitudes)`: float64 tensors keyed by contract output name, and for outputs bounded
as a dot product, `|a| @ |b|`-style magnitudes of the same shape. `rows` (a LongTensor or None)
restricts the computation to those rows of the leading token/row axis, so a 32768-token prefill
case costs a few hundred rows of float64 work instead of the whole matmul.

A reference never rounds to the output dtype: rounding is error the contract's bound allows,
unless every correct kernel must round there (then it is written here and documented).
These are the operator's mathematics, not a copy of the deployed kernel; when the two differ by
more than the bound, find out which is wrong before relaxing anything.
"""

import math


def _torch():
    import torch
    return torch


def _rows(t, rows):
    return t if rows is None else t[rows]


# ----------------------------------------------------------------------------- decoders
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def decode_e2m1(packed):
    """Packed MXFP4 (two e2m1 values per byte, low nibble first) -> float64, last dim doubled."""
    torch = _torch()
    b = packed.contiguous().view(torch.uint8).to(torch.int64)
    table = torch.tensor(E2M1_VALUES, dtype=torch.float64, device=b.device)
    lo, hi = table[b & 0xF], table[(b >> 4) & 0xF]
    return torch.stack([lo, hi], dim=-1).reshape(*b.shape[:-1], b.shape[-1] * 2)


def decode_e8m0(scales):
    """E8M0 exponents (uint8 or float8_e8m0fnu) -> float64 2**(e - 127); 0xFF is NaN."""
    torch = _torch()
    e = scales.contiguous().view(torch.uint8).to(torch.float64)
    out = torch.exp2(e - 127.0)
    return out.masked_fill(e == 255, float("nan"))


def expand_blocks(scale, block, axis, length):
    """Repeat a per-block scale `block` times along `axis` and trim to `length`."""
    return scale.repeat_interleave(block, dim=axis).narrow(axis, 0, length)


def dequant_mxfp4(packed, scales, block=32):
    """MXFP4 operand (row-major, unshuffled) -> float64 values."""
    v = decode_e2m1(packed)
    return v * expand_blocks(decode_e8m0(scales), block, -1, v.shape[-1])


# ----------------------------------------------------------------------------- gemm
def _matmul(a, b_nk, bias=None):
    out = a @ b_nk.mT
    mag = a.abs() @ b_nk.abs().mT
    if bias is not None:
        out = out + bias
        mag = mag + bias.abs()
    return out, mag


def gemm(L, sym, rows=None):
    """out = a @ b.T + bias; b is (n, k) when trans_b else (k, n)."""
    b = L["b"] if sym.get("trans_b", True) else L["b"].mT
    out, mag = _matmul(_rows(L["a"], rows), b, L.get("bias"))
    return {"out": out}, {"out": mag}


def gemm_a8w8(L, sym, rows=None):
    """Per-token a_scale (m, 1) and per-channel b_scale (n, 1)."""
    a = _rows(L["a"] * L["a_scale"].reshape(-1, 1), rows)
    b = L["b"] * L["b_scale"].reshape(-1, 1)
    out, mag = _matmul(a, b, L.get("bias"))
    return {"out": out}, {"out": mag}


def gemm_a8w8_blockscale(L, sym, rows=None):
    """a_scale per 1 x 128 of a (m, ceil(k/128)), b_scale per 128 x 128 of b."""
    k, n = L["a"].shape[-1], L["b"].shape[0]
    block_n, block_k = int(sym.get("block_n", 128)), int(sym.get("block_k", 128))
    a_scale = L["a_scale"] if sym.get("scale_major", "K") == "K" else L["a_scale"].mT
    a = L["a"] * expand_blocks(a_scale, block_k, -1, k)
    bs = expand_blocks(expand_blocks(L["b_scale"], block_n, 0, n), block_k, 1, k)
    out, mag = _matmul(_rows(a, rows), L["b"] * bs, L.get("bias"))
    return {"out": out}, {"out": mag}


def gemm_a4w4_mxfp4(L, sym, rows=None):
    """MXFP4 a and b with e8m0 scales per 1 x 32. The quantizer hook hands over each packed
    operand's logical value already dequantized (values x scales, unshuffled), so this is the
    float64 product of exactly the numbers the kernel multiplies."""
    out, mag = _matmul(_rows(L["a"], rows), L["b"], L.get("bias"))
    return {"out": out}, {"out": mag}


# ----------------------------------------------------------------------------- moe
def _act(x, name):
    torch = _torch()
    if name == "silu":
        return x * torch.sigmoid(x)
    if name == "gelu":
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
    if name == "gelu_tanh":
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    raise ValueError("activation %r" % name)


def fused_moe(L, sym, rows=None):
    """out[t] = sum_k w[t,k] * w2[e] @ (act(w1_gate[e] @ x) * (w1_up[e] @ x)), e = ids[t,k].

    w1 is (E, 2*I, D) with gate rows first (gate_mode "separated"); the task's hooks hand over
    dequantized float64 weights whatever the stored format. Expert ids index the local experts;
    an expert-parallel rank's mask is applied by the hook (absent experts contribute zero)."""
    torch = _torch()
    x = _rows(L["hidden_states"], rows)
    ids = _rows(L["topk_ids"], rows).to(torch.int64)
    w = _rows(L["topk_weights"], rows)
    w1, w2 = L["w1"], L["w2"]
    inter = w1.shape[1] // 2
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float64, device=x.device)
    act = sym.get("activation", "silu")
    for e in ids.unique().tolist():
        if e < 0 or e >= w1.shape[0]:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ w1[e].mT
        y = (_act(h[:, :inter], act) * h[:, inter:]) @ w2[e].mT
        out.index_add_(0, tok, y * w[tok, slot][:, None])
    return {"out": out}, {}


def moe_grouped_topk(L, sym, rows=None):
    """DeepSeek-style biased grouped top-k routing.

    s = sigmoid(logits) (or softmax); selection ranks s + bias: keep the `topk_group` groups whose
    best-two sum is largest, then the top `topk` experts inside them; weights are the UNbiased s of
    the chosen experts, renormalized when `renormalize`, times `routed_scaling_factor`."""
    torch = _torch()
    logits = _rows(L["gating_output"], rows)
    s = torch.sigmoid(logits) if sym.get("scoring", "sigmoid") == "sigmoid" else torch.softmax(logits, -1)
    biased = s + L["correction_bias"] if "correction_bias" in L else s
    t, e = biased.shape
    g, tg, k = int(sym["n_group"]), int(sym["topk_group"]), int(sym["topk"])
    grouped = biased.view(t, g, e // g)
    group_score = grouped.topk(2, dim=-1).values.sum(-1) if sym.get("group_score", "top2") == "top2" \
        else grouped.max(-1).values
    keep = group_score.topk(tg, dim=-1).indices
    mask = torch.zeros(t, g, dtype=torch.bool, device=logits.device).scatter_(1, keep, True)
    masked = biased.masked_fill(~mask.repeat_interleave(e // g, dim=1), float("-inf"))
    ids = masked.topk(k, dim=-1).indices
    w = s.gather(1, ids)
    if sym.get("renormalize", True):
        w = w / w.sum(-1, keepdim=True)
    w = w * float(sym.get("routed_scaling_factor", 1.0))
    return {"topk_ids": ids, "topk_weights": w, "selection_scores": masked}, {}


# ----------------------------------------------------------------------------- norms, activations
def rmsnorm(L, sym, rows=None):
    """x' = x + residual (when present); out = x' / sqrt(mean(x'^2) + eps) * weight.
    Outputs `out` and, with a residual, `residual_out` = x'."""
    torch = _torch()
    x = _rows(L["x"], rows)
    res = {}
    if "residual" in L:
        x = x + _rows(L["residual"], rows)
        res["residual_out"] = x
    eps = float(sym.get("eps", 1e-6))
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * L["weight"]
    res["out"] = y
    return res, {}


def act_mul(L, sym, rows=None):
    """out = act(x[..., :d]) * x[..., d:] with d = x.shape[-1] // 2."""
    x = _rows(L["x"], rows)
    d = x.shape[-1] // 2
    return {"out": _act(x[..., :d], sym.get("activation", "silu")) * x[..., d:]}, {}


def rope(L, sym, rows=None):
    """Rotary embedding on the first `rot_dim` features of every head of q and k.

    cos_sin_cache is (max_pos, rot_dim): cos halves then sin halves (vLLM/SGLang layout).
    `style` "neox" rotates the two halves; "gptj" rotates interleaved pairs."""
    torch = _torch()
    pos = _rows(L["positions"], rows).to(torch.int64)
    cache = L["cos_sin_cache"][pos]
    rot = cache.shape[-1]
    cos, sin = cache[..., : rot // 2], cache[..., rot // 2:]
    out = {}
    for name in ("query", "key"):
        if name not in L:
            continue
        x = _rows(L[name], rows)
        heads = x.shape[-1] // int(sym["head_size"])
        x = x.reshape(x.shape[0], heads, int(sym["head_size"]))
        xr, xp = x[..., :rot], x[..., rot:]
        c, s = cos[:, None, :], sin[:, None, :]
        if sym.get("style", "neox") == "neox":
            x1, x2 = xr[..., : rot // 2], xr[..., rot // 2:]
            xr = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
        else:
            x1, x2 = xr[..., 0::2], xr[..., 1::2]
            xr = torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).flatten(-2)
        out[name + "_out"] = torch.cat([xr, xp], dim=-1).reshape(x.shape[0], -1)
    return out, {}


# ----------------------------------------------------------------------------- selection
def topk(L, sym, rows=None):
    """Top-`topk` positions among the first lengths[r] scores of each row, -1 padded."""
    torch = _torch()
    s = _rows(L["scores"], rows).clone()
    k = int(sym["topk"])
    if "lengths" in L:
        n = _rows(L["lengths"], rows).to(torch.int64)
        col = torch.arange(s.shape[-1], device=s.device)
        s = s.masked_fill(col[None, :] >= n[:, None], float("-inf"))
    v, i = s.topk(k, dim=-1)
    i = i.masked_fill(torch.isinf(v) & (v < 0), -1)
    return {"indices": i}, {}


def dsa_indexer(L, sym, rows=None):
    """DeepSeek sparse-attention lightning indexer, then top-k.

    score[t, s] = sum_h weights[t, h] * relu(q[t, h] . k[s]) (times `k_scale[s]` when present),
    over keys ks[t] <= s < ke[t]; others -inf. `indices` are the top-`topk` positions. Returns the
    float64 `scores` too, which a fused score+top-k bounds its indices against."""
    torch = _torch()
    q = _rows(L["q"], rows)
    kk = L["k"] * L["k_scale"].reshape(-1, 1) if "k_scale" in L else L["k"]
    w = _rows(L["weights"], rows)
    logits = torch.einsum("thd,sd->ths", q, kk).clamp_min(0)
    score = torch.einsum("ths,th->ts", logits, w)
    col = torch.arange(score.shape[-1], device=score.device)
    if "ks" in L:
        ks, ke = _rows(L["ks"], rows).to(torch.int64), _rows(L["ke"], rows).to(torch.int64)
        score = score.masked_fill((col[None] < ks[:, None]) | (col[None] >= ke[:, None]), float("-inf"))
    out = {"scores": score}
    if "topk" in sym:
        v, i = score.topk(min(int(sym["topk"]), score.shape[-1]), dim=-1)
        out["indices"] = i.masked_fill(torch.isinf(v), -1)
    return out, {}


# ----------------------------------------------------------------------------- attention
def _softmax_attend(q, k, v, scale, valid, sinks=None):
    """q (h, d), k/v (s, d|dv), valid (s,) bool -> (h, dv) output and (h,) lse."""
    torch = _torch()
    logits = (q @ k.mT) * scale
    logits = logits.masked_fill(~valid[None, :], float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    norm = lse if sinks is None else torch.logaddexp(lse, sinks)
    p = torch.exp(logits - norm[:, None]).nan_to_num(0.0)
    return p @ v, lse


def sparse_mla(L, sym, rows=None):
    """Sparse MLA: each token attends to the KV-pool slots its indices name (-1 skipped).

    q is (T, H, D) with D = dv + rope (or q_nope/q_rope concatenated); kv is (pool, 1, D) with
    values its first dv features. out (T, H, dv), lse (T, H). A row with no valid slot has
    output 0 and lse -inf, as sglang's triton_sparse_mla returns (FlashMLA's sparse API returns
    +inf there; a task on that entry overrides the lse bound accordingly)."""
    torch = _torch()
    q = L["q"] if "q" in L else torch.cat([L["q_nope"], L["q_rope"]], dim=-1)
    q = _rows(q, rows)
    idx = _rows(L["indices"], rows).reshape(q.shape[0], -1).to(torch.int64)
    kv = L["kv"].reshape(L["kv"].shape[0], -1)
    dv = int(sym["d_v"])
    scale = float(sym["sm_scale"])
    sinks = L.get("sinks")
    out = torch.zeros(q.shape[0], q.shape[1], dv, dtype=torch.float64, device=q.device)
    lse = torch.full((q.shape[0], q.shape[1]), float("-inf"), dtype=torch.float64, device=q.device)
    for t in range(q.shape[0]):
        valid = idx[t] >= 0
        if not bool(valid.any()):
            continue
        rows_kv = kv[idx[t].clamp_min(0)]
        o, l = _softmax_attend(q[t], rows_kv, rows_kv[:, :dv], scale, valid, sinks)
        out[t], lse[t] = o, l
    return {"output": out, "lse": lse}, {}


def paged_attention_decode(L, sym, rows=None):
    """One query token per sequence over a paged KV cache.

    k_cache/v_cache are LOGICAL (num_blocks, block_size, kv_heads, head_dim) whatever the stored
    layout (the task's hook un-shuffles them); block_table (B, max_blocks); seq_lens (B,).
    Query head h reads kv head h // (heads // kv_heads)."""
    torch = _torch()
    q = _rows(L["query"], rows)
    bt = _rows(L["block_table"], rows).to(torch.int64)
    lens = _rows(L["seq_lens"], rows).to(torch.int64)
    kc, vc = L["k_cache"], L["v_cache"]
    bs, hkv = kc.shape[1], kc.shape[2]
    heads = q.shape[1]
    group = heads // hkv
    scale = float(sym["scale"])
    out = torch.zeros(q.shape[0], heads, vc.shape[-1], dtype=torch.float64, device=q.device)
    for b in range(q.shape[0]):
        n = int(lens[b])
        slots = torch.arange(n, device=q.device)
        blocks = bt[b][slots // bs]
        k = kc[blocks, slots % bs]           # (n, hkv, d)
        v = vc[blocks, slots % bs]
        valid = torch.ones(n, dtype=torch.bool, device=q.device)
        for hk in range(hkv):
            hs = slice(hk * group, (hk + 1) * group)
            out[b, hs], _ = _softmax_attend(q[b, hs], k[:, hk], v[:, hk], scale, valid)
    return {"out": out}, {}


def varlen_attention(L, sym, rows=None):
    """Prefill attention over packed sequences (cu_seqlens_q/k), optionally causal (bottom-right
    aligned: query i of a length-Lq chunk over Lk keys sees keys < Lk - Lq + i + 1)."""
    torch = _torch()
    q, k, v = L["q"], L["k"], L["v"]
    cq, ck = L["cu_seqlens_q"].to(torch.int64), L["cu_seqlens_k"].to(torch.int64)
    heads, hkv = q.shape[1], k.shape[1]
    group = heads // hkv
    scale = float(sym["scale"])
    out = torch.zeros(q.shape[0], heads, v.shape[-1], dtype=torch.float64, device=q.device)
    lse = torch.full((q.shape[0], heads), float("-inf"), dtype=torch.float64, device=q.device)
    want = None if rows is None else set(rows.tolist())
    for b in range(cq.numel() - 1):
        q0, q1, k0, k1 = int(cq[b]), int(cq[b + 1]), int(ck[b]), int(ck[b + 1])
        lq, lk = q1 - q0, k1 - k0
        for i in range(lq):
            t = q0 + i
            if want is not None and t not in want:
                continue
            n = lk - lq + i + 1 if sym.get("causal", True) else lk
            valid = torch.arange(lk, device=q.device) < n
            for hk in range(hkv):
                hs = slice(hk * group, (hk + 1) * group)
                out[t, hs], lse[t, hs] = _softmax_attend(q[t, hs], k[k0:k1, hk], v[k0:k1, hk], scale, valid)
    if rows is not None:
        out, lse = out[rows], lse[rows]
    return {"out": out, "lse": lse}, {}


# ----------------------------------------------------------------------------- collectives
def all_reduce(L, sym, rows=None):
    """Sum of every rank's input: `inputs` is (world, ...) stacked by the multi-rank harness."""
    x = L["inputs"]
    return {"out": _rows(x.sum(0), rows)}, {}


REFERENCES = {f.__name__: f for f in (gemm, gemm_a8w8, gemm_a8w8_blockscale, gemm_a4w4_mxfp4,
                                      fused_moe, moe_grouped_topk, rmsnorm, act_mul, rope, topk,
                                      dsa_indexer, sparse_mla, paged_attention_decode,
                                      varlen_attention, all_reduce)}


def reference(name, logical, sym, rows=None):
    if name not in REFERENCES:
        raise KeyError("no float64 reference %r; known: %s" % (name, sorted(REFERENCES)))
    return REFERENCES[name](logical, sym, rows)
