#!/usr/bin/env python3
"""Candidate rewrites of the Kimi-K3 attention-residual Triton pair.

Stock (`sglang/srt/layers/attn_residual.py`):

  _score_kernel   grid (T, nvb+1), num_warps=8, BLOCK_H=1024
      for h0 in static_range(0, 7168, 1024):
          sumsq += tl.sum(v*v);  dotv += tl.sum(v*cw)
      -> 14 block-wide reductions per CTA (2 per H block x 7 blocks), and with
         512 lanes over a 1024-element block each lane loads 2 bf16 = 4 B, a
         quarter of the 16 B/lane the memory path wants.

  _combine_kernel grid (T, 7), num_warps=4, BLOCK_H=1024
      for j in range(nvb+1):  acc += tl.sum(tl.where(offs_b == j, p, 0)) * v
      -> one cross-lane reduce per row just to broadcast a scalar weight,
         and 4 B/lane loads again.

Candidates keep the semantics byte-for-byte in intent (same fp32 accumulate,
same eps, same softmax) and change only how the work is laid out.
"""
import torch
import triton
import triton.language as tl

_MAX_ROWS = 16


# --------------------------------------------------------------------------
# score: one CTA per (token, row); single reduction, vectorized loads
# --------------------------------------------------------------------------
@triton.jit
def _score_kernel_v(
    prefix_ptr,
    bank_ptr,
    cw_ptr,
    scores_ptr,
    NVB,
    eps,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Identical contract to _score_kernel, but the two cross-lane reductions
    are hoisted out of the H loop: the loop body only does FMAs into two
    per-lane accumulators, and the block reduce happens once.

    The bank/prefix select stays *inside* the loop and yields a value, as in the
    stock kernel.  Hoisting it into a base pointer chosen by an `if` is the
    obvious rewrite and it does not compile: TritonAMDGPUCanonicalizePointers
    asserts on an scf.if that yields fat pointers with different bases
    (`expected then fat ptr canNarrow and else fat ptr canNarrow to be equal`,
    CanonicalizePointers.cpp:1441).  See analysis/triton_amd_ptr_if_bug.md."""
    pid_t = tl.program_id(0)
    j = tl.program_id(1)
    if j > NVB:
        return

    acc_sq = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_dot = tl.zeros([BLOCK_H], dtype=tl.float32)
    for h0 in tl.static_range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        if j < NVB:
            v = tl.load(
                bank_ptr + pid_t * stride_bm + j * stride_bb + offs_h
            ).to(tl.float32)
        else:
            v = tl.load(prefix_ptr + pid_t * stride_pm + offs_h).to(tl.float32)
        cw = tl.load(cw_ptr + offs_h)
        acc_sq += v * v
        acc_dot += v * cw
    sumsq = tl.sum(acc_sq)
    dotv = tl.sum(acc_dot)
    rrms = 1.0 / tl.sqrt(sumsq / H + eps)
    tl.store(scores_ptr + pid_t * stride_sm + j, dotv * rrms)


# --------------------------------------------------------------------------
# combine: one CTA per (token, H chunk); rows as a 2D tile, no per-row reduce
# --------------------------------------------------------------------------
@triton.jit
def _combine_kernel_v(
    prefix_ptr,
    bank_ptr,
    scores_ptr,
    out_ptr,
    NVB,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    stride_om,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    """Weighted sum as a single [BLOCK_R, BLOCK_H] tile reduction.

    The stock kernel loops over rows and pays a 16-lane cross-lane reduce per
    row just to turn p[j] into a scalar multiplier.  Loading the rows as one
    2D tile makes the weight a plain broadcast along axis 0 and the mix a
    single tl.sum(axis=0)."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    h0 = pid_h * BLOCK_H

    offs_b = tl.arange(0, MAX_ROWS)
    mask_b = offs_b <= NVB
    raw = tl.load(
        scores_ptr + pid_t * stride_sm + offs_b, mask=mask_b, other=float("-inf")
    )
    m = tl.max(raw, axis=0)
    e = tl.where(mask_b, tl.exp(raw - m), 0.0)
    p = e / tl.sum(e, axis=0)

    offs_h = h0 + tl.arange(0, BLOCK_H)
    offs_r = tl.arange(0, BLOCK_R)

    # bank rows 0..NVB-1 as one tile
    mask_r = offs_r < NVB
    vb = tl.load(
        bank_ptr + pid_t * stride_bm + offs_r[:, None] * stride_bb + offs_h[None, :],
        mask=mask_r[:, None],
        other=0.0,
    ).to(tl.float32)
    pb = tl.where(offs_r < NVB, tl.sum(tl.where(offs_b[None, :] == offs_r[:, None],
                                                p[None, :], 0.0), axis=1), 0.0)
    acc = tl.sum(pb[:, None] * vb, axis=0)

    # the prefix row (index NVB) is not in the bank
    vp = tl.load(prefix_ptr + pid_t * stride_pm + offs_h).to(tl.float32)
    p_last = tl.sum(tl.where(offs_b == NVB, p, 0.0), axis=0)
    acc += p_last * vp

    tl.store(out_ptr + pid_t * stride_om + offs_h, acc.to(out_ptr.dtype.element_ty))


# --------------------------------------------------------------------------
# host wrappers
# --------------------------------------------------------------------------
H_HID = 7168

# tuned per-shape; overridden by the sweep in bench_attn_res.py
CFG = {
    "score_block_h": 1024,
    "score_warps": 2,
    "combine_block_h": 1024,
    "combine_warps": 4,
    "combine_block_r": 0,  # 0 = size the row tile to nvb (see below)
}


def _pow2_ge(n):
    p = 1
    while p < n:
        p *= 2
    return p


def mix_cand(prefix, bank, nvb, cw, scores, out, cfg=None):
    c = cfg or CFG
    T, Hh = prefix.shape
    _score_kernel_v[(T, nvb + 1)](
        prefix, bank, cw, scores, nvb, 1e-6,
        prefix.stride(0), bank.stride(0), bank.stride(1), scores.stride(0),
        H=Hh, BLOCK_H=c["score_block_h"], num_warps=c["score_warps"],
    )
    bh = c["combine_block_h"]
    # nvb is a host value, so the row tile can be specialized to it.  A fixed
    # 8-row tile costs 32% at nvb=1: [8, BLOCK_H] fp32 is 32 KB of registers per
    # CTA whether or not the rows are masked off.
    br = c["combine_block_r"] or _pow2_ge(max(nvb, 1))
    _combine_kernel_v[(T, Hh // bh)](
        prefix, bank, scores, out, nvb,
        prefix.stride(0), bank.stride(0), bank.stride(1),
        scores.stride(0), out.stride(0),
        BLOCK_H=bh, BLOCK_R=br, MAX_ROWS=_MAX_ROWS,
        num_warps=c["combine_warps"],
    )
    return out
