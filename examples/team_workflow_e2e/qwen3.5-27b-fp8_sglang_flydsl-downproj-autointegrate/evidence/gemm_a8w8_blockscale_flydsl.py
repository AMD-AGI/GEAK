# SPDX-License-Identifier: MIT
# Engineer r1_d0 (algorithm) — FUSED fp8 a8w8 BLOCKSCALE GEMM core for the down-proj
# seam aiter.ops.triton.gemm_a8w8_blockscale:gemm_a8w8_blockscale (N=5120, K=17408).
#
# WHAT THIS IS
#   A self-contained Triton fused fp8 block-scale GEMM. Operands stay fp8 the WHOLE
#   way (NO bf16 repeat_interleave / dequant materialization of X or W anywhere) and
#   the per-128-K-block scales are folded into the fp32 accumulator at every K-tile
#   boundary (tile_k == 128 == GROUP_K). This is the correct "per-K-block scaled
#   split-K accumulation in ONE fp8 GEMM pass" — math:
#       Y[m,n] = sum_kb  x_scale[m,kb] * w_scale[n//128,kb] * (sum_{k in kb} Xq*Wq)
#   accumulated in fp32 across all 136 K-blocks. Matches the immutable fp32 dequant
#   oracle within tol=0.06.
#
#   gfx942 (CDNA3) has NO native block-scaled MFMA (mfma_scale_f32_16x16x128 is
#   gfx950-only), so the per-block scale is emulated in software exactly as above.
#
# WHY IT BEATS THE bf16-dequant baseline
#   The placeholder baseline materialized BOTH operands to bf16 [M,17408]/[5120,17408]
#   every call (the dominant prefill cost + e2e-forbidden footprint). This kernel
#   never expands either operand: one fp8 MFMA pass with inline scale fold.
#
# TUNING (this engineer owns the inner GEMM math + launch shape)
#   Per-M-bucket config: deep-K / narrow-N prefill wants bigger BLOCK_SIZE_M and
#   GROUP_SIZE_M (L2 weight reuse over the 40 N-tiles); tiny-M decode wants a
#   skinny tile + split-K to fill the 304 CUs (GRID_MN=40 otherwise idles ~87%).

from typing import Optional

import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd


# ----------------------------------------------------------------------------- #
# Fused fp8 block-scale GEMM kernel (single-pass + optional split-K).
# C = A @ B^T with per-(row,1x128-K) act scale and per-(128x128) weight scale,
# folded into fp32 accumulation at each BLOCK_SIZE_K(==GROUP_K==128) boundary.
# ----------------------------------------------------------------------------- #
@triton.heuristics(
    {
        "EVEN_K": lambda a: a["K"] % a["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda a: triton.cdiv(a["M"], a["BLOCK_SIZE_M"])
        * triton.cdiv(a["N"], a["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _fused_blockscale_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ascale_m > 0)
    tl.assume(stride_ascale_k > 0)
    tl.assume(stride_bscale_k > 0)
    tl.assume(stride_bscale_n > 0)

    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        remap_xcd(pid, GRID_MN)
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE) < K:
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )

        offs_k_scale = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
        a_scale_ptrs = (
            a_scale_ptr + offs_am * stride_ascale_m + offs_k_scale * stride_ascale_k
        )
        offs_b_scale_n = offs_bn // GROUP_N
        b_scale_ptrs = (
            b_scale_ptr
            + offs_k_scale * stride_bscale_k
            + offs_b_scale_n * stride_bscale_n
        )
        offs_ks_step = BLOCK_SIZE_K // GROUP_K

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
                )
                b = tl.load(
                    b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                )

            a_scale = tl.load(a_scale_ptrs)
            b_scale = tl.load(b_scale_ptrs)

            accumulator += (
                tl.dot(a, b, input_precision="ieee")
                * a_scale[:, None]
                * b_scale[None, :]
            )

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            a_scale_ptrs += offs_ks_step * stride_ascale_k
            b_scale_ptrs += offs_ks_step * stride_bscale_k

        c = accumulator.to(c_ptr.type.element_ty)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _reduce_kernel(
    c_in_ptr,
    c_out_ptr,
    M,
    N,
    stride_c_in_k,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_m,
    stride_c_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, MAX_KSPLIT)
    c_in_ptrs = (
        c_in_ptr
        + (offs_k[:, None, None] * stride_c_in_k)
        + (offs_m[None, :, None] * stride_c_in_m)
        + (offs_n[None, None, :] * stride_c_in_n)
    )
    if ACTUAL_KSPLIT == MAX_KSPLIT:
        c = tl.load(c_in_ptrs)
    else:
        c = tl.load(c_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0)
    c = tl.sum(c, axis=0)
    c = c.to(c_out_ptr.type.element_ty)
    c_out_ptrs = (
        c_out_ptr
        + (offs_m[:, None] * stride_c_out_m)
        + (offs_n[None, :] * stride_c_out_n)
    )
    tl.store(c_out_ptrs, c)


# ----------------------------------------------------------------------------- #
# Per-(N,K,M-bucket) config selection. GROUP_K == BLOCK_SIZE_K == 128 is REQUIRED
# (one act/weight scale per 128-K block). cache_modifier ".cg" streams W (reused
# across M-tiles only via L2; weight is read once per N-tile per M-tile).
# ----------------------------------------------------------------------------- #
_BLOCK_K = 128  # == GROUP_K (scale block along K). Must stay 128.


def _select_config(M: int, N: int, K: int) -> dict:
    # Decode (tiny M): few M rows. Keep BM small to avoid wasted lanes; split-K to
    # fill CUs (GRID_MN=40 tiles otherwise idles most of 304 CUs).
    if M <= 16:
        return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_K=_BLOCK_K,
                    GROUP_SIZE_M=1, NUM_KSPLIT=8, num_warps=4, num_stages=2,
                    cache_modifier=None)
    if M <= 4096:
        # NUM_KSPLIT must keep each split-K partition a multiple of GROUP_K(=128) and
        # non-empty; KS=4 over 136 K-blocks is safe (KS>=16 leaves empty/misaligned
        # partitions -> wrong). split-K fills the 304 CUs the 40 N-tiles can't.
        return dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=_BLOCK_K,
                    GROUP_SIZE_M=1, NUM_KSPLIT=4, num_warps=4, num_stages=2,
                    cache_modifier=None)
    # Prefill (large M): deep-K / narrow-N. BM=256 (tall M-tile) amortizes the deep
    # 136-block K-loop over more rows and lifts MFMA occupancy; BN=128 (N is only 40
    # tiles wide), GROUP_SIZE_M=1, no split-K (enough M tiles already), no .cg cache
    # modifier (streaming W through L2 with default caching is faster here). ns=2 is
    # the LDS ceiling at BM=256 (ns>=3 OOMs LDS). Measured ~1.33x vs recorded baseline.
    return dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=128, BLOCK_SIZE_K=_BLOCK_K,
                GROUP_SIZE_M=1, NUM_KSPLIT=1, num_warps=8, num_stages=2,
                cache_modifier=None)


def gemm_a8w8_blockscale(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    skip_reduce: Optional[bool] = False,
) -> torch.Tensor:
    """Fused fp8 a8w8 block-scale GEMM. Y[M,N] = dequant(x) @ dequant(w)^T, bf16 out.

    Operands stay fp8; per-128-K-block scales folded into fp32 accumulation. Drop-in
    for aiter.ops.triton.gemm_a8w8_blockscale.
    """
    assert x.dim() == 2 and w.dim() == 2, "x,w must be 2-D"
    M, K = x.shape
    N, Kw = w.shape
    assert K == Kw, f"K mismatch: x K={K} vs w K={Kw}"
    out_dtype = dtype if dtype is not None else torch.bfloat16

    # w arrives as (N,K); kernel reads it transposed as (K,N) via strides.
    # w_scale arrives as (scale_n, scale_k); kernel wants (scale_k, scale_n).
    w_t = w.t()  # (K, N) view
    w_scale_t = w_scale.t()

    cfg = config if config is not None else _select_config(M, N, K)
    BLOCK_SIZE_M = cfg["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = cfg["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = cfg["BLOCK_SIZE_K"]
    GROUP_SIZE_M = cfg["GROUP_SIZE_M"]
    NUM_KSPLIT = cfg["NUM_KSPLIT"]
    num_warps = cfg.get("num_warps", 8)
    num_stages = cfg.get("num_stages", 2)
    cache_modifier = cfg.get("cache_modifier", None)

    GROUP_K = BLOCK_SIZE_K
    GROUP_N = triton.next_power_of_2(triton.cdiv(N, w_scale_t.shape[1]))

    if y is None:
        y = torch.empty((M, N), dtype=out_dtype, device=x.device)

    SPLITK_BLOCK_SIZE = triton.cdiv(K, NUM_KSPLIT)
    # Keep split-K partitions aligned to the 128-K scale block.
    if NUM_KSPLIT > 1:
        SPLITK_BLOCK_SIZE = triton.cdiv(SPLITK_BLOCK_SIZE, GROUP_K) * GROUP_K
        ACTUAL_KSPLIT = triton.cdiv(K, SPLITK_BLOCK_SIZE)
        y_pp = torch.empty((NUM_KSPLIT, M, N), dtype=torch.float32, device=x.device)
        c_target = y_pp
        stride_ck = y_pp.stride(0)
        stride_cm = y_pp.stride(1)
        stride_cn = y_pp.stride(2)
    else:
        ACTUAL_KSPLIT = 1
        y_pp = None
        c_target = y
        stride_ck = 0
        stride_cm = y.stride(0)
        stride_cn = y.stride(1)

    grid = lambda META: (  # noqa: E731
        META["NUM_KSPLIT"]
        * triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _fused_blockscale_kernel[grid](
        x,
        w_t,
        c_target,
        x_scale,
        w_scale_t,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_t.stride(0),  # stride_bk: stepping along K of W^T[K,N]
        w_t.stride(1),  # stride_bn: stepping along N of W^T[K,N]
        stride_ck,
        stride_cm,
        stride_cn,
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale_t.stride(0),
        w_scale_t.stride(1),
        GROUP_K=GROUP_K,
        GROUP_N=GROUP_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_KSPLIT=NUM_KSPLIT,
        SPLITK_BLOCK_SIZE=SPLITK_BLOCK_SIZE,
        cache_modifier=cache_modifier,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if NUM_KSPLIT > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _reduce_kernel[grid_reduce](
            y_pp,
            y,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(NUM_KSPLIT),
        )

    return y
