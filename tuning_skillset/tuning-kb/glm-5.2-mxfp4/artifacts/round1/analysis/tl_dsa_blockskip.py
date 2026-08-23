#!/usr/bin/env python3
"""Candidate A: skip index blocks that are entirely padding in the DSA sparse-MLA
fp8 partial kernel.

The stock kernel (sglang/kernels/ops/attention/dsa/tilelang_kernel.py,
`sparse_mla_fwd_decode_partial_fp8`) walks all `topk // block_I` index blocks for every
query and *masks* the -1 padding slots: it still gathers 64x576 bytes of KV per block and
still issues all five GEMMs, then throws the result away via a -inf score.

At this workload's shape that is a lot of provably dead work. The chunk is 16384 tokens =
2 requests x ISL 8192, and DSA selects index_topk=2048 keys per query, so every query at
position < 2048 within its sequence has a causal prefix shorter than topk and gets a
padded index row. Counting whole blocks of 64:

    processed today : 8192 * 32                              = 262144 blocks/sequence
    actually needed : sum_p min(ceil((p+1)/64), 32)          = 230400 blocks/sequence
    dead            : 31744                                  = 12.1%

Skipping an all-padding block is *bit-exact*, not an approximation. Every score in such a
block is -inf, so exp2 gives 0: it adds nothing to `sumexp` and nothing to `acc_o`, and
`m_i` is reduced with clear=False so a block of -inf cannot lower it either. The only
change is that the arithmetic is not performed.

Two skip predicates are implemented, selected by `skip_mode`:

  "reduce" (default, and what ships in patches/01) -- a block-uniform any-valid reduce over
      the validity mask the kernel already builds. Correct no matter how padding is laid
      out, and costs one LDS reduce against the 36 KB gather it elides.

  "first" -- test only the block's first slot. Valid because padding is a contiguous suffix
      today: `naive_paged_transform` in kernels/aot/csrc/elementwise/deepseek_v4_topk.hip
      writes page_to_slot(i) for i < length and -1 above it, the radix_topk branch pads not
      at all, and `_pad_topk_indices` / `_allocate_prefill_result` fill whole trailing rows.
      Faster in both regimes (prefill 1.067x vs 1.039x, decode 0.961x vs 0.927x) but it
      rests on an invariant that all four topk backends would each have to preserve, and a
      violation drops real KV keys silently. Measured, not adopted -- see FINDINGS.md 1b.

Either way the guard is dispatched prefill-only (`q.shape[0] >= SKIP_EMPTY_MIN_TOKENS`):
at decode no block is ever all-padding, so the guard never fires and both modes are pure
overhead. `skip_empty=False` is a parse-time Python bool, so the branch, the reduce and the
mask write vanish from the emitted HIP entirely and the factory returns a byte-identical
kernel to the stock one.
"""
import tilelang
import tilelang.language as T
import torch

from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
    _is_fp8_fnuz,
    _pick_inner_iter,
    pass_configs,
    sparse_mla_fwd_decode_combine,
)


# Token count at or above which the empty-block guard is compiled in.
#
# The guard only pays when index rows contain -1 padding, and padding only exists for
# queries whose causal prefix is shorter than index_topk=2048 -- a prefill property. At
# decode every sequence here is >= ISL 8192, no block is ever all-padding, and the guard
# is pure overhead (measured 34.55 -> 37.32 us, 0.926x). The two regimes are far apart in
# token count: decode batches are captured at HIP-graph sizes <= 512, prefill chunks are
# --chunked-prefill-size 16384. 1024 sits in the empty middle, so the split is not
# sensitive to where exactly it is put. This is a performance switch only -- both variants
# compute the same thing, so a misdispatch costs speed, never correctness.
SKIP_EMPTY_MIN_TOKENS = 1024


@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def sparse_mla_fwd_decode_partial_fp8_skip(
    num_heads: int,
    d_v: int,
    d_tail: int,
    topk: int,
    *,
    sm_scale=None,
    block_I=64,
    inner_iter=1,
    threads=256,
    h_per_block=16,
    skip_empty=True,
    skip_mode="reduce",
):
    assert d_v == 512, f"only support d_v=512"
    assert (
        topk % block_I == 0
    ), "otherwise will load some index=0 thus causing wrong kv to be loaded"

    fp8_dtype = "float8_e4m3fnuz" if _is_fp8_fnuz else "float8_e4m3fn"
    fp8_max_val = 240.0 if _is_fp8_fnuz else 448.0
    s_inv_scale_const = fp8_max_val
    s_scale_const = 1.0 / fp8_max_val

    BI = block_I
    group_size = 128
    dim_quant_fp8 = d_v + d_tail
    rope_offset_fp8 = d_v
    n_groups = topk // (BI * inner_iter)

    if sm_scale is None:
        sm_scale = (1.0 / (d_v + d_tail)) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    # h_per_block is the head tile. Stock hardcodes 16; at this model's tp_q_head_num=8
    # that means half of every tile -- and half of the fp32 output accumulators, which are
    # the kernel's dominant register cost -- is padding. Exposed here so it can be swept.
    assert (
        num_heads <= h_per_block or num_heads % h_per_block == 0
    ), "num_heads must be <= h_per_block or divisible by it"
    head_blocks_per_seq = (num_heads + h_per_block - 1) // h_per_block

    batch = 1
    kv_group = 1
    seq_len = T.symbolic("seq_len")
    num_pages = T.symbolic("num_pages")

    q_fp8_shape = [batch, seq_len, num_heads, d_v + d_tail]
    kv_fp8_shape = [batch, num_pages, kv_group, dim_quant_fp8]
    idx_shape = [batch, seq_len, kv_group, topk]
    partial_o_shape = [batch, seq_len, n_groups, num_heads, d_v]
    partial_lse_shape = [batch, seq_len, n_groups, num_heads]

    accum_dtype = T.float32
    dtype_bf16 = T.bfloat16

    @T.prim_func
    def main(
        q_fp8: T.Tensor(q_fp8_shape, fp8_dtype),
        kv_fp8: T.Tensor(kv_fp8_shape, fp8_dtype),
        indices: T.Tensor(idx_shape, T.int32),
        partial_o: T.Tensor(partial_o_shape, dtype_bf16),
        partial_lse: T.Tensor(partial_lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len * head_blocks_per_seq, n_groups, threads=threads) as (
            bx,
            by,
        ):
            b_i, g_i = 0, 0
            s_i = bx // head_blocks_per_seq
            group_i = by
            H0 = (bx % head_blocks_per_seq) * h_per_block
            H1 = H0 + h_per_block

            q_tile0 = T.alloc_shared([h_per_block, group_size], fp8_dtype)
            q_tile1 = T.alloc_shared([h_per_block, group_size], fp8_dtype)
            q_tile2 = T.alloc_shared([h_per_block, group_size], fp8_dtype)
            q_tile3 = T.alloc_shared([h_per_block, group_size], fp8_dtype)
            kv_tile0 = T.alloc_shared([BI, group_size], fp8_dtype)
            kv_tile1 = T.alloc_shared([BI, group_size], fp8_dtype)
            kv_tile2 = T.alloc_shared([BI, group_size], fp8_dtype)
            kv_tile3 = T.alloc_shared([BI, group_size], fp8_dtype)
            q_tail_buf = T.alloc_shared([h_per_block, d_tail], fp8_dtype)
            k_tail_shared = T.alloc_shared([BI, d_tail], fp8_dtype)
            s_fp8_shared = T.alloc_shared([h_per_block, BI], fp8_dtype)
            page_idx_shared = T.alloc_shared([BI], T.int32)

            mask = T.alloc_fragment([BI], T.bool)
            # Block-uniform "does this index block contain any real key at all".
            valid_i32 = T.alloc_fragment([1, BI], T.int32)
            any_valid = T.alloc_fragment([1], T.int32)
            acc_s = T.alloc_fragment([h_per_block, BI], accum_dtype)
            acc_tile = T.alloc_fragment([h_per_block, BI], accum_dtype)
            sv_tile = T.alloc_fragment([h_per_block, group_size], accum_dtype)
            sumexp = T.alloc_fragment([h_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([h_per_block], accum_dtype)
            alpha = T.alloc_fragment([h_per_block], accum_dtype)
            m_i = T.alloc_fragment([h_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([h_per_block], accum_dtype)
            inv_denom = T.alloc_fragment([h_per_block], accum_dtype)

            acc_o_tile0 = T.alloc_fragment([h_per_block, group_size], accum_dtype)
            acc_o_tile1 = T.alloc_fragment([h_per_block, group_size], accum_dtype)
            acc_o_tile2 = T.alloc_fragment([h_per_block, group_size], accum_dtype)
            acc_o_tile3 = T.alloc_fragment([h_per_block, group_size], accum_dtype)

            T.fill(acc_o_tile0, 0)
            T.fill(acc_o_tile1, 0)
            T.fill(acc_o_tile2, 0)
            T.fill(acc_o_tile3, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            T.copy(q_fp8[b_i, s_i, H0:H1, d_v:], q_tail_buf)
            T.copy(q_fp8[b_i, s_i, H0:H1, 0 * group_size : 1 * group_size], q_tile0)
            T.copy(q_fp8[b_i, s_i, H0:H1, 1 * group_size : 2 * group_size], q_tile1)
            T.copy(q_fp8[b_i, s_i, H0:H1, 2 * group_size : 3 * group_size], q_tile2)
            T.copy(q_fp8[b_i, s_i, H0:H1, 3 * group_size : 4 * group_size], q_tile3)

            for k_i in T.serial(inner_iter):
                topk_block_i = group_i * inner_iter + k_i

                # The index read is 256 B and happens either way; it is the 36 KB KV
                # gather and the five GEMMs behind it that the guard elides.
                for bi_i in T.Parallel(BI):
                    idx = indices[b_i, s_i, g_i, topk_block_i * BI + bi_i]
                    valid = idx >= 0
                    page_idx_shared[bi_i] = T.if_then_else(valid, idx, 0)
                    mask[bi_i] = valid
                    if skip_empty and skip_mode == "reduce":
                        valid_i32[0, bi_i] = T.if_then_else(valid, 1, 0)

                if skip_empty and skip_mode == "reduce":
                    T.reduce_max(valid_i32, any_valid, dim=1, clear=True)

                # Block-uniform branch: every thread in the block evaluates the same
                # reduced scalar, so the __syncthreads inside T.copy/T.gemm below stay
                # collective.
                #
                # `skip_empty` is a Python bool resolved at parse time, so with it False
                # `guard` is the Python literal True and the TVM Script parser drops the
                # branch, the cross-lane reduce and the valid_i32 write entirely (verified
                # against the emitted HIP). That matters: at decode every sequence is
                # longer than topk, so no index block is ever all-padding, the guard never
                # fires, and it costs a measured +8% -- more than the prefill gain it buys.
                #
                # Two ways to compute it, selected by `skip_mode`:
                #
                #   "reduce" -- OR the per-lane validity across the block. Assumes nothing
                #     about how the -1s are arranged, but costs a cross-lane AllReduce plus
                #     its LDS workspace (which also shifts page_idx_shared and eats into
                #     the shared-memory budget).
                #
                #   "first" -- test only indices[..., topk_block_i * BI]. This is a single
                #     scalar load, already block-uniform, no reduce and no extra LDS. It is
                #     correct because the -1s are always a contiguous suffix of the row:
                #     deepseek_v4_topk.hip's naive_paged_transform writes real slots for
                #     i < length and -1 for i >= length, and the radix_topk branch (length
                #     > topk) writes no -1 at all. Those are the only two producers of this
                #     tensor. If a third ever interleaves -1s, this mode silently drops
                #     real keys -- hence it stays an explicit, named choice.
                if not skip_empty:
                    guard = True
                elif skip_mode == "first":
                    guard = indices[b_i, s_i, g_i, topk_block_i * BI] >= 0
                else:
                    guard = any_valid[0] > 0
                if guard:
                    for bi_i, j in T.Parallel(BI, group_size):
                        page = page_idx_shared[bi_i]
                        kv_tile0[bi_i, j] = kv_fp8[b_i, page, g_i, 0 * group_size + j]
                        kv_tile1[bi_i, j] = kv_fp8[b_i, page, g_i, 1 * group_size + j]
                        kv_tile2[bi_i, j] = kv_fp8[b_i, page, g_i, 2 * group_size + j]
                        kv_tile3[bi_i, j] = kv_fp8[b_i, page, g_i, 3 * group_size + j]

                    for bi_i, j in T.Parallel(BI, d_tail):
                        page = page_idx_shared[bi_i]
                        k_tail_shared[bi_i, j] = kv_fp8[
                            b_i, page, g_i, rope_offset_fp8 + j
                        ]

                    for h_i, bi_i in T.Parallel(h_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            mask[bi_i], 0, -T.infinity(acc_s.dtype)
                        )

                    T.gemm(q_tile0, kv_tile0, acc_s, transpose_B=True, clear_accum=False)
                    T.gemm(q_tile1, kv_tile1, acc_tile, transpose_B=True, clear_accum=True)
                    for h_i, bi_i in T.Parallel(h_per_block, BI):
                        acc_s[h_i, bi_i] += acc_tile[h_i, bi_i]
                    T.gemm(q_tile2, kv_tile2, acc_tile, transpose_B=True, clear_accum=True)
                    for h_i, bi_i in T.Parallel(h_per_block, BI):
                        acc_s[h_i, bi_i] += acc_tile[h_i, bi_i]
                    T.gemm(q_tile3, kv_tile3, acc_tile, transpose_B=True, clear_accum=True)
                    for h_i, bi_i in T.Parallel(h_per_block, BI):
                        acc_s[h_i, bi_i] += acc_tile[h_i, bi_i]
                    T.gemm(
                        q_tail_buf,
                        k_tail_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(h_per_block):
                        alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(h_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(
                            acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                        )
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(h_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                    for h_i, j in T.Parallel(h_per_block, group_size):
                        acc_o_tile0[h_i, j] = acc_o_tile0[h_i, j] * alpha[h_i]
                        acc_o_tile1[h_i, j] = acc_o_tile1[h_i, j] * alpha[h_i]
                        acc_o_tile2[h_i, j] = acc_o_tile2[h_i, j] * alpha[h_i]
                        acc_o_tile3[h_i, j] = acc_o_tile3[h_i, j] * alpha[h_i]

                    for h_i, bi_i in T.Parallel(h_per_block, BI):
                        s_fp8_shared[h_i, bi_i] = T.clamp(
                            acc_s[h_i, bi_i] * s_inv_scale_const,
                            -fp8_max_val,
                            fp8_max_val,
                        )
                    T.gemm(s_fp8_shared, kv_tile0, sv_tile, clear_accum=True)
                    for h_i, j in T.Parallel(h_per_block, group_size):
                        acc_o_tile0[h_i, j] = (
                            acc_o_tile0[h_i, j] + sv_tile[h_i, j] * s_scale_const
                        )

                    T.gemm(s_fp8_shared, kv_tile1, sv_tile, clear_accum=True)
                    for h_i, j in T.Parallel(h_per_block, group_size):
                        acc_o_tile1[h_i, j] = (
                            acc_o_tile1[h_i, j] + sv_tile[h_i, j] * s_scale_const
                        )

                    T.gemm(s_fp8_shared, kv_tile2, sv_tile, clear_accum=True)
                    for h_i, j in T.Parallel(h_per_block, group_size):
                        acc_o_tile2[h_i, j] = (
                            acc_o_tile2[h_i, j] + sv_tile[h_i, j] * s_scale_const
                        )

                    T.gemm(s_fp8_shared, kv_tile3, sv_tile, clear_accum=True)
                    for h_i, j in T.Parallel(h_per_block, group_size):
                        acc_o_tile3[h_i, j] = (
                            acc_o_tile3[h_i, j] + sv_tile[h_i, j] * s_scale_const
                        )

            for h_i in T.Parallel(h_per_block):
                denom = T.if_then_else(sumexp[h_i] == 0.0, 1.0, sumexp[h_i])
                inv_denom[h_i] = 1.0 / denom
            for h_i, j in T.Parallel(h_per_block, group_size):
                acc_o_tile0[h_i, j] = acc_o_tile0[h_i, j] * inv_denom[h_i]
                acc_o_tile1[h_i, j] = acc_o_tile1[h_i, j] * inv_denom[h_i]
                acc_o_tile2[h_i, j] = acc_o_tile2[h_i, j] * inv_denom[h_i]
                acc_o_tile3[h_i, j] = acc_o_tile3[h_i, j] * inv_denom[h_i]

            for h_i in T.Parallel(h_per_block):
                sumexp[h_i] = T.if_then_else(
                    sumexp[h_i] == 0.0,
                    -(2**30),
                    T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale,
                )

            T.copy(
                acc_o_tile0,
                partial_o[b_i, s_i, group_i, H0:H1, 0 * group_size : 1 * group_size],
            )
            T.copy(
                acc_o_tile1,
                partial_o[b_i, s_i, group_i, H0:H1, 1 * group_size : 2 * group_size],
            )
            T.copy(
                acc_o_tile2,
                partial_o[b_i, s_i, group_i, H0:H1, 2 * group_size : 3 * group_size],
            )
            T.copy(
                acc_o_tile3,
                partial_o[b_i, s_i, group_i, H0:H1, 3 * group_size : 4 * group_size],
            )

            T.copy(sumexp, partial_lse[b_i, s_i, group_i, H0:H1])

    return main


def tilelang_sparse_fwd_skip(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    block_I: int = 64,
    threads: int = 256,
    block_per_cu: int = 2,
    h_per_block: int = 16,
    skip_empty: bool = None,
    skip_mode: str = "reduce",
) -> torch.Tensor:
    """Drop-in for tilelang_sparse_fwd on the gfx95 + fp8-KV path."""
    num_heads = q.shape[1]
    tail_dim = q.shape[2] - d_v
    topk = indices.shape[-1]
    assert topk == 2048

    if q.dtype != kv.dtype:
        q = q.to(kv.dtype)
    cu = 256
    ni = topk // block_I
    inner_iter = _pick_inner_iter(q.shape[0], ni, cu, block_per_cu)
    if skip_empty is None:
        skip_empty = q.shape[0] >= SKIP_EMPTY_MIN_TOKENS
    kernel_partial = sparse_mla_fwd_decode_partial_fp8_skip(
        num_heads,
        d_v,
        tail_dim,
        topk,
        sm_scale=sm_scale,
        block_I=block_I,
        inner_iter=inner_iter,
        threads=threads,
        h_per_block=h_per_block,
        skip_empty=skip_empty,
        skip_mode=skip_mode,
    )
    partial_o, partial_lse = kernel_partial(
        q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0)
    )
    n_groups = ni // inner_iter
    kernel_combine = sparse_mla_fwd_decode_combine(
        num_heads,
        d_v,
        n_groups * block_I,
        head_per_block=4,
        block_I=block_I,
        threads=threads,
    )
    return kernel_combine(partial_o, partial_lse)
