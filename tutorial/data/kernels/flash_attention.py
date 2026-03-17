import os
import torch
import triton
import triton.language as tl
from maybe_triton_jit import maybe_triton_jit

# from .utils import is_hopper, is_ampere

# configs = [
#     triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
#     for BM in [32, 64, 128]
#     for BN in [32, 64, 128]
#     for s in [1, 2, 3, 4]
#     for w in [4, 8]
# ]


# The following best configs are obtained by autotuning under dim64.
# Currently, only sm80(A100/A800), sm89(L), and sm90(H20) are supported.
# Developers can autotune according to your own needs.
def get_fwd_configs():
    if 0:  # is_hopper():
        return [
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=2, num_warps=4)
        ]
    elif 0:  # is_ampere():
        return [
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=3, num_warps=4)
        ]
    else:
        return [
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=2, num_warps=4)
        ]


def get_bwd_preprocess_configs():
    if 0:  # is_hopper():
        return [triton.Config({"BLOCK_M": 32}, num_stages=4, num_warps=4)]
    elif 0:  # is_ampere():
        return [triton.Config({"BLOCK_M": 32}, num_stages=3, num_warps=4)]
    else:
        return [triton.Config({"BLOCK_M": 32}, num_stages=3, num_warps=8)]


def get_bwd_q_configs():
    if 0:  # is_hopper():
        return [
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=4, num_warps=4)
        ]
    elif 0:  # is_ampere():
        return [
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=1, num_warps=4)
        ]
    else:
        return [
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=1, num_warps=4)
        ]


def get_bwd_kv_configs():
    if 0:  # is_hopper():
        return [
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=4, num_warps=4)
        ]
    elif 0:  # is_ampere():
        return [
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=1, num_warps=4)
        ]
    else:
        return [
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=1, num_warps=4)
        ]


if os.environ.get("TRITON_DEBUG") == "1":
    configs = [triton.Config({"BLOCK_M": 32, "BLOCK_N": 16}, num_stages=1, num_warps=1)]
    bwd_preprocess_configs = [triton.Config({"BLOCK_M": 32}, num_stages=1, num_warps=1)]


def keep(config):
    m = config.kwargs["BLOCK_M"]
    n = config.kwargs["BLOCK_N"]
    if torch.mlu.is_available() and torch.mlu.get_device_properties(0).major == 9:
        if m == 64 and config.num_warps == 8:
            return False
    return m % n == 0


@triton.jit
def load_if(block_ptr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        return tl.load(block_ptr)
    elif EVEN_M:
        return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
    elif EVEN_N:
        return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
    else:
        return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")


@triton.jit
def store_if(block_ptr, value, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        tl.store(block_ptr, value)
    elif EVEN_N:
        tl.store(block_ptr, value, boundary_check=(0,))
    elif EVEN_M:
        tl.store(block_ptr, value, boundary_check=(1,))
    else:
        tl.store(block_ptr, value, boundary_check=(0, 1))


@triton.jit
def mask_fn(q_attn_arg, k_attn_arg, q_offset, k_offset, TYPE: tl.constexpr):
    tril_causal = q_offset[:, None] >= k_offset[None, :]
    triu_causal = q_offset[:, None] <= k_offset[None, :]
    # attn_arg = 0 代表 sequence，非 0 代表 query，不同 query 用不同的 attn_arg
    if TYPE == 1:
        return (
            triu_causal
            & (
                (q_attn_arg[:, None] == k_attn_arg[None, :])
                | (k_attn_arg[None, :] == 0)
            )
        ) | (q_offset[:, None] == k_offset[None, :])
    if TYPE == 2:
        return (
            tril_causal
            & (
                (q_attn_arg[:, None] == k_attn_arg[None, :])
                | (k_attn_arg[None, :] == 0)
            )
        ) | (q_offset[:, None] == k_offset[None, :])


@maybe_triton_jit(
    list(filter(keep, get_fwd_configs())),
    key=["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"],
)
@triton.jit
def fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    l_ptr,
    q_attn_arg_ptr,
    k_attn_arg_ptr,
    cu_seqlens_q,
    cu_seqlens_k,
    q_head,
    kv_head,
    scale,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    MASK_FN: tl.constexpr,
    SPARSE_OPT: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = o_ptr.type.element_ty
    start_m = tl.program_id(0)
    start_qh = tl.program_id(1)
    start_b = tl.program_id(2)
    start_kvh = start_qh // (q_head // kv_head)

    q_start = tl.load(cu_seqlens_q + start_b)
    q_end = tl.load(cu_seqlens_q + start_b + 1)
    q_len = q_end - q_start
    if start_m * BLOCK_M >= q_len:
        return

    k_start = tl.load(cu_seqlens_k + start_b)
    k_end = tl.load(cu_seqlens_k + start_b + 1)
    k_len = k_end - k_start
    if SPARSE_OPT:
        begin = 0
        if k_len == 0:
            return
        end = k_len
    else:
        if MASK_FN & 1:
            begin = start_m * BLOCK_M
            if begin >= k_len:
                return
            end = k_len
        else:
            begin = 0
            end = tl.minimum((start_m + 1) * BLOCK_M, k_len)

    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = scale * log2e
    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    q_start = q_start.to(tl.int64)
    k_start = k_start.to(tl.int64)
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM,
        shape=(q_len, QK_DIM),
        strides=(q_head * QK_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, QK_DIM),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_start * kv_head * QK_DIM + start_kvh * QK_DIM,
        shape=(QK_DIM, k_len),
        strides=(1, kv_head * QK_DIM),
        offsets=(0, begin),
        block_shape=(QK_DIM, BLOCK_N),
        order=(0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + k_start * kv_head * V_DIM + start_kvh * V_DIM,
        shape=(k_len, V_DIM),
        strides=(kv_head * V_DIM, 1),
        offsets=(begin, 0),
        block_shape=(BLOCK_N, V_DIM),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + q_start * q_head * V_DIM + start_qh * V_DIM,
        shape=(q_len, V_DIM),
        strides=(q_head * V_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, V_DIM),
        order=(1, 0),
    )
    l_block_ptr = tl.make_block_ptr(
        base=l_ptr + q_start * q_head + start_qh,
        shape=(q_len,),
        strides=(q_head,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    q_attn_arg_block_ptr = tl.make_block_ptr(
        base=q_attn_arg_ptr + q_start,
        shape=(q_len,),
        strides=(1,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    k_attn_arg_block_ptr = tl.make_block_ptr(
        base=k_attn_arg_ptr + k_start,
        shape=(k_len,),
        strides=(1,),
        offsets=(begin,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )

    acc = tl.zeros((BLOCK_M, V_DIM), dtype=tl.float32)
    m = tl.full((BLOCK_M,), value=-(2**30), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)

    q = load_if(q_block_ptr, False, True)
    q_attn_arg = load_if(q_attn_arg_block_ptr, False, True)

    for start_n in range(begin, end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_attn_arg = load_if(k_attn_arg_block_ptr, False, True)
        offset_n = start_n + tl.arange(0, BLOCK_N)
        mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n, MASK_FN)
        if not SPARSE_OPT or tl.sum(mask.cast(tl.int32)) != 0:
            k = load_if(k_block_ptr, True, False)
            # v = load_if(v_block_ptr, False, True)
            s = tl.dot(q, k)
            boundary_mask = (offset_n < k_len)[None, :]
            s = tl.where(mask & boundary_mask, s, -(2**30))
            m_new = tl.maximum(m, tl.max(s, 1))
            alpha = tl.math.exp2((m - m_new) * qk_scale)
            p = tl.math.exp2((s - m_new[:, None]) * qk_scale)
            p_sum = tl.sum(p, 1)
            acc *= alpha[:, None]
            v = load_if(v_block_ptr, False, True)
            acc += tl.dot(p.to(dtype), v)
            l = l * alpha + p_sum
            m = m_new
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))
        k_attn_arg_block_ptr = tl.advance(k_attn_arg_block_ptr, (BLOCK_N,))

    acc = acc / l[:, None]
    l = m * scale + tl.log(l)

    store_if(o_block_ptr, acc.to(dtype), False, True)
    store_if(l_block_ptr, l, False, True)


@maybe_triton_jit(get_bwd_preprocess_configs(), key=["V_DIM"])
@triton.jit
def bwd_preprocess(
    o_ptr,
    do_ptr,
    d_ptr,
    cu_seqlens_q,
    q_head,
    V_DIM: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    start_h = tl.program_id(1)
    start_b = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q + start_b)
    q_end = tl.load(cu_seqlens_q + start_b + 1)
    q_len = q_end - q_start
    if start_m * BLOCK_M >= q_len:
        return

    q_start = q_start.to(tl.int64)
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + q_start * q_head * V_DIM + start_h * V_DIM,
        shape=(q_len, V_DIM),
        strides=(q_head * V_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, V_DIM),
        order=(1, 0),
    )
    do_block_ptr = tl.make_block_ptr(
        base=do_ptr + q_start * q_head * V_DIM + start_h * V_DIM,
        shape=(q_len, V_DIM),
        strides=(q_head * V_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, V_DIM),
        order=(1, 0),
    )
    d_block_ptr = tl.make_block_ptr(
        base=d_ptr + q_start * q_head + start_h,
        shape=(q_len,),
        strides=(q_head,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    o = load_if(o_block_ptr, False, True).to(tl.float32)
    do = load_if(do_block_ptr, False, True).to(tl.float32)
    d = tl.sum(o * do, 1)
    store_if(d_block_ptr, d, False, True)


@maybe_triton_jit(
    list(filter(keep, get_bwd_kv_configs())),
    key=["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"],
)
@triton.jit
def bwd_kv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    dk_ptr,
    dv_ptr,
    do_ptr,
    l_ptr,
    d_ptr,
    q_attn_arg_ptr,
    k_attn_arg_ptr,
    cu_seqlens_q,
    cu_seqlens_k,
    q_head,
    kv_head,
    scale,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    MASK_FN: tl.constexpr,
    SPARSE_OPT: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = k_ptr.type.element_ty
    start_n = tl.program_id(0)
    start_qh = tl.program_id(1)
    start_b = tl.program_id(2)
    start_kvh = start_qh // (q_head // kv_head)

    k_start = tl.load(cu_seqlens_k + start_b)
    k_end = tl.load(cu_seqlens_k + start_b + 1)
    k_len = k_end - k_start
    if start_n * BLOCK_N >= k_len:
        return

    q_start = tl.load(cu_seqlens_q + start_b)
    q_end = tl.load(cu_seqlens_q + start_b + 1)
    q_len = q_end - q_start

    if SPARSE_OPT:
        begin = 0
        end = q_len
    else:
        if MASK_FN & 1:
            begin = 0
            end = tl.minimum(start_n * BLOCK_N // BLOCK_M * BLOCK_M + BLOCK_M, q_len)
        else:
            begin = start_n * BLOCK_N // BLOCK_M * BLOCK_M
            end = q_len

    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = scale * log2e
    offset_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    q_start = q_start.to(tl.int64)
    k_start = k_start.to(tl.int64)
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM,
        shape=(q_len, QK_DIM),
        strides=(q_head * QK_DIM, 1),
        offsets=(begin, 0),
        block_shape=(BLOCK_M, QK_DIM),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_start * kv_head * QK_DIM + start_kvh * QK_DIM,
        shape=(QK_DIM, k_len),
        strides=(1, kv_head * QK_DIM),
        offsets=(0, start_n * BLOCK_N),
        block_shape=(QK_DIM, BLOCK_N),
        order=(0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + k_start * kv_head * V_DIM + start_kvh * V_DIM,
        shape=(V_DIM, k_len),
        strides=(1, kv_head * V_DIM),
        offsets=(0, start_n * BLOCK_N),
        block_shape=(V_DIM, BLOCK_N),
        order=(0, 1),
    )
    dk_block_ptr = tl.make_block_ptr(
        base=dk_ptr + k_start * q_head * QK_DIM + start_qh * QK_DIM,
        shape=(k_len, QK_DIM),
        strides=(q_head * QK_DIM, 1),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, QK_DIM),
        order=(1, 0),
    )
    dv_block_ptr = tl.make_block_ptr(
        base=dv_ptr + k_start * q_head * V_DIM + start_qh * V_DIM,
        shape=(k_len, V_DIM),
        strides=(q_head * V_DIM, 1),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, V_DIM),
        order=(1, 0),
    )
    do_block_ptr = tl.make_block_ptr(
        base=do_ptr + q_start * q_head * V_DIM + start_qh * V_DIM,
        shape=(q_len, V_DIM),
        strides=(q_head * V_DIM, 1),
        offsets=(begin, 0),
        block_shape=(BLOCK_M, V_DIM),
        order=(1, 0),
    )
    l_block_ptr = tl.make_block_ptr(
        base=l_ptr + q_start * q_head + start_qh,
        shape=(q_len,),
        strides=(q_head,),
        offsets=(begin,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    d_block_ptr = tl.make_block_ptr(
        base=d_ptr + q_start * q_head + start_qh,
        shape=(q_len,),
        strides=(q_head,),
        offsets=(begin,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    q_attn_arg_block_ptr = tl.make_block_ptr(
        base=q_attn_arg_ptr + q_start,
        shape=(q_len,),
        strides=(1,),
        offsets=(begin,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    k_attn_arg_block_ptr = tl.make_block_ptr(
        base=k_attn_arg_ptr + k_start,
        shape=(k_len,),
        strides=(1,),
        offsets=(start_n * BLOCK_N,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )

    dk = tl.zeros((BLOCK_N, QK_DIM), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, V_DIM), dtype=tl.float32)

    k = load_if(k_block_ptr, True, False)
    v = load_if(v_block_ptr, True, False)
    k_attn_arg = load_if(k_attn_arg_block_ptr, False, True)

    for start_m in range(begin, end, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        q_attn_arg = load_if(q_attn_arg_block_ptr, False, True)
        offset_m = start_m + tl.arange(0, BLOCK_M)
        mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n, MASK_FN)
        if not SPARSE_OPT or tl.sum(mask.cast(tl.int32)) != 0:
            q = load_if(q_block_ptr, False, True)
            s = tl.dot(q, k)
            l = load_if(l_block_ptr, False, True)
            p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
            p = tl.where(mask, p, 0.0)
            do = load_if(do_block_ptr, False, True)
            dv += tl.dot(tl.trans(p).to(dtype), do)
            d = load_if(d_block_ptr, False, True)
            dp = tl.dot(do, v)
            ds = p * (dp - d[:, None])
            ds = tl.where(mask, ds, 0.0)
            dk += tl.dot(tl.trans(ds).to(dtype), q)
        q_block_ptr = tl.advance(q_block_ptr, (BLOCK_M, 0))
        do_block_ptr = tl.advance(do_block_ptr, (BLOCK_M, 0))
        l_block_ptr = tl.advance(l_block_ptr, (BLOCK_M,))
        d_block_ptr = tl.advance(d_block_ptr, (BLOCK_M,))
        q_attn_arg_block_ptr = tl.advance(q_attn_arg_block_ptr, (BLOCK_M,))

    dk *= scale
    store_if(dk_block_ptr, dk.to(dtype), False, True)
    store_if(dv_block_ptr, dv.to(dtype), False, True)


@maybe_triton_jit(
    list(filter(keep, get_bwd_q_configs())),
    key=["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"],
)
@triton.jit
def bwd_q_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    dq_ptr,
    do_ptr,
    l_ptr,
    d_ptr,
    q_attn_arg_ptr,
    k_attn_arg_ptr,
    cu_seqlens_q,
    cu_seqlens_k,
    q_head,
    kv_head,
    scale,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    MASK_FN: tl.constexpr,
    SPARSE_OPT: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = q_ptr.type.element_ty
    start_m = tl.program_id(0)
    start_qh = tl.program_id(1)
    start_b = tl.program_id(2)
    start_kvh = start_qh // (q_head // kv_head)

    q_start = tl.load(cu_seqlens_q + start_b)
    q_end = tl.load(cu_seqlens_q + start_b + 1)
    q_len = q_end - q_start
    if start_m * BLOCK_M >= q_len:
        return

    k_start = tl.load(cu_seqlens_k + start_b)
    k_end = tl.load(cu_seqlens_k + start_b + 1)
    k_len = k_end - k_start

    if SPARSE_OPT:
        begin = 0
        end = k_len
    else:
        if MASK_FN & 1:
            begin = start_m * BLOCK_M
            if begin >= k_len:
                return
            end = k_len
        else:
            begin = 0
            end = tl.minimum((start_m + 1) * BLOCK_M, k_len)

    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = scale * log2e
    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    q_start = q_start.to(tl.int64)
    k_start = k_start.to(tl.int64)
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM,
        shape=(q_len, QK_DIM),
        strides=(q_head * QK_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, QK_DIM),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_start * kv_head * QK_DIM + start_kvh * QK_DIM,
        shape=(k_len, QK_DIM),
        strides=(kv_head * QK_DIM, 1),
        offsets=(begin, 0),
        block_shape=(BLOCK_N, QK_DIM),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + k_start * kv_head * V_DIM + start_kvh * V_DIM,
        shape=(V_DIM, k_len),
        strides=(1, kv_head * V_DIM),
        offsets=(0, begin),
        block_shape=(V_DIM, BLOCK_N),
        order=(0, 1),
    )
    dq_block_ptr = tl.make_block_ptr(
        base=dq_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM,
        shape=(q_len, QK_DIM),
        strides=(q_head * QK_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, QK_DIM),
        order=(1, 0),
    )
    do_block_ptr = tl.make_block_ptr(
        base=do_ptr + q_start * q_head * V_DIM + start_qh * V_DIM,
        shape=(q_len, V_DIM),
        strides=(q_head * V_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, V_DIM),
        order=(1, 0),
    )
    l_block_ptr = tl.make_block_ptr(
        base=l_ptr + q_start * q_head + start_qh,
        shape=(q_len,),
        strides=(q_head,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    d_block_ptr = tl.make_block_ptr(
        base=d_ptr + q_start * q_head + start_qh,
        shape=(q_len,),
        strides=(q_head,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    q_attn_arg_block_ptr = tl.make_block_ptr(
        base=q_attn_arg_ptr + q_start,
        shape=(q_len,),
        strides=(1,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    k_attn_arg_block_ptr = tl.make_block_ptr(
        base=k_attn_arg_ptr + k_start,
        shape=(k_len,),
        strides=(1,),
        offsets=(begin,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )

    dq = tl.zeros((BLOCK_M, QK_DIM), dtype=tl.float32)

    q = load_if(q_block_ptr, False, True)
    do = load_if(do_block_ptr, False, True)
    l = load_if(l_block_ptr, False, True)
    d = load_if(d_block_ptr, False, True)
    q_attn_arg = load_if(q_attn_arg_block_ptr, False, True)

    for start_n in range(begin, end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_attn_arg = load_if(k_attn_arg_block_ptr, False, True)
        offset_n = start_n + tl.arange(0, BLOCK_N)
        mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n, MASK_FN)
        if not SPARSE_OPT or tl.sum(mask.cast(tl.int32)) != 0:
            k = load_if(k_block_ptr, False, True)
            v = load_if(v_block_ptr, True, False)
            s = tl.dot(q, tl.trans(k))
            p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
            dp = tl.dot(do, v)
            ds = p * (dp - d[:, None])
            boundary_mask = (offset_n < k_len)[None, :]
            ds = tl.where(mask & boundary_mask, ds, 0.0)
            dq += tl.dot(ds.to(dtype), k)
        k_block_ptr = tl.advance(k_block_ptr, (BLOCK_N, 0))
        v_block_ptr = tl.advance(v_block_ptr, (0, BLOCK_N))
        k_attn_arg_block_ptr = tl.advance(k_attn_arg_block_ptr, (BLOCK_N,))

    dq *= scale
    store_if(dq_block_ptr, dq.to(dtype), False, True)


# q: [total_q_seq, head, dim]
# k: [total_kv_seq, head, dim]
class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        q_attn_arg,
        k_attn_arg,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        scale,
        mask_fn,
        sparse_opt,
    ):
        q_len, q_head, qk_dim = q.shape
        k_len, kv_head, v_dim = v.shape
        batch_size = cu_seqlens_q.shape[0] - 1
        o = q.new_empty(q_len, q_head, v_dim)
        l = q.new_empty(q_len, q_head, dtype=torch.float32)
        grid = lambda META: (
            triton.cdiv(max_seqlen_q, META["BLOCK_M"]),
            q_head,
            batch_size,
        )
        fwd_kernel[grid](
            q,
            k,
            v,
            o,
            l,
            q_attn_arg,
            k_attn_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            q_head,
            kv_head,
            scale,
            QK_DIM=qk_dim,
            V_DIM=v_dim,
            MASK_FN=mask_fn,
            SPARSE_OPT=sparse_opt,
            DTYPE=(19 if q.dtype == torch.float16 else 14),
        )
        ctx.save_for_backward(
            q, k, v, o, l, q_attn_arg, k_attn_arg, cu_seqlens_q, cu_seqlens_k
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.scale = scale
        ctx.mask_fn = mask_fn
        ctx.sparse_opt = sparse_opt
        ctx.k_len = k_len
        ctx.q_head = q_head
        ctx.kv_head = kv_head
        ctx.qk_dim = qk_dim
        ctx.v_dim = v_dim
        ctx.batch_size = batch_size
        ctx.max_seqlen_k = max_seqlen_k
        ctx.dtype = 19 if q.dtype == torch.float16 else 14
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l, q_attn_arg, k_attn_arg, cu_seqlens_q, cu_seqlens_k = (
            ctx.saved_tensors
        )
        dq = torch.empty_like(q)
        dk = k.new_empty(ctx.k_len, ctx.q_head, ctx.qk_dim)
        dv = v.new_empty(ctx.k_len, ctx.q_head, ctx.v_dim)
        d = torch.empty_like(l)
        grid = lambda META: (
            triton.cdiv(ctx.max_seqlen_q, META["BLOCK_M"]),
            ctx.q_head,
            ctx.batch_size,
        )
        bwd_preprocess[grid](
            o,
            do,
            d,
            cu_seqlens_q,
            ctx.q_head,
            V_DIM=ctx.v_dim,
            DTYPE=ctx.dtype,
        )
        grid = lambda META: (
            triton.cdiv(ctx.max_seqlen_k, META["BLOCK_N"]),
            ctx.q_head,
            ctx.batch_size,
        )
        bwd_kv_kernel[grid](
            q,
            k,
            v,
            dk,
            dv,
            do,
            l,
            d,
            q_attn_arg,
            k_attn_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.q_head,
            ctx.kv_head,
            ctx.scale,
            QK_DIM=ctx.qk_dim,
            V_DIM=ctx.v_dim,
            MASK_FN=ctx.mask_fn,
            SPARSE_OPT=ctx.sparse_opt,
            DTYPE=ctx.dtype,
        )
        grid = lambda META: (
            triton.cdiv(ctx.max_seqlen_q, META["BLOCK_M"]),
            ctx.q_head,
            ctx.batch_size,
        )
        bwd_q_kernel[grid](
            q,
            k,
            v,
            dq,
            do,
            l,
            d,
            q_attn_arg,
            k_attn_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.q_head,
            ctx.kv_head,
            ctx.scale,
            QK_DIM=ctx.qk_dim,
            V_DIM=ctx.v_dim,
            MASK_FN=ctx.mask_fn,
            SPARSE_OPT=ctx.sparse_opt,
            DTYPE=ctx.dtype,
        )
        head_group = ctx.q_head // ctx.kv_head
        if head_group > 1:
            dk = dk.reshape(ctx.k_len, ctx.kv_head, head_group, ctx.qk_dim).sum(2)
            dv = dv.reshape(ctx.k_len, ctx.kv_head, head_group, ctx.v_dim).sum(2)
        return (dq, dk, dv) + (None,) * 9
