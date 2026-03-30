#!/usr/bin/env python3
"""
Fused RMSNorm + FP8 Quantization Kernel Implementation

Based on aiter's fused_fp8_quant implementation (ROCm/aiter):
- Fuses RMSNorm normalization with FP8 quantization
- Supports per-tensor static and per-token group quantization
- Supports residual add, second input RMSNorm, activation+mul, and reduction variants
- Reduces memory bandwidth by avoiding intermediate tensors

All 6 variants are included:
  1. fused_rms_fp8_per_tensor_static_quant
  2. fused_rms_fp8_group_quant
  3. fused_flatten_fp8_group_quant
  4. fused_reduce_act_mul_fp8_group_quant
  5. fused_reduce_rms_fp8_group_quant
  6. fused_silu_mul_fp8_per_tensor_static_quant
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.libdevice import fast_dividef, fast_expf
except ImportError:
    try:
        from triton.language.extra.cuda.libdevice import fast_dividef, fast_expf
    except ImportError:
        from triton.language.math import fast_dividef, fast_expf


fp8_dtype = torch.float8_e4m3fnuz


# ======
# INLINED: aiter/ops/triton/_triton_kernels/activation.py (subset)
# ======


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _silu_exp2(x):
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _gelu(x):
    M_SQRT1_2 = 0.70710678118654752440
    ALPHA = M_SQRT1_2
    return 0.5 * x * (1.0 + tl.erf(x * ALPHA))


@triton.jit
def _gelu_tanh(x):
    M_SQRT2 = 1.41421356237309504880
    M_2_SQRTPI = 1.12837916709551257390
    BETA = M_SQRT2 * M_2_SQRTPI * 0.5
    KAPPA = 0.044715
    x_cube = x * x * x
    inner = BETA * (x + KAPPA * x_cube)
    return 0.5 * x * (1.0 + _tanh(inner))


@triton.jit
def _relu(x):
    return tl.maximum(0.0, x)


def _get_activation_from_str(activation: str):
    mapping = {
        "gelu": _gelu,
        "gelu_tanh": _gelu_tanh,
        "silu": _silu,
        "silu_exp2": _silu_exp2,
        "relu": _relu,
    }
    return mapping[activation]


# ======
# TRITON KERNELS
# ======


@triton.jit
def _rmsmorm_op(row, weight, n_cols, epsilon):
    row_norm = row * row
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)
    rms_norm = row * norm_factor * weight
    return rms_norm


@triton.jit
def _fp8_quant_op(
    x,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
):
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, QUANT_BLOCK_SIZE)
    m = tl.maximum(tl.max(tl.abs(x), axis=-1), 1e-10)
    scale_out = m.to(tl.float32) / DTYPE_MAX
    scale_recip = 1.0 / scale_out.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, 1)
    x = tl.clamp(x * scale_recip, DTYPE_MIN, DTYPE_MAX)
    return x, scale_out


@triton.jit
def _fused_rms_fp8_per_tensor_static_quant_kernel(
    inp1_ptr, weight1_ptr, inp2_ptr, weight2_ptr, res1_ptr,
    out1_fp8_ptr, out2_ptr, out_res1_ptr, out1_ptr, scale_ptr,
    eps1, eps2, n_rows, inp1_n_cols, inp2_n_cols,
    inp1_row_stride, inp2_row_stride, inp1_col_stride, inp2_col_stride,
    res1_row_stride, res1_col_stride,
    out1_fp8_row_stride, out1_fp8_col_stride,
    out2_row_stride, out2_col_stride,
    out_res1_row_stride, out_res1_col_stride,
    out1_row_stride, out1_col_stride,
    BLOCK_SIZE_N: tl.constexpr, DTYPE_MAX: tl.constexpr, DTYPE_MIN: tl.constexpr,
    HAVE_SECOND_INPUT: tl.constexpr, FIRST_INPUT_RES: tl.constexpr,
    FIRST_INPUT_OUT: tl.constexpr, RMSNORM_CONVERT_TO_INP1_TYPE: tl.constexpr,
):
    m_pid = tl.program_id(0)
    n_offs = tl.arange(0, BLOCK_SIZE_N)
    mask1 = n_offs < inp1_n_cols
    inp1 = tl.load(
        inp1_ptr + m_pid * inp1_row_stride + n_offs * inp1_col_stride,
        mask=mask1, other=0.0, cache_modifier=".cg",
    ).to(tl.float32)
    if FIRST_INPUT_RES:
        res1 = tl.load(
            res1_ptr + m_pid * res1_row_stride + n_offs * res1_col_stride,
            mask=mask1, other=0.0, cache_modifier=".cg",
        ).to(tl.float32)
        inp1 = inp1 + res1
    w1 = tl.load(weight1_ptr + n_offs, mask=mask1, other=0.0).to(tl.float32)
    norm1 = _rmsmorm_op(inp1, w1, inp1_n_cols, eps1)
    if FIRST_INPUT_OUT:
        mask1 = n_offs < inp1_n_cols
        tl.store(out1_ptr + m_pid * out1_row_stride + n_offs * out1_col_stride, norm1, mask=mask1)
    if RMSNORM_CONVERT_TO_INP1_TYPE:
        norm1 = norm1.to(inp1_ptr.dtype.element_ty)
        norm1 = norm1.to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    scale_recip = 1.0 / scale
    out1_fp8 = tl.clamp(norm1 * scale_recip, DTYPE_MIN, DTYPE_MAX)
    tl.store(
        out1_fp8_ptr + m_pid * out1_fp8_row_stride + n_offs * out1_fp8_col_stride,
        out1_fp8.to(out1_fp8_ptr.dtype.element_ty), mask=mask1,
    )
    if HAVE_SECOND_INPUT:
        mask2 = n_offs < inp2_n_cols
        inp2 = tl.load(
            inp2_ptr + m_pid * inp2_row_stride + n_offs * inp2_col_stride,
            mask=mask2, other=0.0, cache_modifier=".cg",
        ).to(tl.float32)
        w2 = tl.load(weight2_ptr + n_offs, mask=mask2, other=0.0).to(tl.float32)
        norm2 = _rmsmorm_op(inp2, w2, inp2_n_cols, eps2)
        tl.store(out2_ptr + m_pid * out2_row_stride + n_offs * out2_col_stride, norm2, mask=mask2)
    if FIRST_INPUT_RES:
        inp1 = inp1.to(out_res1_ptr.dtype.element_ty)
        tl.store(
            out_res1_ptr + m_pid * out_res1_row_stride + n_offs * out_res1_col_stride,
            inp1, mask=mask1,
        )


@triton.jit
def _fused_rms_fp8_group_quant_kernel(
    inp1_ptr, weight1_ptr, inp2_ptr, weight2_ptr, res1_ptr,
    out1_fp8_ptr, out1_bs_ptr, out2_ptr, out_res1_ptr, out1_ptr,
    eps1, eps2, n_rows, inp1_n_cols, inp2_n_cols,
    inp1_row_stride, inp2_row_stride, inp1_col_stride, inp2_col_stride,
    res1_row_stride, res1_col_stride,
    out1_fp8_row_stride, out1_fp8_col_stride,
    out1_bs_row_stride, out1_bs_col_stride,
    out2_row_stride, out2_col_stride,
    out_res1_row_stride, out_res1_col_stride,
    out1_row_stride, out1_col_stride,
    BLOCK_SIZE_N: tl.constexpr, QUANT_BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr, DTYPE_MIN: tl.constexpr,
    HAVE_SECOND_INPUT: tl.constexpr, FIRST_INPUT_RES: tl.constexpr,
    FIRST_INPUT_OUT: tl.constexpr,
):
    m_pid = tl.program_id(0)
    tl.assume(inp1_row_stride > 0)
    tl.assume(inp1_col_stride > 0)
    tl.assume(out1_fp8_row_stride > 0)
    tl.assume(out1_fp8_col_stride > 0)
    tl.assume(out1_bs_row_stride > 0)
    tl.assume(out1_bs_col_stride > 0)
    n_offs = tl.arange(0, BLOCK_SIZE_N)
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE
    mask1 = n_offs < inp1_n_cols
    inp1 = tl.load(
        inp1_ptr + m_pid * inp1_row_stride + n_offs * inp1_col_stride,
        mask=mask1, other=0.0, cache_modifier=".cg",
    ).to(tl.float32)
    if FIRST_INPUT_RES:
        res1 = tl.load(
            res1_ptr + m_pid * res1_row_stride + n_offs * res1_col_stride,
            mask=mask1, other=0.0, cache_modifier=".cg",
        ).to(tl.float32)
        inp1 = inp1 + res1
    w1 = tl.load(weight1_ptr + n_offs, mask=mask1, other=0.0).to(tl.float32)
    norm1 = _rmsmorm_op(inp1, w1, inp1_n_cols, eps1)
    if FIRST_INPUT_OUT:
        tl.store(out1_ptr + m_pid * out1_row_stride + n_offs * out1_col_stride, norm1, mask=mask1)
    out1_fp8, out1_block_scales = _fp8_quant_op(norm1, 1, BLOCK_SIZE_N, QUANT_BLOCK_SIZE, DTYPE_MAX, DTYPE_MIN)
    out1_fp8 = tl.ravel(out1_fp8)
    out1_block_scales = tl.ravel(out1_block_scales)
    tl.store(
        out1_fp8_ptr + m_pid * out1_fp8_row_stride + n_offs * out1_fp8_col_stride,
        out1_fp8.to(out1_fp8_ptr.dtype.element_ty), mask=mask1,
    )
    g_offs = tl.arange(0, NUM_QUANT_BLOCKS)
    num_bs_cols = (inp1_n_cols + QUANT_BLOCK_SIZE - 1) // QUANT_BLOCK_SIZE
    tl.store(
        out1_bs_ptr + m_pid * out1_bs_row_stride + g_offs * out1_bs_col_stride,
        out1_block_scales.to(out1_bs_ptr.dtype.element_ty), mask=g_offs < num_bs_cols,
    )
    if HAVE_SECOND_INPUT:
        mask2 = n_offs < inp2_n_cols
        inp2 = tl.load(
            inp2_ptr + m_pid * inp2_row_stride + n_offs * inp2_col_stride,
            mask=mask2, other=0.0, cache_modifier=".cg",
        ).to(tl.float32)
        w2 = tl.load(weight2_ptr + n_offs, mask=mask2, other=0.0).to(tl.float32)
        norm2 = _rmsmorm_op(inp2, w2, inp2_n_cols, eps2)
        tl.store(out2_ptr + m_pid * out2_row_stride + n_offs * out2_col_stride, norm2, mask=mask2)
    if FIRST_INPUT_RES:
        inp1 = inp1.to(out_res1_ptr.dtype.element_ty)
        tl.store(
            out_res1_ptr + m_pid * out_res1_row_stride + n_offs * out_res1_col_stride,
            inp1, mask=mask1,
        )


@triton.jit
def _fused_flatten_fp8_group_quant_kernel(
    x_ptr, out_ptr, out_scales_ptr,
    x_stride_m, x_stride_n1, x_stride_n2,
    out_stride_m, out_stride_n,
    out_scales_stride_m, out_scales_stride_n,
    N2,
    BLOCK_SIZE_N2: tl.constexpr, QUANT_BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr, DTYPE_MIN: tl.constexpr,
):
    m = tl.program_id(0)
    n1 = tl.program_id(1)
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N2 // QUANT_BLOCK_SIZE
    n2_offs = tl.arange(0, BLOCK_SIZE_N2)
    x_offs = m * x_stride_m + n1 * x_stride_n1 + n2_offs * x_stride_n2
    x = tl.load(x_ptr + x_offs, mask=n2_offs < N2)
    out, out_block_scales = _fp8_quant_op(x, 1, BLOCK_SIZE_N2, QUANT_BLOCK_SIZE, DTYPE_MAX, DTYPE_MIN)
    out = tl.ravel(out)
    out_block_scales = tl.ravel(out_block_scales)
    tl.store(
        out_ptr + m * out_stride_m + (n1 * BLOCK_SIZE_N2 + n2_offs) * out_stride_n,
        out.to(out_ptr.dtype.element_ty), mask=n2_offs < N2,
    )
    block_scale_offs = tl.arange(0, NUM_QUANT_BLOCKS)
    tl.store(
        out_scales_ptr + m * out_scales_stride_m + (n1 * NUM_QUANT_BLOCKS + block_scale_offs) * out_scales_stride_n,
        out_block_scales.to(out_scales_ptr.dtype.element_ty),
        mask=block_scale_offs < tl.cdiv(N2, QUANT_BLOCK_SIZE),
    )


@triton.jit
def _fused_reduce_act_mul_fp8_group_quant(
    x_ptr, y_ptr, y_scale_ptr, x2_ptr, y2_ptr,
    M, N1, N2,
    stride_x_spk, stride_x_m, stride_x_n,
    stride_y_m, stride_y_n, stride_y_scale_m, stride_y_scale_n,
    stride_x2_spk, stride_x2_m, stride_x2_n, stride_y2_m, stride_y2_n,
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE_M2: tl.constexpr, BLOCK_SIZE_N1: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr, QUANT_BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr, DTYPE_MIN: tl.constexpr,
    X_HAS_SPLITK: tl.constexpr, X_NUM_KSPLIT: tl.constexpr,
    X_NUM_KSPLIT_POW2: tl.constexpr, X_MASK: tl.constexpr,
):
    tl.assume(stride_x_spk > 0)
    tl.assume(stride_x_m > 0)
    tl.assume(stride_x_n > 0)
    tl.assume(stride_y_m > 0)
    tl.assume(stride_y_n > 0)
    tl.assume(stride_y_scale_m > 0)
    tl.assume(stride_y_scale_n > 0)
    tl.assume(stride_x2_spk > 0)
    tl.assume(stride_x2_m > 0)
    tl.assume(stride_x2_n > 0)
    tl.assume(stride_y2_m > 0)
    tl.assume(stride_y2_n > 0)

    m_pid = tl.program_id(axis=0)
    if X_HAS_SPLITK and m_pid >= M:
        pid2 = m_pid - M
        num_pid_n2 = tl.cdiv(N2, BLOCK_SIZE_N2)
        pid_m2 = pid2 // num_pid_n2
        pid_n2 = pid2 % num_pid_n2
        offs_m2 = (pid_m2 * BLOCK_SIZE_M2 + tl.arange(0, BLOCK_SIZE_M2)) % M
        offs_n2 = (pid_n2 * BLOCK_SIZE_N2 + tl.arange(0, BLOCK_SIZE_N2)) % N2
        offs_spk = tl.arange(0, X_NUM_KSPLIT_POW2)
        x2_ptrs = (
            x2_ptr + offs_spk[:, None, None] * stride_x2_spk
            + offs_m2[None, :, None] * stride_x2_m + offs_n2[None, None, :] * stride_x2_n
        )
        if X_NUM_KSPLIT_POW2 == X_NUM_KSPLIT:
            x2 = tl.load(x2_ptrs)
        else:
            x2 = tl.load(x2_ptrs, mask=offs_spk[:, None, None] < X_NUM_KSPLIT, other=0.0)
        x2 = tl.sum(x2, axis=0)
        x2 = x2.to(y2_ptr.type.element_ty)
        y2_out_ptrs = y2_ptr + (offs_m2[:, None] * stride_y2_m) + (offs_n2[None, :] * stride_y2_n)
        tl.store(y2_out_ptrs, x2)
        return

    n_offs = tl.arange(0, BLOCK_SIZE_N1)
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N1 // QUANT_BLOCK_SIZE
    mask = None
    other = None
    if X_HAS_SPLITK:
        offs_spk = tl.arange(0, X_NUM_KSPLIT_POW2)
        x_ptrs = x_ptr + offs_spk[:, None] * stride_x_spk + m_pid * stride_x_m + n_offs[None, :] * stride_x_n
        if X_MASK:
            mask = (offs_spk[:, None] < X_NUM_KSPLIT) & (n_offs[None, :] < N1)
            other = 0.0
        else:
            mask = offs_spk[:, None] < X_NUM_KSPLIT
            other = 0.0
    else:
        x_ptrs = x_ptr + m_pid * stride_x_m + n_offs * stride_x_n
        if X_MASK:
            mask = n_offs < N1
            other = 0.0
    x = tl.load(x_ptrs, mask=mask, other=other, cache_modifier=".cg").to(tl.float32)
    x_mul = tl.load(x_ptrs + N1 * stride_x_n, mask=mask, other=other, cache_modifier=".cg").to(tl.float32)
    if X_HAS_SPLITK:
        x = tl.sum(x, axis=0)
        x_mul = tl.sum(x_mul, axis=0)
    x = ACTIVATION(x) * x_mul
    y, y_scale = _fp8_quant_op(x, 1, BLOCK_SIZE_N1, QUANT_BLOCK_SIZE, DTYPE_MAX, DTYPE_MIN)
    y = tl.ravel(y)
    y_scale = tl.ravel(y_scale)
    if X_MASK:
        mask = n_offs < N1
    else:
        mask = n_offs < N1
    tl.store(y_ptr + m_pid * stride_y_m + n_offs * stride_y_n, y.to(y_ptr.dtype.element_ty), mask=mask)
    g_offs = tl.arange(0, NUM_QUANT_BLOCKS)
    num_bs_cols = (N1 + QUANT_BLOCK_SIZE - 1) // QUANT_BLOCK_SIZE
    tl.store(
        y_scale_ptr + m_pid * stride_y_scale_m + g_offs * stride_y_scale_n,
        y_scale.to(y_scale_ptr.dtype.element_ty), mask=g_offs < num_bs_cols,
    )


@triton.jit
def _fused_reduce_rms_fp8_group_quant_kernel(
    inp1_ptr, weight1_ptr, inp2_ptr, weight2_ptr, inp3_ptr, res1_ptr,
    out1_fp8_ptr, out1_bs_ptr, out2_ptr, out_res1_ptr, out1_ptr, out3_ptr,
    eps1, eps2, n_rows, inp1_n_cols, inp2_n_cols, inp3_n_cols,
    inp1_spk_stride, inp2_spk_stride, inp3_spk_stride,
    inp1_row_stride, inp2_row_stride, inp3_row_stride,
    inp1_col_stride, inp2_col_stride, inp3_col_stride,
    res1_row_stride, res1_col_stride,
    out1_fp8_row_stride, out1_fp8_col_stride,
    out1_bs_row_stride, out1_bs_col_stride,
    out2_row_stride, out2_col_stride,
    out_res1_row_stride, out_res1_col_stride,
    out1_row_stride, out1_col_stride,
    out3_row_stride, out3_col_stride,
    BLOCK_SIZE_N1: tl.constexpr, BLOCK_SIZE_N2: tl.constexpr,
    BLOCK_SIZE_N3: tl.constexpr,
    N_MASK1: tl.constexpr, N_MASK2: tl.constexpr, N_MASK3: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr, DTYPE_MAX: tl.constexpr, DTYPE_MIN: tl.constexpr,
    HAVE_SECOND_INPUT: tl.constexpr, FIRST_INPUT_RES: tl.constexpr,
    FIRST_INPUT_OUT: tl.constexpr,
    HAS_SPLITK: tl.constexpr, NUM_SPLITK: tl.constexpr,
    NUM_SPLITK_POW2: tl.constexpr,
):
    m_pid = tl.program_id(0)
    if m_pid < n_rows:
        n1_offs = tl.arange(0, BLOCK_SIZE_N1)
        NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N1 // QUANT_BLOCK_SIZE
        if N_MASK1:
            mask1 = n1_offs < inp1_n_cols
            other1 = 0.0
        else:
            mask1 = None
            other1 = None
        if HAS_SPLITK:
            spk_offs = tl.arange(0, NUM_SPLITK_POW2)
            if NUM_SPLITK_POW2 != NUM_SPLITK:
                if N_MASK1:
                    mask1_in = (spk_offs[:, None] < NUM_SPLITK) & (n1_offs[None, :] < inp1_n_cols)
                else:
                    mask1_in = spk_offs[:, None] < NUM_SPLITK
                other1_in = 0.0
            else:
                if N_MASK1:
                    mask1_in = mask1[None, :]
                else:
                    mask1_in = mask1
                other1_in = other1
            inp1 = tl.load(
                inp1_ptr + spk_offs[:, None] * inp1_spk_stride + m_pid * inp1_row_stride + n1_offs[None, :] * inp1_col_stride,
                mask=mask1_in, other=other1_in, cache_modifier=".cg",
            ).to(tl.float32)
            inp1 = tl.sum(inp1, axis=0)
        else:
            inp1 = tl.load(
                inp1_ptr + m_pid * inp1_row_stride + n1_offs * inp1_col_stride,
                mask=mask1, other=other1, cache_modifier=".cg",
            ).to(tl.float32)
        if FIRST_INPUT_RES:
            res1 = tl.load(
                res1_ptr + m_pid * res1_row_stride + n1_offs * res1_col_stride,
                mask=mask1, other=other1, cache_modifier=".cg",
            ).to(tl.float32)
            inp1 = inp1 + res1
        w1 = tl.load(weight1_ptr + n1_offs, mask=mask1, other=other1).to(tl.float32)
        norm1 = _rmsmorm_op(inp1, w1, inp1_n_cols, eps1)
        if FIRST_INPUT_OUT:
            tl.store(out1_ptr + m_pid * out1_row_stride + n1_offs * out1_col_stride, norm1, mask=mask1)
        out1_fp8, out1_block_scales = _fp8_quant_op(norm1, 1, BLOCK_SIZE_N1, QUANT_BLOCK_SIZE, DTYPE_MAX, DTYPE_MIN)
        out1_fp8 = tl.ravel(out1_fp8)
        out1_block_scales = tl.ravel(out1_block_scales)
        tl.store(
            out1_fp8_ptr + m_pid * out1_fp8_row_stride + n1_offs * out1_fp8_col_stride,
            out1_fp8.to(out1_fp8_ptr.dtype.element_ty), mask=mask1,
        )
        g_offs = tl.arange(0, NUM_QUANT_BLOCKS)
        num_bs_cols = (inp1_n_cols + QUANT_BLOCK_SIZE - 1) // QUANT_BLOCK_SIZE
        tl.store(
            out1_bs_ptr + m_pid * out1_bs_row_stride + g_offs * out1_bs_col_stride,
            out1_block_scales.to(out1_bs_ptr.dtype.element_ty), mask=g_offs < num_bs_cols,
        )
        if FIRST_INPUT_RES:
            inp1 = inp1.to(out_res1_ptr.dtype.element_ty)
            tl.store(
                out_res1_ptr + m_pid * out_res1_row_stride + n1_offs * out_res1_col_stride,
                inp1, mask=mask1,
            )
    elif m_pid < 2 * n_rows:
        m_pid -= n_rows
        if HAS_SPLITK:
            spk_offs = tl.arange(0, NUM_SPLITK_POW2)
        if HAVE_SECOND_INPUT:
            n2_offs = tl.arange(0, BLOCK_SIZE_N2)
            if N_MASK2:
                mask2 = n2_offs < inp1_n_cols
                other2 = 0.0
            else:
                mask2 = None
                other2 = None
            if HAS_SPLITK:
                if NUM_SPLITK_POW2 != NUM_SPLITK:
                    if N_MASK2:
                        mask2_in = (spk_offs[:, None] < NUM_SPLITK) & (n2_offs[None, :] < inp2_n_cols)
                    else:
                        mask2_in = spk_offs[:, None] < NUM_SPLITK
                    other2_in = 0.0
                else:
                    if N_MASK2:
                        mask2_in = mask2[None, :]
                    else:
                        mask2_in = mask2
                    other2_in = other2
                inp2 = tl.load(
                    inp2_ptr + spk_offs[:, None] * inp2_spk_stride + m_pid * inp2_row_stride + n2_offs[None, :] * inp2_col_stride,
                    mask=mask2_in, other=other2_in, cache_modifier=".cg",
                ).to(tl.float32)
                inp2 = tl.sum(inp2, axis=0)
            else:
                inp2 = tl.load(
                    inp2_ptr + m_pid * inp2_row_stride + n2_offs * inp2_col_stride,
                    mask=mask2, other=other2, cache_modifier=".cg",
                ).to(tl.float32)
            w2 = tl.load(weight2_ptr + n2_offs, mask=mask2, other=other2).to(tl.float32)
            norm2 = _rmsmorm_op(inp2, w2, inp2_n_cols, eps2)
            tl.store(out2_ptr + m_pid * out2_row_stride + n2_offs * out2_col_stride, norm2, mask=mask2)
    elif m_pid < 3 * n_rows:
        m_pid -= 2 * n_rows
        if HAS_SPLITK:
            spk_offs = tl.arange(0, NUM_SPLITK_POW2)
            n3_offs = tl.arange(0, BLOCK_SIZE_N3)
            if N_MASK3:
                mask3 = n3_offs < inp3_n_cols
                other3 = 0.0
            else:
                mask3 = None
                other3 = None
            if NUM_SPLITK_POW2 != NUM_SPLITK:
                if N_MASK3:
                    mask3_in = (spk_offs[:, None] < NUM_SPLITK) & (n3_offs[None, :] < inp3_n_cols)
                else:
                    mask3_in = spk_offs[:, None] < NUM_SPLITK
                other3_in = 0.0
            else:
                if N_MASK3:
                    mask3_in = mask3[None, :]
                else:
                    mask3_in = mask3
                other3_in = other3
            inp3 = tl.load(
                inp3_ptr + spk_offs[:, None] * inp3_spk_stride + m_pid * inp3_row_stride + n3_offs[None, :] * inp3_col_stride,
                mask=mask3_in, other=other3_in, cache_modifier=".cg",
            ).to(tl.float32)
            inp3 = tl.sum(inp3, axis=0)
            tl.store(out3_ptr + m_pid * out3_row_stride + n3_offs * out3_col_stride, inp3, mask=mask3)


@triton.jit
def _fused_silu_mul_fp8_per_tensor_static_quant_kernel(
    inp_ptr, out_fp8_ptr, scale_ptr,
    n_rows, n_cols, row_stride, col_stride,
    out_fp8_row_stride, out_fp8_col_stride,
    BLOCK_SIZE_N: tl.constexpr, DTYPE_MAX: tl.constexpr, DTYPE_MIN: tl.constexpr,
    SILU_CONVERT_TO_INP_TYPE: tl.constexpr,
):
    m_pid = tl.program_id(0)
    n_offs = tl.arange(0, BLOCK_SIZE_N)
    first_half_ptrs = inp_ptr + m_pid * row_stride + n_offs * col_stride
    second_half_ptrs = inp_ptr + m_pid * row_stride + (n_cols + n_offs) * col_stride
    mask = n_offs < n_cols
    a = tl.load(first_half_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
    b = tl.load(second_half_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
    silu_a = fast_dividef(a, (1 + fast_expf(-a)))
    silu_o = silu_a * b
    if SILU_CONVERT_TO_INP_TYPE:
        silu_o = silu_o.to(inp_ptr.dtype.element_ty)
        silu_o = silu_o.to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    scale_recip = 1.0 / scale
    quant_fp8_out = tl.clamp(silu_o * scale_recip, DTYPE_MIN, DTYPE_MAX)
    tl.store(
        out_fp8_ptr + m_pid * out_fp8_row_stride + n_offs * out_fp8_col_stride,
        quant_fp8_out.to(out_fp8_ptr.dtype.element_ty), mask=mask,
    )


# ======
# PYTHON WRAPPERS (all 6 variants)
# ======


def fused_rms_fp8_per_tensor_static_quant(
    inp1, inp1_weight, inp1_epsilon, inp1_scale,
    inp2=None, inp2_weight=None, inp2_epsilon=None,
    dtype_quant=fp8_dtype, res1=None, output_unquantized_inp1=False,
    rmsnorm_convert_to_inp1_type=False,
):
    M, N1 = inp1.shape
    BLOCK_SIZE_N = triton.next_power_of_2(N1)
    N2 = 0
    if inp2 is not None:
        M2, N2 = inp2.shape
        BLOCK_SIZE_N = triton.next_power_of_2(N2)
        assert M == M2
    out1_fp8 = torch.empty((M, N1), dtype=dtype_quant, device=inp1.device)
    out2, out2_row_stride, out2_col_stride = None, 0, 0
    inp2_row_stride, inp2_col_stride = 0, 0
    if inp2 is not None:
        out2 = torch.empty((M, N2), dtype=inp1.dtype, device=inp1.device)
        inp2_row_stride, inp2_col_stride = inp2.stride(0), inp2.stride(1)
        out2_row_stride, out2_col_stride = out2.stride(0), out2.stride(1)
    out1, out1_row_stride, out1_col_stride = None, 0, 0
    if output_unquantized_inp1:
        out1 = torch.empty((M, N1), dtype=inp1.dtype, device=inp1.device)
        out1_row_stride, out1_col_stride = out1.stride(0), out1.stride(1)
    out_res1, res1_row_stride, res1_col_stride = None, 0, 0
    out_res1_row_stride, out_res1_col_stride = 0, 0
    if res1 is not None:
        out_res1 = torch.empty((M, N1), dtype=inp1.dtype, device=inp1.device)
        res1_row_stride, res1_col_stride = res1.stride(0), res1.stride(1)
        out_res1_row_stride, out_res1_col_stride = out_res1.stride(0), out_res1.stride(1)
    if BLOCK_SIZE_N <= 64:
        num_warps = 1
    elif BLOCK_SIZE_N <= 256:
        num_warps = 2
    elif BLOCK_SIZE_N <= 1024:
        num_warps = 4
    elif BLOCK_SIZE_N <= 4096:
        num_warps = 8
    else:
        num_warps = 16
    DTYPE_MAX = torch.finfo(out1_fp8.dtype).max if torch.is_floating_point(out1_fp8) else torch.iinfo(out1_fp8.dtype).max
    _fused_rms_fp8_per_tensor_static_quant_kernel[(M,)](
        inp1, inp1_weight, inp2, inp2_weight, res1,
        out1_fp8, out2, out_res1, out1, inp1_scale,
        inp1_epsilon, inp2_epsilon, M, N1, N2,
        inp1.stride(0), inp2_row_stride, inp1.stride(1), inp2_col_stride,
        res1_row_stride, res1_col_stride,
        out1_fp8.stride(0), out1_fp8.stride(1),
        out2_row_stride, out2_col_stride,
        out_res1_row_stride, out_res1_col_stride,
        out1_row_stride, out1_col_stride,
        BLOCK_SIZE_N=BLOCK_SIZE_N, DTYPE_MAX=DTYPE_MAX, DTYPE_MIN=-DTYPE_MAX,
        HAVE_SECOND_INPUT=(inp2 is not None), FIRST_INPUT_RES=(res1 is not None),
        FIRST_INPUT_OUT=output_unquantized_inp1,
        RMSNORM_CONVERT_TO_INP1_TYPE=rmsnorm_convert_to_inp1_type,
        num_warps=num_warps,
    )
    return out1_fp8, out1, out2, out_res1


def fused_rms_fp8_group_quant(
    inp1, inp1_weight, inp1_epsilon,
    inp2=None, inp2_weight=None, inp2_epsilon=None,
    group_size=128, dtype_quant=fp8_dtype, res1=None,
    output_unquantized_inp1=False, transpose_scale=False,
):
    M, N1 = inp1.shape
    BLOCK_SIZE_N = max(triton.next_power_of_2(N1), group_size)
    N2 = 0
    if inp2 is not None:
        M2, N2 = inp2.shape
        BLOCK_SIZE_N = max(triton.next_power_of_2(N2), BLOCK_SIZE_N)
        assert M == M2
    out1_fp8 = torch.empty((M, N1), dtype=dtype_quant, device=inp1.device)
    num_bs_cols = (N1 + group_size - 1) // group_size
    if transpose_scale:
        out1_bs = torch.empty((num_bs_cols, M), dtype=torch.float32, device=inp1.device)
    else:
        out1_bs = torch.empty((M, num_bs_cols), dtype=torch.float32, device=inp1.device)
    out2, out2_row_stride, out2_col_stride = None, 0, 0
    inp2_row_stride, inp2_col_stride = 0, 0
    if inp2 is not None:
        out2 = torch.empty((M, N2), dtype=inp1.dtype, device=inp1.device)
        inp2_row_stride, inp2_col_stride = inp2.stride(0), inp2.stride(1)
        out2_row_stride, out2_col_stride = out2.stride(0), out2.stride(1)
    out1, out1_row_stride, out1_col_stride = None, 0, 0
    if output_unquantized_inp1:
        out1 = torch.empty((M, N1), dtype=inp1.dtype, device=inp1.device)
        out1_row_stride, out1_col_stride = out1.stride(0), out1.stride(1)
    BLOCK_SIZE_N = max(BLOCK_SIZE_N, group_size)
    out_res1, res1_row_stride, res1_col_stride = None, 0, 0
    out_res1_row_stride, out_res1_col_stride = 0, 0
    if res1 is not None:
        out_res1 = torch.empty((M, N1), dtype=inp1.dtype, device=inp1.device)
        res1_row_stride, res1_col_stride = res1.stride(0), res1.stride(1)
        out_res1_row_stride, out_res1_col_stride = out_res1.stride(0), out_res1.stride(1)
    # Better num_warps tuning based on block size
    if BLOCK_SIZE_N <= 64:
        num_warps = 1
    elif BLOCK_SIZE_N <= 256:
        num_warps = 2
    elif BLOCK_SIZE_N <= 1024:
        num_warps = 4
    elif BLOCK_SIZE_N <= 4096:
        num_warps = 8
    else:
        num_warps = 16
    DTYPE_MAX = torch.finfo(out1_fp8.dtype).max if torch.is_floating_point(out1_fp8) else torch.iinfo(out1_fp8.dtype).max
    if transpose_scale:
        out1_bs_row_stride, out1_bs_col_stride = out1_bs.stride(1), out1_bs.stride(0)
    else:
        out1_bs_row_stride, out1_bs_col_stride = out1_bs.stride(0), out1_bs.stride(1)
    _fused_rms_fp8_group_quant_kernel[(M,)](
        inp1, inp1_weight, inp2, inp2_weight, res1,
        out1_fp8, out1_bs, out2, out_res1, out1,
        inp1_epsilon, inp2_epsilon, M, N1, N2,
        inp1.stride(0), inp2_row_stride, inp1.stride(1), inp2_col_stride,
        res1_row_stride, res1_col_stride,
        out1_fp8.stride(0), out1_fp8.stride(1),
        out1_bs_row_stride, out1_bs_col_stride,
        out2_row_stride, out2_col_stride,
        out_res1_row_stride, out_res1_col_stride,
        out1_row_stride, out1_col_stride,
        BLOCK_SIZE_N=BLOCK_SIZE_N, QUANT_BLOCK_SIZE=group_size,
        DTYPE_MAX=DTYPE_MAX, DTYPE_MIN=-DTYPE_MAX,
        HAVE_SECOND_INPUT=(inp2 is not None), FIRST_INPUT_RES=(res1 is not None),
        FIRST_INPUT_OUT=output_unquantized_inp1,
        num_warps=num_warps,
        num_stages=2,
    )
    if transpose_scale:
        out1_bs = out1_bs.view(M, num_bs_cols)
    return (out1_fp8, out1_bs), out1, out2, out_res1


def fused_flatten_fp8_group_quant(x, group_size, dtype_quant=fp8_dtype):
    M, N1, N2 = x.shape
    BLOCK_SIZE_N2 = max(triton.next_power_of_2(N2), group_size)
    N = N1 * N2
    out = torch.empty((M, N), dtype=dtype_quant, device=x.device)
    out_block_scales = torch.empty((M, triton.cdiv(N, group_size)), dtype=torch.float32, device=x.device)
    DTYPE_MAX = torch.finfo(out.dtype).max if torch.is_floating_point(out) else torch.iinfo(out.dtype).max
    _fused_flatten_fp8_group_quant_kernel[(M, N1)](
        x, out, out_block_scales, *x.stride(), *out.stride(), *out_block_scales.stride(), N2,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2, QUANT_BLOCK_SIZE=group_size,
        DTYPE_MAX=DTYPE_MAX, DTYPE_MIN=-DTYPE_MAX,
    )
    return out, out_block_scales


def fused_reduce_act_mul_fp8_group_quant(
    x, activation="silu", x2=None, group_size=128,
    dtype_quant=fp8_dtype, dtype=torch.bfloat16,
):
    assert x.dim() == 2 or x.dim() == 3
    X_HAS_SPLITK = False
    x_num_splitk, N2, y2 = 1, 1, None
    if x.dim() == 3:
        x_num_splitk, M, N1 = x.shape
        x_num_splitk, _, N2 = x2.shape
        X_HAS_SPLITK = True
        y2 = torch.empty((M, N2), dtype=dtype, device=x2.device)
    else:
        M, N1 = x.shape
    assert N1 % 2 == 0
    N1 = N1 // 2
    y = torch.empty((M, N1), dtype=dtype_quant, device=x.device)
    y_scale = torch.empty((M, (N1 + group_size - 1) // group_size), dtype=torch.float32, device=x.device)
    BLOCK_SIZE_N1 = max(triton.next_power_of_2(N1), group_size)
    BLOCK_SIZE_N2 = max(triton.next_power_of_2(N2), 32)
    BLOCK_SIZE_M2 = 1 if M <= 128 else 4
    X_MASK = N1 % BLOCK_SIZE_N1 != 0
    DTYPE_MAX = torch.finfo(y.dtype).max if torch.is_floating_point(y) else torch.iinfo(y.dtype).max
    num_pid = M
    if X_HAS_SPLITK:
        num_pid += triton.cdiv(M, BLOCK_SIZE_M2) * triton.cdiv(N2, BLOCK_SIZE_N2)
    _fused_reduce_act_mul_fp8_group_quant[(num_pid,)](
        x, y, y_scale, x2, y2, M, N1, N2,
        0 if not X_HAS_SPLITK else x.stride(0),
        x.stride(0) if not X_HAS_SPLITK else x.stride(1),
        x.stride(1) if not X_HAS_SPLITK else x.stride(2),
        y.stride(0), y.stride(1), y_scale.stride(0), y_scale.stride(1),
        0 if not X_HAS_SPLITK else x2.stride(0),
        0 if not X_HAS_SPLITK else x2.stride(1),
        0 if not X_HAS_SPLITK else x2.stride(2),
        0 if not X_HAS_SPLITK else y2.stride(0),
        0 if not X_HAS_SPLITK else y2.stride(1),
        ACTIVATION=_get_activation_from_str(activation) if activation else "",
        BLOCK_SIZE_M2=BLOCK_SIZE_M2, BLOCK_SIZE_N1=BLOCK_SIZE_N1,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2, QUANT_BLOCK_SIZE=group_size,
        DTYPE_MAX=DTYPE_MAX, DTYPE_MIN=-DTYPE_MAX,
        X_HAS_SPLITK=X_HAS_SPLITK, X_NUM_KSPLIT=x_num_splitk,
        X_NUM_KSPLIT_POW2=triton.next_power_of_2(x_num_splitk), X_MASK=X_MASK,
        num_warps=1 if max(BLOCK_SIZE_N1, BLOCK_SIZE_N2) <= 512 else 4,
    )
    return (y, y_scale), y2


def fused_reduce_rms_fp8_group_quant(
    inp1, inp1_weight, inp1_epsilon,
    inp2=None, inp2_weight=None, inp2_epsilon=None, inp3=None,
    group_size=128, dtype_quant=fp8_dtype, dtype=None, res1=None,
    output_unquantized_inp1=False, out3=None, transpose_scale=False,
):
    out_dtype = dtype if dtype is not None else inp1.dtype
    SPK, HAS_SPLITK = 1, False
    inp1_spk_stride, inp1_row_stride, inp1_col_stride = 0, 0, 0
    if inp1.dim() == 3:
        SPK, M, N1 = inp1.shape
        assert SPK > 1
        HAS_SPLITK = True
        inp1_spk_stride, inp1_row_stride, inp1_col_stride = inp1.stride(0), inp1.stride(1), inp1.stride(2)
    else:
        M, N1 = inp1.shape
        inp1_row_stride, inp1_col_stride = inp1.stride(0), inp1.stride(1)
    BLOCK_SIZE_N1 = max(triton.next_power_of_2(N1), group_size)
    N2, N3, BLOCK_SIZE_N2, BLOCK_SIZE_N3 = 0, 0, 1, 1
    if inp2 is not None:
        N2 = inp2.shape[-1]
        BLOCK_SIZE_N2 = triton.next_power_of_2(N2)
    if inp3 is not None:
        N3 = inp3.shape[-1]
        BLOCK_SIZE_N3 = triton.next_power_of_2(N3)
    out1_fp8 = torch.empty((M, N1), dtype=dtype_quant, device=inp1.device)
    num_bs_cols = (N1 + group_size - 1) // group_size
    if transpose_scale:
        out1_bs = torch.empty((num_bs_cols, M), dtype=torch.float32, device=inp1.device)
    else:
        out1_bs = torch.empty((M, num_bs_cols), dtype=torch.float32, device=inp1.device)
    if transpose_scale:
        out1_bs_row_stride, out1_bs_col_stride = out1_bs.stride(1), out1_bs.stride(0)
    else:
        out1_bs_row_stride, out1_bs_col_stride = out1_bs.stride(0), out1_bs.stride(1)
    out2, inp2_spk_stride, out2_row_stride, out2_col_stride = None, 0, 0, 0
    inp2_row_stride, inp2_col_stride = 0, 0
    if inp2 is not None:
        out2 = torch.empty((M, N2), dtype=out_dtype, device=inp1.device)
        if SPK > 1:
            inp2_spk_stride, inp2_row_stride, inp2_col_stride = inp2.stride(0), inp2.stride(1), inp2.stride(2)
        else:
            inp2_row_stride, inp2_col_stride = inp2.stride(0), inp2.stride(1)
        out2_row_stride, out2_col_stride = out2.stride(0), out2.stride(1)
    inp3_spk_stride, out3_row_stride, out3_col_stride = 0, 0, 0
    inp3_row_stride, inp3_col_stride = 0, 0
    if inp3 is not None:
        if out3 is None:
            out3 = torch.empty((M, N3), dtype=out_dtype, device=inp1.device)
        inp3_spk_stride, inp3_row_stride, inp3_col_stride = inp3.stride(0), inp3.stride(1), inp3.stride(2)
        out3_row_stride, out3_col_stride = out3.stride(0), out3.stride(1)
    out1, out1_row_stride, out1_col_stride = None, 0, 0
    if output_unquantized_inp1:
        out1 = torch.empty((M, N1), dtype=out_dtype, device=inp1.device)
        out1_row_stride, out1_col_stride = out1.stride(0), out1.stride(1)
    out_res1, res1_row_stride, res1_col_stride = None, 0, 0
    out_res1_row_stride, out_res1_col_stride = 0, 0
    if res1 is not None:
        out_res1 = torch.empty((M, N1), dtype=out_dtype, device=inp1.device)
        res1_row_stride, res1_col_stride = res1.stride(0), res1.stride(1)
        out_res1_row_stride, out_res1_col_stride = out_res1.stride(0), out_res1.stride(1)
    max_BN = max(BLOCK_SIZE_N1, BLOCK_SIZE_N2, BLOCK_SIZE_N3)
    if max_BN <= 64:
        num_warps = 1
    elif max_BN <= 256:
        num_warps = 2
    elif max_BN <= 1024:
        num_warps = 4
    elif max_BN <= 4096:
        num_warps = 8
    else:
        num_warps = 16
    DTYPE_MAX = torch.finfo(out1_fp8.dtype).max if torch.is_floating_point(out1_fp8) else torch.iinfo(out1_fp8.dtype).max
    _fused_reduce_rms_fp8_group_quant_kernel[(3 * M if HAS_SPLITK else 2 * M,)](
        inp1, inp1_weight, inp2, inp2_weight, inp3, res1,
        out1_fp8, out1_bs, out2, out_res1, out1, out3,
        inp1_epsilon, inp2_epsilon, M, N1, N2, N3,
        inp1_spk_stride, inp2_spk_stride, inp3_spk_stride,
        inp1_row_stride, inp2_row_stride, inp3_row_stride,
        inp1_col_stride, inp2_col_stride, inp3_col_stride,
        res1_row_stride, res1_col_stride,
        out1_fp8.stride(0), out1_fp8.stride(1),
        out1_bs_row_stride, out1_bs_col_stride,
        out2_row_stride, out2_col_stride,
        out_res1_row_stride, out_res1_col_stride,
        out1_row_stride, out1_col_stride,
        out3_row_stride, out3_col_stride,
        BLOCK_SIZE_N1=BLOCK_SIZE_N1, BLOCK_SIZE_N2=BLOCK_SIZE_N2, BLOCK_SIZE_N3=BLOCK_SIZE_N3,
        N_MASK1=(BLOCK_SIZE_N1 != N1), N_MASK2=(BLOCK_SIZE_N2 != N2), N_MASK3=(BLOCK_SIZE_N3 != N3),
        QUANT_BLOCK_SIZE=group_size, DTYPE_MAX=DTYPE_MAX, DTYPE_MIN=-DTYPE_MAX,
        HAVE_SECOND_INPUT=(inp2 is not None), FIRST_INPUT_RES=(res1 is not None),
        FIRST_INPUT_OUT=output_unquantized_inp1,
        HAS_SPLITK=HAS_SPLITK, NUM_SPLITK=SPK, NUM_SPLITK_POW2=triton.next_power_of_2(SPK),
        num_warps=num_warps,
    )
    if transpose_scale:
        out1_bs = out1_bs.view(M, num_bs_cols)
    return (out1_fp8, out1_bs), out1, out2, out_res1, out3


def fused_silu_mul_fp8_per_tensor_static_quant(
    inp, inp_scale, dtype_quant=fp8_dtype, silu_convert_to_inp_type=False,
):
    M, N2 = inp.shape
    assert N2 % 2 == 0
    N = N2 // 2
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    out_fp8 = torch.empty((M, N), dtype=dtype_quant, device=inp.device)
    num_warps = 1 if BLOCK_SIZE_N <= 512 else (4 if BLOCK_SIZE_N <= 2048 else (8 if BLOCK_SIZE_N <= 4096 else 16))
    DTYPE_MAX = torch.finfo(out_fp8.dtype).max if torch.is_floating_point(out_fp8) else torch.iinfo(out_fp8.dtype).max
    _fused_silu_mul_fp8_per_tensor_static_quant_kernel[(M,)](
        inp, out_fp8, inp_scale, M, N,
        inp.stride(0), inp.stride(1), out_fp8.stride(0), out_fp8.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N, DTYPE_MAX=DTYPE_MAX, DTYPE_MIN=-DTYPE_MAX,
        SILU_CONVERT_TO_INP_TYPE=silu_convert_to_inp_type,
        num_warps=num_warps,
    )
    return out_fp8


##################################################################################################################################################

##################################################################################################################################################

# ======
# TEST CONFIGURATIONS
# ======

# (M, N1, N2) -- batch/tokens, hidden dimension 1, hidden dimension 2
ALL_SHAPES = [
    (1, 128, 128),
    (4, 128, 128),
    (1, 128, 4096),
    (8, 128, 128),
    (1, 128, 7168),
    (1, 4096, 4096),
    (1, 128, 8192),
    (1, 4096, 8192),
    (1, 7168, 7168),
    (1, 8192, 8192),
    (32, 128, 128),
    (4, 4096, 4096),
    (8, 4096, 4096),
    (16, 4096, 4096),
    (256, 128, 128),
    (32, 128, 7168),
    (1024, 128, 128),
    (256, 128, 7168),
    (256, 4096, 4096),
    (8192, 128, 128),
    (32, 7168, 7168),
    (256, 7168, 7168),
    (1024, 4096, 4096),
    (1024, 8192, 8192),
    (8192, 7168, 7168),
]

seen = set()
unique_shapes = []
for s in ALL_SHAPES:
    if s not in seen:
        seen.add(s)
        unique_shapes.append(s)
ALL_SHAPES = sorted(unique_shapes, key=lambda s: s[0] * (s[1] + s[2]))

# HARNESS_SHAPES: uniformly sample 25 shapes from ALL_SHAPES
_n_all = len(ALL_SHAPES)
if _n_all <= 25:
    HARNESS_SHAPES = ALL_SHAPES
else:
    _harness_indices = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_SHAPES = [ALL_SHAPES[i] for i in _harness_indices]

# PROFILE_SHAPES: exactly 5 shapes evenly spaced
_profile_indices = [int(round(i * (_n_all - 1) / 4)) for i in range(5)]
PROFILE_SHAPES = [ALL_SHAPES[i] for i in _profile_indices]

# For backward compatibility
EVAL_CONFIGS = HARNESS_SHAPES
PROFILE_CONFIGS = PROFILE_SHAPES

RTOL, ATOL = 0.1, 0.1


# ======
# REFERENCE IMPLEMENTATIONS
# ======


def rmsnorm(input, weight, eps=1e-6):
    row_norm = input * input
    row_norm = torch.sum(row_norm, dim=-1)
    norm_factor = torch.rsqrt((row_norm / input.shape[1]) + eps)
    rms_norm = input * norm_factor[:, None] * weight[None, :]
    return rms_norm


def per_token_fp8_group_quant(x, dtype_quant, group_size=128):
    import torch.nn.functional as F
    DTYPE_MAX = torch.finfo(dtype_quant).max
    M, N = x.shape
    if N % group_size > 0:
        num_pad = group_size - (N % group_size)
        x_reshape = F.pad(x, (0, num_pad, 0, 0), "constant", 0)
        x_reshape = x_reshape.reshape(
            M, (N + group_size - 1) // group_size, group_size
        ).to(torch.float32)
    else:
        x_reshape = x.reshape(M, N // group_size, group_size).to(torch.float32)
    x_max = torch.max(torch.abs(x_reshape), dim=-1, keepdim=True)[0]
    x_max = torch.where(x_max < 1e-10, 1e-10, x_max).to(torch.float32)
    x_scale = x_max / DTYPE_MAX
    scale_recip = 1.0 / x_scale
    x_quant = torch.clamp(x_reshape * scale_recip, -DTYPE_MAX, DTYPE_MAX).to(
        dtype_quant
    )
    x_quant = x_quant.reshape(M, (N + group_size - 1) // group_size * group_size)[:, :N]
    x_scale = x_scale.squeeze(-1)
    return x_quant, x_scale


def upcast(x, s, dtype, group_size=128):
    x_N = x.shape[1]
    x = x.reshape(-1, x_N // group_size, group_size).to(torch.float32) * s.reshape(
        -1, s.shape[1], 1
    )
    x = x.reshape(-1, x_N)
    return x.to(dtype=dtype)


def run_torch_rms_fp8_group_quant(
    x1, w1, eps1, x2, w2, eps2, res1, dtype_quant, group_size
):
    s = x1 + res1
    y1 = rmsnorm(s, w1, eps1)
    y2 = rmsnorm(x2, w2, eps2)
    y1_q, y1_s = per_token_fp8_group_quant(y1, dtype_quant, group_size)
    return (y1_q, y1_s), y1.to(x1.dtype), y2.to(x1.dtype), s.to(x1.dtype)


# ======
# INPUT GENERATION
# ======


def generate_inputs(M, N1, N2, dtype=torch.bfloat16):
    """Generate inputs on CPU then move to GPU."""
    torch.manual_seed(42)
    x1 = (torch.randn((M, N1), dtype=dtype, device="cpu") / 10).to("cuda")
    x2 = (torch.randn((M, N2), dtype=dtype, device="cpu") / 10).to("cuda")
    w1 = torch.ones((N1,), dtype=torch.float32, device="cpu").to("cuda")
    w2 = torch.ones((N2,), dtype=torch.float32, device="cpu").to("cuda")
    res1 = (torch.randn((M, N1), dtype=dtype, device="cpu") / 10).to("cuda")
    return x1, w1, x2, w2, res1


# ======
# TEST HARNESS
# ======


def run_correctness(shapes=None, verbose=True):
    if shapes is None:
        shapes = HARNESS_SHAPES
    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")

    group_size = 128
    dtype = torch.bfloat16
    results, failures = [], []

    for i, (M, N1, N2) in enumerate(shapes):
        try:
            x1, w1, x2, w2, res1 = generate_inputs(M, N1, N2, dtype)

            (y1_q_torch, y1_s_torch), y1_torch, y2_torch, y1_res_torch = \
                run_torch_rms_fp8_group_quant(
                    x1, w1, 1e-6, x2, w2, 1e-6, res1, fp8_dtype, group_size
                )

            (y1_q_triton, y1_s_triton), y1_triton, y2_triton, y1_res_triton = \
                fused_rms_fp8_group_quant(
                    x1, w1, 1e-6,
                    inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                    group_size=group_size,
                    dtype_quant=fp8_dtype,
                    res1=res1,
                    output_unquantized_inp1=True,
                )

            torch.testing.assert_close(y1_torch, y1_triton, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(y2_torch, y2_triton, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(y1_res_torch, y1_res_triton, atol=ATOL, rtol=RTOL)

            y1_upcast_torch = upcast(
                y1_q_torch, y1_s_torch, dtype=torch.float32, group_size=group_size
            )
            y1_upcast_triton = upcast(
                y1_q_triton, y1_s_triton, dtype=torch.float32, group_size=group_size
            )
            torch.testing.assert_close(y1_upcast_torch, y1_upcast_triton, atol=ATOL, rtol=RTOL)

            results.append({"config": (M, N1, N2), "correct": True})
            if verbose:
                print(f"  PASS: ({M}, {N1}, {N2})")

            del x1, x2, w1, w2, res1
            torch.cuda.empty_cache()
        except Exception as e:
            failures.append({"config": (M, N1, N2), "error": str(e)})
            if verbose:
                print(f"  FAIL: ({M}, {N1}, {N2}) - {str(e)[:50]}")

    if verbose:
        print("-" * 62)
        print(
            f"{'Status:':<22} {'ALL PASS' if not failures else f'FAILED ({len(failures)}/{len(shapes)})'}"
        )

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
    }


def run_profile(shapes=None, warmup=50, iters=200, verbose=True):
    if shapes is None:
        shapes = PROFILE_SHAPES
    group_size = 128
    dtype = torch.bfloat16

    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    for M, N1, N2 in shapes:
        x1, w1, x2, w2, res1 = generate_inputs(M, N1, N2, dtype)
        for _ in range(warmup):
            _ = fused_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                group_size=group_size,
                dtype_quant=fp8_dtype,
                res1=res1,
                output_unquantized_inp1=True,
            )
        torch.cuda.synchronize()
        for _ in range(iters):
            _ = fused_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                group_size=group_size,
                dtype_quant=fp8_dtype,
                res1=res1,
                output_unquantized_inp1=True,
            )
        torch.cuda.synchronize()
        if verbose:
            print(f"  ({M},{N1},{N2}) done")
        del x1, x2, w1, w2, res1
        torch.cuda.empty_cache()


def run_benchmark(shapes=None, warmup=50, iters=200, verbose=True):
    if shapes is None:
        shapes = HARNESS_SHAPES
    group_size = 128
    dtype = torch.bfloat16
    latencies = []
    speedups = []

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations each...")
    print(f"{'Config (M,N1,N2)':<22} {'PyTorch':>10} {'Triton':>10} {'Speedup':>10}")
    print("-" * 62)

    for M, N1, N2 in shapes:
        x1, w1, x2, w2, res1 = generate_inputs(M, N1, N2, dtype)

        for _ in range(warmup):
            _ = fused_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                group_size=group_size,
                dtype_quant=fp8_dtype,
                res1=res1,
                output_unquantized_inp1=True,
            )
        torch.cuda.synchronize()

        triton_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = fused_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                group_size=group_size,
                dtype_quant=fp8_dtype,
                res1=res1,
                output_unquantized_inp1=True,
            )
            end.record()
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end))

        triton_ms = sorted(triton_times)[len(triton_times) // 2]

        torch_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = run_torch_rms_fp8_group_quant(
                x1, w1, 1e-6, x2, w2, 1e-6, res1, fp8_dtype, group_size
            )
            end.record()
            torch.cuda.synchronize()
            torch_times.append(start.elapsed_time(end))

        torch_ms = sorted(torch_times)[len(torch_times) // 2]
        speedup = torch_ms / triton_ms if triton_ms > 0 else 1.0

        latencies.append(triton_ms)
        speedups.append(speedup)

        marker = " *" if speedup > 1.0 else ""
        if verbose:
            print(f"({M:>6}, {N1:>5}, {N2:>5}){' ':4} {torch_ms:>8.4f}ms {triton_ms:>8.4f}ms {speedup:>8.2f}x{marker}", flush=True)

    log_sum = sum(math.log(l) for l in latencies)
    geomean_latency = math.exp(log_sum / len(latencies))

    log_sum_speedup = sum(math.log(s) for s in speedups)
    geomean_speedup = math.exp(log_sum_speedup / len(speedups))

    print("-" * 62)
    print(f"{'Geometric mean latency:':<22} {geomean_latency:.4f} ms")
    print(f"{'Geometric mean speedup:':<22} {geomean_speedup:.2f}x")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)
    print(f"GEAK_RESULT_SPEEDUP={geomean_speedup:.2f}", flush=True)

    return {
        "geomean_latency_ms": geomean_latency,
        "geomean_speedup": geomean_speedup,
    }


# ======
# MAIN
# ======

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fused RMS + FP8 Kernel Test Harness")
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Run correctness tests on benchmark shapes",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Run minimal profiling workload"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark on HARNESS_SHAPES (25 uniformly sampled)",
    )
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Run benchmark on ALL_SHAPES (complete set)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of benchmark iterations (default: 200)",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("Fused RMSNorm + FP8 Quantization Kernel")
    print("=" * 62)

    if args.correctness:
        print("\n[Correctness Mode]")
        run_correctness(HARNESS_SHAPES)
    elif args.profile:
        print("\n[Profile Mode]")
        run_profile(PROFILE_SHAPES, warmup=args.warmup, iters=args.iterations)
    elif args.full_benchmark:
        print("\n[Full Benchmark Mode]")
        run_benchmark(ALL_SHAPES, warmup=args.warmup, iters=args.iterations)
    else:
        print("\n[Benchmark Mode]")
        run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iters=args.iterations)

    print("=" * 62)