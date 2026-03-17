import torch
import triton
import triton.language as tl
from maybe_triton_jit import maybe_triton_jit
from triton.language.extra import libdevice

# from .utils import is_hopper, is_ampere


def diff(x, y):
    assert x.shape == y.shape
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    diff_max = torch.max(torch.abs(x - y)).item()
    diff_sum = torch.sum(torch.abs(x - y)).item()
    return f"diff.max: {diff_max:.3f}, diff.avg: {100.0 * diff_sum / (torch.sum(torch.abs(x)).item() + 1e-10):.3f}%"


def naive_torch_swiglu(x, w_g, w_fc, b_g, b_fc):
    gate = torch.nn.functional.silu(torch.nn.functional.linear(x, w_g.T, b_g))
    fc = torch.nn.functional.linear(x, w_fc.T, b_fc)
    y = gate * fc
    return y


@triton.jit
def fast_sigmoid(x):
    return libdevice.fast_dividef(1.0, 1.0 + libdevice.fast_expf(-x))


@triton.jit
def fast_silu(x):
    dtype = x.type.element_ty
    x = x.to(tl.float32)
    return libdevice.fast_dividef(x, 1.0 + libdevice.fast_expf(-x)).to(dtype)


@triton.jit
def fast_silu_bwd(dy, x):
    dtype = x.type.element_ty
    dy = dy.to(tl.float32)
    x = x.to(tl.float32)
    sigmoid = fast_sigmoid(x)
    return (dy * sigmoid * (1 + x * (1 - sigmoid))).to(dtype)


@triton.jit
def atomic_store(out_ptr, acc, mask, LOCKS, SPLIT_K):
    LOCKS = LOCKS + tl.program_id(0)
    COUNT = LOCKS + tl.num_programs(0)
    while tl.atomic_cas(LOCKS, 0, 1) == 1:
        pass
    count = tl.load(COUNT)
    if count == 0:
        tl.store(out_ptr, acc, mask=mask)
    else:
        cur = tl.load(out_ptr, mask=mask, other=0.0)
        tl.store(out_ptr, acc + cur, mask=mask)
    tl.atomic_xchg(COUNT, (count + 1) % SPLIT_K)
    tl.atomic_xchg(LOCKS, 0)


def fwd_autotune_config():
    if 0:  # is_hopper():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=4,
            )
        ]
    elif 0:  # is_ampere():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=4,
            )
        ]
    else:
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=1,
                num_warps=4,
            )
        ]


def bwd_b_autotune_config():
    if 0:  # is_hopper():
        return [
            triton.Config(
                {"BLOCK_SIZE_M": 512, "BLOCK_SIZE_N": 16}, num_stages=3, num_warps=8
            )
        ]
    elif 0:  # is_ampere():
        return [
            triton.Config(
                {"BLOCK_SIZE_M": 512, "BLOCK_SIZE_N": 16}, num_stages=4, num_warps=8
            )
        ]
    else:
        return [
            triton.Config(
                {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 32}, num_stages=3, num_warps=8
            )
        ]


def bwd_x_autotune_config():
    if 0:  # is_hopper():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=4,
            )
        ]
    elif 0:  # is_ampere():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=4,
            )
        ]
    else:
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=2,
                num_warps=4,
            )
        ]


def bwd_w_autotune_config():
    if 0:  # is_hopper():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 64,
                    "SPLIT_K": 1,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=4,
            )
        ]
    elif 0:  # is_ampere():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 64,
                    "SPLIT_K": 1,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=4,
            )
        ]
    else:
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 64,
                    "SPLIT_K": 1,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=4,
            )
        ]


@maybe_triton_jit(
    configs=fwd_autotune_config(),
    key=["N", "K", "IS_TRAINING"],
)
@triton.jit
def fused_swiglu_fwd_kernel(
    x_ptr,
    w_g_ptr,
    w_fc_ptr,
    b_g_ptr,
    b_fc_ptr,
    y_ptr,
    g_ptr,
    fc_ptr,
    M,
    N,
    K,
    IS_TRAINING: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    dtype = y_ptr.type.element_ty
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if (pid_m * BLOCK_SIZE_M >= M) or (pid_n * BLOCK_SIZE_N >= N):
        return

    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offset_xm[:, None] * K + offset_k[None, :])
    w_g_ptrs = w_g_ptr + (offset_k[:, None] * N + offset_wn[None, :])
    w_fc_ptrs = w_fc_ptr + (offset_k[:, None] * N + offset_wn[None, :])
    b_g_ptrs = b_g_ptr + offset_wn
    b_fc_ptrs = b_fc_ptr + offset_wn
    b_g = tl.load(b_g_ptrs, mask=offset_wn < N, other=0.0)
    b_fc = tl.load(b_fc_ptrs, mask=offset_wn < N, other=0.0)

    accumulator_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator_fc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offset_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w_g = tl.load(
            w_g_ptrs, mask=offset_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
        )
        w_fc = tl.load(
            w_fc_ptrs, mask=offset_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
        )
        accumulator_g = tl.dot(x, w_g, accumulator_g)
        accumulator_fc = tl.dot(x, w_fc, accumulator_fc)
        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K
        w_g_ptrs += BLOCK_SIZE_K * N
        w_fc_ptrs += BLOCK_SIZE_K * N
    accumulator_g += b_g[None, :]
    accumulator_fc += b_fc[None, :]
    accumulator_g = accumulator_g.to(dtype)
    accumulator_fc = accumulator_fc.to(dtype)
    silu_g = fast_silu(accumulator_g)
    hadamard_product = silu_g.to(tl.float32) * accumulator_fc.to(tl.float32)
    y = hadamard_product.to(dtype)

    offset_ym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = y_ptr + N * offset_ym[:, None] + offset_yn[None, :]
    y_mask = (offset_ym[:, None] < M) & (offset_yn[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)
    accumulator_g = accumulator_g.to(dtype)
    accumulator_fc = accumulator_fc.to(dtype)
    if IS_TRAINING:
        g_ptrs = g_ptr + N * offset_ym[:, None] + offset_yn[None, :]
        fc_ptrs = fc_ptr + offset_ym[:, None] * N + offset_yn[None, :]
        tl.store(g_ptrs, accumulator_g, mask=y_mask)
        tl.store(fc_ptrs, accumulator_fc, mask=y_mask)


@maybe_triton_jit(
    configs=bwd_b_autotune_config(),
    key=["N"],
)
@triton.jit
def fused_swiglu_bwd_b_kernel(
    dy_ptr,
    g_ptr,
    fc_ptr,
    dg_ptr,
    dfc_ptr,
    db_g_ptr,
    db_fc_ptr,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    dtype = dy_ptr.type.element_ty
    col_idx = tl.program_id(axis=0)
    col_off = col_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_off = tl.arange(0, BLOCK_SIZE_M)
    dy_ptrs = dy_ptr + (row_off[None, :] * N + col_off[:, None])
    g_ptrs = g_ptr + (row_off[None, :] * N + col_off[:, None])
    fc_ptrs = fc_ptr + (row_off[None, :] * N + col_off[:, None])
    dg_ptrs = dg_ptr + (row_off[None, :] * N + col_off[:, None])
    dfc_ptrs = dfc_ptr + (row_off[None, :] * N + col_off[:, None])
    sum_b_g = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    sum_b_fc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    for row_idx in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        mask = (row_off[None, :] < M - row_idx * BLOCK_SIZE_M) & (col_off[:, None] < N)
        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptrs, mask=mask, other=0.0)
        fc = tl.load(fc_ptrs, mask=mask, other=0.0).to(tl.float32)
        silu_g = fast_silu(g)
        dg = (dy * fc).to(dtype)
        dg = fast_silu_bwd(dg, g)
        dfc = (dy * silu_g.to(tl.float32)).to(dtype)
        sum_b_g += dg.to(tl.float32)
        sum_b_fc += dfc.to(tl.float32)
        tl.store(dg_ptrs, dg, mask=mask)
        tl.store(dfc_ptrs, dfc, mask=mask)
        dy_ptrs += BLOCK_SIZE_M * N
        g_ptrs += BLOCK_SIZE_M * N
        fc_ptrs += BLOCK_SIZE_M * N
        dg_ptrs += BLOCK_SIZE_M * N
        dfc_ptrs += BLOCK_SIZE_M * N
    tl.store(db_g_ptr + col_off, tl.sum(sum_b_g, 1), mask=col_off < N)
    tl.store(db_fc_ptr + col_off, tl.sum(sum_b_fc, 1), mask=col_off < N)


@maybe_triton_jit(
    configs=bwd_x_autotune_config(),
    key=["N", "K"],
)
@triton.jit
def fused_swiglu_bwd_x_kernel(
    dg_ptr,
    dfc_ptr,
    w_g_ptr,
    w_fc_ptr,
    dx_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    dtype = dx_ptr.type.element_ty
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if (pid_m * BLOCK_SIZE_M >= M) or (pid_n * BLOCK_SIZE_N >= N):
        return

    offset_dym = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    dg_ptrs = dg_ptr + (offset_dym[:, None] * K + offset_k[None, :])
    dfc_ptrs = dfc_ptr + (offset_dym[:, None] * K + offset_k[None, :])
    w_g_ptrs = w_g_ptr + (offset_k[:, None] + offset_wn[None, :] * K)
    w_fc_ptrs = w_fc_ptr + (offset_k[:, None] + offset_wn[None, :] * K)

    accumulator_dx_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator_dx_fc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        dg = tl.load(dg_ptrs, mask=offset_k[None, :] < k_remaining, other=0.0)
        dfc = tl.load(dfc_ptrs, mask=offset_k[None, :] < k_remaining, other=0.0)
        w_g = tl.load(w_g_ptrs, mask=offset_k[:, None] < k_remaining, other=0.0)
        w_fc = tl.load(w_fc_ptrs, mask=offset_k[:, None] < k_remaining, other=0.0)
        # bwd x for gate
        accumulator_dx_g = tl.dot(dg, w_g, accumulator_dx_g)
        # bwd x for fc
        accumulator_dx_fc = tl.dot(dfc, w_fc, accumulator_dx_fc)
        # Advance the ptrs to the next K block.
        dg_ptrs += BLOCK_SIZE_K
        dfc_ptrs += BLOCK_SIZE_K
        w_g_ptrs += BLOCK_SIZE_K
        w_fc_ptrs += BLOCK_SIZE_K
    dx = (
        accumulator_dx_g.to(dtype).to(tl.float32)
        + accumulator_dx_fc.to(dtype).to(tl.float32)
    ).to(dtype)

    offset_dxm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_dxn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dx_ptrs = dx_ptr + offset_dxm[:, None] * N + offset_dxn[None, :]
    dx_mask = (offset_dxm[:, None] < M) & (offset_dxn[None, :] < N)
    tl.store(dx_ptrs, dx, mask=dx_mask)


@maybe_triton_jit(
    configs=bwd_w_autotune_config(),
    key=["M", "N"],
)
@triton.jit
def fused_swiglu_bwd_w_kernel(
    dg_ptr,
    dfc_ptr,
    x_ptr,
    dw_g_ptr,
    dw_fc_ptr,
    LOCK_G,
    LOCK_FC,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    dtype = dw_g_ptr.type.element_ty
    pid = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if ((pid_m * BLOCK_SIZE_M >= M) or (pid_n * BLOCK_SIZE_N >= N)) or (
        pid_k * BLOCK_SIZE_K >= K
    ):
        return

    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_dyn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offset_xm[:, None] + offset_k[None, :] * M)
    dg_ptrs = dg_ptr + (offset_k[:, None] * N + offset_dyn[None, :])
    dfc_ptrs = dfc_ptr + (offset_k[:, None] * N + offset_dyn[None, :])

    accumulator_dw_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator_dw_fc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_SIZE_K * SPLIT_K):
        x = tl.load(x_ptrs, mask=offset_k[None, :] < k, other=0.0)
        dg = tl.load(dg_ptrs, mask=offset_k[:, None] < k, other=0.0)
        dfc = tl.load(dfc_ptrs, mask=offset_k[:, None] < k, other=0.0)
        # bwd w_gate
        accumulator_dw_g = tl.dot(x, dg, accumulator_dw_g)
        # bwd w_fc
        accumulator_dw_fc = tl.dot(x, dfc, accumulator_dw_fc)
        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K * SPLIT_K * M
        dg_ptrs += BLOCK_SIZE_K * SPLIT_K * N
        dfc_ptrs += BLOCK_SIZE_K * SPLIT_K * N
    dw_g = accumulator_dw_g.to(dtype)
    dw_fc = accumulator_dw_fc.to(dtype)

    offset_dwm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_dwn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw_g_ptrs = dw_g_ptr + offset_dwm[:, None] * N + offset_dwn[None, :]
    dw_fc_ptrs = dw_fc_ptr + offset_dwm[:, None] * N + offset_dwn[None, :]
    dw_mask = (offset_dwm[:, None] < M) & (offset_dwn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(dw_g_ptrs, dw_g, mask=dw_mask)
        tl.store(dw_fc_ptrs, dw_fc, mask=dw_mask)
    else:
        atomic_store(dw_g_ptrs, dw_g, dw_mask, LOCK_G, SPLIT_K)
        atomic_store(dw_fc_ptrs, dw_fc, dw_mask, LOCK_FC, SPLIT_K)


# x: [total_len, dim]
# w_g: [in_dim, out_dim]
# w_fc: [in_dim, out_dim]
# b_g: [out_dim]
# b_fc: [out_dim]
# y: [total_len, out_dim]
class FusedSwiglu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w_g, w_fc, b_g, b_fc, is_training=True, is_recompute=False):
        # Check constraints.
        assert w_g.shape == w_fc.shape
        assert b_g.shape == b_fc.shape
        assert x.shape[1] == w_g.shape[0], "Incompatible dimensions"
        assert b_g.shape[0] == w_g.shape[1]
        assert (
            x.is_contiguous() and w_g.is_contiguous() and w_fc.is_contiguous()
        ), "Tensors must be contiguous"
        assert x.dtype == w_g.dtype == w_fc.dtype and x.dtype in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ]
        total_len, in_dim = x.shape
        out_dim = w_g.shape[1]
        # Allocates output.
        if is_recompute:
            g = x.new_empty(1)
            fc = x.new_empty(1)
        else:
            g = x.new_empty(total_len, out_dim)
            fc = x.new_empty(total_len, out_dim)
        y = x.new_empty(total_len, out_dim)
        grid = lambda META: (
            triton.cdiv(total_len, META["BLOCK_SIZE_M"])
            * triton.cdiv(out_dim, META["BLOCK_SIZE_N"]),
        )
        fused_swiglu_fwd_kernel[grid](
            x,
            w_g,
            w_fc,
            b_g,
            b_fc,
            y,
            g,
            fc,
            total_len,
            out_dim,
            in_dim,
            IS_TRAINING=is_training and not is_recompute,
        )

        ctx.is_recompute = is_recompute
        ctx.save_for_backward(x, w_g, w_fc, b_g, b_fc, g, fc)
        return y

    @staticmethod
    def backward(ctx, dy):
        device = dy.device
        is_recompute = ctx.is_recompute
        x, w_g, w_fc, b_g, b_fc, g, fc = ctx.saved_tensors
        total_len, out_dim = dy.shape
        in_dim, out_dim = w_g.shape
        dx = torch.empty_like(x)
        dw_g = torch.zeros_like(w_g)
        dw_fc = torch.zeros_like(w_fc)
        db_g = torch.zeros(out_dim, dtype=dy.dtype, device=device)
        db_fc = torch.zeros(out_dim, dtype=dy.dtype, device=device)
        dg = torch.empty_like(g)
        dfc = torch.empty_like(fc)
        if is_recompute:
            dg = dy.new_empty(total_len, out_dim)
            dfc = dy.new_empty(total_len, out_dim)
            # recompute g
            g = torch.nn.functional.linear(x, w_g.T, b_g)
            # recompute fc1
            fc = torch.nn.functional.linear(x, w_fc.T, b_fc)
        # dg & dfc & db_g & db_fc backward
        grid = lambda META: (triton.cdiv(out_dim, META["BLOCK_SIZE_N"]),)
        fused_swiglu_bwd_b_kernel[grid](
            dy,
            g,
            fc,
            dg,
            dfc,
            db_g,
            db_fc,
            total_len,
            out_dim,
        )
        # x backward
        # M: total_len, N: in_dim, K: out_dim (reduce_axis)
        grid = lambda META: (
            triton.cdiv(total_len, META["BLOCK_SIZE_M"])
            * triton.cdiv(in_dim, META["BLOCK_SIZE_N"]),
        )
        fused_swiglu_bwd_x_kernel[grid](
            dg,
            dfc,
            w_g,
            w_fc,
            dx,
            total_len,
            in_dim,
            out_dim,
        )
        # weight backward
        # M: in_dim, N: out_dim, K: total_len (reduce_axis)
        # allocate locks for split-k
        lock_g = torch.zeros(32 * 1024, dtype=torch.int32, device=device)
        lock_fc = torch.zeros(32 * 1024, dtype=torch.int32, device=device)
        grid = lambda META: (
            triton.cdiv(in_dim, META["BLOCK_SIZE_M"])
            * triton.cdiv(out_dim, META["BLOCK_SIZE_N"]),
            META["SPLIT_K"],
        )
        fused_swiglu_bwd_w_kernel[grid](
            dg,
            dfc,
            x,
            dw_g,
            dw_fc,
            lock_g,
            lock_fc,
            in_dim,
            out_dim,
            total_len,
        )

        return dx, dw_g, dw_fc, db_g, db_fc, None, None


if __name__ == "__main__":
    dtype = torch.bfloat16
    x = torch.randn((220000, 512), dtype=dtype, device="cuda").requires_grad_()
    w_g = torch.randn((512, 1024), dtype=dtype, device="cuda").requires_grad_()
    w_fc = torch.randn((512, 1024), dtype=dtype, device="cuda").requires_grad_()
    b_g = torch.randn((1024), dtype=dtype, device="cuda").requires_grad_()
    b_fc = torch.randn((1024), dtype=dtype, device="cuda").requires_grad_()
    dy = torch.randn((220000, 1024), dtype=dtype, device="cuda")
    y = FusedSwiglu.apply(x, w_g, w_fc, b_g, b_fc, True, False)
    y.backward(dy)
    triton_dx, x.grad = x.grad.clone(), None
    triton_dw_g, w_g.grad = w_g.grad.clone(), None
    triton_dw_fc, w_fc.grad = w_fc.grad.clone(), None
    triton_db_g, b_g.grad = b_g.grad.clone(), None
    triton_db_fc, b_fc.grad = b_fc.grad.clone(), None

    ref = naive_torch_swiglu(x, w_g, w_fc, b_g, b_fc)
    ref.backward(dy)
    torch_dx, x.grad = x.grad.clone(), None
    torch_dw_g, w_g.grad = w_g.grad.clone(), None
    torch_dw_fc, w_fc.grad = w_fc.grad.clone(), None
    torch_db_g, b_g.grad = b_g.grad.clone(), None
    torch_db_fc, b_fc.grad = b_fc.grad.clone(), None
    print(diff(ref, y))
    print(diff(torch_dx, triton_dx))
    print(diff(torch_dw_g, triton_dw_g))
    print(diff(torch_dw_fc, triton_dw_fc))
    print(diff(torch_db_g, triton_db_g))
    print(diff(torch_db_fc, triton_db_fc))
    atol = 1e-2
    rtol = 1e-2
    if torch.allclose(y, ref, atol=atol, rtol=rtol):
        print("✅ [Fwd]Triton and Torch match")
    else:
        print("❌ [Fwd]Triton and Torch differ")
    if torch.allclose(triton_dx, torch_dx, atol=atol, rtol=rtol):
        print("✅ [Bwd X]Triton and Torch match")
    else:
        print("❌ [Bwd X]Triton and Torch differ")
    if torch.allclose(triton_dw_g, torch_dw_g, atol=atol, rtol=rtol):
        print("✅ [Bwd WG]Triton and Torch match")
    else:
        print("❌ [Bwd WG]Triton and Torch differ")
    if torch.allclose(triton_dw_fc, torch_dw_fc, atol=atol, rtol=rtol):
        print("✅ [Bwd WFC]Triton and Torch match")
    else:
        print("❌ [Bwd WFC]Triton and Torch differ")
    if torch.allclose(triton_db_g, torch_db_g, atol=atol, rtol=rtol):
        print("✅ [Bwd BG]Triton and Torch match")
    else:
        print("❌ [Bwd BG]Triton and Torch differ")
    if torch.allclose(triton_db_fc, torch_db_fc, atol=atol, rtol=rtol):
        print("✅ [Bwd BFC]Triton and Torch match")
    else:
        print("❌ [Bwd BFC]Triton and Torch differ")
