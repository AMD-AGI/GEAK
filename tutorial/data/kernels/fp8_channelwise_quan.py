import triton
import torch
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
            },
            num_stages=8,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
            },
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
            },
            num_stages=8,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
            },
            num_stages=8,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
            },
            num_stages=8,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
            },
            num_stages=7,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
            },
            num_stages=8,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
            },
            num_stages=8,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
            },
            num_stages=8,
            num_warps=8,
        ),
    ],
    key=["M", "N"],
)
@triton.jit
def channel_granul_fp8_quant_kernel(
    x_ptr,
    y_ptr,
    scale_ptr,
    B,
    M,
    N,
    stride_x_b,
    stride_x_m,
    stride_x_n,
    stride_y_b,
    stride_y_m,
    stride_y_n,
    stride_s_b,
    stride_s_m,
    stride_s_n,
    fp8_max: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    REDUCE_ALONG_N: tl.constexpr,  # True: 规约 N（per-N）；False: 规约 M（per-M）
    X_ROW_MAJOR: tl.constexpr,  # True: x沿着N维度连续；False: x沿着M维度连续
    Y_ROW_MAJOR: tl.constexpr,  # True: y沿着N维度连续；False: y沿着M维度连续
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    x_base = x_ptr + pid * stride_x_b
    y_base = y_ptr + pid * stride_y_b

    if REDUCE_ALONG_N:
        m0 = bid * BLOCK_M
        amax_m = tl.zeros((BLOCK_M,), dtype=tl.float32)
        x_bp = tl.make_block_ptr(
            base=x_base,
            shape=(M, N),
            strides=(stride_x_m, stride_x_n),
            offsets=(m0, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if X_ROW_MAJOR else (0, 1),
        )
        x_bp1 = x_bp
        for _ in range(0, tl.cdiv(N, BLOCK_N)):
            x_tile = tl.load(x_bp1, boundary_check=(0, 1), padding_option="zero")
            x_abs = tl.abs(x_tile)
            amax_m = tl.maximum(amax_m, tl.max(x_abs, axis=1))
            x_bp1 = tl.advance(x_bp1, (0, BLOCK_N))
        safe_amax = tl.maximum(amax_m, EPS)
        scale_m = tl.div_rn(fp8_max, tl.cast(safe_amax, tl.float32))
        reciprocal_scale_m = tl.div_rn(tl.cast(safe_amax, tl.float32), fp8_max)
        scale_bp = tl.make_block_ptr(
            base=scale_ptr + pid * stride_s_b,
            shape=(M, 1),
            strides=(stride_s_m, stride_s_n),
            offsets=(m0, 0),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )
        tl.store(scale_bp, reciprocal_scale_m[:, None], boundary_check=(0,))
        x_bp2 = x_bp
        y_bp2 = tl.make_block_ptr(
            base=y_base,
            shape=(M, N),
            strides=(stride_y_m, stride_y_n),
            offsets=(m0, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if Y_ROW_MAJOR else (0, 1),
        )
        for _ in range(0, tl.cdiv(N, BLOCK_N)):
            x_tile = tl.load(x_bp2, boundary_check=(0, 1), padding_option="zero")
            y_tile = tl.cast(x_tile, tl.float32) * scale_m[:, None]  # 广播到 (BM, BN)
            y_tile = tl.clamp(y_tile, min=-fp8_max, max=fp8_max)
            y_fp8 = y_tile.to(y_ptr.dtype.element_ty)
            tl.store(y_bp2, y_fp8, boundary_check=(0, 1))
            x_bp2 = tl.advance(x_bp2, (0, BLOCK_N))
            y_bp2 = tl.advance(y_bp2, (0, BLOCK_N))

    else:
        n0 = bid * BLOCK_N
        amax_n = tl.zeros((BLOCK_N,), dtype=tl.float32)
        x_bp = tl.make_block_ptr(
            base=x_base,
            shape=(M, N),
            strides=(stride_x_m, stride_x_n),
            offsets=(0, n0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if X_ROW_MAJOR else (0, 1),
        )
        x_bp1 = x_bp
        for _ in range(0, tl.cdiv(M, BLOCK_M)):
            x_tile = tl.load(
                x_bp1, boundary_check=(0, 1), padding_option="zero"
            )  # 检查 M,N 维度的边界
            x_abs = tl.abs(x_tile)
            amax_n = tl.maximum(amax_n, tl.max(x_abs, axis=0))
            x_bp1 = tl.advance(x_bp1, (BLOCK_M, 0))
        safe_amax = tl.maximum(amax_n, EPS)
        scale_n = tl.div_rn(fp8_max, tl.cast(safe_amax, tl.float32))
        reciprocal_scale_n = tl.div_rn(tl.cast(safe_amax, tl.float32), fp8_max)
        # 把 scale 写到 keepdim (B, 1, N)：只在 m=0 行写入
        scale_bp = tl.make_block_ptr(
            base=scale_ptr + pid * stride_s_b,
            shape=(1, N),
            strides=(
                stride_s_m,
                stride_s_n,
            ),  # 注意stride_s_m 应对应 keepdim 的 m 维（这里为 1）
            offsets=(0, n0),
            block_shape=(1, BLOCK_N),
            order=(1, 0),
        )
        # 保存 reciprocal_scale
        tl.store(scale_bp, reciprocal_scale_n[None, :], boundary_check=(1,))
        x_bp2 = x_bp
        y_bp2 = tl.make_block_ptr(
            base=y_base,
            shape=(M, N),
            strides=(stride_y_m, stride_y_n),
            offsets=(0, n0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if Y_ROW_MAJOR else (0, 1),
        )
        for _ in range(0, tl.cdiv(M, BLOCK_M)):
            x_tile = tl.load(x_bp2, boundary_check=(0, 1), padding_option="zero")
            y_tile = tl.cast(x_tile, tl.float32) * scale_n[None, :]  # 广播到 (BM, BN)
            y_tile = tl.clamp(y_tile, min=-fp8_max, max=fp8_max)
            y_fp8 = y_tile.to(y_ptr.dtype.element_ty)
            tl.store(y_bp2, y_fp8, boundary_check=(0, 1))
            x_bp2 = tl.advance(x_bp2, (BLOCK_M, 0))
            y_bp2 = tl.advance(y_bp2, (BLOCK_M, 0))


def channel_granul_fp8_quant(
    x: torch.Tensor,
    float8_dtype: torch.dtype,
    axiswise_dim: int,
    output_row_major: bool = True,
    scale_tol: float = 1e-12,
) -> list[torch.Tensor]:
    assert x.dim() == 3, "only support 3D tensor now"
    if axiswise_dim not in (-1, -2):
        raise ValueError("axiswise_dim must be -1 or -2")
    reduce_along_n = True if axiswise_dim == -1 else False

    x_row_major = x.is_contiguous()

    B, M, N = x.shape
    if output_row_major:
        y = torch.empty((B, M, N), device=x.device, dtype=float8_dtype)
    else:
        y = torch.empty((B, N, M), device=x.device, dtype=float8_dtype)
        y = y.transpose(-1, -2)
    reciprocal_scale = torch.empty(
        (B, M, 1) if reduce_along_n else (B, 1, N), dtype=torch.float32, device=x.device
    )
    stride_scale_b, stride_scale_m, stride_scale_n = reciprocal_scale.stride()

    fp8_max = float(torch.finfo(float8_dtype).max)
    grid = lambda META: (
        (
            B,
            triton.cdiv(M, META["BLOCK_M"]),
        )
        if reduce_along_n
        else (
            B,
            triton.cdiv(N, META["BLOCK_N"]),
        )
    )
    channel_granul_fp8_quant_kernel[grid](
        x,
        y,
        reciprocal_scale,
        B,
        M,
        N,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        stride_scale_b,
        stride_scale_m,
        stride_scale_n,
        fp8_max=fp8_max,
        REDUCE_ALONG_N=reduce_along_n,
        X_ROW_MAJOR=x_row_major,
        Y_ROW_MAJOR=output_row_major,
        EPS=scale_tol,
    )

    return y, reciprocal_scale
