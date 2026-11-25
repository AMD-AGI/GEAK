import triton
import triton.language as tl


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % args["SPLITK_BLOCK_SIZE"] == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0),
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _gemm_a16_w16_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
    activation: tl.constexpr,
    use_activation: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    SKIP_REDUCE: tl.constexpr,
):
    """Kernel for computing C = A x B."""
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_ck > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    pid_unified = tl.program_id(axis=0)
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)
    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    if split_k_start < K:
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = split_k_start + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )
        acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
        if ADD_BIAS:
            if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                accumulator = tl.load(bias_ptr + offs_bn).to(dtype=acc_dtype)
                accumulator = tl.broadcast_to(
                    accumulator[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                )
            else:
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        else:
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
        k_span = split_k_end - split_k_start
        num_k_iter = tl.cdiv(k_span, BLOCK_SIZE_K)
        for k in range(num_k_iter):
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=offs_k[None, :] < k_span - k * BLOCK_SIZE_K,
                    other=0.0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < k_span - k * BLOCK_SIZE_K,
                    other=0.0,
                    cache_modifier=cache_modifier,
                )
            accumulator += tl.dot(a, b, input_precision="ieee")
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if use_activation and NUM_KSPLIT == 1:
            accumulator = activation(accumulator)
        c = accumulator.to(c_ptr.type.element_ty)
        offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)
