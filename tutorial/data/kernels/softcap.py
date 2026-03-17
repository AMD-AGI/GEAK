import torch
import triton
import triton.language as tl
from maybe_triton_jit import maybe_triton_jit
from triton.language.extra import libdevice

# from .utils import is_hopper, is_ampere, is_mlu590


def get_fwd_config():
    return [triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=1)]
    if is_hopper():
        return [triton.Config({"BLOCK_SIZE": 2048}, num_stages=2, num_warps=4)]
    elif is_ampere():
        return [triton.Config({"BLOCK_SIZE": 2048}, num_stages=4, num_warps=8)]
    elif is_mlu590():
        return [triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=1)]
    else:
        return [triton.Config({"BLOCK_SIZE": 2048}, num_stages=2, num_warps=16)]


def get_bwd_config():
    return [triton.Config({"BLOCK_SIZE": 8192}, num_stages=3, num_warps=1)]
    if is_hopper():
        return [triton.Config({"BLOCK_SIZE": 2048}, num_stages=4, num_warps=4)]
    elif is_ampere():
        return [triton.Config({"BLOCK_SIZE": 2048}, num_stages=3, num_warps=8)]
    elif is_mlu590():
        return [triton.Config({"BLOCK_SIZE": 8192}, num_stages=3, num_warps=1)]
    else:
        return [triton.Config({"BLOCK_SIZE": 512}, num_stages=2, num_warps=8)]


@maybe_triton_jit(
    configs=get_fwd_config(),
    key=[],
)
@triton.jit
def softcap_fwd_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    softcap,
    BLOCK_SIZE: tl.constexpr,
):
    # 0. calc addr of ptr
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 1. load x
    x = tl.load(x_ptr + offsets, mask=mask)

    # 2. softcap compute
    y = softcap * (libdevice.tanh(x.to(tl.float32) / softcap)).to(x.dtype)

    # 3. store y
    tl.store(y_ptr + offsets, y, mask=mask)


@maybe_triton_jit(
    configs=get_bwd_config(),
    key=[],
)
@triton.jit
def softcap_bwd_kernel(
    dy_ptr,
    x_ptr,
    dx_ptr,
    n_elements,
    softcap,
    BLOCK_SIZE: tl.constexpr,
):
    # 0. calc addr of ptr
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 1. load dy & x
    dy = tl.load(dy_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    # 2. softcap backward compute
    y = libdevice.tanh(x.to(tl.float32) / softcap).to(x.dtype)
    dx = dy * (1 - y * y)

    # 3. store dx
    tl.store(dx_ptr + offsets, dx, mask=mask)


class Softcap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, softcap):
        assert x.is_contiguous(), "Tensors must be contiguous"
        assert x.dtype in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ]
        numel = x.numel()
        # Allocates output.
        y = torch.empty_like(x)
        grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"]),)
        softcap_fwd_kernel[grid](
            x,
            y,
            numel,
            softcap,
        )

        ctx.save_for_backward(x)
        ctx.softcap = softcap
        return y

    @staticmethod
    def backward(ctx, dy):
        x = ctx.saved_tensors[0]
        numel = x.numel()
        dx = torch.empty_like(x)
        grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"]),)
        softcap_bwd_kernel[grid](
            dy,
            x,
            dx,
            numel,
            ctx.softcap,
        )

        return dx, None


if __name__ == "__main__":
    dtype = torch.float16
    softcap = 50.0
    x = torch.randn((1024, 1024), dtype=dtype, device="mlu").requires_grad_()
    y = Softcap.apply(x, softcap)
    dy = torch.randn_like(y)
    y.backward(dy)
    triton_dx, x.grad = x.grad.clone(), None

    ref = softcap * torch.tanh(x.to(torch.float32) / softcap).to(dtype)
    ref.backward(dy)
    torch_dx, x.grad = x.grad.clone(), None
    atol = 1e-3
    rtol = 1e-3
    if torch.allclose(y, ref, atol=atol, rtol=rtol):
        print("✅ [Fwd]Triton and Torch match")
    else:
        print("❌ [Fwd]Triton and Torch differ")
    if torch.allclose(triton_dx, torch_dx, atol=atol, rtol=rtol):
        print("✅ [Bwd]Triton and Torch match")
    else:
        print("❌ [Bwd]Triton and Torch differ")
