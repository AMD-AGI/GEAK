---
layer: "flydsl"
category: "translation"
tags: ["flydsl", "translation", "conv2d", "maxpool", "batchnorm"]
last_updated: 2026-04-23
---

# FlyDSL Translation: Conv2d, MaxPool2d, BatchNorm2d

## Conv2d via im2col + Preshuffle GEMM

Convolution is equivalent to im2col (unfolding input patches into a 2D matrix)
followed by a matrix multiplication with the filter weights. Use `F.unfold` for
the im2col step and `compile_preshuffle_gemm_a8` for the GEMM.

**Conv2d GEMM uses fp16 preshuffle even for fp32 input data.** This is an
acceptable precision tradeoff because BatchNorm after convolution absorbs small
numerical differences. The weight matrix (shared across all batch elements) is
preshuffled once, making this very efficient.

**If correctness fails**, fall back to `im2col + torch.mm` (fp32) instead.

**Do NOT use `torch.bmm` for conv.** The weight matrix is shared across the
batch, NOT per-batch. Use a single preshuffle GEMM call with the batch folded
into the M dimension.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from tests.utils import shuffle_weight

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self._gemm = None
        self._weight_shuffled = None

    def forward(self, x):
        B, C_in, H_in, W_in = x.shape
        kH, kW = self.kernel_size
        C_out = self.conv.weight.shape[0]

        # im2col: unfold input patches into columns
        # Output shape: (B, C_in * kH * kW, L) where L = H_out * W_out
        patches = F.unfold(x, kernel_size=self.kernel_size,
                           stride=self.stride, padding=self.padding)
        L = patches.shape[2]
        K = C_in * kH * kW

        # Reshape for GEMM: fold batch into M dimension
        # (B, K, L) -> transpose -> (B, L, K) -> reshape -> (B*L, K)
        patches_2d = patches.transpose(1, 2).reshape(B * L, K).half().contiguous()
        M = B * L

        # Compile GEMM and preshuffle weight (once)
        if self._gemm is None:
            self._gemm = compile_preshuffle_gemm_a8(
                M=0, N=C_out, K=K,
                tile_m=64, tile_n=128, tile_k=128,
                in_dtype="fp16", out_dtype="fp16", lds_stage=2,
            )
            # Conv weight: (C_out, C_in, kH, kW) -> (C_out, K) -> preshuffle
            w = self.conv.weight.data.reshape(C_out, -1).half().contiguous()
            self._weight_shuffled = shuffle_weight(w, layout=(16, 16))

        output = torch.empty(M, C_out, device=x.device, dtype=torch.float16)
        scale = torch.empty(0, device=x.device, dtype=torch.float32)
        self._gemm(
            output.view(-1), patches_2d.view(-1),
            self._weight_shuffled.view(-1), scale, scale,
            M, C_out, torch.cuda.current_stream(),
        )

        # Reshape back to NCHW
        H_out = (H_in + 2 * self.padding - kH) // self.stride + 1
        W_out = (W_in + 2 * self.padding - kW) // self.stride + 1
        out = output.float().reshape(B, H_out, W_out, C_out).permute(0, 3, 1, 2).contiguous()

        return out
```

### Conv2d Fallback: im2col + torch.mm

If the preshuffle GEMM path fails correctness checks (tolerance too tight for
fp16), fall back to fp32 `torch.mm`. This keeps `F.unfold` for im2col but uses
PyTorch for the matmul:

```python
# In forward(), replace the GEMM section with:
W = self.conv.weight.data.reshape(C_out, -1)  # (C_out, K) -- fp32
patches_2d = patches.transpose(1, 2).reshape(B * L, K)  # (B*L, K) -- fp32
out = torch.mm(patches_2d, W.t())  # (B*L, C_out) -- fp32
out = out.reshape(B, H_out, W_out, C_out).permute(0, 3, 1, 2).contiguous()
```

## MaxPool2d via Custom FlyDSL Kernel

MaxPool2d computes the maximum over a sliding window. Each output element
reads `kernel_size * kernel_size` input elements and takes the max.

Flatten the batch dimension into channels (`B*C` channels) to process the
entire batch in a single kernel launch without Python loops.

```python
import torch
import torch.nn as nn
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, gpu
from flydsl.expr.typing import T

BLOCK_DIM = 256

@flyc.kernel
def maxpool2d_kernel(
    Input: fx.Tensor,
    Output: fx.Tensor,
    C: fx.Constexpr[int],
    H_in: fx.Constexpr[int],
    W_in: fx.Constexpr[int],
    H_out: fx.Constexpr[int],
    W_out: fx.Constexpr[int],
    kernel_size: fx.Constexpr[int],
    stride: fx.Constexpr[int],
    block_dim: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    global_idx = bid * block_dim + tid
    total_out = C * H_out * W_out

    copy_atom = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
    reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
    reg_layout = fx.make_layout(1, 1)

    in_div = fx.logical_divide(Input, fx.make_layout(1, 1))
    out_div = fx.logical_divide(Output, fx.make_layout(1, 1))

    if arith.cmpi(arith.CmpIPredicate.ult, global_idx, fx.Int32(total_out)):
        c = global_idx // fx.Int32(H_out * W_out)
        rem = global_idx % fx.Int32(H_out * W_out)
        oh = rem // fx.Int32(W_out)
        ow = rem % fx.Int32(W_out)

        ih_start = oh * fx.Int32(stride)
        iw_start = ow * fx.Int32(stride)

        neg_inf = arith.constant(-1e30, type=T.f32)
        max_val = neg_inf

        for kh in fx.range_constexpr(kernel_size):
            for kw in fx.range_constexpr(kernel_size):
                ih = ih_start + fx.Int32(kh)
                iw = iw_start + fx.Int32(kw)
                in_idx = c * fx.Int32(H_in * W_in) + ih * fx.Int32(W_in) + iw
                r_in = fx.memref_alloca(reg_ty, reg_layout)
                fx.copy_atom_call(copy_atom, fx.slice(in_div, (None, in_idx)), r_in)
                val = vector.extract(fx.memref_load_vec(r_in), static_position=[0])
                max_val = arith.maximumf(max_val, val)

        r_out = fx.memref_alloca(reg_ty, reg_layout)
        vec_ty = T.vec(1, T.f32)
        fx.memref_store_vec(vector.from_elements(vec_ty, [max_val]), r_out)
        fx.copy_atom_call(copy_atom, r_out, fx.slice(out_div, (None, global_idx)))


@flyc.jit
def maxpool2d_launch(
    Input: fx.Tensor, Output: fx.Tensor,
    total: fx.Int32,
    C: fx.Constexpr[int], H_in: fx.Constexpr[int], W_in: fx.Constexpr[int],
    H_out: fx.Constexpr[int], W_out: fx.Constexpr[int],
    kernel_size: fx.Constexpr[int], stride: fx.Constexpr[int],
    block_dim: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (total + block_dim - 1) // block_dim
    maxpool2d_kernel(Input, Output, C, H_in, W_in, H_out, W_out,
                     kernel_size, stride, block_dim).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream,
    )


class Model(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        kernel_size = 2
        stride = 2
        H_out = H // stride
        W_out = W // stride

        # Flatten batch into channel dim for single kernel launch
        x_flat = x.reshape(1, B * C, H, W)
        output = torch.empty(1, B * C, H_out, W_out, device=x.device, dtype=x.dtype)
        total = B * C * H_out * W_out
        inp = x_flat.contiguous().view(-1)
        out = output.contiguous().view(-1)
        maxpool2d_launch(
            inp, out, total,
            B * C, H, W, H_out, W_out, kernel_size, stride, BLOCK_DIM,
            stream=torch.cuda.current_stream(),
        )
        return output.reshape(B, C, H_out, W_out)
```

## BatchNorm2d (Eval Mode)

In inference mode, BatchNorm2d is a simple per-channel affine transform.
Use `F.batch_norm` for this — it calls into the optimized MIOpen backend:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.bn.eval()

    def forward(self, x):
        return F.batch_norm(
            x, self.bn.running_mean, self.bn.running_var,
            self.bn.weight, self.bn.bias, training=False, eps=self.bn.eps,
        )
```

`F.batch_norm` is an acceptable PyTorch fallback — there is no FlyDSL
pre-built BatchNorm kernel.

## Composing Multiple Ops

For models with multiple ops (e.g., Conv2d + BatchNorm + ReLU + MaxPool2d + Linear),
compose these patterns in a single `Model` class. Each op gets its own
kernel/launcher, initialized lazily in `forward()`.

## Summary

| PyTorch op | FlyDSL translation | Approach |
|-----------|-------------------|----------|
| `nn.Conv2d` | im2col + preshuffle GEMM (fp16) | `F.unfold` + `compile_preshuffle_gemm_a8` + reshape NCHW. Fallback: `F.unfold` + `torch.mm` |
| `F.max_pool2d` | Custom `@flyc.kernel` | `arith.maximumf` over window elements, batch flattened into channels |
| `nn.BatchNorm2d` | `F.batch_norm` | Acceptable PyTorch fallback (MIOpen backend) |
