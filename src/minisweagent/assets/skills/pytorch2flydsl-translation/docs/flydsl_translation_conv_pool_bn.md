---
layer: "flydsl"
category: "translation"
tags: ["flydsl", "translation", "conv2d", "maxpool", "batchnorm"]
last_updated: 2026-04-23
---

# FlyDSL Translation: Conv2d, MaxPool2d, BatchNorm2d

## Conv2d via im2col + Preshuffle GEMM (Full Replacement)

Do NOT keep `nn.Conv2d` — replace it entirely with `nn.Parameter` for weights,
`F.unfold` for im2col, and `compile_preshuffle_gemm_a8` for the GEMM. The
`nn.Module` wrapper is required by the translation harness for `.cuda()` / `.to()`
parameter management — only the internals must be FlyDSL-native.

### Step-by-step Conv2d translation

1. Extract conv weights from the original model, reshape to `(out_ch, in_ch * kH * kW)`
2. Store as `nn.Parameter` (raw tensors, not `nn.Conv2d`)
3. Preshuffle weight once with `shuffle_weight(w, layout=(16, 16))`
4. Compile GEMM once with `compile_preshuffle_gemm_a8(...)`
5. In `forward()`: `F.unfold(x, ...)` produces `[N, C*kH*kW, L]` columns matrix
6. Fold batch into M dimension: `(B*L, K)` = A matrix for GEMM
7. Call the pre-compiled GEMM launcher
8. Reshape output back to `[N, out_ch, H_out, W_out]`

**Do NOT use `torch.bmm`** — the weight matrix is shared across the batch.
Fold B into M and use a single preshuffle GEMM call.

### Complete Conv2d Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from tests.utils import shuffle_weight

class Model(nn.Module):  # nn.Module required by translation harness
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        kH = kW = kernel_size
        K = in_channels * kH * kW

        # Store conv weights as raw nn.Parameter (NOT nn.Conv2d)
        self.conv_weight = nn.Parameter(
            torch.randn(out_channels, K, dtype=torch.float16))

        # Preshuffle weight ONCE — register_buffer so .cuda() moves it
        self.register_buffer("w_shuffled",
            shuffle_weight(self.conv_weight.data.contiguous(), layout=(16, 16)))

        # Compile GEMM ONCE
        self.gemm_fn = compile_preshuffle_gemm_a8(
            M=0, N=out_channels, K=K,
            tile_m=64, tile_n=128, tile_k=128,
            in_dtype="fp16", out_dtype="fp16", lds_stage=2,
        )

        self.stride = stride
        self.padding = padding
        self.kernel_size = (kH, kW)

    def forward(self, x):
        B, C_in, H_in, W_in = x.shape
        kH, kW = self.kernel_size
        C_out = self.conv_weight.shape[0]
        K = self.conv_weight.shape[1]

        # Step 1: im2col via F.unfold → (B, C_in*kH*kW, L)
        patches = F.unfold(x, kernel_size=self.kernel_size,
                           stride=self.stride, padding=self.padding)
        L = patches.shape[2]

        # Step 2: fold batch into M → (B*L, K)
        patches_2d = patches.transpose(1, 2).reshape(B * L, K).half().contiguous()
        M = B * L

        # Step 3: GEMM
        output = torch.empty(M, C_out, device=x.device, dtype=torch.float16)
        scale = torch.empty(0, device=x.device, dtype=torch.float32)
        self.gemm_fn(
            output.view(-1), patches_2d.view(-1),
            self.w_shuffled.view(-1), scale, scale,
            M, C_out, torch.cuda.current_stream(),
        )

        # Step 4: reshape back to NCHW
        H_out = (H_in + 2 * self.padding - kH) // self.stride + 1
        W_out = (W_in + 2 * self.padding - kW) // self.stride + 1
        return output.reshape(B, H_out, W_out, C_out).permute(0, 3, 1, 2).contiguous()

def get_inputs():
    return [torch.randn(8, 3, 32, 32, device="cuda")]

def get_init_inputs():
    return [3, 64, 3, 1, 1]
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

## BatchNorm2d via Custom @flyc.kernel

In inference mode, BatchNorm2d is a per-channel affine transform:
`output = input * scale + shift` where `scale = weight / sqrt(var + eps)`
and `shift = bias - mean * scale`. Pre-compute scale/shift once in `__init__`.

Do NOT use `F.batch_norm` or `nn.BatchNorm2d` — use a custom `@flyc.kernel`.

```python
import torch
import torch.nn as nn
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector
from flydsl.expr.typing import T

BLOCK_DIM = 256

@flyc.kernel
def batchnorm_kernel(
    Input: fx.Tensor,
    Scale: fx.Tensor,
    Shift: fx.Tensor,
    Output: fx.Tensor,
    C: fx.Constexpr[int],
    HW: fx.Constexpr[int],
    block_dim: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    global_idx = bid * block_dim + tid

    copy_atom = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
    reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
    reg_layout = fx.make_layout(1, 1)

    in_div = fx.logical_divide(Input, fx.make_layout(1, 1))
    out_div = fx.logical_divide(Output, fx.make_layout(1, 1))
    sc_div = fx.logical_divide(Scale, fx.make_layout(1, 1))
    sh_div = fx.logical_divide(Shift, fx.make_layout(1, 1))

    total = fx.Int32(C * HW)
    if arith.cmpi(arith.CmpIPredicate.ult, global_idx, total):
        c_idx = global_idx // fx.Int32(HW)

        r_in = fx.memref_alloca(reg_ty, reg_layout)
        fx.copy_atom_call(copy_atom, fx.slice(in_div, (None, global_idx)), r_in)
        val = vector.extract(fx.memref_load_vec(r_in), static_position=[0])

        r_sc = fx.memref_alloca(reg_ty, reg_layout)
        fx.copy_atom_call(copy_atom, fx.slice(sc_div, (None, c_idx)), r_sc)
        sc = vector.extract(fx.memref_load_vec(r_sc), static_position=[0])

        r_sh = fx.memref_alloca(reg_ty, reg_layout)
        fx.copy_atom_call(copy_atom, fx.slice(sh_div, (None, c_idx)), r_sh)
        sh = vector.extract(fx.memref_load_vec(r_sh), static_position=[0])

        result = arith.addf(arith.mulf(val, sc), sh)

        r_out = fx.memref_alloca(reg_ty, reg_layout)
        vec_ty = T.vec(1, T.f32)
        fx.memref_store_vec(vector.from_elements(vec_ty, [result]), r_out)
        fx.copy_atom_call(copy_atom, r_out, fx.slice(out_div, (None, global_idx)))


@flyc.jit
def batchnorm_launch(
    Input: fx.Tensor, Scale: fx.Tensor, Shift: fx.Tensor, Output: fx.Tensor,
    total: fx.Int32,
    C: fx.Constexpr[int], HW: fx.Constexpr[int], block_dim: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (total + block_dim - 1) // block_dim
    batchnorm_kernel(Input, Scale, Shift, Output, C, HW, block_dim).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self._scale = None
        self._shift = None

    def _precompute(self):
        scale = self.weight / torch.sqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        self._scale = scale.contiguous()
        self._shift = shift.contiguous()

    def forward(self, x):
        if self._scale is None:
            self._precompute()
        B, C, H, W = x.shape
        HW = H * W
        x_flat = x.reshape(B * C, HW).contiguous().view(-1)
        output = torch.empty_like(x_flat)
        total = B * C * HW
        # Expand scale/shift to match batch: repeat B times
        scale_expanded = self._scale.repeat(B).contiguous().view(-1)
        shift_expanded = self._shift.repeat(B).contiguous().view(-1)
        batchnorm_launch(
            x_flat, scale_expanded, shift_expanded, output,
            total, B * C, HW, BLOCK_DIM,
            stream=torch.cuda.current_stream())
        return output.reshape(B, C, H, W)
```

## Conv2d Optimization: 1x1 Convolutions Skip F.unfold

For **1x1 stride-1** convolutions (e.g., ResNet skip connections, channel projections),
`F.unfold` is unnecessary overhead -- the patch matrix is just a reshape of the input:

```python
if kernel_size == 1 and stride == 1 and padding == 0:
    # No im2col needed: (B, C_in, H, W) -> (B*H*W, C_in)
    patches_2d = x.permute(0, 2, 3, 1).reshape(B * H * W, C_in).half().contiguous()
else:
    patches = F.unfold(x, kernel_size=kernel_size, stride=stride, padding=padding)
    L = patches.shape[2]
    patches_2d = patches.transpose(1, 2).reshape(B * L, K).half().contiguous()
```

This avoids materializing the patch matrix and saves memory for large feature maps.

## Conv2d Optimization: Small-K GEMM Guidance

Preshuffle GEMM with very small K (e.g., K=27 for a 3x3 conv on 3-channel input,
padded to 32) is **overhead-dominated** because:
- `tile_k=128` means the K-loop runs once with most elements zeroed
- GEMM launch overhead exceeds useful compute

For K < 64, use `tile_k=32` (minimum) and consider smaller tile configs:
```python
gemm_fn = compile_preshuffle_gemm_a8(
    M=0, N=out_ch, K=K,
    tile_m=32, tile_n=64, tile_k=32,  # smaller tiles for small K
    in_dtype="fp16", out_dtype="fp16", lds_stage=2,
)
```

The GEMM is still correct (K is padded internally) but the agent should be aware
that small-K GEMMs may not outperform cuDNN's fused conv kernel. The goal is
FlyDSL-native code, not necessarily faster than cuDNN for every individual op.

## BatchNorm2d: Train vs Eval Mode

The pre-computed `scale`/`shift` pattern above **only works in eval mode** where
`running_mean` and `running_var` are fixed. In training mode, BN computes batch
statistics dynamically.

**IMPORTANT:** The translation harness does NOT call `.eval()`. Models run in
their default mode (training). Check the source kernel: if it does not call
`.eval()`, assume training mode.

**Eval mode** (source kernel calls `.eval()` or sets `self.eval()` in `__init__`):
- Use the pre-computed scale/shift pattern above

**Training mode** (default -- no `.eval()` call in source kernel):
- Do NOT fuse BN into conv weights (batch stats change each forward pass)
- Compute per-channel mean and variance from the current batch:
  `mean = x.mean(dim=(0, 2, 3))` and `var = x.var(dim=(0, 2, 3), unbiased=False)`
  (`.mean()` and `.var()` are basic tensor reductions, not forbidden compute ops)
- Compute dynamic scale/shift: `scale = weight / sqrt(var + eps)`,
  `shift = bias - mean * scale`
- Apply element-wise: `out = x * scale + shift` via a custom `@flyc.kernel`
- Perform the mean/var computation in fp32 for numerical stability, then cast
  scale/shift back to fp16 before passing to the FlyDSL kernel

## Composing Multiple Ops (ResNet Pattern)

For models like ResNet with `Conv2d + BatchNorm + ReLU + Conv2d + BatchNorm + skip + ReLU`:

1. **Each Conv2d** gets its own preshuffle GEMM (compiled once in `__init__`)
2. **Each BatchNorm** uses pre-computed scale/shift from its running stats
3. **ReLU** should be applied via a separate `@flyc.kernel` (or fused into the BN kernel as `bn_relu_kernel`)
4. **Residual add + ReLU** as a final element-wise `@flyc.kernel`

When the pattern is Conv+BN+ReLU, the preferred approach is:
- Use GEMM for Conv, then a fused BN+ReLU `@flyc.kernel` that applies `max(0, x * scale + shift)`
- This keeps the GEMM clean and handles bias/activation in a single element-wise kernel

## Summary

| PyTorch op | FlyDSL translation | Approach |
|-----------|-------------------|----------|
| `nn.Conv2d` | im2col + preshuffle GEMM (fp16) | `F.unfold` + `compile_preshuffle_gemm_a8` + reshape NCHW. Skip unfold for 1x1 convs. |
| `F.max_pool2d` | Custom `@flyc.kernel` | `arith.maximumf` over window elements, batch flattened into channels. |
| `nn.BatchNorm2d` | Custom `@flyc.kernel` (eval) | Pre-compute scale/shift in `__init__`, element-wise apply in forward. Eval mode only. |
| Conv+BN+ReLU | GEMM + fused BN+ReLU kernel | GEMM for conv, then `@flyc.kernel` applying `max(0, x * scale + shift)`. |
