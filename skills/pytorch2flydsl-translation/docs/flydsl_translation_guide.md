---
layer: "flydsl"
category: "translation"
tags: ["flydsl", "pytorch", "translation", "guide", "op-mapping"]
last_updated: 2026-04-08
---

# PyTorch to FlyDSL Translation Guide

## Overview

Translate PyTorch `nn.Module` kernels to FlyDSL using the real `flydsl.compiler`
and `flydsl.expr` API. The translated code must use `@flyc.kernel` / `@flyc.jit`
and wrap everything in a `Model(nn.Module)` for test harness compatibility.

## Key Principle: FlyDSL Uses Layout Algebra

Unlike CUDA/Triton where you use `input[idx]`, FlyDSL uses layout algebra:
1. **Partition** tensors with `fx.logical_divide(tensor, layout)`
2. **Select** tiles with `fx.slice(tensor, (None, index))`
3. **Copy** data through copy atoms between global memory and registers
4. **Compute** on register-resident vector values
5. **Store** results back through copy atoms

## Step 1: Analyze the PyTorch Kernel

Identify the computational pattern:
1. **Element-wise**: Each output depends only on corresponding input(s) → custom `@flyc.kernel`
2. **Reduction**: Output has fewer elements (sum, mean, softmax) → manual reduction or pre-built kernel
3. **GEMM/Linear**: Matrix multiplication → `compile_preshuffle_gemm_a8` + `shuffle_weight`
4. **Normalization**: LayerNorm, RMSNorm → `build_layernorm_module` / `build_rmsnorm_module`
5. **Convolution**: Conv2d → im2col (`F.unfold`) + `compile_preshuffle_gemm_a8` (fp16 cast)

## Step 2: Choose Translation Strategy

**Always prefer FlyDSL over PyTorch.** Replace ALL PyTorch compute modules
(`nn.Linear`, `nn.Conv2d`) with FlyDSL-native patterns using `nn.Parameter`
for weight storage and pre-built FlyDSL kernels for compute.

### Replacing nn.Linear with FlyDSL GEMM

Do NOT keep `nn.Linear` — replace it entirely:
1. Extract weight and bias from the original model
2. Store as `nn.Parameter` (raw tensors)
3. Preshuffle weight once with `shuffle_weight(w, layout=(16, 16))`
4. Compile GEMM once with `compile_preshuffle_gemm_a8(...)` — use fused epilogue if followed by activation
5. Call the compiled launcher in `forward()`

### Replacing nn.Conv2d with im2col + FlyDSL GEMM

Do NOT keep `nn.Conv2d` — replace it entirely:
1. Extract conv weight, reshape to `(out_channels, in_channels * kH * kW)`
2. Store as `nn.Parameter`, preshuffle once
3. In forward: `F.unfold(x, ...)` produces `[N, C*kH*kW, L]` columns matrix
4. Compile GEMM once, call in forward with unfolded input
5. Reshape output back to `[N, out_channels, H_out, W_out]`

See `flydsl_translation_conv_pool_bn.md` for the complete worked example.

### Pre-built FlyDSL kernels

- **GEMM**: `compile_preshuffle_gemm_a8` — replaces `nn.Linear`, `torch.matmul`, `F.linear`
- **Flash Attention**: `build_flash_attn_func_module` — replaces `F.scaled_dot_product_attention`
- **Softmax**: `build_softmax_module`
- **LayerNorm/RMSNorm**: `build_layernorm_module` / `build_rmsnorm_module`

### Custom FlyDSL kernels

Write `@flyc.kernel` + `@flyc.jit` with layout algebra for element-wise ops
and simple reductions.

### Acceptable PyTorch fallbacks (minimal set)

- `F.unfold` (im2col for Conv2d — structural, not compute)
- `F.batch_norm` (MIOpen backend, no FlyDSL equivalent)
- `torch.mm` (fp32 GEMM only — when fp16 preshuffle fails correctness)
- `F.max_pool2d` (if custom FlyDSL maxpool kernel fails)

## Step 3: Element-wise Translation (Complete Example)

### ReLU Translation

**PyTorch:**
```python
class Model(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(16, 16384).cuda()]

def get_init_inputs():
    return []
```

**FlyDSL:**
```python
import torch
import torch.nn as nn
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector
from flydsl.expr.typing import T, Int32

BLOCK_DIM = 256
VEC_WIDTH = 4

@flyc.kernel
def relu_kernel(
    Input: fx.Tensor,
    Output: fx.Tensor,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    tile_elems = block_dim * vec_width

    tI = fx.logical_divide(Input, fx.make_layout(tile_elems, 1))
    tO = fx.logical_divide(Output, fx.make_layout(tile_elems, 1))
    tI = fx.slice(tI, (None, bid))
    tO = fx.slice(tO, (None, bid))

    tI = fx.logical_divide(tI, fx.make_layout(vec_width, 1))
    tO = fx.logical_divide(tO, fx.make_layout(vec_width, 1))

    copy_bits = vec_width * 32
    RegTy = fx.MemRefType.get(T.f32(), fx.LayoutType.get(vec_width, 1),
                               fx.AddressSpace.Register)
    reg_layout = fx.make_layout(vec_width, 1)
    copyAtom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)

    rI = fx.memref_alloca(RegTy, reg_layout)
    rO = fx.memref_alloca(RegTy, reg_layout)

    fx.copy_atom_call(copyAtom, fx.slice(tI, (None, tid)), rI)

    vI = fx.memref_load_vec(rI)
    zero_vec = vector.broadcast(T.vec(vec_width, T.f32()), arith.constant(0.0, type=T.f32()))
    vO = fx.arith.maximumf(vI, zero_vec)

    fx.memref_store_vec(vO, rO)
    fx.copy_atom_call(copyAtom, rO, fx.slice(tO, (None, tid)))


@flyc.jit
def relu_launch(
    Input: fx.Tensor,
    Output: fx.Tensor,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    tile_elems = block_dim * vec_width
    grid_x = (n + tile_elems - 1) // tile_elems
    relu_kernel(Input, Output, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream,
    )


class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        n = x.numel()
        tX = flyc.from_dlpack(x).mark_layout_dynamic(leading_dim=0, divisibility=VEC_WIDTH)
        relu_launch(tX, output, n, n, BLOCK_DIM, VEC_WIDTH,
                    stream=torch.cuda.current_stream())
        return output


def get_inputs():
    return [torch.randn(16, 16384).cuda()]

def get_init_inputs():
    return []
```

### Swish (x * sigmoid(x)) Translation

For sigmoid, use: `1 / (1 + exp2(-x * log2(e)))`.
FlyDSL provides `rocdl.exp2` for fast hardware exp2.

```python
from flydsl.expr import rocdl

# Inside kernel, after loading vI:
LOG2E = 1.4426950408889634
c_log2e = arith.constant(LOG2E, type=T.f32())
c_one = arith.constant(1.0, type=T.f32())

# For vector operations, broadcast constants:
log2e_vec = vector.broadcast(vec_type, c_log2e)
one_vec = vector.broadcast(vec_type, c_one)

neg_x = fx.arith.negf(vI)
neg_x_log2e = fx.arith.mulf(neg_x, log2e_vec)
# exp2 operates element-wise on vectors via ArithValue.exp2
exp_neg_x = neg_x_log2e.exp2()
denom = fx.arith.addf(one_vec, exp_neg_x)
sigmoid = fx.arith.divf(one_vec, denom)
swish = fx.arith.mulf(vI, sigmoid)
```

## Step 4: Using Pre-built Kernels

For operations with pre-built FlyDSL kernels, use the builder API:

### Softmax Example

```python
from kernels.softmax_kernel import build_softmax_module

class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self._softmax = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M, N = x.shape[0], x.shape[1]
        if self._softmax is None:
            self._softmax = build_softmax_module(M, N, dtype_str="f32")
        output = torch.empty_like(x)
        self._softmax(x, output, M, stream=torch.cuda.current_stream())
        return output
```

### GEMM / Linear Layer

```python
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from tests.utils import shuffle_weight

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Store weights as raw nn.Parameter (NOT nn.Linear)
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float16))
        # Preshuffle weight ONCE
        self.w_shuffled = shuffle_weight(self.weight.data.contiguous(), layout=(16, 16))
        # Compile GEMM ONCE — fuse bias+relu if needed
        N, K = out_features, in_features
        self.gemm_fn = compile_preshuffle_gemm_a8(
            M=0, N=N, K=K,
            tile_m=64, tile_n=128, tile_k=128,
            in_dtype="fp16", out_dtype="fp16", lds_stage=2,
        )

    def forward(self, x):
        x = x.half()
        M = x.shape[0]
        N = self.weight.shape[0]
        output = torch.empty(M, N, device=x.device, dtype=torch.float16)
        scale = torch.empty(0, device=x.device, dtype=torch.float32)
        self.gemm_fn(
            output.view(-1), x.contiguous().view(-1),
            self.w_shuffled.view(-1), scale, scale, M, N,
            torch.cuda.current_stream(),
        )
        return output + self.bias
```

### Flash Attention

```python
from kernels.flash_attn_func import build_flash_attn_func_module

class Model(nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self._flash_attn = build_flash_attn_func_module(
            num_heads=num_heads, head_dim=head_dim,
            causal=True, dtype_str="f16",
        )

    def forward(self, q, k, v):
        # q, k, v: (batch, seq_len, num_heads, head_dim) — BSHD layout
        B, S, H, D = q.shape
        output = torch.empty_like(q)
        self._flash_attn(
            q.contiguous().view(-1), k.contiguous().view(-1),
            v.contiguous().view(-1), output.view(-1),
            B, S,  # num_heads is baked into builder, NOT passed here
            stream=torch.cuda.current_stream(),
        )
        return output
```

## Step 5: Hybrid Translation for Complex Models

For L2/L3 kernels with multiple operation types, compose FlyDSL kernels:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from tests.utils import shuffle_weight

class Model(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        K = in_ch * kernel_size * kernel_size
        # Conv weights as raw nn.Parameter (NOT nn.Conv2d)
        self.conv_weight = nn.Parameter(torch.randn(out_ch, K, dtype=torch.float16))
        self.conv_bias = nn.Parameter(torch.randn(out_ch, dtype=torch.float16))
        self.w_shuffled = shuffle_weight(self.conv_weight.data.contiguous(), layout=(16, 16))
        self.conv_gemm = compile_preshuffle_gemm_a8(
            M=0, N=out_ch, K=K, tile_m=64, tile_n=128, tile_k=128,
            in_dtype="fp16", out_dtype="fp16", lds_stage=2,
            epilogue="bias_relu")  # fuse bias + ReLU
        self.kernel_size = kernel_size

    def forward(self, x):
        # Conv2d via im2col + preshuffle GEMM
        patches = F.unfold(x, kernel_size=self.kernel_size, padding=1)
        B, K, L = patches.shape
        patches_2d = patches.transpose(1, 2).reshape(B * L, K).half().contiguous()
        # ... GEMM call, reshape to NCHW ...
        return out
```

## Common Pitfalls

1. **Wrong import**: Use `import flydsl.compiler as flyc` NOT `import flycompute as flyc`
2. **Missing layout algebra**: FlyDSL doesn't support `tensor[idx]` — use `fx.logical_divide` + `fx.slice` + copy atoms
3. **Forgetting `fx.Constexpr`**: Tile sizes and vec widths must be `fx.Constexpr[int]`
4. **Wrong copy atom size**: `fx.UniversalCopy(bits)` where bits = vec_width * element_bits (e.g., 4 * 32 = 128 for 4 float32)
5. **Missing `from_dlpack`**: For vectorized kernels, wrap input with `flyc.from_dlpack(x).mark_layout_dynamic(leading_dim=0, divisibility=vec_width)`
6. **N not aligned to tile**: If tensor size isn't a multiple of `block_dim * vec_width`, either pad or use the scalar path
7. **Forgetting stream**: Always pass `stream=torch.cuda.current_stream()` to `@flyc.jit` launchers
8. **No Triton/CUDA**: Do NOT use Triton or CUDA. Only use `flydsl.compiler` and `flydsl.expr`.

## Decision Tree

```
What operation type?
├── Element-wise (relu, sigmoid, tanh, swish, clamp, ...)
│   └── Write custom @flyc.kernel with layout algebra pattern
├── Reduction (sum, mean, max, ...)
│   └── Write custom kernel with wave/block reduce pattern
├── Softmax
│   └── Use build_softmax_module() from kernels.softmax_kernel
├── LayerNorm / RMSNorm
│   └── Use build_layernorm_module() / build_rmsnorm_module()
├── GEMM / Linear / torch.matmul
│   ├── fp32 required? Check GEMM dtype table. No fp32 → use torch.mm
│   └── fp16/bf16 → Use compile_preshuffle_gemm_a8() [NOT torch.matmul / F.linear]
├── Attention (self-attention, SDPA, Flash)
│   └── Use build_flash_attn_func_module() from kernels.flash_attn_func
│       (fallback to decomposed GEMM+softmax if constraints not met)
├── Conv2d
│   ├── F.unfold (im2col) to get patches (B, K_patch, L)
│   ├── Transpose+reshape to (B*L, K_patch) = A matrix
│   ├── Weight (C_out, K_patch) → preshuffle once (fp16 cast)
│   ├── compile_preshuffle_gemm_a8(fp16) → reshape output to NCHW
│   └── If correctness fails → im2col + torch.mm fallback
├── MaxPool2d
│   └── Custom @flyc.kernel with arith.maximumf over window elements
├── BatchNorm2d
│   └── F.batch_norm (acceptable PyTorch fallback, MIOpen backend)
└── Complex model (L2/L3)
    └── FlyDSL for ALL ops; Conv2d via im2col+GEMM, MaxPool via custom kernel
```

**IMPORTANT**: Do NOT use `torch.matmul`, `F.linear`, `nn.Linear`, or
`F.scaled_dot_product_attention` when FlyDSL pre-built kernels are available.
Acceptable PyTorch fallbacks: `F.unfold` (im2col), `F.batch_norm`, `torch.mm`
(fp32 GEMM only). Conv2d uses im2col + preshuffle GEMM, NOT `nn.Conv2d`.
