---
layer: "flydsl"
category: "translation"
tags: ["flydsl", "pytorch", "translation", "api-reference"]
last_updated: 2026-04-08
---

# FlyDSL API Reference for Translation

FlyDSL is a Python DSL and MLIR stack for GPU kernels on AMD GPUs (MI300X, MI350).
Kernels are JIT-compiled through MLIR on first call, then cached.

## Imports

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, vector, range_constexpr
from flydsl.expr.typing import T, Int32
```

## Kernel Structure (Three-Layer Pattern)

Every FlyDSL kernel follows `@flyc.kernel` + `@flyc.jit` + `Model(nn.Module)`:

```python
import torch
import torch.nn as nn
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu

@flyc.kernel
def my_kernel(Input: fx.Tensor, Output: fx.Tensor, block_dim: fx.Constexpr[int]):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    # ... kernel body (layout algebra + arith ops) ...

@flyc.jit
def my_launch(Input: fx.Tensor, Output: fx.Tensor, n: fx.Int32,
              const_n: fx.Constexpr[int], block_dim: fx.Constexpr[int],
              stream: fx.Stream = fx.Stream(None)):
    grid_x = (n + block_dim - 1) // block_dim
    my_kernel(Input, Output, block_dim).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream,
    )

class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        n = x.numel()
        my_launch(x, output, n, n, 256, stream=torch.cuda.current_stream())
        return output
```

## Parameter Types

| Type | Usage | Description |
|------|-------|-------------|
| `fx.Tensor` | Kernel + JIT args | GPU tensor (auto-converted from PyTorch via DLPack) |
| `fx.Constexpr[int]` | Kernel + JIT args | Compile-time constant (different values = different compiled kernels) |
| `fx.Int32` | JIT args | Runtime integer |
| `fx.Stream` | JIT args | CUDA/HIP stream |

## Thread / Block Indexing

```python
tid = fx.thread_idx.x      # thread index within workgroup (alias: gpu.thread_idx.x)
bid = fx.block_idx.x        # block (workgroup) index
bdim = fx.block_dim.x       # block dimension size
```

## Element-wise Kernel Pattern (Layout Algebra)

FlyDSL uses layout algebra for all memory access. The canonical element-wise pattern:

```python
@flyc.kernel
def elementwise_kernel(
    A: fx.Tensor, B: fx.Tensor,
    block_dim: fx.Constexpr[int], vec_width: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    tile_elems = block_dim * vec_width

    # 1. Partition tensor into block-sized tiles
    tA = fx.logical_divide(A, fx.make_layout(tile_elems, 1))
    tB = fx.logical_divide(B, fx.make_layout(tile_elems, 1))
    tA = fx.slice(tA, (None, bid))   # select this block's tile
    tB = fx.slice(tB, (None, bid))

    # 2. Sub-partition by vec_width (each thread handles vec_width elements)
    tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
    tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))

    # 3. Allocate register buffers
    copy_bits = vec_width * 32  # for float32
    RegTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(vec_width, 1),
                               fx.AddressSpace.Register)
    reg_layout = fx.make_layout(vec_width, 1)
    copyAtom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)

    rA = fx.memref_alloca(RegTy, reg_layout)
    rB = fx.memref_alloca(RegTy, reg_layout)

    # 4. Load data via copy atoms
    fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

    # 5. Compute (vector ops on register contents)
    vA = fx.memref_load_vec(rA)
    vB = fx.memref_load_vec(rB)
    vC = fx.arith.addf(vA, vB)  # element-wise add

    # 6. Store result
    rC = fx.memref_alloca(RegTy, reg_layout)
    fx.memref_store_vec(vC, rC)
    fx.copy_atom_call(copyAtom, rC, fx.slice(tB, (None, tid)))  # store to output
```

### Key Layout Algebra Functions

| Function | Description |
|----------|-------------|
| `fx.logical_divide(tensor, layout)` | Partition tensor by layout |
| `fx.slice(tensor, (None, index))` | Select a partition at index |
| `fx.make_layout(shape, stride)` | Create a layout (shape, stride) pair |
| `fx.make_copy_atom(copy_type, dtype)` | Create a copy atom for memory transfers |
| `fx.memref_alloca(type, layout)` | Allocate register storage |
| `fx.copy_atom_call(atom, src, dst)` | Execute copy between memory/registers |
| `fx.memref_load_vec(reg)` | Load register contents as vector value |
| `fx.memref_store_vec(vec, reg)` | Store vector value to register |

### Copy Atom Types

| Type | Bits | Usage |
|------|------|-------|
| `fx.UniversalCopy(bits)` | Variable | General copy (32, 64, 128, ...) |
| `fx.UniversalCopy32b()` | 32 | Single float32 element |
| `fx.rocdl.BufferCopy32b()` | 32 | AMD buffer instruction copy |
| `fx.rocdl.BufferCopy128b()` | 128 | AMD buffer instruction 4x float32 |

### Register Type Construction

```python
# For vec_width float32 elements:
RegTy = fx.MemRefType.get(
    fx.T.f32(),                          # element type
    fx.LayoutType.get(vec_width, 1),     # layout (shape, stride)
    fx.AddressSpace.Register             # address space
)
```

## Arithmetic Operations

```python
from flydsl.expr import arith

# Vector arithmetic (operate on fx.memref_load_vec results)
result = fx.arith.addf(a, b)          # a + b (float)
result = fx.arith.subf(a, b)          # a - b (float)
result = fx.arith.mulf(a, b)          # a * b (float)
result = fx.arith.divf(a, b)          # a / b (float)
result = fx.arith.negf(a)             # -a (float)
result = fx.arith.maximumf(a, b)      # max(a, b)
result = fx.arith.minimumf(a, b)      # min(a, b)

# Scalar arithmetic
c = arith.constant(0.0, type=T.f32)   # constant
c = fx.Float32(3.14)                   # float32 constant
c = fx.Int32(42)                       # int32 constant
c = fx.Index(0)                        # index constant

# Comparisons (returns predicate)
cond = arith.cmpi(arith.CmpIPredicate.ult, a, b)   # unsigned less than
cond = arith.cmpf(arith.CmpFPredicate.ogt, a, b)   # ordered greater than

# Select
result = arith.select(cond, true_val, false_val)

# Type casting
idx = arith.index_cast(T.index, int_val)
```

## Vector Operations

```python
from flydsl.expr import vector

# Extract scalar from vector
scalar = vector.extract(vec, static_position=[0])

# Build vector from scalars
vec = vector.from_elements(vec_type, [a, b, c, d])

# Vector broadcast (scalar to vector)
vec = vector.broadcast(vec_type, scalar)

# Vector reduction
max_val = vector.reduction(T.f32, vector.CombiningKind.MAXNUMF, vec)
sum_val = vector.reduction(T.f32, vector.CombiningKind.ADD, vec)

# Bitcast
result = vector.bitcast(target_type, vec)
```

## Scalar Element-wise Pattern (for arbitrary N)

When N is not a multiple of tile size, use scalar copy atoms:

```python
@flyc.kernel
def scalar_kernel(Input: fx.Tensor, Output: fx.Tensor, N: fx.Constexpr[int]):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    copy_atom = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
    reg_ty = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1),
                                fx.AddressSpace.Register)
    reg_layout = fx.make_layout(1, 1)

    in_div = fx.logical_divide(Input, fx.make_layout(1, 1))
    out_div = fx.logical_divide(Output, fx.make_layout(1, 1))

    idx = bid * 256 + tid
    if arith.cmpi(arith.CmpIPredicate.ult, idx, Int32(N)):
        r_in = fx.memref_alloca(reg_ty, reg_layout)
        fx.copy_atom_call(copy_atom, fx.slice(in_div, (None, idx)), r_in)
        val = vector.extract(fx.memref_load_vec(r_in), static_position=[0])

        # ... compute on val ...
        result = arith.maximumf(val, arith.constant(0.0, type=T.f32))

        r_out = fx.memref_alloca(reg_ty, reg_layout)
        vec_ty = T.vec(1, fx.T.f32())
        fx.memref_store_vec(vector.from_elements(vec_ty, [result]), r_out)
        fx.copy_atom_call(copy_atom, r_out, fx.slice(out_div, (None, idx)))
```

## Synchronization

```python
from flydsl.expr import gpu
gpu.barrier()   # workgroup barrier
```

## Launch Configuration

```python
@flyc.jit
def launch(Input: fx.Tensor, Output: fx.Tensor, n: fx.Int32,
           const_n: fx.Constexpr[int], block_dim: fx.Constexpr[int],
           stream: fx.Stream = fx.Stream(None)):
    grid_x = (n + block_dim - 1) // block_dim
    my_kernel(Input, Output, block_dim).launch(
        grid=(grid_x, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )
```

Grid/block dimensions accept int, ir.Value, or tuple of 1-3 values.

## DLPack Tensor Wrapping

For tensors needing dynamic layout annotation:

```python
tA = flyc.from_dlpack(pytorch_tensor).mark_layout_dynamic(
    leading_dim=0, divisibility=vec_width
)
```

Regular PyTorch tensors can also be passed directly to `@flyc.jit` functions.

## Pre-built Kernels (ALWAYS prefer over PyTorch equivalents)

FlyDSL provides highly optimized pre-built kernels. **Use these instead of PyTorch ops.**

```python
# GEMM — replaces nn.Linear, torch.matmul, F.linear
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from tests.utils import shuffle_weight

launch_fn = compile_preshuffle_gemm_a8(
    M=0, N=N, K=K, tile_m=64, tile_n=128, tile_k=128,
    in_dtype="fp16", out_dtype="fp16", lds_stage=2,
)
# B must be preshuffled ONCE (e.g. in __init__):
B_shuffled = shuffle_weight(B.contiguous(), layout=(16, 16))
# For non-quantized: use empty scale tensors
scale = torch.empty(0, device=device, dtype=torch.float32)
# Launch — ALL tensors must be .view(-1):
launch_fn(C.view(-1), A.view(-1), B_shuffled.view(-1), scale, scale, M, N, stream)

# Flash Attention — replaces F.scaled_dot_product_attention
from kernels.flash_attn_func import build_flash_attn_func_module

flash_fn = build_flash_attn_func_module(
    num_heads=H, head_dim=D, causal=True, dtype_str="f16",
)
# Q/K/V/O: contiguous 1D (BSHD flattened). num_heads baked in at build time.
flash_fn(Q_flat, K_flat, V_flat, O_flat, batch_size, seq_len, stream=stream)
# Requires: head_dim>=64, head_dim%32==0, seq_len%128==0

# Softmax — replaces torch.softmax, F.softmax
from kernels.softmax_kernel import build_softmax_module
executor = build_softmax_module(M=batch, N=dim, dtype_str="f32")
# Usage: executor(input_2d, output_2d, M, stream=stream)

# LayerNorm — replaces nn.LayerNorm
from kernels.layernorm_kernel import build_layernorm_module
executor = build_layernorm_module(M=batch, N=dim, dtype_str="f32")
# Usage: executor(input, gamma, beta, output, M, stream=stream)

# RMSNorm — replaces custom RMSNorm
from kernels.rmsnorm_kernel import build_rmsnorm_module
executor = build_rmsnorm_module(M=batch, N=dim, dtype_str="f32")
# Usage: executor(input, gamma, output, M, stream=stream)
```

**Only these ops have NO FlyDSL equivalent**: Conv2d, Conv3d, MaxPool2d, AvgPool2d, BatchNorm2d.

## Common PyTorch -> FlyDSL Op Mapping

| PyTorch | FlyDSL | Type |
|---------|--------|------|
| `x + y` | `fx.arith.addf(vX, vY)` | Custom kernel |
| `x * y` | `fx.arith.mulf(vX, vY)` | Custom kernel |
| `x - y` | `fx.arith.subf(vX, vY)` | Custom kernel |
| `x / y` | `fx.arith.divf(vX, vY)` | Custom kernel |
| `torch.relu(x)` | `arith.maximumf(val, zero)` | Custom kernel |
| `torch.sigmoid(x)` | `1 / (1 + exp(-x))` using arith + exp2 | Custom kernel |
| `torch.clamp(x, min=a)` | `arith.maximumf(val, a_const)` | Custom kernel |
| `torch.mean(x)` | Parallel reduction (see reduction patterns) | Custom kernel |
| `torch.softmax(x)` | `build_softmax_module()` | **Pre-built** |
| `torch.matmul(A, B)` | `compile_preshuffle_gemm_a8()` | **Pre-built** |
| `nn.Linear` | `compile_preshuffle_gemm_a8()` | **Pre-built** |
| `F.linear` | `compile_preshuffle_gemm_a8()` | **Pre-built** |
| `F.scaled_dot_product_attention` | `build_flash_attn_func_module()` | **Pre-built** |
| `nn.LayerNorm` | `build_layernorm_module()` | **Pre-built** |
| `nn.RMSNorm` | `build_rmsnorm_module()` | **Pre-built** |
| `nn.Conv2d` | `torch.nn.Conv2d` (NO FlyDSL equiv) | PyTorch only |
| `F.max_pool2d` | `torch.nn.functional` (NO FlyDSL equiv) | PyTorch only |
| `nn.BatchNorm2d` | `torch.nn.BatchNorm2d` (NO FlyDSL equiv) | PyTorch only |
