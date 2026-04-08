---
layer: "flydsl"
category: "translation"
tags: ["flydsl", "pytorch", "translation", "api-reference"]
last_updated: 2026-03-23
---

# FlyDSL Translation API Reference

This document provides the FlyDSL API surface needed for translating PyTorch kernels.

## Core Concepts

FlyDSL is a Python-embedded DSL for AMD GPU kernels. It compiles via MLIR/ROCm
and runs on AMD Instinct GPUs. Kernels are written in Python using FlyDSL
primitives, then JIT-compiled at first invocation.

## Kernel Structure (Three-Layer Pattern)

Every FlyDSL kernel follows a three-layer structure:

```python
import flycompute as flyc

# Layer 1: Device kernel (runs on GPU threads)
@flyc.kernel
def my_kernel(input_ptr, output_ptr, N: int):
    tid = flyc.thread_id()
    if tid < N:
        output_ptr[tid] = input_ptr[tid] * 2.0

# Layer 2: JIT-compiled launch wrapper
@flyc.jit
def my_jit(input_tensor, output_tensor, N):
    grid = (N + 255) // 256
    my_kernel[grid, 256](input_tensor, output_tensor, N)

# Layer 3: PyTorch nn.Module wrapper (host-side interface)
class Model(torch.nn.Module):
    def forward(self, x):
        output = torch.empty_like(x)
        my_jit(x, output, x.numel())
        return output
```

## Parameter Types

- **Pointer parameters**: `flyc.Pointer[flyc.float32]`, `flyc.Pointer[flyc.float16]`
- **Scalar parameters**: `int`, `float`, Python scalars
- **Tensor parameters**: In `@flyc.jit` functions, pass PyTorch tensors directly

## Thread and Grid Primitives

- `flyc.thread_id()` — global thread ID
- `flyc.block_id()` — block (workgroup) ID
- `flyc.block_dim()` — threads per block
- `flyc.grid_dim()` — total blocks in grid

## Memory Operations

- `flyc.load(ptr, offset)` — load from global memory
- `flyc.store(ptr, offset, value)` — store to global memory
- `flyc.shared_memory(shape, dtype)` — allocate shared (LDS) memory
- `flyc.syncthreads()` — block-level barrier

## Math Operations

- `flyc.exp(x)`, `flyc.log(x)`, `flyc.sqrt(x)`
- `flyc.max(a, b)`, `flyc.min(a, b)`, `flyc.abs(x)`
- `flyc.sigmoid(x)` — 1 / (1 + exp(-x))
- `flyc.tanh(x)`

## Reduction Primitives

- `flyc.atomic_add(ptr, offset, value)` — atomic addition to global memory
- `flyc.warp_reduce_sum(value)` — warp-level sum reduction
- `flyc.warp_reduce_max(value)` — warp-level max reduction

## Launch Configuration

```python
# Launch: kernel[grid_size, block_size](args...)
my_kernel[grid, block](ptr, N)

# 2D grid:
my_kernel[(grid_x, grid_y), (block_x, block_y)](ptr, M, N)
```

## PyTorch Interop

FlyDSL kernels receive PyTorch tensors in `@flyc.jit` functions:
- `tensor.data_ptr()` — raw device pointer (implicit in kernel calls)
- `tensor.shape`, `tensor.stride()` — layout info
- `tensor.dtype` — element type

The `Model(nn.Module)` wrapper provides the same interface as the PyTorch
original, making it a drop-in replacement for correctness testing.

## Common PyTorch -> FlyDSL Op Mapping

| PyTorch | FlyDSL |
|---------|--------|
| `x + y` (element-wise) | `flyc.load + flyc.store` per element |
| `torch.relu(x)` | `flyc.max(val, 0.0)` |
| `torch.sigmoid(x)` | `flyc.sigmoid(val)` |
| `x * torch.sigmoid(x)` (swish) | `val * flyc.sigmoid(val)` |
| `torch.sum(x)` | Parallel reduction with `flyc.warp_reduce_sum` |
| `torch.mean(x)` | Sum reduction / N |
| `torch.matmul(A, B)` | Tiled GEMM with shared memory |
