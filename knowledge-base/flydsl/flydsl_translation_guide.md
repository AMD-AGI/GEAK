---
layer: "flydsl"
category: "translation"
tags: ["flydsl", "pytorch", "translation", "guide", "op-mapping"]
last_updated: 2026-03-23
---

# PyTorch to FlyDSL Translation Guide

## Overview

This guide covers the systematic process of translating PyTorch `nn.Module`
kernels to FlyDSL. The key transformation is converting high-level PyTorch
tensor operations into explicit GPU thread-parallel code.

## Step 1: Analyze the PyTorch Kernel

Identify the computational pattern:
1. **Element-wise**: Each output element depends only on the corresponding input element(s)
2. **Reduction**: Output has fewer elements than input (sum, mean, softmax)
3. **GEMM**: Matrix multiplication patterns
4. **Attention**: Scaled dot-product attention variants

## Step 2: Map the Interface

PyTorch kernels in KernelBench follow this interface:

```python
class Model(torch.nn.Module):
    def __init__(self, *init_args):
        super().__init__()
        # Store parameters

    def forward(self, *args) -> torch.Tensor:
        # Computation
        return result

def get_inputs() -> list:
    return [torch.randn(batch, dim).cuda()]

def get_init_inputs() -> list:
    return []
```

The FlyDSL translation MUST preserve this exact interface so the test harness
can compare outputs.

## Step 3: Translate Operations

### Element-wise Operations

PyTorch broadcasts automatically. In FlyDSL, each thread handles one or more elements:

```python
# PyTorch: output = x * torch.sigmoid(x)
# FlyDSL:
@flyc.kernel
def swish_kernel(x_ptr, out_ptr, N: int):
    tid = flyc.thread_id()
    if tid < N:
        val = flyc.load(x_ptr, tid)
        out = val * flyc.sigmoid(val)
        flyc.store(out_ptr, tid, out)
```

### Reduction Operations

Reductions require parallel reduction patterns:

```python
# PyTorch: output = torch.mean(x)
# FlyDSL: parallel sum then divide by N
# See flydsl_translation_reductions.md for detailed patterns
```

### Key Differences from PyTorch

| Aspect | PyTorch | FlyDSL |
|--------|---------|--------|
| Parallelism | Implicit (CUDA backend) | Explicit (thread IDs) |
| Memory | Automatic allocation | Manual `torch.empty_like` + pointer ops |
| Broadcasting | Automatic | Manual indexing |
| Reductions | Single call | Multi-pass parallel reduction |
| Dtype handling | Automatic casting | Explicit pointer types |

## Step 4: Handle Host-Side Bridging

The `@flyc.jit` function bridges PyTorch tensors to kernel arguments:

```python
@flyc.jit
def swish_jit(x, output, N):
    block_size = 256
    grid_size = (N + block_size - 1) // block_size
    swish_kernel[grid_size, block_size](x, output, N)

class Model(torch.nn.Module):
    def forward(self, x):
        output = torch.empty_like(x)
        swish_jit(x, output, x.numel())
        return output
```

## Common Pitfalls

1. **Off-by-one in grid size**: Always use `(N + block_size - 1) // block_size`
2. **Missing bounds check**: Always guard with `if tid < N` in kernels
3. **Forgetting output allocation**: PyTorch allocates implicitly; FlyDSL needs explicit `torch.empty_like`
4. **Wrong pointer type**: Match `flyc.Pointer[dtype]` to tensor dtype
5. **Missing synchronization**: Use `flyc.syncthreads()` after shared memory writes
6. **Incorrect reduction**: Parallel reductions need wave-level primitives, not serial loops

## Worked Example: ReLU

**PyTorch:**
```python
class Model(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)
```

**FlyDSL:**
```python
import flycompute as flyc
import torch

@flyc.kernel
def relu_kernel(x_ptr, out_ptr, N: int):
    tid = flyc.thread_id()
    if tid < N:
        val = flyc.load(x_ptr, tid)
        flyc.store(out_ptr, tid, flyc.max(val, 0.0))

@flyc.jit
def relu_jit(x, output, N):
    block = 256
    grid = (N + block - 1) // block
    relu_kernel[grid, block](x, output, N)

class Model(torch.nn.Module):
    def forward(self, x):
        output = torch.empty_like(x)
        relu_jit(x, output, x.numel())
        return output

def get_inputs():
    return [torch.randn(16, 16384).cuda()]

def get_init_inputs():
    return []
```
