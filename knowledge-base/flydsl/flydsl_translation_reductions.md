---
layer: "flydsl"
category: "translation"
tags: ["flydsl", "translation", "reductions", "softmax", "layernorm"]
last_updated: 2026-03-23
---

# Translating Reductions from PyTorch to FlyDSL

## Overview

Reduction operations (sum, mean, softmax, layer norm) are among the most
challenging translations because PyTorch hides the parallel reduction
complexity behind single function calls.

## Parallel Reduction Pattern

### Wave-Level Reduction

AMD GPUs execute 64 threads per wavefront. Use wave-level primitives first:

```python
@flyc.kernel
def sum_kernel(x_ptr, partial_ptr, N: int):
    tid = flyc.thread_id()
    bid = flyc.block_id()
    bdim = flyc.block_dim()
    local_id = tid - bid * bdim

    # Each thread loads one element
    val = flyc.load(x_ptr, tid) if tid < N else 0.0

    # Wave-level reduction (64 threads)
    wave_sum = flyc.warp_reduce_sum(val)

    # First thread in each wave writes to shared memory
    smem = flyc.shared_memory((bdim // 64,), flyc.float32)
    wave_id = local_id // 64
    if local_id % 64 == 0:
        flyc.store(smem, wave_id, wave_sum)
    flyc.syncthreads()

    # First wave reduces across waves
    if local_id < bdim // 64:
        partial = flyc.load(smem, local_id)
        block_sum = flyc.warp_reduce_sum(partial)
        if local_id == 0:
            flyc.atomic_add(partial_ptr, bid, block_sum)
```

### Multi-Pass Reduction

For large tensors, use a two-pass approach:
1. First pass: each block reduces to a partial sum
2. Second pass: reduce partial sums to final result

## Translating `torch.softmax`

Softmax requires three passes: max, exp-sum, normalize.

```python
# PyTorch: output = torch.softmax(x, dim=-1)

# FlyDSL: row-wise softmax (each block handles one row)
@flyc.kernel
def softmax_kernel(x_ptr, out_ptr, cols: int):
    row = flyc.block_id()
    tid = flyc.thread_id() - row * flyc.block_dim()
    row_offset = row * cols

    # Pass 1: find row max (for numerical stability)
    max_val = -1e30
    for i in range(tid, cols, flyc.block_dim()):
        val = flyc.load(x_ptr, row_offset + i)
        max_val = flyc.max(max_val, val)
    max_val = flyc.warp_reduce_max(max_val)
    # (cross-wave reduction via shared memory omitted for brevity)

    # Pass 2: compute exp(x - max) and sum
    exp_sum = 0.0
    for i in range(tid, cols, flyc.block_dim()):
        val = flyc.load(x_ptr, row_offset + i)
        exp_val = flyc.exp(val - max_val)
        flyc.store(out_ptr, row_offset + i, exp_val)
        exp_sum += exp_val
    exp_sum = flyc.warp_reduce_sum(exp_sum)

    flyc.syncthreads()

    # Pass 3: normalize
    for i in range(tid, cols, flyc.block_dim()):
        val = flyc.load(out_ptr, row_offset + i)
        flyc.store(out_ptr, row_offset + i, val / exp_sum)
```

## Translating `torch.mean` / `torch.sum`

For global reductions:

```python
# PyTorch: output = torch.mean(x)
# FlyDSL: two-pass global reduction

@flyc.jit
def mean_jit(x, output, N):
    block = 256
    grid = (N + block - 1) // block
    partials = torch.zeros(grid, device=x.device)
    sum_kernel[grid, block](x, partials, N)
    # Final reduction of partials
    final_sum = partials.sum()
    output.fill_(final_sum / N)
```

## Translating Layer Norm / RMS Norm

Layer norm combines mean, variance, and normalization:

```python
# PyTorch: output = F.layer_norm(x, [hidden_dim])
# FlyDSL: each block handles one row (batch element)
# 1. Compute mean (reduction)
# 2. Compute variance (reduction)
# 3. Normalize: (x - mean) / sqrt(variance + eps) * weight + bias
```

## Key Considerations

- **Shared memory (LDS)**: AMD MI300X has 64 KB LDS per workgroup. Use for
  cross-wave communication within a block.
- **Wavefront size**: AMD uses 64-thread wavefronts (not 32 like NVIDIA warps).
- **Synchronization**: `flyc.syncthreads()` synchronizes all threads in a block.
  There is no global synchronization — use multiple kernel launches.
- **Numerical stability**: Always subtract the max before `exp()` in softmax.
