---
layer: "flydsl"
category: "translation"
tags: ["flydsl", "translation", "attention", "flash-attention"]
last_updated: 2026-03-23
---

# Translating Attention Kernels from PyTorch to FlyDSL

## Overview

Attention mechanisms (scaled dot-product attention, multi-head attention) are
the most complex translation targets. Direct translation of the mathematical
formula is possible but inefficient; Flash Attention algorithms are preferred
for production use.

## Why Direct Translation Is Insufficient

The naive attention formula `softmax(Q @ K^T / sqrt(d)) @ V` requires
materializing the full `[seq_len, seq_len]` attention matrix, which:
- Uses O(seq_len^2) memory
- Is bandwidth-bound on GPU
- Cannot fit in LDS for long sequences

## Flash Attention Algorithm

Flash Attention computes attention without materializing the full matrix
by processing Q in tiles and maintaining running statistics:

```
For each Q tile (q_tile):
    Initialize running max = -inf, running sum = 0, output = 0
    For each K,V tile (k_tile, v_tile):
        scores = q_tile @ k_tile^T / sqrt(d)
        new_max = max(running_max, row_max(scores))
        correction = exp(running_max - new_max)
        exp_scores = exp(scores - new_max)
        output = output * correction + exp_scores @ v_tile
        running_sum = running_sum * correction + row_sum(exp_scores)
        running_max = new_max
    output = output / running_sum
```

## FlyDSL Translation Pattern

```python
# PyTorch: output = F.scaled_dot_product_attention(Q, K, V)

@flyc.kernel
def attention_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                     seq_len: int, head_dim: int):
    # Each block handles one query position
    q_pos = flyc.block_id()
    tid = flyc.thread_id() - q_pos * flyc.block_dim()

    # Load Q row into registers
    # Tile over K/V in chunks
    # Maintain running softmax statistics
    # Accumulate weighted V
    # Store final output
    pass  # See full implementation in FlyDSL repo docs

@flyc.jit
def attention_jit(Q, K, V, output, seq_len, head_dim):
    grid = seq_len  # one block per query position
    block = head_dim  # threads process one head dimension
    attention_kernel[grid, block](Q, K, V, output, seq_len, head_dim)
```

## Multi-Head Attention

```python
# PyTorch: output = nn.MultiheadAttention(embed_dim, num_heads)(Q, K, V)

class Model(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Projection weights (stored as parameters)

    def forward(self, Q, K, V):
        batch, seq_len, _ = Q.shape
        # 1. Project Q, K, V through linear layers
        # 2. Reshape to [batch, num_heads, seq_len, head_dim]
        # 3. Run attention per head (can parallelize via batch*num_heads blocks)
        # 4. Concat heads and project output
        pass
```

## Causal Masking

For causal (autoregressive) attention, mask future positions:

```python
# In the attention kernel, when computing scores:
if k_pos > q_pos:
    score = -1e30  # mask out future positions
```

## Memory Staging

- **Q row** (~128 floats for head_dim=128): fits in registers
- **K/V tiles** (tile_size x head_dim): load into shared memory
- **Running statistics** (max, sum per row): kept in registers
- **Output accumulator**: kept in registers, written to global at end

## Key Considerations

- Attention translation is the highest-complexity translation target.
- For correctness validation, a naive implementation (materializing full
  attention matrix) is acceptable.
- Flash Attention optimization should be done post-translation in the
  optimization phase.
- Use `flyc.warp_reduce_max` and `flyc.warp_reduce_sum` for the online
  softmax statistics.
