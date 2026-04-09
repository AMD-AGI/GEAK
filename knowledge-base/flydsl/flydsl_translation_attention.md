---
layer: "flydsl"
category: "translation"
tags: ["flydsl", "translation", "attention", "transformer", "flash-attention"]
last_updated: 2026-03-23
---

# FlyDSL Translation: Attention Patterns

## FlyDSL Has Flash Attention

FlyDSL provides a high-performance Flash Attention kernel in `kernels/flash_attn_func.py`.
This is an MFMA32-based implementation with online softmax, LDS prefetch, and XOR swizzle.
**Always use it instead of PyTorch `F.scaled_dot_product_attention`.**

### Strategy 1: Pre-built Flash Attention (Preferred)

```python
from kernels.flash_attn_func import build_flash_attn_func_module

class Model(nn.Module):
    def __init__(self, num_heads, head_dim, seq_len):
        super().__init__()
        self._flash_attn = build_flash_attn_func_module(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=True,           # set False for non-causal
            dtype_str="f16",       # or "bf16"
        )

    def forward(self, q, k, v):
        # q, k, v: (batch, seq_len, num_heads, head_dim) — BSHD layout
        B, S, H, D = q.shape
        output = torch.empty_like(q)
        # Flatten to 1D (BSHD contiguous layout)
        self._flash_attn(
            q.contiguous().view(-1),
            k.contiguous().view(-1),
            v.contiguous().view(-1),
            output.view(-1),
            B, S,                              # batch_size, seq_len
            stream=torch.cuda.current_stream(),
        )
        return output
```

**Builder signature:**
```python
build_flash_attn_func_module(
    num_heads: int,       # number of attention heads
    head_dim: int,        # dimension per head (>= 64, % 32 == 0)
    causal: bool = True,  # causal masking
    dtype_str: str = "f16",  # "f16" or "bf16"
    waves_per_eu: int = 2,
)
```

**Launcher signature** (returned function):
```python
launcher(Q_flat, K_flat, V_flat, O_flat, batch_size, seq_len, stream=None)
```

Note: `num_heads` is baked in at build time. The launcher only takes `batch_size`
and `seq_len` as runtime parameters (not `num_heads`).

**Constraints:**
- `head_dim % 32 == 0` and `head_dim >= 64`
- `seq_len % 128 == 0`
- Q/K/V/O must be contiguous 1D (BSHD flattened layout)
- Supports f16 and bf16
- Auto-selects BLOCK_M (128 or 256) based on num_heads

### Strategy 2: Decomposed Attention with Pre-built Kernels

When flash attention constraints aren't met, decompose into pre-built components:

```python
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from kernels.softmax_kernel import build_softmax_module
from tests.utils import shuffle_weight

class Model(nn.Module):
    def forward(self, q, k, v):
        # QK^T via FlyDSL GEMM (after preshuffling K)
        scores = ...  # use compile_preshuffle_gemm_a8

        # Softmax via FlyDSL
        scores_2d = scores.reshape(-1, N)
        softmax_fn = build_softmax_module(scores_2d.shape[0], N, "f32")
        attn_weights = torch.empty_like(scores_2d)
        softmax_fn(scores_2d, attn_weights, scores_2d.shape[0],
                   stream=torch.cuda.current_stream())

        # V projection via FlyDSL GEMM
        return ...  # use compile_preshuffle_gemm_a8
```

### Strategy 3: PyTorch Fallback (Last Resort)

Only use `F.scaled_dot_product_attention` when:
- head_dim < 64 or head_dim not divisible by 32
- seq_len not divisible by 128
- Batched attention with irregular shapes

```python
attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

## Causal Masking

The FlyDSL flash attention kernel supports causal masking natively via `causal=True`
in the builder. For decomposed attention, apply the mask before softmax:

```python
mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```

## Full Multi-Head Attention Block Translation

For a full multi-head attention block (e.g., minGPT):

```python
import torch
import torch.nn as nn
from kernels.flash_attn_func import build_flash_attn_func_module

class Model(nn.Module):
    def __init__(self, n_embd, n_head, block_size, ...):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # QKV projection
        self.c_proj = nn.Linear(n_embd, n_embd)       # output projection
        head_dim = n_embd // n_head
        self._flash_attn = build_flash_attn_func_module(
            num_heads=n_head,
            head_dim=head_dim,
            causal=True,
            dtype_str="f16",
        )

    def forward(self, x):
        B, T, C = x.size()
        # QKV projection (nn.Linear kept for simplicity, or replace with GEMM)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        head_dim = C // self.n_head
        # Reshape to BSHD layout
        q = q.view(B, T, self.n_head, head_dim).half()
        k = k.view(B, T, self.n_head, head_dim).half()
        v = v.view(B, T, self.n_head, head_dim).half()

        # Attention via FlyDSL Flash Attention
        y = torch.empty_like(q)
        self._flash_attn(
            q.contiguous().view(-1),
            k.contiguous().view(-1),
            v.contiguous().view(-1),
            y.view(-1),
            B, T,
            stream=torch.cuda.current_stream(),
        )
        y = y.float().view(B, T, C)

        # Output projection
        return self.c_proj(y)
```

## Decision Summary

```
Attention pattern?
├── Standard SDPA (head_dim>=64, head_dim%32==0, seq%128==0)
│   └── Use build_flash_attn_func_module() from kernels.flash_attn_func
├── Decomposed Q@K, softmax, @V
│   └── Use compile_preshuffle_gemm_a8 + build_softmax_module
├── Non-standard shapes (head_dim<64, seq not %128)
│   └── F.scaled_dot_product_attention (PyTorch fallback, last resort)
└── Causal attention
    └── FlyDSL flash attention supports causal=True natively
```
