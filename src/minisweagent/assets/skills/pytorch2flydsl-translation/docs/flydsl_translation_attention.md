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

### CRITICAL: Never Decompose When Flash Attention Fits

If head_dim >= 64, head_dim % 32 == 0, and seq_len % 128 == 0:
**YOU MUST use `build_flash_attn_func_module()`**. Do NOT decompose into
separate GEMM + softmax + GEMM calls. Decomposed attention with Python
for-loops over batch*heads is 5-10x slower than flash attention.

**Anti-pattern (DO NOT DO THIS):**
```python
# BAD: Python loop over batch*heads calling GEMM one at a time
for i in range(batch_size * num_heads):
    gemm_fn(scores[i], Q[i], K[i], ...)
softmax_fn(scores, attn_weights, ...)
for i in range(batch_size * num_heads):
    gemm_fn(output[i], attn_weights[i], V[i], ...)
```

### Strategy 2: Pad to Flash Attention Constraints

When head_dim or seq_len don't meet flash attention constraints, **pad** to
the next valid size, run flash attention, and slice back. NEVER fall back to
`F.scaled_dot_product_attention`.

```python
class Model(nn.Module):
    def __init__(self, num_heads, head_dim, seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.padded_head_dim = ((head_dim + 31) // 32) * 32
        if self.padded_head_dim < 64:
            self.padded_head_dim = 64
        self.padded_seq_len = ((seq_len + 127) // 128) * 128
        self._flash_attn = build_flash_attn_func_module(
            num_heads=num_heads,
            head_dim=self.padded_head_dim,
            causal=True, dtype_str="f16",
        )

    def forward(self, q, k, v):
        B, S, H, D = q.shape
        # Pad head_dim if needed
        if D < self.padded_head_dim:
            pad_d = self.padded_head_dim - D
            q = F.pad(q, (0, pad_d))
            k = F.pad(k, (0, pad_d))
            v = F.pad(v, (0, pad_d))
        # Pad seq_len if needed
        if S < self.padded_seq_len:
            pad_s = self.padded_seq_len - S
            q = F.pad(q, (0, 0, 0, 0, 0, pad_s))
            k = F.pad(k, (0, 0, 0, 0, 0, pad_s))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_s))
        output = torch.empty_like(q)
        self._flash_attn(
            q.contiguous().view(-1), k.contiguous().view(-1),
            v.contiguous().view(-1), output.view(-1),
            B, self.padded_seq_len,
            stream=torch.cuda.current_stream(),
        )
        # Slice back to original dimensions
        return output[:, :S, :, :D]
```

### Strategy 3: Decomposed Attention with Pre-built Kernels

ONLY when padding is impractical (e.g., very large padding ratios),
decompose into FlyDSL pre-built components. NEVER use `F.scaled_dot_product_attention`.

```python
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from kernels.softmax_kernel import build_softmax_module
from tests.utils import shuffle_weight

class Model(nn.Module):
    def forward(self, q, k, v):
        # QK^T via FlyDSL GEMM (after preshuffling K^T)
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

## Causal Masking

The FlyDSL flash attention kernel supports causal masking natively via `causal=True`
in the builder. For decomposed attention, apply the mask before softmax:

```python
mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```

## Full Multi-Head Attention Block Translation

For a full multi-head attention block (e.g., minGPT), replace ALL `nn.Linear`
with FlyDSL preshuffle GEMM:

```python
import torch
import torch.nn as nn
from kernels.flash_attn_func import build_flash_attn_func_module
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from tests.utils import shuffle_weight

class Model(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        head_dim = n_embd // n_head

        # QKV projection — raw nn.Parameter, NOT nn.Linear
        self.c_attn_weight = nn.Parameter(torch.randn(3 * n_embd, n_embd, dtype=torch.float16))
        self.c_attn_bias = nn.Parameter(torch.randn(3 * n_embd, dtype=torch.float16))
        self.register_buffer("c_attn_w_shuffled",
            shuffle_weight(self.c_attn_weight.data.contiguous(), layout=(16, 16)))
        self.c_attn_gemm = compile_preshuffle_gemm_a8(
            M=0, N=3 * n_embd, K=n_embd,
            tile_m=64, tile_n=128, tile_k=128,
            in_dtype="fp16", out_dtype="fp16", lds_stage=2)

        # Output projection — raw nn.Parameter, NOT nn.Linear
        self.c_proj_weight = nn.Parameter(torch.randn(n_embd, n_embd, dtype=torch.float16))
        self.c_proj_bias = nn.Parameter(torch.randn(n_embd, dtype=torch.float16))
        self.register_buffer("c_proj_w_shuffled",
            shuffle_weight(self.c_proj_weight.data.contiguous(), layout=(16, 16)))
        self.c_proj_gemm = compile_preshuffle_gemm_a8(
            M=0, N=n_embd, K=n_embd,
            tile_m=64, tile_n=128, tile_k=128,
            in_dtype="fp16", out_dtype="fp16", lds_stage=2)

        self._flash_attn = build_flash_attn_func_module(
            num_heads=n_head, head_dim=head_dim, causal=True, dtype_str="f16")

    def forward(self, x):
        B, T, C = x.size()
        stream = torch.cuda.current_stream()
        scale = torch.empty(0, device=x.device, dtype=torch.float32)

        # QKV projection via FlyDSL GEMM + bias
        x_2d = x.half().reshape(B * T, C).contiguous()
        qkv = torch.empty(B * T, 3 * C, device=x.device, dtype=torch.float16)
        self.c_attn_gemm(qkv.view(-1), x_2d.view(-1), self.c_attn_w_shuffled.view(-1),
                         scale, scale, B * T, 3 * C, stream)
        qkv = qkv + self.c_attn_bias.unsqueeze(0)
        q, k, v = qkv.view(B, T, 3, self.n_head, C // self.n_head).unbind(dim=2)

        # Flash Attention
        y = torch.empty_like(q)
        self._flash_attn(q.contiguous().view(-1), k.contiguous().view(-1),
                         v.contiguous().view(-1), y.view(-1), B, T, stream=stream)
        y = y.reshape(B * T, C)

        # Output projection via FlyDSL GEMM + bias
        out = torch.empty(B * T, C, device=x.device, dtype=torch.float16)
        self.c_proj_gemm(out.view(-1), y.contiguous().view(-1), self.c_proj_w_shuffled.view(-1),
                         scale, scale, B * T, C, stream)
        out = out + self.c_proj_bias.unsqueeze(0)
        return out.view(B, T, C).float()
```

## Attention Matmul: Preshuffle GEMM vs Flash Attention vs torch.bmm

Preshuffle GEMM (`compile_preshuffle_gemm_a8`) is **weight-stationary**: one operand
(B-matrix) must be a fixed weight that is pre-shuffled once at init time. It is
**not suitable** for attention score computation (Q@K^T, att@V) where both operands
are dynamic activations that change every forward pass.

### Which op for which matmul

| Matmul | Operands | Use |
|--------|----------|-----|
| QKV projection (x @ W_qkv) | x=activation, W=fixed weight | `compile_preshuffle_gemm_a8` (preshuffle W once) |
| Output projection (attn_out @ W_proj) | attn_out=activation, W=fixed weight | `compile_preshuffle_gemm_a8` (preshuffle W once) |
| Q @ K^T (attention scores) | Q=activation, K=activation | `build_flash_attn_func_module` (handles Q@K^T + softmax + @V) |
| att @ V (attention output) | att=activation, V=activation | `build_flash_attn_func_module` (part of flash attention) |
| Activation @ activation (no flash attn fit) | both dynamic | `torch.bmm` as fallback (acceptable for fp32 or non-standard shapes) |

### When torch.bmm is acceptable

`torch.bmm` is an acceptable fallback for **activation-activation matmuls** when:
- Flash attention doesn't apply (non-standard activation function, not softmax-based)
- Both operands vary per batch element (cannot preshuffle either side)
- The matmul is fp32 (FlyDSL preshuffle GEMM only supports fp16/bf16/int8/fp8)

Examples where `torch.bmm` is acceptable:
- ReLU-attention: Q@K^T with ReLU instead of softmax (flash attention only supports softmax)
- Custom attention patterns with non-standard masking
- fp32 batched matmul where both sides are dynamic

### Anti-pattern: DO NOT preshuffle activations

```python
# BAD: Preshuffling K every forward pass
K_shuffled = shuffle_weight(K_transposed, layout=(16, 16))  # expensive, per-batch!
preshuffle_gemm(scores, Q, K_shuffled, ...)  # defeats the purpose of preshuffling
```

Preshuffling is a heavyweight operation designed to be done **once** at init. Calling
it every forward pass adds overhead that far exceeds any GEMM speedup.

## Decision Summary

```
Matmul type?
├── Linear projection (x @ W where W is fixed weight)
│   └── compile_preshuffle_gemm_a8 + nn.Parameter [NO nn.Linear]
│       Preshuffle W once at init. Works for QKV proj, output proj, FFN layers.
├── Attention scores (Q @ K^T → softmax → @ V)
│   ├── Standard SDPA (head_dim>=64, head_dim%32==0, seq%128==0)
│   │   └── build_flash_attn_func_module() [NO F.scaled_dot_product_attention]
│   ├── Non-standard dims
│   │   └── Pad Q/K/V, run flash attention, slice back
│   └── Non-softmax attention (e.g., ReLU-attention)
│       └── torch.bmm is acceptable for activation-activation matmuls
├── Activation @ activation (non-attention, both sides dynamic)
│   └── torch.bmm as fallback (DO NOT preshuffle activations)
└── Causal masking
    └── FlyDSL flash attention supports causal=True natively
```
