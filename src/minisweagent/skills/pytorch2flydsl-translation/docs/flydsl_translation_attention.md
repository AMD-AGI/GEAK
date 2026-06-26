---
layer: "flydsl"
category: "translation"
tags: ["flydsl", "translation", "attention", "transformer", "flash-attention", "mla", "decode", "paged-kv"]
last_updated: 2026-06-10
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

ONLY when padding is impractical (e.g., very large padding ratios, paged KV),
decompose into FlyDSL pre-built components. NEVER use `F.scaled_dot_product_attention`.

Use a **mixed strategy**:
- **`hgemm_splitk_`** for activation@activation matmuls (`Q@K^T`, `attn@V`).
- **`compile_preshuffle_gemm_a8`** for fixed-weight projections (`x@W_qkv`, `out@W_proj`).
- Do **not** preshuffle dynamic K/V cache tensors each forward.
- For paged decode (MLA latent cache or PagedAttention K/V cache), see § Decode Attention
  below: wrap a matching prebuilt fused kernel when one exists, otherwise decompose (batch
  KV gather, pre-scale Q, stacked `batch*nheads`, f16 softmax, `SPLIT_K=1` to start;
  page-tiled online softmax for long context). Do not preshuffle dynamic K/V.

```python
import torch
import torch.nn as nn
from kernels.hgemm_splitk import hgemm_splitk_
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from kernels.softmax_kernel import build_softmax_module
from tests.utils import shuffle_weight

class Model(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # Fixed-weight projection path -> preshuffle GEMM
        self.w_qkv = nn.Parameter(torch.randn(3 * n_embd, n_embd, dtype=torch.float16))
        self.register_buffer("w_qkv_shuffled", shuffle_weight(self.w_qkv.data.contiguous(), layout=(16, 16)))
        self.qkv_gemm = compile_preshuffle_gemm_a8(
            M=0, N=3 * n_embd, K=n_embd,
            tile_m=64, tile_n=128, tile_k=128,
            in_dtype="fp16", out_dtype="fp16", lds_stage=2
        )

    def forward(self, q, k, v):
        # (Optional) projection example: x -> qkv via preshuffle GEMM
        # self.qkv_gemm(x_2d.view(-1), self.w_qkv_shuffled.view(-1), ..., stream=stream)

        # QK^T: C = Q @ K^T  —  q: (M, K), k: (N, K) with N=seq_len
        hgemm_splitk_(scores, q_flat, k, hgemm_kwargs=self._gemm_kwargs, stream=stream)

        # Softmax via FlyDSL
        self._softmax(scores, attn, M, stream=stream)

        # attn @ V: v_t = v.t()  —  (V_dim, seq_len)
        hgemm_splitk_(out, attn, v_t, hgemm_kwargs=self._gemm_kwargs, stream=stream)
        return out
```

See `flydsl_translation_gemm.md` § Split-K GEMM for shapes, tile config, and MLA examples.

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
| Activation @ activation (no flash attn fit) | both dynamic, fp16/bf16 | `hgemm_splitk_` — see `flydsl_translation_gemm.md` § Split-K GEMM |
| Activation @ activation (fp32 or rare shapes) | both dynamic | `torch.bmm` only if FlyDSL path unavailable |

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

## Decode Attention (Paged: MLA & PagedAttention)

Decode-mode attention reads a **paged KV cache** through a `block_table` /
`page_table` with `seqlen_q == 1`. This covers MLA (latent cache) and PagedAttention
(standard / GQA KV cache); both share the same decomposed FlyDSL strategy and
optimizations. This is **not** standard BSHD flash attention — do **not** use
`build_flash_attn_func_module` for it.

- **MLA** reads a latent cache where each row is a single compressed vector: K uses the
  full `headdim_qk`, and V is the **leading `headdim_v` slice of that same row**
  (`headdim_v <= headdim_qk`). Signals: a `MultiHeadLatentAttention`-style module,
  `kv_cache` + `block_table` + `cache_seqlens`, asymmetric `headdim_qk` / `headdim_v`.
- **PagedAttention** reads separate `k_cache`/`v_cache` with symmetric `headdim` and may
  use GQA (`nheads_q % nheads_kv == 0`). Reconstruct the cache and expand KV heads by the
  group count in a single batched gather before the GEMMs.

Define shape symbols from the source (not a fixed benchmark): `B`=batch, `Sq`=seqlen_q,
`H`=nheads, `Dqk`/`Dv`=QK/V head dims (equal for PagedAttention), `T`=cache length,
`M_row = H*Sq`, `M_tot = B*M_row`.

**Do NOT wrap the kernel in a CUDA graph to report the speedup.** Graph capture only
removes host-side launch overhead — it is not a FlyDSL kernel improvement, and it makes
the comparison against the (non-captured) PyTorch baseline misleading. Measure the FlyDSL
kernel WITHOUT CUDA graphs so the uplift reflects the translation itself. The real
kernel-level wins come from the optimizations below.

### Strategy

1. **Reuse a prebuilt fused MLA kernel when one matches the shape.** Fused MLA
   kernels bake head count, head dims, page size, and dtype in at compile time, so
   they only apply when the source shape matches. When one fits, wrap its launcher
   (build/cache any metadata buffers in `__init__`) instead of writing a kernel.
2. **Otherwise decompose** with `hgemm_splitk_` (for `Q@K^T` and `attn@V`) +
   `build_softmax_module`, applying the optimizations below.
3. **For long context / memory pressure**, use a **page-tiled online softmax** so the
   full `(rows, seq)` attention matrix is never materialized (see below).

### Decomposed path: key optimizations

Decode (`Sq == 1`) MLA is **memory-bound** and **launch-bound** — `M_tot = B*M_row`
is tiny (e.g. 64 rows), so runtime is dominated by the **number of kernel launches**
and KV-cache bytes read, not FLOPs. A naive per-batch decomposition stalls at a few×;
the items below are what take it to ~10×. The first three are the highest-leverage.

| Optimization | What to do |
|--------------|------------|
| **Stack the softmax (biggest win)** | Write every batch's QK result into **one** `(M_tot, T)` scores buffer and run a **single** `build_softmax_module(M_tot, T)` over all `B*M_row` rows. Even when a per-batch `block_table` forces per-batch `Q@K^T` / `attn@V` GEMMs, the softmax must stay one stacked launch — **never** call `build_softmax_module(M_row, T)` once per batch inside the loop. |
| **Gather + transpose outside the loop** | When `cache_seqlens` are uniform, gather all KV once via `block_table` → `(B, T, Dqk)`, and build the batched `V^T` → `(B, Dv, T)` **before** the GEMM loop. Never `.t().contiguous()` / `.contiguous()` per batch inside the loop. |
| **Fewer launches** | Aim for ~`2B + 1` launches (per-batch QK + 1 stacked softmax + per-batch PV), not `~3B` + extra copies. Per-batch loops are only acceptable for the QK/PV GEMMs (because K differs per request), not for softmax or layout ops. |
| **No host sync** | Index `block_table` / `cache_seqlens` on the GPU; never `.item()` / `.tolist()` inside the loop. |
| **Pre-scale Q once** | Fold `1/sqrt(Dqk)` into Q before `Q@K^T`; avoid per-batch `scores.float().mul_(scale)`. |
| **Right output dtype + no layout copy** | Allocate the output in the GEMM dtype and write GEMM results straight into it; avoid a trailing `output.to(dtype)` cast. When `Sq == 1` the `(B, M_row, Dv)` GEMM output is already the `(B, Sq, H, Dv)` layout — reshape, don't `transpose(...).contiguous()`. |
| **Match dtypes** | Use an f16/bf16 `build_softmax_module` to match `hgemm_splitk_` I/O; reserve fp32 accumulation for the online path. |
| **Tune tiles from shapes** | Derive `(M, N, K)` per GEMM (QK: `(M_tot, T, Dqk)`, PV: `(M_row, Dv, T)`); start `SPLIT_K=1` at small M and profile up. When `N` (=`T` or `Dv`) is a multiple of 128, prefer `TILE_N=128` over 64 — this is typically the ~7×→~10× jump for decode. |
| **Reuse buffers** | One `(M, seq)` buffer for scores→attn (in-place softmax); skip the mask entirely when `Sq == 1`. |
| **Lazy-compile** | Cache compiled softmax / GEMM modules keyed by shape to avoid re-JIT every forward. |

Concrete decode skeleton (uniform `T`; this stacked structure is what reaches ~10×):

```python
# Sq == 1, M_row = H, M_tot = B*M_row
q_scaled = (q.half() * scale).reshape(B, M_row, Dqk)      # pre-scale once
kv = gather_pages(kv_cache, block_table, T)              # (B, T, Dqk) — one gather
vt = kv[:, :, :Dv].transpose(1, 2).contiguous()         # (B, Dv, T) — batched once

scores = empty(M_tot, T, fp16)
for b in range(B):                                      # per-batch QK only (K differs)
    hgemm_splitk_(scores[b*M_row:(b+1)*M_row], q_scaled[b], kv[b], hgemm_kwargs=qk_kwargs)

softmax_fn(scores, attn, M_tot)                          # ONE stacked softmax launch

for b in range(B):                                      # per-batch PV only
    hgemm_splitk_(out[b*M_row:(b+1)*M_row], attn[b*M_row:(b+1)*M_row], vt[b], hgemm_kwargs=pv_kwargs)
```

For prefill (`Sq > 1`), `M_row = H*Sq` grows and the causal mask is live; raise
`TILE_M` and re-profile `SPLIT_K` for the larger row count.

### Page-tiled online softmax (long context)

Iterate the KV cache in page-sized tiles and keep a running softmax state per row, so
no full-sequence attention buffer is allocated:

```python
m = full((M, 1), -inf, f32)   # running max
l = zeros((M, 1), f32)        # running exp-sum
o = zeros((M, Dv), f32)       # running output

for each page tile t:
    K_t, V_t = load_tile_from_paged_cache(...)
    scores_t = hgemm_splitk(Q, K_t) * scale
    m_new = maximum(m, rowmax(scores_t))
    alpha = exp(m - m_new)
    p_t   = exp(scores_t - m_new)
    l = alpha * l + rowsum(p_t)
    o = alpha * o + hgemm_splitk(p_t, V_t^T)
    m = m_new

out = o / l
```

Keep `m/l/o` in fp32 here. A single fused `@flyc.kernel` that runs this loop
internally (page loop + online `m/l/o` + PV) is usually the largest win for decode,
since it collapses the whole attention into one launch.

### Anti-patterns

1. Using flash attention (`build_flash_attn_func_module`) for paged MLA — wrong API and layout.
2. Per-batch **softmax** or **layout** ops inside the loop (`for b: build_softmax_module(M_row, T)`, `for b: v.t().contiguous()`) — stack the softmax into one `(M_tot, T)` launch and batch the gather/transpose outside the loop. (Per-batch QK/PV GEMMs are fine because K differs per request; softmax and layout are not.)
3. Host syncs in `forward` (`.item()`, `.tolist()`) that serialize host↔device every step.
4. Preshuffling K/V with `shuffle_weight` every forward — preshuffle is for fixed weights only.
5. Standalone scale / mask passes and double KV copies instead of folding them in.
6. f32 softmax feeding an f16 GEMM in the decomposed path (avoidable dtype cast).
7. Forcing a shape-specialized fused kernel onto a different shape by editing its baked-in constants — that is a different kernel; decompose or write a shape-appropriate kernel instead.

### Reference

- Split-K GEMM shapes and tile config for the decomposed path: `flydsl_translation_gemm.md` § Split-K GEMM.

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
│   ├── Paged decode (seqlen_q=1: MLA kv_cache or PagedAttention k/v_cache + block_table)
│   │   ├── Prebuilt fused kernel matches the shape → wrap its launcher [see § Decode Attention]
│   │   └── No match → decompose: hgemm_splitk_ + build_softmax_module [see § Decode Attention]
│   ├── Flash infeasible (paged KV, non-BSHD)
│   │   ├── Baseline: hgemm_splitk_ + build_softmax_module
│   │   └── Preferred: page-tiled online softmax (`m/l/o` + tile hgemm_splitk_)
│   │       [see § Decode Attention; NO shuffle_weight on K/V]
│   └── Non-softmax attention (e.g., ReLU-attention)
│       └── hgemm_splitk_ or torch.bmm for activation-activation matmuls
├── Activation @ activation (non-attention, both sides dynamic)
│   └── hgemm_splitk_ (fp16/bf16); torch.bmm only if FlyDSL unavailable
└── Causal masking
    └── FlyDSL flash attention supports causal=True natively
```
