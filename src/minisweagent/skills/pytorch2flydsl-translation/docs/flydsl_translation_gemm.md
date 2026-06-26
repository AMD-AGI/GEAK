---
layer: "flydsl"
category: "translation"
tags: ["flydsl", "translation", "gemm", "matmul", "linear", "hgemm", "splitk", "decode"]
last_updated: 2026-06-09
---

# FlyDSL Translation: GEMM / Matrix Multiplication

## Always Use FlyDSL Pre-built GEMM

FlyDSL provides highly optimized GEMM kernels. **Do NOT fall back to PyTorch
`torch.matmul` / `F.linear` / `nn.Linear` when FlyDSL GEMM is available.**

### Primary: Preshuffle GEMM

```python
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from tests.utils import shuffle_weight

# Compile a GEMM launcher (JIT-compiled on first call)
launch_fn = compile_preshuffle_gemm_a8(
    M=0, N=N, K=K,             # M=0 for dynamic batch size
    tile_m=64, tile_n=128, tile_k=128,  # tile sizes
    in_dtype="fp16",            # "fp8", "int8", "int4", "fp16", "bf16", "fp4"
    out_dtype="fp16",           # "fp16" or "bf16"
    lds_stage=2,                # ping-pong LDS (tuned)
)

# B-matrix MUST be preshuffled (done once, e.g. in __init__):
B_shuffled = shuffle_weight(B.contiguous(), layout=(16, 16))

# Launch call — ALL tensors must be .view(-1) (flattened to 1D):
C = torch.empty(M, N, device=x.device, dtype=torch.float16)
scale_a = torch.empty(0, device=x.device, dtype=torch.float32)
scale_b = torch.empty(0, device=x.device, dtype=torch.float32)
launch_fn(
    C.contiguous().view(-1),
    A.contiguous().view(-1),
    B_shuffled.contiguous().view(-1),
    scale_a, scale_b,
    M, N,
    torch.cuda.current_stream(),
)
```

### CRITICAL: Weight Preshuffling

The preshuffle GEMM **requires** B in a permuted layout. Use `shuffle_weight`:

```python
from tests.utils import shuffle_weight

# For fp16/bf16 weights:
weight_shuffled = shuffle_weight(weight.contiguous(), layout=(16, 16))

# For int8 weights:
weight_shuffled = shuffle_weight(weight_i8.contiguous(), layout=(16, 16))
```

`shuffle_weight` permutes the weight tensor in blocks of (16, 32) — the N-dimension
is split into blocks of 16 rows, and K into blocks of 32 elements. This matches the
MFMA tile register layout for maximum throughput.

**You MUST call `shuffle_weight` once in `__init__` and cache the result.** Do NOT
call it in every `forward()` pass.

### Scales for Non-quantized GEMM

For fp16/bf16, scale tensors are unused but still required as arguments. Use empty tensors:

```python
scale_a = torch.empty(0, device=device, dtype=torch.float32)
scale_b = torch.empty(0, device=device, dtype=torch.float32)
```

### Supported Data Types

| `in_dtype` | A type | B type | C type | Notes |
|-----------|--------|--------|--------|-------|
| `"fp16"` | fp16 | fp16 | fp16 | Default for most translations |
| `"bf16"` | bf16 | bf16 | bf16 | |
| `"fp8"` | fp8 | fp8 | fp16 | With per-token scaling |
| `"int8"` | int8 | int8 | int32 | |
| `"int4"` | int8 | int4(packed) | int32 | W4A8 quantization |
| `"fp4"` | fp8 | fp4 | fp16 | Requires gfx950 (MI350) |

### Tile Configuration Guide

| M range | Recommended `tile_m` | Notes |
|---------|---------------------|-------|
| 1-16 | 16 | Small batch |
| 16-64 | 32 or 64 | Medium batch |
| 64+ | 64 or 128 | Large batch |

`tile_n`: 128. `tile_k`: 128 for fp16/bf16, 256 for fp8/int8. Use `lds_stage=2`.

### Bias and Activation After GEMM

`compile_preshuffle_gemm_a8` computes `C = A @ B` only. It does **not** support
fused bias or activation epilogues. When the original PyTorch code includes
bias addition or activation (e.g. `F.relu(F.linear(x, w, b))`), handle them
as separate operations after the GEMM:

- **Bias**: add via a simple `@flyc.kernel` or `torch.add`
- **Activation**: apply via a `@flyc.kernel` (e.g. `arith.maximumf` for ReLU)
- **Fused bias+activation**: write a single `@flyc.kernel` that computes
  `output = max(0, gemm_output + bias)` in one pass

### Alternative: hgemm_splitk (FP16/BF16 Split-K GEMM)

For **dynamic activation × activation** matmuls (especially **small M**, e.g.
decode with `M = seqlen_q * num_heads`), use `hgemm_splitk_` instead of
preshuffle GEMM. It does **not** require `shuffle_weight`; B can change every
forward (paged KV, attention scores).

Use when:
- Both operands are activations (not a fixed weight to preshuffle once)
- `M` is small and standard preshuffle `tile_m` under-fills the GPU
- Flash attention does not apply (paged cache, MLA, non-BSHD layout)

**Full API, constraints, tile guide, and attention examples** are documented in
the [§ Split-K GEMM (hgemm_splitk)](#split-k-gemm-hgemm_splitk-dynamic-activations--small-m-decode)
section below.

### Constraints (preshuffle GEMM only)

The following apply to `compile_preshuffle_gemm_a8`, **not** to `hgemm_splitk_`:

- `tile_k * elem_bytes` must be divisible by 64
- `M` and `N` can be 0 (dynamic) — resolved at launch time
- B must be preshuffled with `shuffle_weight(b, layout=(16, 16))`
- Scale tensors required (use `torch.empty(0)` for non-quantized)
- All tensor args must be `.view(-1)` (flattened 1D)

## Complete nn.Linear Translation Example

```python
import torch
import torch.nn as nn
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from tests.utils import shuffle_weight

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float16))
        self._gemm = None
        self._weight_shuffled = None

    def forward(self, x):
        x = x.half()  # ensure fp16
        M = x.shape[0]
        N, K = self.weight.shape  # out_features, in_features

        if self._gemm is None:
            self._gemm = compile_preshuffle_gemm_a8(
                M=0, N=N, K=K,
                tile_m=64, tile_n=128, tile_k=128,
                in_dtype="fp16", out_dtype="fp16", lds_stage=2,
            )
            self._weight_shuffled = shuffle_weight(
                self.weight.data.contiguous(), layout=(16, 16)
            )

        output = torch.empty(M, N, device=x.device, dtype=torch.float16)
        scale = torch.empty(0, device=x.device, dtype=torch.float32)
        self._gemm(
            output.contiguous().view(-1),
            x.contiguous().view(-1),
            self._weight_shuffled.contiguous().view(-1),
            scale, scale,
            M, N,
            torch.cuda.current_stream(),
        )
        # Add bias separately
        output = output + self.bias.unsqueeze(0)

        return output

def get_inputs():
    return [torch.randn(1024, 4096, device="cuda")]

def get_init_inputs():
    return [4096, 4096]
```

## Split-K GEMM (hgemm_splitk): Dynamic Activations / Small-M Decode

Use `hgemm_splitk` from `kernels.hgemm_splitk` when **both operands are dynamic
activations** (not fixed weights) and preshuffle GEMM does not apply.

| Scenario | Use |
|----------|-----|
| `nn.Linear`, fixed weight `W` | Preshuffle GEMM (`compile_preshuffle_gemm_a8` + `shuffle_weight`, once in `__init__`) — see above |
| Standard SDPA (contiguous Q/K/V, head_dim/seq constraints) | `build_flash_attn_func_module()` |
| **Activation @ activation**, small **M** (decode, few rows) | **`hgemm_splitk_`** |
| **Activation @ activation**, both sides change every forward (paged KV, attention scores) | **`hgemm_splitk_`** |
| Large M, static B weight | Preshuffle GEMM |

**Do NOT** call `shuffle_weight` on K/V every forward pass to force preshuffle GEMM.
Preshuffle is weight-stationary; per-forward shuffling defeats its purpose.

Typical shapes: decode attention (`seqlen_q=1`, `M = seqlen_q * num_heads` small),
MLA with paged KV cache, batched matmul where B varies per batch element.

**Exception:** For paged decode (MLA latent cache with asymmetric qk/v dims, or
PagedAttention k/v cache), see `flydsl_translation_attention.md` § Decode Attention — wrap
a matching prebuilt fused kernel when one exists, otherwise use the decomposed split-K
path described there.

### Math and Layout

The kernel computes:

```
C = A @ B^T
```

| Tensor | Shape | Role |
|--------|-------|------|
| `A` (`a`) | `(M, K)` | Left operand (e.g. Q flattened over heads) |
| `B` (`b`) | `(N, K)` | Right operand stored row-major as `(N, K)` — **not** transposed |
| `C` (`c`) | `(M, N)` | Output (pre-allocated) |

Equivalent PyTorch: `torch.mm(A, B.T)` or `A @ B.transpose(-2, -1)` when `B` is `(N, K)`.

For **Q @ K^T** with `K` of shape `(seq_len, K_dim)`, pass `B = K` (already `(N, K)`).

For **attn @ V** with `V` of shape `(seq_len, V_dim)`, transpose first:

```python
vt = v.t()  # (V_dim, seq_len) — here N=V_dim, K=seq_len
hgemm_splitk_(out, attn, vt, hgemm_kwargs=kwargs, stream=stream)
```

### High-Level API (Preferred)

```python
from kernels.hgemm_splitk import hgemm_splitk_

# C, A, B: fp16 or bf16, CUDA. Shapes as above.
hgemm_splitk_(
    c,           # (M, N) output, pre-allocated
    a,           # (M, K)
    b,           # (N, K)
    bias=None,   # optional (N,) — rarely used in translations
    hgemm_kwargs={...},  # tile config; see below
    stream=torch.cuda.current_stream(),
)
```

- JIT-compiles on first call for each `(dtype, N, K, **hgemm_kwargs)` tuple (cached).
- `M` is dynamic at launch time; **`N` and `K` are fixed at compile time** (from `b.shape`).
- No preshuffling, no scale tensors, no `.view(-1)` requirement (internally reshapes to 2D).
- `get_default_kwargs(m, n, k)` supplies tuned tiles for common LLM shapes; override via `hgemm_kwargs`.

### Low-Level API

For repeated launches with the same `(N, K)` and tile config, compile once:

```python
from kernels.hgemm_splitk import compile_hgemm_kernel, get_semaphore

launch_fn = compile_hgemm_kernel(
    "f16",          # or "bf16"
    n=N, k=K,       # fixed at compile time
    TILE_M=16, TILE_N=128, TILE_K=64,
    SPLIT_K=1,
    BLOCK_M_WARPS=1, BLOCK_N_WARPS=2, BLOCK_K_WARPS=1,
    B_TO_LDS=True,
    HAS_BIAS=False,
)
semaphore, signal = get_semaphore(stream, device)
launch_fn(c, a, b, bias_placeholder, m, semaphore, signal, stream=stream)
```

Prefer `hgemm_splitk_()` unless you need explicit compile caching control.

### Tile Configuration (`hgemm_kwargs`)

| Key | Meaning |
|-----|---------|
| `TILE_M` | M-dimension tile (16 for small decode M) |
| `TILE_N` | N-dimension tile; **`N` must be divisible by `TILE_N`** |
| `TILE_K` | K-dimension tile; **`K` must be divisible by `TILE_K * SPLIT_K` logic** |
| `SPLIT_K` | Split K across blocks (>1 improves occupancy for large K, small M) |
| `BLOCK_M_WARPS` | Warps along M (product with N/K warps ≤ 8) |
| `BLOCK_N_WARPS` | Warps along N |
| `BLOCK_K_WARPS` | Warps along K (K-slicing within block) |
| `B_TO_LDS` | Stage B matrix in LDS (often `True`) |

#### Recommended starting points

| M range | `TILE_M` | Notes |
|---------|----------|-------|
| 1–16 | 16 | Decode / few query rows |
| 17–64 | 32 or 64 | Medium batch |
| 64+ | 64–128 | May still use splitk; compare vs preshuffle if B is static |

`TILE_N`: 64 or 128 (must divide `N`). `TILE_K`: 64 or 128/256 for large K.

**Decode (small M, large K): try `SPLIT_K > 1` when profiling shows benefit.** With
`M ≈ 16` (single batch slice) and `K ≥ 512`, split-K can raise occupancy. With
**stacked** `M = batch * nheads` (e.g. 64) in decomposed MLA teaching kernels,
**start with `SPLIT_K=1`** — semaphore sync often dominates at tiny per-launch M.
See `flydsl_translation_attention.md` § Decode Attention (decomposed path).

Example (single-batch MLA decode, `M=16`, `N=512`, `K=576` — profile both):

```python
gemm_kwargs = {
    "TILE_M": 16, "TILE_N": 128, "TILE_K": 64,
    "SPLIT_K": 2,          # >1 for small-M / large-K decode occupancy
    "BLOCK_M_WARPS": 1, "BLOCK_N_WARPS": 2,
    "BLOCK_K_WARPS": 1, "B_TO_LDS": True,
}
```

### Constraints (split-K GEMM only)

- **Dtypes**: `torch.float16` or `torch.bfloat16` only.
- **`N % TILE_N == 0`** (compile-time `n` from `b.shape[0]`).
- **`K`**: must satisfy divisibility checks in `compile_hgemm_kernel` (`K % TILE_K`, split-K splits, etc.).
- **`M`**: dynamic; partial final M-tile handled in kernel.
- **GPU arch**: tested on `gfx942`, `gfx950` (see FlyDSL `test_hgemm_splitk.py`).
- **No preshuffle**: unlike `compile_preshuffle_gemm_a8`, B is used as-is.
- When `SPLIT_K > 1`, internal semaphore buffer size limits grid (`bm * bn`).

### Decomposed Attention Pattern

Use when flash attention does not apply (paged KV, non-standard head dims, MLA, etc.):

```python
from kernels.hgemm_splitk import hgemm_splitk_
from kernels.softmax_kernel import build_softmax_module

# 1) scores = Q @ K^T   —  A: (M, K_qk), B: K as (seq, K_qk) -> (N=seq, K=K_qk)
hgemm_splitk_(scores, q_flat, k, hgemm_kwargs=gemm_kwargs, stream=stream)

# 2) scale + mask (element-wise / PyTorch structural ops)

# 3) softmax
softmax_fn(scores, attn, M, stream=stream)

# 4) out = attn @ V  —  V^T as (N=v_dim, K=seq)
vt = v.t().contiguous()
hgemm_splitk_(out, attn, vt, hgemm_kwargs=gemm_kwargs, stream=stream)
```

Store `gemm_kwargs` on the module; compile once per `(N, K)` shape if `seq_len` is bounded
(use `max_seq_len` buffers and slice, as in MLA translations).

### Preshuffle GEMM vs Split-K GEMM

| | Preshuffle GEMM | hgemm_splitk |
|--|-----------------|--------------|
| B operand | Fixed weight, shuffled once | Any `(N, K)` tensor, no shuffle |
| Best for | Linear, conv GEMM | Dynamic activations, small M |
| M | Dynamic (`M=0` in compile) | Dynamic at launch |
| N, K | Dynamic at launch | **Fixed at compile** from `b.shape` |
| Scales | Required (`empty(0)` for fp16) | Not used |
| Launch args | All `.view(-1)` | 2D tensors OK |

### Split-K Pitfalls

1. **Wrong B layout**: Pass `(N, K)`, not `(K, N)`, unless you explicitly transpose to match `C = A @ B^T`.
2. **Recompile every forward**: Changing `N` or `K` (e.g. growing `seq_len` past compile bounds) triggers new JIT. Pre-allocate for `max_seq_len` and slice.
3. **Using preshuffle for K/V**: Do not `shuffle_weight` on cache tensors each step.
4. **Large M + static weight**: Use preshuffle GEMM instead for better throughput.
5. **Flash-eligible SDPA**: If Q/K/V are contiguous BSHD and constraints hold, flash attention beats decomposed splitk.

### Split-K Reference Implementations

- FlyDSL tests: `FlyDSL/tests/kernels/test_hgemm_splitk.py`

## GEMM + Reduction Fusion: Replace GEMM with Custom Kernel

When a GEMM is **immediately followed by a reduction** (e.g., `sum`, `mean`), the
full computation can often be simplified mathematically and implemented as a single
fused `@flyc.kernel`, completely eliminating the rocBLAS GEMM call.

### When to Apply

Check for this pattern:
```python
# PyTorch original
y = x @ W.T          # GEMM: (B, K) @ (K, N) -> (B, N)
y = y / divisor       # element-wise
y = y.sum(dim=1)      # reduction along N -> (B,)
y = y * scale         # element-wise
```

If the GEMM output is reduced along the N dimension, the entire sequence collapses
to a **dot product per row** against a precomputed vector:

```
# Math simplification:
# y[i] = sum_j( x[i,j] * W[j,:].sum() ) * (scale / divisor)
# w_sum = W.sum(dim=0)  -- precompute once (constant)
# y[i] = dot(x[i,:], w_sum) * fused_scale
```

### Implementation Pattern

**In `__init__` / `build_model()` — precompute weight-side reduction:**
```python
w_sum = weight.sum(dim=0)  # (K,) -- done once, weight is constant
fused_scale = scaling_factor / divisor  # fold scalar ops
```

**Replace GEMM + reduction with a custom `@flyc.kernel`:**
```python
@flyc.kernel
def fused_dot_scale_kernel(X: fx.Tensor, W_sum: fx.Tensor, Out: fx.Tensor):
    bid = fx.block_idx.x   # one block per row
    tid = fx.thread_idx.x
    # Each thread accumulates partial dot product using FMA
    acc = arith.constant(0.0, type=T.f32)
    for base in range_constexpr(0, K, BLOCK_THREADS):
        idx = tid + base
        x_val = ...  # load X[bid, idx]
        w_val = ...  # load W_sum[idx]
        acc = arith.fma(x_val, w_val, acc, fastmath=fast)
    # Block-wide reduction (wave shuffle + shared memory)
    total = block_reduce_sum(acc)
    # Thread 0 writes: Out[bid] = total * fused_scale
```

### Why This Is Fast

- **No rocBLAS launch**: Eliminates GEMM kernel launch overhead entirely
- **No intermediate tensor**: The `(B, N)` GEMM output is never materialized
- **Precomputed constants**: Weight reduction and scalar folding happen once at init
- **FMA accumulation**: Numerically stable fused multiply-add in the inner loop
- **Single kernel**: One launch per batch instead of GEMM + element-wise + reduction

### Verified Results

`14_Gemm_Divide_Sum_Scaling`: enriched achieved **15.7x** speedup (vs baseline 1.02x
using `torch.matmul`) by fusing the entire GEMM + divide + sum + scale into a single
`@flyc.kernel` with precomputed `w_sum`.

### Applicability Limits

- Only works when the reduction dimension matches the GEMM output dimension
- Weight must be constant (not an activation) so the weight-side reduction is a one-time cost
- If the GEMM output is used for multiple operations (not just reduction), keep the GEMM

## No PyTorch GEMM Fallbacks

Do NOT use `torch.mm`, `torch.bmm`, `torch.matmul`, `F.linear`, or `nn.Linear`.
ALL matrix multiplications must use FlyDSL preshuffle GEMM.

### fp32 inputs

Cast to fp16 before calling `compile_preshuffle_gemm_a8`. FlyDSL preshuffle GEMM
handles all GEMM operations. Do NOT use `torch.mm` for fp32 GEMM.

### Batched matmul (replacing torch.bmm)

- **Attention pattern (Q@K^T)**: Use `build_flash_attn_func_module()` — flash attention
  handles the full Q@K^T → softmax → @V pipeline natively.
- **Shared B-matrix**: reshape `(B, M, K)` to `(B*M, K)`, use single `compile_preshuffle_gemm_a8`,
  then reshape back. B-matrix is preshuffled once.
- **Varying B-matrix per batch (activations)**: fold batch into M and use `hgemm_splitk_`
  (no preshuffle). See § Split-K GEMM (hgemm_splitk) above.
  Use preshuffle GEMM only when each B-slice is a **static weight** shuffled once.

### Conv2d internal GEMM

Conv2d uses im2col (`F.unfold`) + preshuffle GEMM with fp16 cast.
Do NOT fall back to `torch.mm`, `torch.bmm`, or `F.conv2d` — always use preshuffle GEMM.
The weight matrix is shared across the batch — fold B into M and call a single GEMM.

### All other GEMM (nn.Linear, torch.matmul, F.linear)

Replace entirely with `compile_preshuffle_gemm_a8` + `shuffle_weight`.
Store weights as `nn.Parameter`, not `nn.Linear`.

## Low-Level CuTe-Style Primitives (FlyDSL 0.1.4+)

FlyDSL 0.1.4+ exposes low-level CuTe-style primitives that give fine-grained
control over GEMM execution. These are useful when `compile_preshuffle_gemm_a8`
is insufficient — for example, when you need fp32 output from a GEMM to avoid
precision loss in multi-layer pipelines (e.g. Conv+BN chains where fp16
truncation between layers compounds).

### Available Primitives

| Module | Primitive | Purpose |
|--------|-----------|---------|
| `rocdl.MFMA` | `mfma_f32_16x16x16f16` | Hardware MFMA: fp16 inputs, fp32 accumulator (16x16x16 tile) |
| `rocdl.MFMA` | `mfma_f32_16x16x4f32` | Hardware MFMA: fp32 inputs, fp32 accumulator |
| `fx` | `fx.make_mma_atom(...)` | Construct a CuTe MMA atom from an MFMA instruction |
| `fx` | `fx.make_tiled_mma(...)` | Tile an MMA atom across thread blocks |
| `fx` | `fx.gemm(...)` | Layout-aware GEMM building block using tiled MMA |
| `fx` | `fx.copy(...)` / `rocdl.BufferCopy` | Async global→shared memory copy primitives |

### When to Use

- **Precision-sensitive pipelines**: When `compile_preshuffle_gemm_a8` (which
  truncates fp32 accumulators to fp16 output) causes correctness failures in
  multi-layer networks. A custom CuTe GEMM can write fp32 accumulators directly
  to global memory, avoiding truncation.
- **Non-standard data types**: When you need fp32-in/fp32-out GEMM or mixed
  precision configurations not supported by the pre-built kernels.
- **Custom tiling**: When the pre-built tile configurations don't match your
  problem shape well.

### Example: Custom fp32-Output GEMM Kernel

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.typing import T

@flyc.kernel
def gemm_fp32_out(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, M: fx.Constexpr[int], ...):
    # Use fx.make_mma_atom with mfma_f32_16x16x16f16
    # Accumulate in fp32 registers
    # Store fp32 result directly (no arith.trunc_f)
    ...
```

Note: Writing a correct CuTe GEMM kernel requires understanding tiled MMA
layouts, shared memory staging, and the MFMA instruction semantics. Prefer
`compile_preshuffle_gemm_a8` when fp16 output precision is acceptable.
