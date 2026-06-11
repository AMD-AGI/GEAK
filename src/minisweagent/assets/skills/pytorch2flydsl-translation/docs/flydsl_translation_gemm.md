---
layer: "flydsl"
category: "translation"
tags: ["flydsl", "translation", "gemm", "matmul", "linear"]
last_updated: 2026-03-23
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

### Alternative: hgemm_splitk (FP16 SplitK GEMM)

For small M (e.g., batch_size=1 decode), standard tile configs may not
fill the GPU. `hgemm_splitk` splits the K dimension across thread blocks:

```python
from kernels.hgemm_splitk import compile_hgemm_splitk
```

Use when M < tile_m and standard GEMM underperforms. Only available in
newer FlyDSL versions — check availability before using.

### Constraints

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
- **Varying B-matrix per batch**: reshape both operands so the batch is folded into the
  M dimension. For `(B, M, K) @ (B, K, N)`: iterate over batch elements calling
  preshuffle GEMM per element (each B-slice is preshuffled separately).
  This is acceptable when flash attention does not apply.

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
