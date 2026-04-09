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

        if self.bias is not None:
            output = output + self.bias

        return output

def get_inputs():
    return [torch.randn(1024, 4096, device="cuda")]

def get_init_inputs():
    return [4096, 4096]
```

## When PyTorch Fallback is Acceptable

Only fall back to PyTorch GEMM (`torch.matmul`, `F.linear`) when:

1. **Batched matmul** (`torch.bmm`) — no FlyDSL batched GEMM yet
2. **Very small dimensions** where GEMM kernel overhead exceeds compute
3. **Conv2d / pooling** — these are NOT GEMM (no FlyDSL equivalent)

**Do NOT use PyTorch fallback for standard `nn.Linear` or `torch.matmul(A, B)` —
these should use `compile_preshuffle_gemm_a8`.**
