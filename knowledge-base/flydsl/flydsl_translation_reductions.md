---
layer: "flydsl"
category: "translation"
tags: ["flydsl", "translation", "reductions", "softmax", "layernorm"]
last_updated: 2026-04-08
---

# FlyDSL Translation: Reduction Patterns

## Warp/Block Reduction in FlyDSL

FlyDSL provides reduction helpers in `kernels/reduce.py`. For custom reductions:

### Warp Reduction (XOR Shuffle)

AMD wavefront size is 64. Reduce using XOR shuffles:

```python
from flydsl.expr import arith, range_constexpr
import math

WARP_SIZE = 64

def wave_reduce_sum(x):
    width_i32 = fx.Int32(WARP_SIZE)
    w = x
    for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
        off = fx.Int32(WARP_SIZE // (2 << _sh_exp))
        peer = w.shuffle_xor(off, width_i32)
        w = w.addf(peer)
    return w

def wave_reduce_max(x):
    width_i32 = fx.Int32(WARP_SIZE)
    w = x
    for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
        off = fx.Int32(WARP_SIZE // (2 << _sh_exp))
        peer = w.shuffle_xor(off, width_i32)
        w = w.maximumf(peer)
    return w
```

### Block Reduction (Multi-Wave via LDS)

For blocks with multiple waves, use shared memory:

```python
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.expr import gpu

NUM_WAVES = BLOCK_SIZE // WARP_SIZE

def block_reduce_sum(tid, val, allocator, red_offset):
    lane = tid % WARP_SIZE
    wave = tid // WARP_SIZE
    base_ptr = allocator.get_base()
    s_red = SmemPtr(base_ptr, red_offset, T.f32, shape=(NUM_WAVES,))

    w = wave_reduce_sum(val)
    if lane == fx.Int32(0):
        wave_idx = arith.index_cast(T.index, wave)
        s_red.store(w, [wave_idx])
    gpu.barrier()

    if wave == fx.Int32(0):
        in_range = lane < NUM_WAVES
        lane_safe = in_range.select(lane, fx.Int32(0))
        v = s_red.load([arith.index_cast(T.index, lane_safe)])
        z = arith.constant(0.0, type=T.f32)
        ww = in_range.select(v, z)
        ww = wave_reduce_sum(ww)
        if lane == fx.Int32(0):
            s_red.store(ww, [fx.Index(0)])
    gpu.barrier()
    return s_red.load([fx.Index(0)])
```

## Using Pre-built Softmax

For `torch.softmax(x, dim=-1)` translation, use the pre-built kernel:

```python
from kernels.softmax_kernel import build_softmax_module

class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._softmax = None
        self.dim = dim

    def forward(self, x):
        if x.dim() == 2:
            M, N = x.shape
        else:
            M = x.shape[0] * x.shape[1] if x.dim() == 3 else x.numel() // x.shape[-1]
            N = x.shape[-1]
            x = x.reshape(M, N)

        if self._softmax is None:
            self._softmax = build_softmax_module(M, N, dtype_str="f32")

        output = torch.empty_like(x)
        self._softmax(x, output, M, stream=torch.cuda.current_stream())
        return output.reshape_as(x)
```

## Using Pre-built LayerNorm / RMSNorm

```python
from kernels.layernorm_kernel import build_layernorm_module
from kernels.rmsnorm_kernel import build_rmsnorm_module

class Model(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self._ln = None

    def forward(self, x):
        M = x.shape[0]
        N = x.shape[-1]
        if self._ln is None:
            self._ln = build_layernorm_module(M, N, dtype_str="f32")
        output = torch.empty_like(x)
        self._ln(x, self.gamma, self.beta, output, M,
                 stream=torch.cuda.current_stream())
        return output
```

## Manual Mean/Sum Reduction

For `torch.mean(x)` or `torch.sum(x)`, combine vector reduction with block reduce:

```python
from flydsl.expr import vector

# Inside kernel: each thread processes vec_width elements
vI = fx.memref_load_vec(rI)  # load vector
partial_sum = vector.reduction(T.f32, vector.CombiningKind.ADD, vI)

# Block-level reduction
total = block_reduce_sum(tid, partial_sum, allocator, red_offset)

# Only thread 0 writes final result
if tid == fx.Int32(0):
    # ... store total to output ...
```

## Key AMD Hardware Considerations

- **Wavefront size**: 64 (not 32 like NVIDIA warp)
- **LDS per CU**: 64 KB on MI300X (gfx942), 160 KB on MI350 (gfx950)
- **Use `exp2` via rocdl**: `rocdl.exp2(T.f32, x)` for fast hardware exp2
- **Numerical stability**: For softmax, use `exp2(x * log2(e))` not `exp(x)`
