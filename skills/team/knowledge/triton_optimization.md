# Triton Kernel Optimization Patterns for AMD MI300X

Detection: `.py` files containing `@triton.jit` or `tl.` patterns.

## Priority Hierarchy

**Priority 0 — Algorithmic kernel-body rewrites**:
Change the computation structure: reduction trees, tiling strategies, decomposition, math simplification, operation fusion within the kernel body.

**Priority 1 — Memory access patterns**:
Vectorized loads, contiguous access, eviction policies, layout optimization, tiling shape.

**Priority 2 — Parallelism and occupancy**:
Block size tuning, `waves_per_eu`, split-K, multi-program cooperation.

**Priority 3 — Compiler hints and autotuning**:
`tl.constexpr`, `@triton.autotune`, `num_stages`, `num_warps`.

**Priority 4 — Wrapper/dispatch** (lowest):
Python-side dispatch logic, dtype routing, shape-based kernel selection.

---

## Algorithmic Patterns (Priority 0)

### Tiling and Blocking
The fundamental Triton pattern — process data in blocks:

```python
@triton.jit
def kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask)
    y = compute(x)
    tl.store(Y_ptr + offsets, y, mask=mask)
```

### Split-K for GEMMs
Partition the reduction dimension across multiple programs:

```python
@triton.jit
def matmul_splitk(A, B, C_partial, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  SPLIT_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)  # K-split index
    
    k_start = pid_k * (K // SPLIT_K)
    k_end = min(k_start + (K // SPLIT_K), K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_start, k_end, BLOCK_K):
        a = tl.load(A + ...)
        b = tl.load(B + ...)
        acc += tl.dot(a, b)
    
    tl.store(C_partial + pid_k * M * N + ..., acc)
```

### Reduction Patterns
Use built-in reductions for maximum efficiency:

```python
# Fast reductions (compiled to hardware instructions)
result = tl.sum(x, axis=0)
result = tl.max(x, axis=0)
result = tl.min(x, axis=0)

# Custom reduction via loop tiling
acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
for i in range(0, N, BLOCK_N):
    x = tl.load(ptr + i + tl.arange(0, BLOCK_N))
    acc += tl.sum(x, axis=0)
```

### Operation Fusion
Fuse multiple operations into a single kernel to eliminate intermediate memory traffic:

```python
@triton.jit
def fused_layernorm_dropout(X, W, B, Y, M, V, ...):
    # Load, normalize, scale, bias, dropout — all in one kernel
    x = tl.load(X + offsets)
    mean = tl.sum(x, axis=1) / N
    var = tl.sum((x - mean) ** 2, axis=1) / N
    x_norm = (x - mean) / tl.sqrt(var + eps)
    y = x_norm * tl.load(W + ...) + tl.load(B + ...)
    # Apply dropout
    rand = tl.rand(seed, offsets)
    y = tl.where(rand > p, y / (1 - p), 0.0)
    tl.store(Y + offsets, y)
```

---

## Memory Patterns (Priority 1)

### Contiguous Access
Ensure consecutive programs access consecutive memory:

```python
# GOOD: contiguous along the fast dimension
offsets = pid * BLOCK + tl.arange(0, BLOCK)
x = tl.load(ptr + offsets)

# BAD: strided access
offsets = tl.arange(0, BLOCK) * stride  # non-contiguous
```

### Eviction Policies
Hint the cache about data reuse:

```python
# Data used once → evict immediately
x = tl.load(ptr + offsets, eviction_policy='evict_first')

# Data reused many times → keep in cache
x = tl.load(ptr + offsets, eviction_policy='evict_last')
```

### Vectorized Loads
Use block sizes that are multiples of 128 bytes / element_size for optimal vectorization:
- FP32: BLOCK divisible by 32 (128 bytes / 4)
- FP16: BLOCK divisible by 64 (128 bytes / 2)

---

## Parallelism (Priority 2)

### Autotuning
Let Triton search for optimal parameters:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
```

### AMD-Specific Settings
- `waves_per_eu`: Controls occupancy. Lower values = more registers per wave.
- `num_warps`: On AMD, 1 warp = 1 wavefront = 64 threads.
- `num_stages`: Software pipelining depth. More stages = more LDS usage.
- `tl.dot` maps to MFMA instructions on gfx942.

---

## Compiler Hints (Priority 3)

### Constexpr
Use `tl.constexpr` for values known at JIT time:

```python
@triton.jit
def kernel(..., BLOCK: tl.constexpr, K: tl.constexpr):
    # Compiler can fully unroll, optimize register allocation
    for i in range(K):  # K is compile-time constant
        ...
```

### Minimizing Live Variables
Reduce register pressure around `tl.dot`:

```python
# BAD: many live tensors across tl.dot
a = tl.load(A + ...)
b = tl.load(B + ...)
c = tl.load(C + ...)  # <- extra live tensor
acc = tl.dot(a, b)
result = acc + c

# GOOD: load c after dot completes
a = tl.load(A + ...)
b = tl.load(B + ...)
acc = tl.dot(a, b)
c = tl.load(C + ...)  # <- loaded after dot, register reuse
result = acc + c
```

### Math Operations
```python
# Fast reciprocal square root
x = tl.math.rsqrt(var + eps)  # Faster than 1.0 / tl.sqrt(...)

# Avoid tl.where in hot loops (generates predicated instructions)
# Prefer mask-based loads/stores instead
```

## Build Cache Warning

Triton caches compiled kernels. After modifying kernel source:
```bash
rm -rf ~/.triton/cache/
```
