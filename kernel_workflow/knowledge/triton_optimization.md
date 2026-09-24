# Triton Kernel Optimization Patterns

Patterns ranked by priority. Higher priority (P0) = higher expected impact.

## P0: Algorithm & Tiling Design

### Tiling Strategy
Triton kernels are fundamentally block-based. The tiling scheme determines performance more than anything else.

**Key decisions:**
- Choose block dimensions that maximize data reuse
- Ensure BLOCK_SIZE is a multiple of the **wavefront size** (64 on Instinct CDNA gfx942/gfx950; **32** on RDNA4 gfx1201)
- Balance tile size vs register pressure vs shared memory usage

```python
@triton.jit
def kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask)
    # Process full block at once
    y = tl.exp(x)
    tl.store(Y_ptr + offsets, y, mask=mask)
```

### Reduction Patterns
For reductions, use hierarchical approach: per-block reduction → cross-block atomic or two-pass.

```python
@triton.jit
def reduce_kernel(X_ptr, OUT_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    result = tl.sum(x, axis=0)
    tl.atomic_add(OUT_ptr, result)
```

### Multi-Dimensional Tiling
For 2D problems (matmul, attention), tile both dimensions independently.

```python
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Accumulate over K in BLOCK_K tiles
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + ...)  # BLOCK_M x BLOCK_K tile
        b = tl.load(B_ptr + ...)  # BLOCK_K x BLOCK_N tile
        acc += tl.dot(a, b)       # Maps to MFMA on AMD
    tl.store(C_ptr + ..., acc)
```

### Fused Operations
Combine multiple elementwise operations into a single kernel pass to reduce memory traffic.

## P1: Memory Access Patterns

### Coalesced Block Loading
Ensure `tl.load` accesses contiguous memory within each block. Use `tl.arange` with stride 1.

```python
# GOOD: Contiguous access
offsets = pid * BLOCK + tl.arange(0, BLOCK)
x = tl.load(ptr + offsets)

# BAD: Strided access
offsets = tl.arange(0, BLOCK) * stride  # Non-contiguous
```

### Block Size Selection
- Minimum: 64 (one wavefront on AMD)
- Sweet spots: 64, 128, 256, 512, 1024
- Larger blocks = more data reuse but more register pressure

### Masking
Always use masks for boundary conditions. Avoid `tl.where` when possible — it generates unnecessary instructions.

```python
# Prefer mask parameter over tl.where
x = tl.load(ptr + offsets, mask=mask, other=0.0)  # Faster
# Avoid: x = tl.where(mask, tl.load(ptr + offsets), 0.0)
```

## P2: Compute Optimization

### Constexpr Hints
Mark values known at compile time as `tl.constexpr` to enable compiler optimizations.

```python
@triton.jit
def kernel(N, BLOCK: tl.constexpr, NUM_STAGES: tl.constexpr):
    # Compiler can unroll loops with constexpr bounds
    for i in range(NUM_STAGES):  # Unrolled
        ...
```

### Dot Product → MFMA (CDNA) / WMMA (RDNA4)
On AMD, `tl.dot` maps to the matrix ISA of the **detected** gfx:

- **gfx942 / gfx950 (CDNA):** MFMA (Matrix Fused Multiply-Add). Input types fp16/bf16/fp32/int8;
  typical min 16×16 tiles; fp32 accumulator. gfx950 also has scaled MX MFMA — see `amd_instinct.md` §3.
- **gfx1201 (RDNA4):** **WMMA**, not MFMA. No scaled MFMA / MX. See `amd_rdna4.md` §3.

### Mixed Precision
Load in fp16/bf16, compute in fp32 for bandwidth savings with precision.

```python
x = tl.load(ptr + offsets).to(tl.float16)  # Load as fp16
acc += tl.dot(x, y)  # fp32 acc via MFMA (CDNA) or WMMA (RDNA4)
```

## P3: AMD-Specific Optimizations

### waves_per_eu
Control occupancy via the `waves_per_eu` parameter in `@triton.autotune`.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2,
                      waves_per_eu=2),  # AMD-specific
    ],
    key=['N'],
)
```

### Wavefront width (detect gfx — do not assume 64)
- **Instinct CDNA (gfx942 / gfx950):** 64-thread wavefronts. Triton's `num_warps` is in wave64 units;
  coalescing width is 64 threads × 4 B = 256 B per access.
- **RDNA4 (gfx1201):** **32-thread** wavefronts. `num_warps` is in wave32 units. See
  `amd_rdna4.md`. Do not copy CDNA `num_warps` / `waves_per_eu` tables onto this box.

### MFMA tile sizes (CDNA only)
Match `BLOCK_M/N/K` to the hardware MFMA tile shapes for best utilization (detect the arch with
`rocminfo`):
- **gfx942 (CDNA3)**: 4x4x4, 16x16x16, 32x32x8 (plus 16x16x32 / 32x32x16 for 8-bit). Prefer
  `matrix_instr_nonkdim=16`.
- **gfx950 (CDNA4)**: adds new/wider MFMA variants and native MXFP4/MXFP6/MXFP8 (block-scaled) matrix
  ops not present on gfx942 — a major low-precision GEMM lever. See `amd_instinct.md` §3.

### WMMA / RDNA4 tiles (gfx1201)
Do **not** use the MFMA table above. Starting Triton knobs (then autotune):
- `BLOCK_M = 64` (128 is often too fat on client RDNA)
- `BLOCK_N = 32` on **gfx1201** (16 is a common RDNA3 seed; 64 is the CDNA FA default)
- `waves_per_eu` occupancy-first (RDNA start ~6 vs CDNA 1–3)
- `num_warps` counted in wave32 units
- Attention under CUDA graphs: keep `int64_strides=true` unless A/B says otherwise (`amd_rdna4.md` §3)

See `amd_rdna4.md` only. Do not follow CDNA attention FMHA Triton cards on R9700 — those are MFMA/FNUZ
recipes (gens gfx90a/gfx942/gfx950) and are not rewritten for gfx1201.

## P4: Autotune Configurations

### @triton.autotune
Define multiple configurations and let Triton pick the fastest.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16},
                      num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],  # Re-tune when these change
)
@triton.jit
def kernel(M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
```

### Key Selection
Choose autotune keys that capture shape-dependent behavior. Include dimensions that affect tiling efficiency.

## Compound Strategy Compatibility

Compose well:
- Tiling + matrix-dot (`tl.dot` → MFMA on CDNA, WMMA on RDNA4) → excellent (standard matmul pattern)
- Fused ops + Coalesced loading → excellent
- Autotune + Multiple tile sizes → excellent (let runtime decide)
- Mixed precision + matrix-dot → excellent (higher matrix throughput)

Conflicts:
- Very large tiles + High num_warps → register pressure
- Many num_stages + Large tiles → shared memory overflow
