---
layer: "flydsl"
category: "translation"
tags: ["flydsl", "translation", "gemm", "mfma", "matmul"]
last_updated: 2026-03-23
---

# Translating GEMM Kernels from PyTorch to FlyDSL

## Overview

Matrix multiplication (`torch.matmul`, `torch.mm`, `torch.bmm`) requires
tiled algorithms with shared memory for efficient GPU execution. This is
significantly more complex than element-wise translations.

## Basic Tiled GEMM Pattern

```python
# PyTorch: C = torch.matmul(A, B)  # A: [M, K], B: [K, N], C: [M, N]

TILE_M = 16
TILE_N = 16
TILE_K = 16

@flyc.kernel
def gemm_kernel(a_ptr, b_ptr, c_ptr, M: int, N: int, K: int):
    # Block coordinates
    bm = flyc.block_id() // ((N + TILE_N - 1) // TILE_N)
    bn = flyc.block_id() % ((N + TILE_N - 1) // TILE_N)
    tid = flyc.thread_id() - flyc.block_id() * flyc.block_dim()

    # Thread coordinates within tile
    tm = tid // TILE_N
    tn = tid % TILE_N

    # Global coordinates
    row = bm * TILE_M + tm
    col = bn * TILE_N + tn

    # Shared memory for tiles
    a_shared = flyc.shared_memory((TILE_M, TILE_K), flyc.float32)
    b_shared = flyc.shared_memory((TILE_K, TILE_N), flyc.float32)

    acc = 0.0
    for k_tile in range(0, K, TILE_K):
        # Load A tile
        if row < M and (k_tile + tn) < K:
            flyc.store(a_shared, (tm, tn), flyc.load(a_ptr, row * K + k_tile + tn))
        # Load B tile
        if (k_tile + tm) < K and col < N:
            flyc.store(b_shared, (tm, tn), flyc.load(b_ptr, (k_tile + tm) * N + col))

        flyc.syncthreads()

        # Compute partial dot product
        for kk in range(TILE_K):
            acc += flyc.load(a_shared, (tm, kk)) * flyc.load(b_shared, (kk, tn))

        flyc.syncthreads()

    # Store result
    if row < M and col < N:
        flyc.store(c_ptr, row * N + col, acc)

@flyc.jit
def gemm_jit(A, B, C, M, N, K):
    grid = ((M + TILE_M - 1) // TILE_M) * ((N + TILE_N - 1) // TILE_N)
    block = TILE_M * TILE_N
    gemm_kernel[grid, block](A, B, C, M, N, K)

class Model(torch.nn.Module):
    def forward(self, A, B):
        M, K = A.shape
        _, N = B.shape
        C = torch.empty(M, N, device=A.device, dtype=A.dtype)
        gemm_jit(A, B, C, M, N, K)
        return C
```

## MFMA Intrinsics (MI300X)

For high-performance GEMM on MI300X, use Matrix Fused Multiply-Add (MFMA):

- `flyc.mfma_f32_16x16x4_f32` — 16x16 output tile, K=4 accumulation
- `flyc.mfma_f32_32x32x8_f16` — 32x32 output tile with FP16 inputs

These map directly to AMD CDNA3 matrix core instructions.

## Batched Matrix Multiplication

```python
# PyTorch: C = torch.bmm(A, B)  # A: [B, M, K], B: [B, K, N]

@flyc.kernel
def batched_gemm_kernel(a_ptr, b_ptr, c_ptr,
                         batch: int, M: int, N: int, K: int):
    # Use block_id to index into batch dimension
    batch_idx = flyc.block_id() // (((M + TILE_M - 1) // TILE_M) *
                                     ((N + TILE_N - 1) // TILE_N))
    # Offset pointers by batch stride
    a_offset = batch_idx * M * K
    b_offset = batch_idx * K * N
    c_offset = batch_idx * M * N
    # ... rest of tiled GEMM with offset pointers
```

## Occupancy Tuning

- **Tile sizes**: Balance shared memory usage vs. parallelism.
  Larger tiles (32x32, 64x64) improve arithmetic intensity but reduce occupancy.
- **Register pressure**: Each thread accumulates TILE_M/thread * TILE_N/thread values.
  MI300X has 256 VGPRs per thread; plan register usage accordingly.
- **LDS usage**: MI300X has 64 KB per workgroup. Two tiles of 64x64 float32 = 32 KB.

## Key Differences from PyTorch

- PyTorch's `torch.matmul` dispatches to highly optimized rocBLAS/hipBLAS.
  FlyDSL translations will likely be slower for large matrices unless MFMA
  intrinsics and optimal tiling are used.
- For correctness validation, a naive tiled GEMM is sufficient.
- Performance optimization of the GEMM kernel is out of scope for translation.
