# Translating to TileLang — op-mapping & structural guide

Goal: rewrite a source kernel (PyTorch / Triton / CK / HIP) into an equivalent **TileLang** kernel that is
numerically identical (within tolerance) and faster on MI300X. Preserve the source's external interface
(same callable signature, output shape & dtype, and `get_inputs()` / `get_init_inputs()` if present).

## General workflow
1. Read the source and identify the computation pattern: GEMM/Linear, attention/SDPA, reduction/norm,
   elementwise, or a fused combination.
2. Pick the TileLang structure (see op map). Prefer the autotuned form.
3. Write the kernel + a thin wrapper that preserves the interface.
4. Validate with `save_and_test` against the harness; fix correctness first, then perf.
5. Submit only when correctness passes AND it is ≥ baseline.

## Op map (source → TileLang)
| Source pattern | TileLang approach |
|---|---|
| `torch.matmul`/`mm`/`F.linear`/`nn.Linear`, Triton matmul, CK GEMM | `T.gemm` with autotuned block_M/N/K, `k_pack`, `GemmWarpPolicy`; `transpose_B=True` for `A·Bᵀ` |
| Attention / SDPA / flash-attn (any source) | FlashAttention tile program: Q in fragments, K/V in shared, `T.Pipelined` over KV tiles, `T.gemm`+`reduce_max`/`reduce_sum`+`T.Parallel` exp |
| softmax / layernorm / rmsnorm / reductions | row-reduction kernel with `T.reduce_max`/`T.reduce_sum` + `T.Parallel` epilogue |
| elementwise (add, mul, silu, gelu, residual, scale) | `T.Parallel` map kernel over the tile; fuse into the producer's epilogue when possible |
| grouped/batched GEMM (MoE expert GEMM, bmm) | tile GEMM looped/grided over groups; one launch, never a Python per-group loop |

## Source-specific notes
- **From Triton**: the Triton `@triton.jit` body already expresses tiling (`tl.program_id`, `tl.load`,
  `tl.dot`, masks). Map `tl.dot` → `T.gemm`, block constants → TileLang block shapes, the program grid →
  `T.Kernel` grid, `tl.load/store` with masks → `T.copy` with bounds. Let the autotuner re-pick tile sizes
  rather than copying Triton's `BLOCK_*` verbatim — the gfx942 optimum often differs.
- **From CK (.cu)**: CK encodes the MFMA tile pipeline in C++ template params (cshuffle, block tile,
  warp tile). Read the launch instantiation to recover M/N/K tiling + dtype, then express the same GEMM /
  attention as a TileLang `T.gemm` / FA program and autotune. Keep the dtype & accumulation exactly.
- **From HIP (`__global__`)**: identify the math, ignore the manual index arithmetic; re-express as a
  TileLang tile program. Don't transliterate thread indexing.
- **From PyTorch**: as above by op type.

## Hard rules
- Preserve the interface and numerics. No silent dtype/shape changes.
- One launch per logical op — never a Python `for` loop calling `T.gemm` per batch/head/group.
- Respect the 64 KB LDS budget (bounds `num_stages` × tile size).
- Prefer `@tilelang.autotune` over hard-coded tile shapes.
- A correct, simple TileLang kernel that beats baseline is better than a clever one that fails correctness.
