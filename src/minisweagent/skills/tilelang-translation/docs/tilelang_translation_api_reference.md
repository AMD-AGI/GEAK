# TileLang translation — API reference (MI300X / gfx942)

TileLang (`tile-ai/tilelang`) is a Python tile DSL on a TVM/TIR backend. It compiles concise tile
programs to AMD via LLVM IR. On MI300X it is **competitive-to-better than Triton** (FA fwd ~1.53× Triton,
~2.7× PyTorch; FlashMLA ~parity with hand-tuned AITER asm) and is one of the two preferred optimization
targets (alongside FlyDSL) because it reaches near-asm performance while staying editable.

## Import surface
```python
import tilelang
import tilelang.language as T
```

## Kernel structure (3 layers)
1. `@tilelang.jit` (or `@tilelang.autotune` + `@tilelang.jit`) — host launcher / compile entry.
2. A kernel function using `T.Kernel(...)` to declare grid/block context.
3. A thin `Model(nn.Module)` (or callable) wrapper that allocates outputs and calls the jit'd kernel.

## Primitive vocabulary
| Primitive | role |
|---|---|
| `T.Kernel(bx, by, ..., threads=N)` | declare kernel grid / block context |
| `T.alloc_shared(shape, dtype)` | allocate an **LDS** (shared-memory) tile |
| `T.alloc_fragment(shape, dtype)` | allocate a **register/fragment** tile (per-lane MFMA storage) |
| `T.alloc_var(dtype)` | scalar accumulator (e.g. running softmax stat) |
| `T.copy(src, dst, coalesced_width=...)` | move global↔shared↔fragment; vectorized/coalesced |
| `T.gemm(A, B, C, transpose_B=..., k_pack=..., policy=GemmWarpPolicy.FullRow)` | MFMA-backed tile GEMM |
| `T.reduce_max(...)` / `T.reduce_sum(...)` | row reductions (FA softmax stats) |
| `T.Parallel(...)` | parallel loop over a tile dim (maps to lanes/threads) |
| `T.Pipelined(range, num_stages=...)` | software-pipelined loop (prefetch depth = `num_stages`) |
| `T.use_swizzle(...)` / `enable_rasterization=True` | block-scheduling swizzle for L2/LLC reuse |

## The performance levers (what the autotuner sweeps)
- **`coalesced_width`** on `T.copy` — set so each lane's global load is ≥128-bit (vectorized).
- **`k_pack`** on `T.gemm` — K elements packed per MFMA operand.
- **`policy=GemmWarpPolicy.FullRow`** — warp→tile mapping (FullRow assigns full output rows to a warp;
  good for the MI300X warp scheduling in AMD's FA blog).
- **`transpose_B=True`** — `S = Q·Kᵀ` without a physical transpose.
- **`num_stages`** on `T.Pipelined` — prefetch depth; larger overlaps loads with MFMA but is bounded by
  MI300X's **64 KB LDS** (over-deep stages spill / get rejected by the autotuner).
- **block/tile shapes** (block_M, block_N, block_K) — primary GEMM/attention tiling knobs.

## Autotune
```python
@tilelang.autotune(configs=[...])   # or pass a search space; TileLang sweeps + caches the best
@tilelang.jit
def kernel(...):
    ...
```
Prefer letting the autotuner sweep block shapes / num_stages / coalesced_width / k_pack / GemmWarpPolicy
rather than hard-coding — the optimum is arch- and shape-specific on gfx942.

## Canonical FlashAttention forward (~80 lines)
Allocate Q in fragments + K/V in shared; loop KV tiles with `T.Pipelined`;
`S = T.gemm(Q, K, transpose_B=True)`; `m = T.reduce_max(S)`; `P = exp(S - m)` via `T.Parallel`;
`l = T.reduce_sum(P)`; `O = O*scale + T.gemm(P, V)`; `T.copy(O, out)`.

## Pitfalls
- `T.annotate_layout` and other low-level layout hints exist but are **not** AMD-validated in the FA blog;
  don't assume a primitive is gfx942-good just because it exists — check a working example.
- Over-deep `num_stages` overruns the 64 KB LDS budget on gfx942.
- `GemmWarpPolicy` interacts with MFMA tile shape — sweep it, don't hard-code blindly.
- TileLang leans on CUTLASS/CK backend calls for some ops — CDNA3-validated, not yet a CDNA4 peak path.
