---
title: moe_dispatch_combine on HIP/C++ — SOTA card
kind: sota_card
operator: moe_dispatch_combine
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - https://gau-nernst.github.io/amd-a2a/
  - https://github.com/ROCm/mori
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/moe_sorting.py
---

# moe_dispatch_combine × HIP/C++

## TL;DR
> HIP/C++ is where dispatch/combine is actually written — MoRI-EP's kernels, vLLM/SGLang's local
> permute/scatter, and the reference single-kernel a2a studies are all HIP. Reach for HIP to author a
> custom intra-node dispatch (P2P symmetric memory + xGMI), to fuse the grouped GEMM into the a2a sweep,
> or to own the exact grid/buffer behavior. This is the Tier-C seam and the source of the 292 µs result.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Reference single-kernel a2a | gau-nernst (GPU MODE AMD challenge) | gfx942, bf16 | **292 µs** (from 93,540 µs ref) @ MI300X, E=256/topk=8/hidden 7168/world=8, 2025 | study/blueprint for a custom intra-node a2a |
| MoRI-EP HIP kernels | `ROCm/mori` | gfx942/950, fp8/bf16 | 307/330 GB/s (see [mori_rccl.md](mori.md)) | production (consume, don't re-author) |
| aiter `moe_sorting` (local permute) | `aiter/moe_sorting.py`, `csrc/kernels/moe_align_block_size_kernels.cu` | gfx942/950 | — | single-GPU token permute/scatter |

Recommend: consume MoRI-EP for production EP; author HIP only for a custom fusion or an unsupported topology.

## Config space / knobs
- **`grid_size = 304`** (the MI300X CU count) — 304 beat 256 by ~3× on combine-send. Size to full CU count.
- **Pre-allocate registered buffers**; **hoist `torch.empty`/memset** out of the kernel (malloc shows in
  traces — the biggest non-obvious win).
- **P2P symmetric memory**: map each peer's buffer, use direct `load/store` / `__builtin_amdgcn_*` over xGMI
  instead of staged copies. Combine with the grouped-GEMM sweep to remove a kernel boundary.
- **Block/warp**: wave64; `__launch_bounds__` to cap VGPRs; one block per CU (304) for the all-to-all.
- fp8 dispatch: quantize+scale in the send path; bf16 combine.

## Numerics / parity
fp8 dispatch quant gate; bf16 combine; unbiased weight; mask static-pad tokens. Round-trip identity test +
greedy parity. See [numerics.md](../numerics.md).

## Integration (rebind seam)
- aiter: kernels JIT/AOT into `aiter/jit/`; `aiter/moe_sorting.py` for local permute.
- MoRI: `python/mori/ops/dispatch_combine.py` over HIP kernels in `src/`.
- vLLM/SGLang: `sgl-kernel`/`csrc` permute ops registered as torch ops; rebuild after editing `.cu`.

## Pitfalls & anti-patterns
- `grid_size != 304` (e.g. a power of two) → leaves combine bandwidth on the table.
- `hipMalloc`/memset inside the steady-state kernel → caching-allocator cost in the trace.
- `hipCooperativeLaunch` raises L2 pressure on Die-Die exchange — use only if required.
- Forgetting xGMI is a **fully-connected mesh** (no switch): all 8 GPUs needed for full bandwidth.

## How to verify
rocprofv3: grid=304, no malloc steady state, overlap with GEMM; bandwidth vs the mori table; round-trip +
greedy parity.

## Alternatives / cross-links
[mori_rccl.md](mori.md) · [aiter.md](aiter.md) · [triton.md](triton.md) ·
[`languages/hip_cpp/`](../../../languages/hip_cpp/overview.md) · [overview.md](../overview.md).

## Sources
- 292 µs / grid_size=304 / malloc+memset hoist / P2P ladder: https://gau-nernst.github.io/amd-a2a/
- MoRI-EP HIP kernels: https://github.com/ROCm/mori
- aiter local permute: `ROCm/aiter@a6bb49937:aiter/moe_sorting.py`, `csrc/kernels/moe_align_block_size_kernels.cu`.
