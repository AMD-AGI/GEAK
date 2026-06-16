---
title: shared_expert_fusion on HIP/asm — SOTA card
kind: sota_card
operator: shared_expert_fusion
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/fused_moe_dp_shared_expert.py
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# shared_expert_fusion × HIP/asm

## TL;DR
> The shared-expert fusion is built from the same HIP/asm grouped-GEMM kernels as the routed MoE
> (`fmoe_stage1_*` asm + CK stage-2) plus an **atomic-add** epilogue that writes the shared output into the
> routed result. HIP is the seam to own the atomic-add scheme, the shared/routed overlap (separate HW
> queues), and the dense shared-GEMM tiling. Reach for it to customize the fusion glue.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| shared stage-1/stage-2 + atomic-add | aiter `fused_moe_dp_shared_expert.py` over `fmoe_stage1_*` asm + CK stage-2 | gfx942/950; bf16, fp8 | shares the up-to-3× fused-MoE + Wide-EP wins | DeepSeek shared fused with routed |
| custom HIP shared MLP + `global_atomic_add` | `languages/hip_cpp/` + `-munsafe-fp-atomics` | gfx942/950 | — | a bespoke overlap/atomic scheme |

## Config space / knobs
- **HW fp atomics**: compile with `-munsafe-fp-atomics` so the shared output atomic-adds via
  `global_atomic_add_f32` (HW), not a CAS loop.
- **Overlap**: shared dense GEMM on a separate stream / `GPU_MAX_HW_QUEUES=2` to run concurrently with the
  routed grouped GEMM.
- **Shared tiling**: shared M = all tokens → dense-GEMM tiling (256×256-style, fill 304 CUs), not skinny
  per-expert tiles.
- MFMA 16×16; fp32 acc; LDS 64 KB (CDNA3) / 160 KB (CDNA4); `__launch_bounds__` for VGPR.

## Numerics / parity
Atomic-add order → benign bf16 deltas; HW fp atomics are fine for inference but non-deterministic order.
Shared weight = 1. fp8 fnuz on gfx942. See [numerics.md](../numerics.md).

## Integration (rebind seam)
aiter compiles the kernels JIT/AOT; the atomic-add target is the routed result buffer passed into
`fused_moe_dp_share_expert`. To customize, edit the aiter asm/HIP and rebuild; or author a standalone shared
MLP that atomic-adds into the MoE output buffer.

## Pitfalls & anti-patterns
- Without `-munsafe-fp-atomics` the atomic-add falls back to a slow CAS loop.
- Atomic contention on a hot output buffer — disjoint token ranges where possible.
- Tiling the shared (dense, large-M) GEMM like a skinny per-expert one → under-utilization.

## How to verify
Disassemble: confirm `global_atomic_add_f32` (not CAS); rocprof for shared/routed overlap; isolated shared
GEMM timing; greedy parity.

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [`languages/hip_cpp/`](../../../languages/hip_cpp/overview.md) ·
[`languages/asm_mfma/`](../../../languages/asm_mfma/overview.md) · [overview.md](../overview.md).

## Sources
- atomic-add fusion: `ROCm/aiter@a6bb49937:aiter/fused_moe_dp_shared_expert.py`.
- HW fp atomics / `-munsafe-fp-atomics`: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
