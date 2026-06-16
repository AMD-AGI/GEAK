---
title: gather_scatter on HIP — SOTA card
kind: sota_card
operator: gather_scatter
backend: hip
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://llvm.org/docs/AMDGPUUsage.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# gather_scatter × HIP

## TL;DR
HIP gives the controls Triton lacks: **HW fp atomics** (`-munsafe-fp-atomics` → `global_atomic_add_f32`),
**direct-to-LDS gather** (scattered global → coalesced LDS, no VGPR staging), and a **fully fused**
down-proj+scatter epilogue (no 2-D-index restriction). Reach for HIP for the production scatter-reduce or
when you need to fold the gather/scatter into a custom kernel.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| one-wg-per-row, `float4` vectorized gather | [[operators/gather_scatter/tuning.md]] | gfx942/950, all dtypes | HBM-bound row payload (measure) | embedding, MoE permute |
| `atomicAdd` scatter-reduce (HW path) | HIP kernel lang | both, fp16/bf16/fp32 | HW atomic, no CAS loop | MoE unpermute |
| `global_load_lds` gather into LDS | [[languages/hip_cpp/lds_async.md]] | gfx942 (`global_load_lds`) / gfx950 (`load.to.lds`) | frees VGPRs, overlaps | rows reused within a wg |

## Config space / knobs
- Vectorize rows (`float4`/`int4`, `__restrict__`, 16-B align) → `global_*_dwordx4`.
- `-munsafe-fp-atomics` for HW fp scatter-add; verify the ISA emits `global_atomic_add_*` not a CAS loop.
- `__launch_bounds__(256,2)`; block multiple of 64; grid ≥1024 (grid-stride over rows for small N).
- Direct-to-LDS gather requires a **coalesced + swizzled** LDS destination ([[operators/transpose/tuning.md]] §3).

## Numerics / parity
gather exact; scatter-reduce atomics → non-deterministic fp (task-accuracy gate). Integer scatter exact.
See [[operators/gather_scatter/numerics.md]].

## Integration (rebind seam)
`.hip` compiled `--offload-arch=gfx942[ gfx950]`, bound via a torch custom op (the Tier-C edit seam). For
MoE this is usually subsumed by aiter's asm/CK sort+grouped-GEMM ([backends/aiter.md](aiter.md)); author HIP
when the catalog lacks your shape or you need a custom fused epilogue.

## Pitfalls & anti-patterns
- ⚠ `atomicAdd` without `-munsafe-fp-atomics` → slow software CAS loop. Check the ISA.
- ⚠ Direct-to-LDS gather without swizzle → bank conflicts (the −28% TFLOPS regression class).
- ⚠ Per-element (un-vectorized) gather → bandwidth-starved.
- ⚠ Heavy atomic contention on a hot output row (skewed indices) — sort/segment-reduce instead.

## How to verify
rocprof memory chart (coalescing/L2, atomic throughput); ISA `global_*_dwordx4` + HW `global_atomic_add_*`;
oracle gather `torch.equal`, scatter `allclose`.

## Alternatives / cross-links
[backends/triton.md](triton.md) · [backends/aiter.md](aiter.md) · [[operators/gather_scatter/tuning.md]] ·
[[languages/hip_cpp/lds_async.md]] · [[languages/hip_cpp/patterns.md]].

## Sources
- Vectorize / LDS-stage guidance: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- HW fp atomics, global_load_lds: https://llvm.org/docs/AMDGPUUsage.html
- HIP kernel lang (atomics, masks): https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
