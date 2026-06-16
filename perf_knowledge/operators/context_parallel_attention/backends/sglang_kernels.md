---
title: context_parallel_attention on SGLang kernels — SOTA card
kind: sota_card
operator: context_parallel_attention
backend: sglang_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill]
status: experimental
updated: 2026-06-08
sources:
  - https://github.com/sgl-project/sglang/issues/22223
  - https://github.com/sgl-project/sglang/blob/main/docs/platforms/amd_gpu.md
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# context_parallel_attention × SGLang kernels

## TL;DR
SGLang is where CP gets **orchestrated** for serving: the prefill-context-parallel proposal combines
**zigzag ring attention** + **split-KV transfer** for long-context (256K+) and PD-disaggregation. The
attention kernels underneath are AITER/Triton (above); SGLang owns the **ring loop, zigzag token
assignment, KV transfer, and the hybrid scheduling**. On ROCm this is **maturing** — the explicit
ring/CP work was first prototyped on Ascend NPU, and MI300X-native CP support is still landing. Use it for
long-context prefill on multi-GPU; expect to verify coverage per model/version.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| SGLang prefill CP (zigzag ring + split-KV transfer) | `sgl-project/sglang` issue #22223 + CP scheduling (`managers/`) | gfx942/950; bf16/fp16 | no published MI300X CP-scaling number; proposal/maturing | 256K+ long-context prefill, PD-disagg |
| SGLang attention backends (local tile) | `--attention-backend {aiter,triton,tilelang}` | gfx942/950 | per [[sglang_kernels]] attention card | the local FA tile inside CP |

> **MI300X-native ring/CP is experimental** (initially prototyped on Ascend NPU; ROCm support landing). No
> measured AMD CP-scaling figure published as of 2026-06.

## Config space / knobs
- CP degree, `--prefill-attention-backend` (local tile), zigzag balancing, split-KV transfer for
  PD-disagg.
- ROCm env: `SGLANG_USE_AITER=1`, `HSA_NO_SCRATCH_RECLAIM=1`, `allgather_reducescatter` collective,
  stay in the 8-GPU XGMI island (TP) before CP. See [tuning.md](../tuning.md).

## Numerics / parity
fp32 LSE-merge; zigzag un-permutation; reduction order differs across AITER/Triton/CK → greedy/temp=0
parity after any backend swap. See [numerics.md](../numerics.md).

## Integration (rebind seam)
CP is configured at launch; the local kernel is the `--attention-backend` choice. The ring/KV-transfer
code is in `managers/` + the disaggregation path — that's the integration surface, not a single kernel.

## Pitfalls & anti-patterns
- ROCm CP is newer than the CUDA path — verify your model/version actually engages CP (not silent TP).
- Cross-island/cross-node CP comm is the bottleneck on MI300X (XGMI < NVLink) — TP-first.
- Backend swap changes reduction order → re-gate accuracy.

## How to verify
Confirm CP engaged (not TP fallback) in the server log; TTFT scaling vs `cp`; parity vs single-GPU at a
fits-in-HBM seq; `rocprofv3` XGMI overlap.

## Alternatives / cross-links
[overview.md](../overview.md) · [triton.md](triton.md) · [aiter.md](aiter.md) · backend: [[sglang_kernels]] ·
[[mori_rccl]] (EP all-to-all) · core: [[attention_prefill_fmha]] · [[chunked_prefill]].

## Sources
- SGLang prefill CP / zigzag ring + split-KV (Ascend-first, ROCm maturing): https://github.com/sgl-project/sglang/issues/22223
- SGLang AMD docs: https://github.com/sgl-project/sglang/blob/main/docs/platforms/amd_gpu.md
- ROCm topology / TP-first / allgather_reducescatter: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
