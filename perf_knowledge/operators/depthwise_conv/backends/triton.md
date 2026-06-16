---
title: depthwise_conv on Triton — SOTA card
kind: sota_card
operator: depthwise_conv
backend: triton
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16]
regimes: [both]
status: competitive
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# depthwise_conv × Triton

## TL;DR
Triton is the fastest way to **author** a depthwise conv and the path PyTorch-Inductor emits when it
lowers `conv` under `max-autotune` and beats MIOpen — but on AMD the default and usually-best production
depthwise backend is **MIOpen** ([miopen.md](miopen.md)). Reach for Triton when you need a **fused**
depthwise+epilogue MIOpen can't express, a non-standard shape MIOpen has no fast solver for, or the
`torch.compile` codegen path. The op is memory-bound (no matrix core), so a Triton kernel competes on
coalesced NHWC loads + occupancy, not on MFMA.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Inductor depthwise-conv lowering (max-autotune) | PyTorch Inductor / Triton | gfx942/950; fp16/bf16/fp32 | competitive only when it beats MIOpen on that shape (Inductor benchmarks both and picks) | torch.compile path; fused epilogue |
| Hand-written Triton depthwise conv | author via kernel layer | gfx942/950 | shape-specific; can win odd shapes MIOpen lacks a fast solver for | custom fusion / unsupported shape |

Honest gap: no on-box measurement (vision tail, not on the LLM box). Treat both rows as "author + e2e-gate
vs MIOpen", not as a quoted speedup.

## Config space / knobs
Memory-bound, so the GEMM knobs don't apply (`matrix_instr_nonkdim`, split-K, `num_stages` deep
pipelines — irrelevant). The real knobs: channel-tile `BLOCK_C` (coalesced, 64–256), spatial tile
`BLOCK_H×BLOCK_W`, `num_warps` (wave64 → 2–4, **not** 8 — VGPR spill = 3–5× slower), `waves_per_eu`,
grid ≥1024 workgroups to fill 304 CUs, NHWC pointers for coalesced loads. Hold the small filter in
registers; LDS-stage the spatial halo only if reused. `num_stages` 1–2.

## Numerics / parity
fp32 accumulate over the spatial window; same-math vs `F.conv2d(groups=C)`, `atol≈1e-2` bf16. See
[../numerics.md](../numerics.md).

## Integration (rebind seam)
On the `torch.compile`/Inductor path Triton is emitted automatically (no manual wiring) when max-autotune
picks it over MIOpen. For a hand-written kernel, register a custom op and rebind the model's conv call,
then e2e-gate. There is no aiter dispatch DB for depthwise conv (contrast dense GEMM).

## Pitfalls & anti-patterns
- AMD Triton: buffer loads not default; verify `global_load_dwordx4` / LDS `ds_read_b128` with
  `AMDGCN_ENABLE_DUMP=1` (see [`../../../languages/triton_amd/isa_verify.md`](../../../languages/triton_amd/isa_verify.md)).
- `num_warps=8` carried from NVIDIA → VGPR spill to scratch (HBM) → 3–5× slowdown. Cut warps first.
- Don't expect to beat MIOpen Winograd on plain 3×3 stride-1 depthwise; the Triton win is fusion or an
  unsupported shape.
- Treat the experimental status honestly — benchmark vs MIOpen on *your* shape before shipping.

## How to verify
`AMDGCN_ENABLE_DUMP=1` ISA check; isolated bench vs the MIOpen `MIOpenDriver --group-count C` run at the
same shape; e2e via Inductor max-autotune log (which backend won) or custom-op rebind + parity.

## Alternatives / cross-links
[miopen.md](miopen.md) (production default) · [hip.md](hip.md) · [../overview.md](../overview.md) ·
language: [`../../../languages/triton_amd/`](../../../languages/triton_amd/) · LLM 1D variant: [[causal_conv1d]].

## Sources
- Optimizing Triton kernels (knobs, ISA verify, num_warps spill): https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- Triton AMD backend HIPOptions / pass pipeline: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- ≥1024 grid, memory-bound tuning, Inductor lowers conv only when it beats the library: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
