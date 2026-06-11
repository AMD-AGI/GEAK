---
title: linear_attention_gated_delta on HIP/C++ — SOTA card
kind: sota_card
operator: linear_attention_gated_delta
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
status: experimental
updated: 2026-06-08
sources:
  - https://github.com/fla-org/flash-linear-attention
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0
---

# linear_attention_gated_delta × HIP/C++

## TL;DR
**No public hand-written HIP/asm Gated-DeltaNet scan kernel exists on AMD as of 2026-06.** The chunked
scan + triangular solve is shipped in **Triton** (FLA / aiter) — see [triton.md](triton.md). HIP/C++'s
role here is the **causal-conv1d pre-step** (which has mature HIP/CK implementations) and any custom
state-update glue. Reach for HIP only if you need a fused conv1d+gate the Triton path can't express, or to
own the exact ISA of the recurrent decode kernel.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| causal conv1d (HIP/CK) pre-step | aiter `causal_conv1d` (HIP) + Triton variants | gfx942/950; bf16/fp16 | standard fused conv | the GDN conv1d pre-step ([[causal_conv1d]]) |
| custom HIP recurrent decode | author | gfx942/950 | none published | last-resort decode latency tuning |

> **Primarily Triton-portable; runs on MI300X via [[triton_amd]]. No hand-tuned CK/asm/HIP GDN scan kernel
> known as of 2026-06.** The portable path is FLA/aiter Triton; HIP covers conv1d + glue only.

## Config space / knobs
- HIP wave64: block = multiple of 64, `__launch_bounds__(…,minWavesPerEU)` to keep the state in registers,
  `__restrict__`, `-munsafe-fp-atomics` if a chunk reduction uses atomics. See [[hip_cpp]].
- The decode recurrent kernel is bandwidth-bound on S → maximize occupancy, keep S in VGPR/LDS.

## Numerics / parity
fp32 state accumulate; bf16 MFMA-intrinsic/dtype match is critical (mismatch corrupts the recurrence). See
[numerics.md](../numerics.md).

## Integration (rebind seam)
HIP conv1d/glue registered as custom ops, called from the GDN layer; the attention matmul/scan stays in
Triton. Verify via `rocprofv3` Top-N.

## Pitfalls & anti-patterns
- Reimplementing the chunked scan in HIP is high-effort and unlikely to beat the FLA/aiter Triton kernels —
  not recommended without a measured reason.
- Decomposing the state update (HBM round-trips) → 10–50× slower.

## How to verify
State parity vs FLA fp32 reference; conv1d unit test; `rocprofv3` to confirm the HIP conv1d isn't the
bottleneck vs the Triton scan.

## Alternatives / cross-links
[overview.md](../overview.md) · [triton.md](triton.md) · [tilelang.md](tilelang.md) · languages: [[hip_cpp]] ·
[[triton_amd]] · op: [[causal_conv1d]].

## Sources
- aiter conv1d (HIP) + Triton GDN: `ROCm/aiter@a6bb49937:aiter/ops/causal_conv1d.py`, `aiter/ops/triton/gated_delta_net/` (on-box).
- HIP wave64 / launch_bounds: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- FLA reference: https://github.com/fla-org/flash-linear-attention
