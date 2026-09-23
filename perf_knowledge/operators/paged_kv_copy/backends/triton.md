---
title: paged_kv_copy on Triton — SOTA card
kind: sota_card
operator: paged_kv_copy
backend: triton
gens: [gfx942, gfx950, gfx1151]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode, prefill, both]
status: competitive
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://docs.vllm.ai/en/latest/design/paged_attention/
---

# paged_kv_copy × Triton

## TL;DR
A Triton `reshape_and_cache` (one program per token, BLOCK over head_size) is the **portable / fallback**
KV write — used when the AITER/vLLM HIP path is missing for a shape, and the path the Triton attention
backend (`TRITON_ATTN`) pairs with. Correct and tunable, but it loses the AMD-specific shuffled-layout / asm
zero-conversion advantage of the HIP path; prefer aiter/vllm_kernels on the hot serving path.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton reshape_and_cache (BLOCK over head_size) | aiter `aiter/ops/triton/*` / vLLM Triton path | gfx942/950, bf16/fp16/FP8 | memory-bound; portable; no public AMD GB/s — measure | TRITON_ATTN, missing HIP shape |

## Config space / knobs
- BLOCK over head_size for 128-bit coalesced writes; `num_warps` 2–4, `num_stages=1` (memory-bound).
- Grid = num_tokens (decode = tiny; graph-capture matters more than the kernel).
- AMD knobs in `triton.Config({...})`.

## Numerics / parity
non-quant exact; FP8 (use **fnuz** dialect on gfx942 — `tl.float8e4b8`, not OCP) → accuracy-gate. See
[[operators/paged_kv_copy/numerics.md]], [[languages/triton_amd/pitfalls.md]].

## Integration (rebind seam)
`@triton.jit` from the attention backend / KV manager; pairs with `--attention-backend TRITON_ATTN`. No
shuffled-layout op — that's the HIP path's edge.

## Pitfalls & anti-patterns
- ⚠ OCP fp8 into a Triton KV op on gfx942 → wrong dialect / 2× error; use fnuz.
- ⚠ Decode launch overhead — graph-capture; the Triton kernel itself is tiny.
- ⚠ Missing the shuffled layout → a per-step conversion the HIP path avoids; don't use Triton KV write under
  `pa_fwd_asm` at high concurrency.

## How to verify
`TRITON_PRINT_AUTOTUNING=1`; rocprofv3 → write inside graph, coalesced; oracle `allclose`.

## Alternatives / cross-links
[backends/aiter.md](aiter.md) · [backends/vllm_kernels.md](vllm_kernels.md) · [backends/hip.md](hip.md) ·
[[languages/triton_amd/patterns.md]] · [[operators/attention_decode_paged/overview.md]].

## Sources
- aiter Triton KV ops: ROCm/aiter@a6bb49937:aiter/ops/triton/.
- Triton AMD tuning / fnuz fp8: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html · [[languages/triton_amd/pitfalls.md]].
- Paged KV layout: https://docs.vllm.ai/en/latest/design/paged_attention/

## On gfx1151 (RDNA3.5, Strix Halo)
`gfx1151` appears in `gens:` because this backend's **source is portable** to RDNA — it
compiles/JITs there with no vendor asset table. It is **not** a claim that anything on this
card was measured on RDNA: every perf number, ranking and tuning recipe above is CDNA
(gfx90a/942/950) evidence. Three differences bite before any of it transfers — WMMA not MFMA,
wave32 not wave64, and 40 CU behind a 32 MB MALL on ~229 GB/s shared LPDDR5X rather than HBM —
so tile shapes, occupancy targets and the roofline ceiling all move. Any `fp8_*` entry in
`dtypes:` above is CDNA-only: gfx1151 has **no fp8 matrix instruction and no block-scaled
FP4/FP6**, so an fp8 candidate there runs EMULATED — it passes correctness and loses
performance silently. See [`../../../hardware/rdna35_gfx1151/`](../../../hardware/rdna35_gfx1151/)
and MEASURE on the box.
