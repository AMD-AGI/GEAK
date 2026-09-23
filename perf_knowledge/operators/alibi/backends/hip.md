---
title: alibi on hip — SOTA card
kind: sota_card
operator: alibi
backend: hip
gens: [gfx908, gfx90a, gfx942, gfx950, gfx1151]
dtypes: [bf16, fp16, fp32]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/csrc/rocm/attention.cu
  - https://arxiv.org/abs/2108.12409
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# alibi × hip

## TL;DR
In HIP, ALiBi is a bias term inside the attention kernel. vLLM's custom paged-attention HIP kernel
(`csrc/rocm/attention.cu`, `paged_attention_ll4mi_*`) takes **alibi slopes** and adds `−m_h·(i−j)` to the
fp32 score before the online softmax. CK/aiter FMHA kernels expose a similar bias parameter. There is no
standalone HIP alibi op — it's an attention-kernel feature.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| vLLM paged-attn HIP with alibi slopes | `vllm/csrc/rocm/attention.cu` (`paged_attention_ll4mi_*`) | gfx942/950, bf16/fp16 | inline bias, near-zero cost | ALiBi decode on ROCM_ATTN |
| CK/aiter FMHA bias arg | aiter MHA / CK-Tile FMHA | gfx942/950 | inline | ALiBi prefill |
| (no standalone hip alibi op) | — | — | — | attention feature |

## Config space / knobs
- Pass `alibi_slopes[H]` (device buffer) to the attention kernel; compute bias inline from `(i, j, m_h)`.
- All knobs are the attention kernel's (MFMA tile, `matrix_instr_nonkdim`, `waves_per_eu`, split-K) — see
  [[backends/vllm_kernels/rocm_kernels]].
- fp32 bias add; causal `−inf` after bias.

## Numerics / parity
fp32 bias before softmax; slope sequence + sign + mask order; deterministic bias; attention reduction order
differs → greedy re-gate. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- vLLM: `--attention-backend ROCM_ATTN`; edit `csrc/rocm/attention.cu` to change/extend the alibi path,
  rebuild (Tier-C).
- aiter/CK: the FMHA kernel's bias argument.

## Pitfalls & anti-patterns
- ⚠ Materializing the bias → `O(seq²)` HBM.
- ⚠ Wrong slopes/sign/non-pow2 heads.
- ⚠ Editing `csrc/rocm/attention.cu` needs a vLLM rebuild.

## How to verify
rocprofv3 confirms the paged-attn kernel ran; attention-with-alibi vs fp64; greedy parity.

## Alternatives / cross-links
[triton.md](triton.md) · [[attention_decode_paged]] · [[backends/vllm_kernels/rocm_kernels]] ·
[[languages/hip_cpp/patterns]] · [[rope]] (mutually exclusive).

## Sources
- vLLM paged-attn alibi: https://github.com/vllm-project/vllm/blob/main/csrc/rocm/attention.cu.
- ALiBi: https://arxiv.org/abs/2108.12409.
- attention kernel knobs: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html.

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
