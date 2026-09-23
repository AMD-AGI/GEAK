---
title: attention_decode_paged on HIP (vLLM custom paged-attn) — SOTA card
kind: sota_card
operator: attention_decode_paged
backend: hip
gens: [gfx942, gfx950, gfx1151]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/tree/main/csrc/rocm
  - https://github.com/vllm-project/vllm/blob/main/csrc/rocm/torch_bindings.cpp
  - https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
---

# attention_decode_paged × HIP (vLLM custom paged-attn)

## TL;DR
vLLM ships its **own hand-written HIP paged-attention** decode kernel (`csrc/rocm/attention.cu`, the
`ROCM_ATTN` backend) — the **editable HIP source** for a Tier-C decode rewrite, and strong on decode
without AITER. Use it when you want to own/modify the decode kernel, or as the `ROCM_ATTN` fallback when
AITER lacks a path. It is splitKV/MFMA-based and templated on `BLOCK_SIZE` / KV dtype / fp8 KV.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| vLLM custom paged-attn (`ROCM_ATTN`) | `vllm-project/vllm:csrc/rocm/attention.cu` | gfx942/950; bf16/fp16/fp8 KV | strong decode without AITER; **2.7–4.4× slower** when KV head size unsupported (falls to Triton) | editable decode; AITER-free path |

Kernels to grep in a profile:
- `paged_attention_ll4mi_QKV_mfma16_kernel` — MFMA-16 main path
- `paged_attention_ll4mi_QKV_mfma4_kernel` — MFMA-4 small-head path
- `paged_attention_ll4mi_reduce_kernel` — cross-split softmax reduce

## Config space / knobs
`VLLM_ROCM_CUSTOM_PAGED_ATTN=1` (engage). Template params: `BLOCK_SIZE` (page size), KV dtype, fp8 KV.
When rewriting: `matrix_instr_nonkdim=16`, `waves_per_eu` (decode is memory-bound → 3-4), splitKV
partition size, KV-cache layout (the reshaped `[2, num_blocks, block_size*kv_heads*head]` with `x` inner
split for 128-bit reads). Exposed via `rocm_ops.def("paged_attention", ...)` in `torch_bindings.cpp`.

## Numerics / parity
fp32 online-softmax accumulate; splitKV reduce. fp8 KV uses scaled reads (fnuz on gfx942 — wrong dialect
off by 2×). Custom-HIP vs AITER vs Triton reduction order differs → re-check greedy temp=0 parity. See
[../numerics.md](../numerics.md).

## Integration (rebind seam)
`--attention-backend ROCM_ATTN`. The HIP source (`attention.cu`) is the **Tier-C edit seam** — edit and
**rebuild vLLM** (not a Python-only change). `torch_bindings.cpp` is the registration surface. Dispatch
order on gfx942 (`vllm/platforms/rocm.py`): ROCM_ATTN → ROCM_AITER_UNIFIED_ATTN → TRITON_ATTN, with
AITER MLA/MHA inserted when enabled.

## Pitfalls & anti-patterns
- **V0-era vars silently ignored on V1** (`VLLM_USE_TRITON_FLASH_ATTN`, `VLLM_USE_ROCM_FP8_FLASH_ATTN`) —
  selection is the `--attention-backend` enum + AITER hierarchy.
- KV head size unsupported by the HIP path → Triton decode fallback (2.7–4.4× slower).
- Editing `csrc/rocm/*.cu` requires a vLLM rebuild.
- fp8 KV is an accuracy gate; fnuz re-check on MI300X.

## How to verify
rocprofv3 kernel-trace → confirm `paged_attention_ll4mi_*` actually ran (not a Triton fallback). Isolated
decode bench vs `ROCM_AITER_FA` / `TRITON_ATTN` at the served batch. Greedy temp=0 parity.

## Alternatives / cross-links
[aiter.md](aiter.md) · [vllm_kernels.md](vllm_kernels.md) · [triton.md](triton.md) ·
`backends/vllm_kernels/rocm_kernels.md` · `languages/hip_cpp/` · [[../overview.md]].

## Sources
- vLLM ROCm custom HIP kernels (`attention.cu`, `ll4mi_*` names): https://github.com/vllm-project/vllm/tree/main/csrc/rocm
- ROCm op registration: https://github.com/vllm-project/vllm/blob/main/csrc/rocm/torch_bindings.cpp
- Dispatch order / ROCM_ATTN fallback cliff: https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py ; https://vllm.ai/blog/2026-02-27-rocm-attention-backend

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
