---
title: attention_decode_paged on vllm_kernels — SOTA card
kind: sota_card
operator: attention_decode_paged
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - https://github.com/vllm-project/vllm/tree/main/csrc/rocm
---

# attention_decode_paged × vllm_kernels

## TL;DR
This card is the **dispatch layer**: how vLLM (V1) on ROCm picks a paged-decode kernel. vLLM does not have
one decode kernel — it routes `--attention-backend` across **two worlds**: AITER (`ROCM_AITER_FA`,
`ROCM_AITER_MLA`) and vLLM's own HIP (`ROCM_ATTN`), with `TRITON_ATTN` as the universal fallback. The
auto-selected defaults (`ROCM_AITER_FA` for MHA, `ROCM_AITER_MLA` for MLA) deliver **1.2–4.4× higher
TPS** and are recommended for all workloads. Use this card to choose; use [aiter.md](aiter.md) /
[hip.md](hip.md) for the kernels themselves.

## SOTA implementation(s) — the backend enum
| `--attention-backend` | file (`v1/attention/backends/`) | kernel | fit |
|---|---|---|---|
| `ROCM_AITER_FA` | `rocm_aiter_fa.py` | AITER flash-attn decode (KV shuffle/gather) | **default MHA decode** |
| `ROCM_AITER_MLA` | `mla/rocm_aiter_mla.py` | AITER MLA decode (asm) | **default DeepSeek MLA** |
| `ROCM_ATTN` | `rocm_attn.py` | vLLM custom HIP paged-attn (`attention.cu`) | strong decode, no AITER |
| `TRITON_ATTN` | `triton_attn.py` | Triton unified attention | universal fallback |

Dispatch order on gfx942 (`vllm/platforms/rocm.py` `get_attn_backend_cls`): ROCM_ATTN →
ROCM_AITER_UNIFIED_ATTN → TRITON_ATTN, with AITER MLA/MHA inserted when
`rocm_aiter_ops.is_mla_enabled()/is_mha_enabled()`.

## Config space / knobs
Master `VLLM_ROCM_USE_AITER=1` (default 0) gates `VLLM_ROCM_USE_AITER_MHA` / `_MLA` (default 1).
`VLLM_ROCM_CUSTOM_PAGED_ATTN=1` (custom HIP decode). `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` (ROCM_AITER_FA,
concurrency ≥32). `--kv-cache-dtype fp8_e4m3` (fnuz on MI300X). `VLLM_ROCM_USE_AITER_FP4BMM=0`
(FP4 crashes gfx942). Recommended ROCm env block in `backends/vllm_kernels/overview.md`.

## Numerics / parity
fp32 accumulate; splitKV reduce. fp8 KV is a task-accuracy gate (fnuz on gfx942). Reduction order differs
across the three worlds → re-check greedy temp=0 parity after a swap. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
Pure dispatch: `--attention-backend {ROCM_AITER_FA, ROCM_AITER_MLA, ROCM_ATTN, TRITON_ATTN}` + the
`VLLM_ROCM_USE_AITER*` hierarchy. Use upstream `vllm/vllm-openai-rocm` image (V1 only; `rocm/vllm-dev`
deprecated Jan 2026).

## Pitfalls & anti-patterns
- V0-era flags (`VLLM_USE_TRITON_FLASH_ATTN`) **silently ignored** on V1.
- `ROCM_ATTN` decode fallback cliff: 2.7–4.4× slower when KV head size unsupported by the HIP path.
- AITER MLA accuracy: caused gsm8k loss with Kimi-K2 DP2TP4 (aiter #1455) — accuracy-gate.
- Confirm the active backend from logs, not the flag alone.

## How to verify
rocprofv3 Top-N → kernel name → world: `paged_attention_ll4mi_*` = vLLM HIP; `fmha_*`/`*ck_*` = AITER/CK;
Python name = Triton. Greedy temp=0 parity after a backend swap; TPOT at served concurrency.

## Alternatives / cross-links
[aiter.md](aiter.md) · [hip.md](hip.md) · [triton.md](triton.md) · `backends/vllm_kernels/overview.md`,
`rocm_kernels.md`, `aiter_integration.md` · [[../overview.md]].

## Sources
- ROCm platform dispatch (backend order, gfx9 list): https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
- 7 backends, defaults, 1.2–4.4× TPS (vendor, 2026-01-29): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- custom HIP kernels: https://github.com/vllm-project/vllm/tree/main/csrc/rocm
