---
title: DeepSeek MLA decode on MI300X via aiter — the 17× decode case
kind: case_study
operator: mla_attention
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py
---

# DeepSeek (MLA) decode on MI300X via aiter

> **The headline numbers on this page are vendor-reported** (AMD ROCm blog + vLLM ROCm
> attention-backend blog), labelled inline. We have not re-measured DeepSeek MLA in the
> `e2e_workflow` eval dirs; treat these as the sourced vendor baseline you should A/B
> against on your own box, not as our own measurement.

## Context
DeepSeek-V3 / R1 (and Kimi) use **Multi-head Latent Attention (MLA)**: the KV is compressed to a
**latent of `kv_lora_rank=512` + a `rope=64` carry** (576-dim), so decode runs as **MQA on the
absorbed latent** rather than full multi-head attention over a large KV cache. The win — and the
risk — is the absorption: it collapses the KV memory traffic, but only if the kernel works
directly on the latent. On AMD the SOTA path is **aiter** (`aiter/mla.py`): `mla_decode_fwd`
(hand-tuned assembly, matrix-absorbed) for decode, `mla_prefill_fwd` / `mla_prefill_ps_fwd`
(persistent) for prefill. See [`../../operators/mla_attention/`](../../operators/mla_attention/)
and the SOTA card [`../../operators/mla_attention/backends/aiter.md`](../../operators/mla_attention/backends/aiter.md).

## Baseline (what aiter MLA is measured against)
- **Naive / decomposed MLA decode** (materializing per-head KV instead of working on the latent)
  — the reference the 17× is quoted against.
- **Triton MLA** (`TRITON_MLA`) — the editable/portable fallback; the vLLM blog ranks it
  **1.33–1.52× slower** than aiter MLA on TPS across gfx942/gfx950.

## What works on AMD (the recipe)
- **Backend selection (server flag):**
  - sglang: `--attention-backend aiter` → dispatches `from aiter.mla import mla_decode_fwd,
    mla_prefill_fwd` (the literal seam in `aiter_backend.py`); optional
    `SGLANG_ROCM_FUSED_DECODE_MLA=1`, persistent prefill `SGLANG_AITER_MLA_PERSIST=1`.
  - vLLM: `--attention-backend ROCM_AITER_MLA` **plus the master switch
    `VLLM_ROCM_USE_AITER=1`** (and `VLLM_ROCM_USE_AITER_MLA=1`).
- **Per-gen ranking (vendor, vLLM blog, ISL=10K/OSL=1K, 64 conc):**
  - **gfx942 (MI300X 192GB / MI325X):** `ROCM_AITER_TRITON_MLA` is **~2–3% higher TPS** than the
    asm MLA — the recommended pick on this gen.
  - **gfx950 (MI355X 288GB):** `ROCM_AITER_MLA` (asm MHA prefill) matches/beats Triton and gives
    best TTFT → keep the default.
- **Config knobs:** leave `num_kv_splits=None` (auto, shape-aware); decode contract is
  `nhead_kv==1`, `page_size==1` (fast unpaged path); fp8 latent/KV via `q_scale`/`kv_scale`.
- **Deployment wiring:** [`../../quantization/deployment_recipes.md`](../../quantization/deployment_recipes.md)
  (the AITER master switch is where the fused-kernel speedups are realized).

## What didn't / the traps (kept honestly)
- **fp8 latent/KV is accuracy-sensitive.** AITER MLA has shown eval regressions (gsm8k loss,
  aiter #1455) → **task-accuracy gate, not byte parity**. aiter MLA decode does **not** support
  fp8 KV-cache in vLLM upstream (vendor). Don't enable fp8 KV on MLA without an accuracy probe.
- **Silent Triton fallback on gfx942 for newest variants** (e.g. sparse MLA) → several × slower.
  Confirm the asm kernel actually fired with `AITER_LOG_MORE=1` before trusting a number.
- **Wrong decode contract** (`nhead_kv`/`page_size` ≠ 1) misses the fast unpaged path.
- **Hand-setting `num_kv_splits`** defeats the shape-aware auto-tuning.

## Final result (numbers, vendor-reported)
| metric | value | source / label |
|---|---|---|
| `mla_decode_fwd` vs naive decode | **up to 17×** (MI300X, 2025-03) | **vendor** — ROCm aiter-mla blog |
| AITER MLA serving vs Triton MLA | **1.2–1.6× TPOT, up to 1.5× TPS** (2026-01-29) | **vendor** — vLLM ROCm blog |
| `TRITON_MLA` penalty vs `AITER_MLA` (TPS) | 1.33–1.52× slower | **vendor** — vLLM ROCm blog |
| `AITER_TRITON_MLA` on gfx942 | +2–3% TPS over asm MLA | **vendor** — vLLM ROCm blog |

The absorption itself is **algebraically exact (bf16 parity-safe)** — the 17× is a memory-traffic
win on decode, not an approximation.

## Lessons
1. **MLA decode is memory-bound on the latent** — the entire win is keeping the latent absorbed
   and never materializing per-head KV. aiter's asm `mla_decode_fwd` is the only path that does
   this at SOTA on AMD.
2. **Pick the backend by gen:** `AITER_TRITON_MLA` on gfx942, `AITER_MLA` on gfx950 — the ranking
   flips, so A/B both rather than assuming.
3. **fp8 on MLA is an accuracy decision, not a free bandwidth win** — gate it.
4. **Prove the asm kernel fired** (`AITER_LOG_MORE=1`); a silent Triton fallback masquerades as
   a slow aiter run.

## Cross-links
- MLA operator + SOTA cards: [`../../operators/mla_attention/`](../../operators/mla_attention/) · aiter card: [`../../operators/mla_attention/backends/aiter.md`](../../operators/mla_attention/backends/aiter.md) · triton fallback: [`../../operators/mla_attention/backends/triton.md`](../../operators/mla_attention/backends/triton.md)
- aiter MLA internals: [`../../backends/aiter/attn_mla.md`](../../backends/aiter/attn_mla.md)
- Backend selection: [`../../kernel_workflow/attention_backend_selection.md`](../../kernel_workflow/attention_backend_selection.md)
- Decode paged / KV: [`../../operators/attention_decode_paged/`](../../operators/attention_decode_paged/) · [`../../operators/kv_cache_quant/`](../../operators/kv_cache_quant/)
- Serving wiring: [`../../quantization/deployment_recipes.md`](../../quantization/deployment_recipes.md)

## Sources
- 17× MLA decode vs naive (vendor, MI300X, 2025-03): https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
- 1.2–1.6× TPOT / up to 1.5× TPS, Triton-MLA penalty, gfx942-vs-gfx950 ranking, fp8 KV unsupported in vLLM (vendor, 2026-01-29): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- Kernel seam (`mla_decode_fwd`, `mla_prefill_fwd`, persistent mode, fp8 scales): `ROCm/aiter@a6bb499:aiter/mla.py`.

<!-- MANIFEST: DeepSeek MLA decode on MI300X via aiter — vendor-reported 17× decode vs naive, 1.2–1.6× TPOT / 1.5× TPS vs Triton MLA; backend = ROCM_AITER_MLA (gfx950) / AITER_TRITON_MLA (gfx942); fp8 KV accuracy-gated. -->
