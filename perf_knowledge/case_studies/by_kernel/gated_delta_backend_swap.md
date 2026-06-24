---
title: Gated-delta hybrid — the --attention-backend triton e2e win and the backend-selection method
kind: case_study
operator: linear_attention_gated_delta
backend: triton_amd
gens: [gfx942]
dtypes: [bf16]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - GEAK/examples/e2e_workflow/qwen3.5-27b_sglang_isl1024_osl1024_conc64/final_report.md
  - GEAK/e2e_workflow/knowledge/gemm_attention_backends.md
  - GEAK/e2e_workflow/knowledge/gemm_tuning/aiter_gemm_tuning.md
  - https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-5-on-amd-instinct-gpus.html
---

# Gated-delta hybrid: the `--attention-backend triton` win

> The e2e numbers here are **measured-by-us** (Qwen3.5-27B, MI300X gfx942, sglang 0.5.11). One
> caveat up front: the *same lever* is reported at **+4.15%** and **+4.44%** in the per-iteration
> eval dirs and **+4.96% / "+~5%"** in the cross-run knowledge ledger — the spread is gfx942 box
> drift across sessions. All are measured; the auditable eval-dir figures are +4.15/+4.44%.

## Context
Qwen3.5-27B is a **hybrid linear-attention model**: 48 of 64 layers are gated-delta (Mamba2-style
linear attention), 16 are full attention (`full_attention_interval=4`). The attention *path* is a
small share of GPU time; the **gated-delta Triton kernels** are the editable mass, and the
full-attention layers can run on different backends. `--attention-backend triton` is a **server
flag** — the cheapest possible e2e lever (no source change) — and on this model it both **wins
e2e** and **opens an editable Triton surface** for downstream kernel work. Operator background:
[`../../operators/linear_attention_gated_delta/`](../../operators/linear_attention_gated_delta/).
Method: [`../../kernel_workflow/attention_backend_selection.md`](../../kernel_workflow/attention_backend_selection.md).

## Baseline
- Default sglang attention path on this model = **CK paged-attention** (the full-attn layers),
  measured at ~1.1% GPU time in the profile.
- True e2e baseline (iter1): **1485.432 tok/s** (median of 3 warm, spread 0.44%), TP=1, GPU0.

## What we tried (the backend-selection method)
The config track is a flag/env sweep, gated on the tight same-session A/B. Two candidates were
relevant:

| candidate | e2e | engagement | parity | verdict |
|---|---|---|---|---|
| **`--attention-backend triton`** (cfg0) | **+4.15%** (1485.4 → 1547.0), spread 0.8% | ✅ server.log: `attention_backend='triton'`, `linear_attn_backend='triton'`, `mamba_backend='triton'` | PASS (greedy temp=0, 5 prompts: 3 byte-identical, 2 benign bf16 tie-break in repeated tail, answers identical) | **ACCEPT** |
| `--chunked-prefill-size 8192` (cfg1, stacked) | −0.42% (in-band, slightly negative) | ✅ `chunked_prefill_size=8192` | — | reject (cuda-graph already on; prefill-dominated → no decode-interleave headroom) |

The procedure is the general one: candidate flags = current accepted config + the one swap,
**gate on `delta% > 0.5% AND cand_min > ref_max`**, prove engagement in the server log, check
parity (a cross-backend bf16 argmax flip is real → ≥10 prompts).

## What worked / what didn't
- **Worked:** `--attention-backend triton`, far over the 0.5% gate, engaged on all three
  sub-paths (attention / linear-attn / mamba), parity clean. **Side benefit:** it converted the
  16 full-attn layers and the linear-attn/mamba paths to **editable Triton kernels**, which is
  what made the downstream FLA/mamba kernel track possible at all.
- **Didn't:** `--chunked-prefill-size 8192` (no decode-interleave headroom in a prefill-dominated
  run). And the **editable gated-delta kernels it exposed** were all carry-forward by Amdahl —
  real isolated speedups (`chunk_gated_delta_rule_fwd_kernel_h` 1.18×, `chunk_fwd_kernel_o`
  1.14×, `_causal_conv1d_fwd_kernel` 1.10× parity 12/12) but each ≤3% GPU → e2e inside the band.
  See [`../by_model/qwen3.5-27b_sglang_e2e.md`](../by_model/qwen3.5-27b_sglang_e2e.md).

## Final result (numbers, measured-by-us)
| stack | e2e tok/s | vs baseline | notes |
|---|---|---|---|
| baseline (CK attn) | 1485.4 | — | iter1 |
| + `--attention-backend triton` (cfg0) | 1547.0 | **+4.15%** | 5-repeat, engaged, parity PASS |
| + triton attn (final retest) | 1551.4 | **+4.44%** | spread 0.33% |
| cross-run ledger figure | — | **+4.96% / "+~5%"** | knowledge ledger, box drift |

It is **orthogonal to the GEMM tune and compounds** — stacking
[`gemm_aiter_db_tuning.md`](gemm_aiter_db_tuning.md) on top reached **≈ +6% cumulative**. AMD's
Qwen3.5 day-0 note confirms the recommended launch uses `--attention-backend triton` for the
hybrid GDN path (vendor).

## Lessons
1. **The default is not always best — A/B the backend swap.** This is the validated
   counter-example: on a gated-delta hybrid, Triton attention beat the CK default by ~4–5%.
2. **It's a `winner_kind=flag` change** — cheapest lever, no source; always try the config track
   first.
3. **Gate every swap** (same-session A/B + engagement in server.log + parity); a cross-backend
   bf16 argmax flip is real, so check ≥10 prompts.
4. **A config swap reshapes the profile** — re-profile after accepting it before extracting
   kernels (the bottleneck stayed GEMM here, but the attention path changed identity).
5. **Stack orthogonal levers** (attention flag + GEMM tune compound to ~+6%); gate the stack vs
   the true baseline.

## Cross-links
- Backend-selection decision tree + per-gen ranking: [`../../kernel_workflow/attention_backend_selection.md`](../../kernel_workflow/attention_backend_selection.md)
- Operator: [`../../operators/linear_attention_gated_delta/`](../../operators/linear_attention_gated_delta/) · triton card: [`../../operators/linear_attention_gated_delta/backends/triton.md`](../../operators/linear_attention_gated_delta/backends/triton.md)
- The GEMM lever it stacks with: [`gemm_aiter_db_tuning.md`](gemm_aiter_db_tuning.md)
- The full run: [`../by_model/qwen3.5-27b_sglang_e2e.md`](../by_model/qwen3.5-27b_sglang_e2e.md)
- e2e flow / gate: [`../../kernel_workflow/optimize_e2e_model.md`](../../kernel_workflow/optimize_e2e_model.md)

## Sources
- The +4.15% / +4.44% measured win, engagement, parity, the chunked-prefill reject: `GEAK/examples/e2e_workflow/qwen3.5-27b_sglang_isl1024_osl1024_conc64/final_report.md`.
- Cross-run +4.96% / "+~5%" ledger figure + the stack-with-GEMM note: `GEAK/e2e_workflow/knowledge/{gemm_attention_backends.md,gemm_tuning/aiter_gemm_tuning.md}`.
- AMD Qwen3.5 day-0 (`--attention-backend triton` for hybrid GDN, vendor): https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-5-on-amd-instinct-gpus.html

<!-- MANIFEST: gated-delta hybrid backend swap on Qwen3.5-27B/MI300X — --attention-backend triton = +4.15–4.44% measured (+4.96% ledger), engaged + parity, opens editable Triton surface, stacks with GEMM tune to ~+6%. -->
