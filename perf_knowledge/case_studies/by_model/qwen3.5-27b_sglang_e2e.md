---
title: Qwen3.5-27B serving throughput on MI300X / sglang — the full e2e optimization run
kind: case_study
operator: dense_gemm
backend: aiter
gens: [gfx942]
dtypes: [bf16]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - GEAK/examples/e2e_workflow/qwen3.5-27b_sglang_isl1024_osl1024_conc64/final_report.md
  - GEAK/examples/e2e_workflow/qwen3.5-27b_sglang_gemm-tuning-win/final_report.md
  - GEAK/e2e_workflow/knowledge/gemm_tuning/aiter_gemm_tuning.md
  - GEAK/e2e_workflow/knowledge/gemm_attention_backends.md
---

# Qwen3.5-27B / sglang / MI300X — full e2e run (the flagship)

> **All e2e numbers on this page are measured-by-us** from the `e2e_workflow` eval dirs
> (two iterations), via the tight same-session A/B protocol. Box drift across sessions shifts
> the absolute tok/s a few % between iterations; this page keeps both iterations' numbers with
> their dates rather than averaging across drift.

## Context
- **Model:** Qwen-Qwen3.5-27B (`Qwen3_5ForConditionalGeneration`), architecture class
  `hybrid_linear_attention_dense`. 64 layers = **48 linear-attention (gated-delta / Mamba2-style)
  + 16 full-attention** (`full_attention_interval=4`); **dense MLP, no MoE**. hidden 5120,
  intermediate 17408, head_dim 256, 24 q / 4 kv heads, vocab 248320, bf16.
- **Stack:** sglang 0.5.11, torch 2.9.1+rocm7.2.0, **TP=1** (single GPU — deterministic Amdahl
  accounting and stable A/B).
- **Hardware:** AMD Instinct MI300X (gfx942), noise band 0.5%.
- **Workload:** ISL/OSL/conc = **1024/1024/64** — **prefill-dominated**.
- **Method:** [`../../kernel_workflow/optimize_e2e_model.md`](../../kernel_workflow/optimize_e2e_model.md)
  (the `e2e_workflow` flow): baseline → profile → config sweep → head-kernel → milestone
  cluster → combined gate.

## Baseline
Two iterations, measured a day apart (box drift visible in the absolute numbers):

| iteration | date | baseline tok/s (median ≥3 warm) | spread |
|---|---|---|---|
| iter1 (config + editable kernels) | 2026-06-07 | **1485.432** (1485.4 / 1479.8 / 1486.4) | 0.44% |
| iter2 (GEMM-tuning win) | 2026-06-07 | **1492.7** | 0.23% |

TTFT median ~3.6 s, TPOT median ~39 ms. **Profile (torch trace):** dense hipBLASLt GEMM is
**~79–81% of GPU time** (rank 1 alone 48.8%, rank 2 17.2%, rank 3 8.6%, rank 4 4.5%); the
gated-delta Triton cluster is **~9%** (`chunk_gated_delta_rule_fwd_kernel_h` 2.9%,
`chunk_fwd_kernel_o` 1.9%, `recompute_w_u` 1.6%, `_causal_conv1d` 1.3%); `act_and_mul` 2.1%;
baseline attention is CK paged-attention at 1.1%. **GEMM is the only lever with e2e headroom
above the noise band** — this is the Amdahl read that drove the whole run.

## What we tried (every lever, both winners and rejects)

### Config track (Tier-0, run first — reshapes the whole profile)
| lever | e2e | verdict | why |
|---|---|---|---|
| **`--attention-backend triton`** | **+4.15%** (iter1: 1485.4 → 1547.0) | **ACCEPT** | only flag over the gate; engaged (`attention_backend/linear_attn_backend/mamba_backend='triton'`); parity clean; *also opens an editable Triton surface* for the FLA/mamba kernels |
| `--chunked-prefill-size 8192` | −0.42% | reject | in-band + slightly negative; cuda-graph already on, prefill-dominated → no decode interleave headroom |
| `--quantization fp8` (+ kv fp8) | — | reject | accuracy gate not passed |
| `--enable-torch-compile` | — | reject | triton version mismatch on this image |

> The `--attention-backend triton` win is reported as **+4.15%** (iter1 cfg0, 5-repeat),
> **+4.44%** (iter1 final retest, 1485.4 → 1551.4), and **+4.96%** in the cross-run knowledge
> ledger ([`gemm_attention_backends.md`](../../../e2e_workflow/knowledge/gemm_attention_backends.md)
> records "+~5%"). The spread is box drift across sessions — all are measured-by-us; the
> per-iteration eval-dir numbers (+4.15 / +4.44%) are the auditable ones. See
> [`../by_kernel/gated_delta_backend_swap.md`](../by_kernel/gated_delta_backend_swap.md).

### Head-kernel track — dense GEMM (the 79% mass)
Two iterations, opposite outcomes — and the difference is the whole lesson:

- **iter1: aiter DB tune FAILED to engage.** The untuned set was synthesized from the profile
  with a **guessed `bias=True`** and synthetic prefill M-buckets. The live sglang dense GEMMs
  are **`bias=False`** with real runtime M-buckets, so every tuned row mismatched the 9-tuple
  lookup key → **0 engagement** (`is tuned on cu_num`=0, `not found tuned config`=258), e2e 0%.
  Isolated speedup was real (~1.032×) but never reached the live path. **Rejected at the
  engagement gate** (the TunableOp lesson: never infer engagement from a wiggle).
- **iter2: bias-correct full-coverage tune WON.** Captured shapes live with
  `AITER_TUNE_GEMM=1` (proved 228/234 are `bias=False`), bucket-reduced 234 → 78, tuned with
  gradlib (FLOPs-DESC order), deployed via `AITER_CONFIG_GEMM_BF16`. **246 engagement hits**,
  **+2.23% e2e** stacked on triton attention. Full write-up:
  [`../by_kernel/gemm_aiter_db_tuning.md`](../by_kernel/gemm_aiter_db_tuning.md).
- **Authored Triton GEMM** (recursive `kernel_workflow`): iso 1.466× on the up/gate shape in
  iter2 (0.99× in iter1 — couldn't beat hipBLASLt), but the e2e gate was won by the aiter env
  path, so the authored kernel did not enter the stack.

### Editable-kernel (milestone) track — the FLA / mamba cluster
Real isolated speedups, all engaged (overlay banner proven), none cleared the e2e gate in the
prefill regime — **kept honestly as expected-Amdahl rejects**:

| kernel | iso speedup | e2e | %gpu | verdict | reason |
|---|---|---|---|---|---|
| `chunk_gated_delta_rule_fwd_kernel_h` | 1.18× | +0.171% | 2.95% | reject | ceiling ~0.45% < band; distributions overlap |
| `chunk_fwd_kernel_o` | 1.14× | −0.035% | 1.98% | reject | ceiling ~0.24% < band |
| `_causal_conv1d_fwd_kernel` | 1.10× | +0.292% | 1.26% | reject / **STACK** | non-negative, parity 12/12, ceiling ~0.11% < band → carry-forward |
| `recompute_w_u_fwd_kernel` | 0.99× | — | 1.6% | reject | no isolated speedup |

Fast Amdahl criterion that predicted all four: `pct_gpu × (1 − 1/iso) < 0.5%` ⇒ carry-forward
only, never solo-passes the gate. See
[`../../operators/linear_attention_gated_delta/`](../../operators/linear_attention_gated_delta/).

## What worked / what didn't (the honest summary)
- **Worked:** `--attention-backend triton` (config, +~4%), and — only after the bias fix — the
  **aiter GEMM DB tune (+2.23%)**. Together **≈ +6% cumulative** over the true baseline (iter2:
  1492.7 → ~1583.5 tok/s).
- **Didn't (and why it's fine):** every editable FLA/mamba kernel. They are genuinely faster in
  isolation and genuinely engaged on the live path, but each is ≤3% of GPU time in a
  prefill-dominated, 79%-GEMM regime, so their e2e contribution falls inside the 0.5% band.
  This is an **expected Amdahl outcome, not an integration bug**.
- **The trap that cost iter1 the GEMM win:** guessing the GEMM `bias` flag instead of capturing
  it live. `bias` is part of the aiter DB lookup key; a wrong guess silently deploys a config
  that never matches.

## Final result (numbers, measured-by-us)

| stack | iter | e2e tok/s | vs true baseline |
|---|---|---|---|
| baseline (CK attn, default GEMM) | iter1 | 1485.4 | — |
| + `--attention-backend triton` | iter1 final | **1551.4** (5-rep, spread 0.33%) | **+4.44%** |
| baseline | iter2 | 1492.7 | — |
| + triton attn (ref leg) | iter2 | 1548.9 (median) | +3.8% |
| **+ triton attn + aiter GEMM tune** | iter2 | **1583.5** (median; cand_min 1573.4 > ref_max 1554.7) | **≈ +6% cumulative** |

The GEMM-tune A/B: ref (triton attn) median 1548.9, cand (triton attn + GEMM tune) median
**1583.5**, **Δ +2.23%, distributions non-overlapping, 246 `is tuned on cu_num` hits** →
accepted. Parity safe (same bf16 math; tuner gated every row at `err_ratio < 0.05`).

> **Caveat (honest):** iter2 was **manually stopped during the milestone phase** because nested
> recursive tuning forked hundreds of `rocm_agent_enumerator` processes (a CPU process storm
> that pollutes e2e timing). So there is **no Director combined-gate Validate number** for the
> full stack including the editable cluster — the authoritative banked numbers are the config
> win and the +2.23% GEMM win, stacked.

## Lessons
1. **Profile first, route by Amdahl mass.** 79% GEMM meant GEMM and config were the only levers
   that could clear the band; the editable kernels were always going to be carry-forward.
2. **Capture the GEMM `bias` and shapes from the LIVE server**, never from a guessed schema or
   `meta.json`. The bias mismatch was the single difference between a 0% and a +2.23% GEMM tune.
3. **Prove engagement before believing an e2e delta** (`is tuned on cu_num > 0`); a deployed-
   but-unmatched config looks live and does nothing.
4. **Trust only same-session, non-overlapping A/B.** gfx942 boxes drift several % across hours;
   a positive median with overlapping runs is noise.
5. **Stack config + GEMM; they're orthogonal and compound.** Gate the GEMM tune *stacked on*
   the accepted attention flag, never in isolation.
6. **Serialize heavy nested tunes.** The process storm is real and corrupts timing — bound the
   tuner shape count and kernel budget.

## Cross-links
- The GEMM lever in depth: [`../by_kernel/gemm_aiter_db_tuning.md`](../by_kernel/gemm_aiter_db_tuning.md)
- The attention flag in depth: [`../by_kernel/gated_delta_backend_swap.md`](../by_kernel/gated_delta_backend_swap.md)
- GEMM recipe: [`../../kernel_workflow/gemm_tuning_workflow.md`](../../kernel_workflow/gemm_tuning_workflow.md) · e2e flow: [`../../kernel_workflow/optimize_e2e_model.md`](../../kernel_workflow/optimize_e2e_model.md)
- Attention selection: [`../../kernel_workflow/attention_backend_selection.md`](../../kernel_workflow/attention_backend_selection.md)
- Operators: [`../../operators/dense_gemm/`](../../operators/dense_gemm/) · [`../../operators/linear_attention_gated_delta/`](../../operators/linear_attention_gated_delta/)
- Backend: [`../../backends/aiter/tuned_gemm.md`](../../backends/aiter/tuned_gemm.md)

## Sources
- iter1 (config win +4.15/+4.44%, editable-kernel cluster, the bias-mismatch GEMM reject): `GEAK/examples/e2e_workflow/qwen3.5-27b_sglang_isl1024_osl1024_conc64/final_report.md`.
- iter2 (the +2.23% GEMM win, 246 engagement hits, the process-storm stop): `GEAK/examples/e2e_workflow/qwen3.5-27b_sglang_gemm-tuning-win/final_report.md`.
- Cross-run provenance ledger (+~5% attn, +1.2–2.4% GEMM ceiling, bias-from-server rule): `GEAK/e2e_workflow/knowledge/{gemm_tuning/aiter_gemm_tuning.md,gemm_attention_backends.md}`.

<!-- MANIFEST: Qwen3.5-27B sglang MI300X flagship e2e — measured triton-attn config win (+4.15–4.44%) stacked with bias-correct aiter GEMM DB tune (+2.23%, 246 hits) ≈ +6% cumulative; editable FLA/mamba cluster all carry-forward by Amdahl. -->
