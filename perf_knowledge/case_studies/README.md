---
title: Case studies — real measured / sourced AMD MI-GPU kernel optimization runs
kind: case_study
gens: [gfx942, gfx950]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - GEAK/examples/e2e_workflow/qwen3.5-27b_sglang_isl1024_osl1024_conc64/final_report.md
  - GEAK/examples/e2e_workflow/qwen3.5-27b_sglang_gemm-tuning-win/final_report.md
  - GEAK/e2e_workflow/knowledge/gemm_tuning/aiter_gemm_tuning.md
  - GEAK/e2e_workflow/knowledge/gemm_attention_backends.md
---

# Case studies

End-to-end and per-kernel optimization runs on AMD Instinct GPUs, written up the way they
actually happened — **including the attempts that failed**. Every number here is either
**measured-by-us** (a same-session A/B from a real eval dir, labelled with the stack/date) or
**vendor-reported** (AMD blog / vLLM blog, labelled as such). Nothing is invented or
extrapolated past its source. Where a figure differs between runs (box drift across sessions),
both are shown with their provenance.

The methodology these follow is in [`../kernel_workflow/optimize_e2e_model.md`](../kernel_workflow/optimize_e2e_model.md)
(e2e flow), [`../kernel_workflow/optimize_single_kernel.md`](../kernel_workflow/optimize_single_kernel.md)
(inner loop), and [`../kernel_workflow/gemm_tuning_workflow.md`](../kernel_workflow/gemm_tuning_workflow.md)
(the GEMM recipe). Read those for the *how*; these case studies are the *what happened*.

## How to read a case study
Each file follows: **Context → Baseline → What we tried → What worked / what didn't (both kept)
→ Final result (numbers) → Lessons → Sources**. The accept gate everywhere is the project
standard: `delta% > 0.5% noise band AND cand_min > ref_max` on a tight same-session A/B, plus
**engagement proof** on the live path and **output parity** (see
[`../kernel_workflow/optimize_e2e_model.md`](../kernel_workflow/optimize_e2e_model.md)).

## by_model/  — full e2e serving runs

| case | model / stack | headline | kind |
|---|---|---|---|
| [`by_model/qwen3.5-27b_sglang_e2e.md`](by_model/qwen3.5-27b_sglang_e2e.md) | Qwen3.5-27B / sglang 0.5.11 / MI300X | **triton attn + GEMM tune ≈ +6% cumulative** (measured) | flagship |
| [`by_model/deepseek_mla_mi300x.md`](by_model/deepseek_mla_mi300x.md) | DeepSeek (MLA) / aiter / MI300X | **17× MLA decode**, 1.2–1.6× TPOT (vendor) | MLA serving |
| [`by_model/llama_fp8_serving.md`](by_model/llama_fp8_serving.md) | Llama-class fp8 / vLLM+sglang / MI300X | fp8 recipe + AITER fused kernels (vendor) | quant serving |

## by_kernel/  — single-kernel / single-lever wins

| case | lever | headline | kind |
|---|---|---|---|
| [`by_kernel/gemm_aiter_db_tuning.md`](by_kernel/gemm_aiter_db_tuning.md) | aiter per-shape GEMM DB tune | **+2.23% e2e** (measured), 246 engagement hits | GEMM tune |
| [`by_kernel/gated_delta_backend_swap.md`](by_kernel/gated_delta_backend_swap.md) | `--attention-backend triton` | **+4.15–4.96% e2e** (measured/KB) | backend swap |
| [`by_kernel/fused_norm_quant_win.md`](by_kernel/fused_norm_quant_win.md) | RMSNorm+fp8-quant fusion | 1–6% e2e latency (vendor, sglang #18466) | fusion |
| [`by_kernel/mfma_tile_selection.md`](by_kernel/mfma_tile_selection.md) | 16×16 vs 32×32 MFMA tile | 16×16 default on MI300X (vendor) | tile choice |

## Cross-links
- Workflows: [`../kernel_workflow/`](../kernel_workflow/) · Operators: [`../operators/`](../operators/)
- aiter backend: [`../backends/aiter/`](../backends/aiter/) · Quantization: [`../quantization/`](../quantization/)
- Optimization techniques: [`../optimization/`](../optimization/) · SOTA matrix: [`../index/sota_matrix.md`](../index/sota_matrix.md)

## Sources
- Flagship measured numbers: the two `examples/e2e_workflow/qwen3.5-27b_sglang_*/final_report.md` iteration reports.
- Provenance ledger for the GEMM/attention levers: `GEAK/e2e_workflow/knowledge/{gemm_tuning/aiter_gemm_tuning.md,gemm_attention_backends.md}`.

<!-- MANIFEST: case_studies index — routes to by_model/{qwen3.5-27b_sglang_e2e,deepseek_mla_mi300x,llama_fp8_serving} and by_kernel/{gemm_aiter_db_tuning,gated_delta_backend_swap,fused_norm_quant_win,mfma_tile_selection}; every number labelled measured-by-us vs vendor-reported. -->
