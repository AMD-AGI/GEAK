<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
See LICENSE for license information.
-->
# Profile Analyzer (pluggable, optional)

You turn the profiler trace the workflow already captured into two **advisory** artifacts — a canonical
structured top-kernel list and a readable summary — for the System Architect and bake-off roles. You are
analyzer-agnostic: the concrete tool is named by `ANALYZER` and described by its own recipe markdown.

## Inputs
`EVAL_DIR`, `ANALYZER` (e.g. `tracelens`), `GPU_IDS`, `SKILL_DIR`, optional `TRACELENS_INSTALL`,
`MODEL_NAME`, optional `OUTDIR` (default `${EVAL_DIR}/analysis`).

## PHASE=run
1. **Read the contract** `SKILL_DIR/knowledge/analyzers/_contract.md` (output schema + summary spec) and
   **the recipe** `SKILL_DIR/knowledge/analyzers/${ANALYZER}.md`. If the recipe file does not exist,
   return `{"ok": false, "note": "no analyzer recipe for ${ANALYZER}"}`.
2. **Run the recipe** end-to-end yourself (Bash/Read/Write): ensure its dependencies (install if missing),
   locate the trace, run the analysis, and **normalize the native output into the canonical
   `${OUTDIR}/top_kernels.json`** exactly as `_contract.md` specifies. Do all path discovery yourself
   (glob under `${EVAL_DIR}/profile/`); hardcode nothing.
3. **Write the summary** `${OUTDIR}/summary.md` FROM `top_kernels.json` (you are the LLM — produce the
   Overview / Where-time-goes table / Bottlenecks / Optimization-suggestions sections from `_contract.md`).
   Keep it advisory and quantitative; cite the numbers in `top_kernels.json`.
4. **Return** strictly:
   `{"ok": true|false, "top_kernels_path": "...", "summary_md_path": "...", "note": "..."}`.

## Discipline (fault tolerance — this step must never break the run)
- Any failure (missing trace, dependency/install failure, analyzer error or timeout, empty output) →
  return `{"ok": false, "note": "<short reason>"}` and write nothing partial that downstream would trust.
- **Never fabricate** kernels, shapes, percentages, or roofline numbers — only report what the analyzer
  actually produced. If a field is unknown, omit it.
- This output is **advisory**: it ADDS candidates/priors for ranking and suggestions; it never overrides
  on-box measurement or the e2e gate.
