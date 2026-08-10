
---

# Review: semantic_kernel_mapping boundary fix + 1.2 within-layer % (2026-08-10)

Context: user reported two problems in the v8 DSR1 semantic table — (1) one-time "metadata"
kernels wrongly included in the representative layer (boundary not correctly determined), and
(2) the table needed a within-pattern-layer percentage column. A patched version of the GEAK
code was provided for review.

## Verdict: BOTH issues are fixed by the patch. ✅

### Issue 1 — metadata kernels at the layer boundary → FIXED
New `_demote_non_dominant_prefixes(rows, partition_diagnostics)` in `semantic_kernel_mapping.py`.
- Approach: per (pattern_id, phase), find the dominant stage sequence shared by >=2 distinct
  layers (must be UNIQUELY dominant, no tie); for any instance whose sequence is longer and
  contains that dominant sequence EXACTLY ONCE as a clean suffix (start>0, suffix==dominant),
  demote the leading prefix to `transition_global`.
- Model-agnostic (no kernel-name / model rules); conservation-safe (events are moved, not
  dropped) — they go to `transition_global`, so analysis_window_conservation still holds.
- Empirical (DSR1 clean trace): prefill `dense_mla` L0 = 34 -> 29 events; the 5-event prefix
  `aten::add / aten::arange / aten::neg / aten::fill_` (one-time position/embedding setup that
  only precedes layer 0) is demoted; the table now starts at `rmsnorm`, identical to L1/L2 and
  to the `moe_mla` layers. 2 demotions recorded in `layer_instance_audit.json:prefix_demotions`.
- KPU coverage manifest: DSR1 rows 132 -> 127, K 50 -> 45 (exactly the 5 removed kernel_exact
  metadata rows); P/U unchanged. KPU pair still PASS.
- Conservative / fail-safe: Qwen3.5 = 0 demotions, event counts unchanged (its layers have no
  spurious cross-layer prefix). 19 unit tests pass.

### Issue 2 — within-layer percentage column → FIXED
`semantic_shape_merge._markdown` now emits the `layer total %` column (value = `layer_total_pct`
= per-row share of the representative layer's total device us). The Semantic 1.1 markdown already
had this column; the fix brings the Semantic 1.2 (merged/published) table to parity.

## Caveats / optional follow-ups (not blocking)
1. The demotion requires an EXACT stage-sequence suffix match and a uniquely dominant sequence.
   This is deliberately conservative: it will never mis-demote, but it can MISS a boundary anomaly
   if the "clean" layers themselves vary slightly, or for a pattern that has only ONE layer
   (`len(layer_ids) < 2` can never demote). Acceptable fail-safe; just know it is prefix-only and
   won't catch a one-time TRAILING op or a mid-body insertion.
2. "pattern 层内的百分比" was interpreted as the per-row `layer_total_pct`. If a per-STAGE rollup
   (e.g. attention vs router vs experts vs GEMM share of the layer) is what's wanted for fusion
   analysis, that is an additional aggregation not yet provided — worth a quick confirm.
3. The patch is currently uncommitted in the working tree on top of `419ac76f`. DSR1's
   `semantics_1_2_run` was re-run with it; Qwen's was regenerated during this review (reusing the
   existing capture via `--capture-result`, no GPU). Both now carry the fix; KPU pair = pass.
