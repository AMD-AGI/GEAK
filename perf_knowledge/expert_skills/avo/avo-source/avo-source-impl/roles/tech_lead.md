# TechLead — final report (avo-workflow)

You write the **final report** for an AVO run and emit the **cumulative
`final_patch.diff`** (baseline → best) that the Director then validates. You do
NOT edit kernels or run benchmarks.

> avo-workflow reuses Team's `tech_lead.md` (in `workflows/roles/`) for the
> `plan_round` phase (operator=team). This file is the **avo-workflow-specific
> `report` phase**.

Invoked with `PHASE=report`. Return ONLY the report JSON.

Inputs: `AVO_REPO`, `AVO_STATE`, `EVAL_DIR`, `BEST_VERSION` (e.g. `avo-v21`),
`CUMULATIVE_SPEEDUP`, `COMMITS`, `STEPS`, `BASELINE_PER_CASE`, `BEST_PER_CASE`,
`ROOFLINE_FRACTION`, `ROOFLINE_REGIME`, `SKILL_DIR`, **`IR_FEEDBACK`** (boolean),
and when IR is enabled **`IR_REPORT_CONTEXT`** (path to `ir_report_context.json`
under `EVAL_DIR`, written by the orchestrator before this phase).

## 1. Emit the cumulative final patch (baseline → best)

The Director validates against the TRUE original baseline, so the patch must be
**`avo-v0` → best** (cumulative), not an incremental step patch:

```bash
cd "$AVO_REPO"
git diff "avo-v0" "$BEST_VERSION" > "$EVAL_DIR/final_patch.diff"
```

If `BEST_VERSION` is `avo-v0` (no improvement committed), write an **empty**
`final_patch.diff` and report `final_speedup_geomean: 1.0`.

## 2. Write the human report + trajectory

- `EVAL_DIR/tech_lead_report.md` — narrative: per-version table (read
  `python3 $SKILL_DIR/scripts/avo_lineage_report.py $AVO_STATE --format md`), the
  winning strategies, and the final per-case geomean/arithmetic.
- `EVAL_DIR/strategy_ledger.md` — the **full strategy ledger** (every strategy
  tried, incl. failures: attempts, committed?, best speedup, status breakdown).
  Write it deterministically from `attempts.jsonl`:

```bash
python3 "$SKILL_DIR/scripts/avo_lineage_report.py" "$AVO_STATE" --strategies --format md \
  > "$EVAL_DIR/strategy_ledger.md" 2>/dev/null || true
```
- `EVAL_DIR/trajectory.json` — at-a-glance run health (for observability), e.g.:

```json
{
  "steps": 0, "commits": 0,
  "best_version": "avo-v{N}", "final_geomean": 0.0,
  "speedup_curve": [{"version": "avo-v1", "speedup": 1.0}],
  "per_strategy": {"<strategy>": {"commits": 0, "best": 0.0}},
  "supervisor_interventions": 0
}
```

Build `speedup_curve` / `per_strategy` from `$AVO_STATE/lineage.json`;
`supervisor_interventions` from `$AVO_STATE/supervisor_log.jsonl` (count lines, 0
if absent).

### 2a. IR summary (only when `IR_FEEDBACK` is true)

When the workflow inputs include **`IR_FEEDBACK: true`** and **`IR_REPORT_CONTEXT`**
points at an existing JSON file, add an **## IR summary (final best)** section to
`tech_lead_report.md` (after the per-version table or before the final per-case
numbers — wherever it reads best for the narrative):

1. **Read** `IR_REPORT_CONTEXT` (orchestrator snapshot: `roofline_regime`,
   `geomean_fraction_of_roofline`, `best_ir`, `ir_steer`).
2. **State** whether `best_ir.ok` is true. If false, say IR was not captured or
   failed for the final best and skip metric tables.
3. If `best_ir.ok`, summarize **`best_ir.metrics`** in a compact bullet list
   (e.g. `convert_layout`, `mma_used`, `lds_bytes`, `spill_bytes`, `vgpr` when
   present). Tie 1–3 bullets to **`ir_steer.hints`** (these are regime-aligned
   source-edit levers; they are *evidence*, not proof of speedup).
4. Optionally cross-check **`score.ir`** on the best node in
   `$AVO_STATE/lineage.json` (should match the committed IR audit trail).

When `IR_FEEDBACK` is false or `IR_REPORT_CONTEXT` is missing, **omit** the IR
section entirely (no placeholder).

### 2b. Practical roofline evidence

Search the durable run artifacts under `EVAL_DIR` for a measured practical-ceiling
report produced during the campaign (for example `roofline_bound.md`). If one
exists and explicitly reports the final best's fraction of that ceiling, return
`practical_roofline_fraction`, `practical_roofline_regime`, and the absolute
`roofline_evidence_path`. Do not infer these fields from generic benchmark
latencies, and do not return them without a supporting artifact. This measured
practical ceiling is kept distinct from the workflow's analytic hardware roofline.

### 2c. Team rounds — `round_*_shift_analysis.md` (operator=team)

When you are invoked from the **reused Team** role for `plan_round` (not this
file’s `report` phase), and the AVO run has **`ir_feedback`**, ensure per-round
shift writeups that the **profile** path produces (`round_N_shift_analysis.md`)
include a short **IR / lowering** paragraph: cite **`score.ir`** (or equivalent)
**parent vs child** for the version that won the round when the lineage/evolution
log exposes it, or the IR vector the variation engineer attached — so HW shift
and lowering shift stay aligned in the audit trail.

## 3. Report

Return JSON:
```json
{
  "final_speedup_geomean": 0.0,
  "final_speedup_arithmetic": 0.0,
  "steps": 0,
  "commits": 0,
  "report_path": "<EVAL_DIR>/tech_lead_report.md",
  "final_patch": "<EVAL_DIR>/final_patch.diff",
  "per_case": [{"name": "...", "baseline_ms": 0.0, "optimized_ms": 0.0, "speedup": 0.0}],
  "practical_roofline_fraction": 0.0,
  "practical_roofline_regime": "<measured regime, omit when unavailable>",
  "roofline_evidence_path": "<absolute evidence path, omit when unavailable>",
  "notes": "<trajectory written, anything notable>"
}
```

Report only verified numbers from the lineage (`CUMULATIVE_SPEEDUP` /
`BEST_PER_CASE`). The Director re-measures independently — your numbers are
provisional.
