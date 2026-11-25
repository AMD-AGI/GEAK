# The AVO Lineage (P_t) — outputs to read

The lineage is the set of **committed** kernel versions, each a verified
improvement over its parent. After an **AVO run** (the native `avo-workflow`
loop, or the optional `geak-avo` fallback) it lives under `<eval_dir>/avo_state/`
and `<eval_dir>/avo_repo/` (git):

| File / dir | Contents |
|------------|----------|
| `avo_state/lineage.json` | committed versions + `best_id` + per-version score (`speedup`, `per_shape`) + `parent_id` |
| `avo_state/attempts.jsonl` | every attempt incl. failures (diagnosis only — NOT the lineage). One line per step appended by the variation engineer (`committed:false`, every exit path); commits add a `committed:true` line. |
| `avo_state/patches/v{N}.patch` | per-version incremental patches |
| `avo_state/evolution_log/` | per-step causal memory (strategy, committed?, speedup, rationale) |
| `avo_state/supervisor_log.jsonl` | supervisor interventions (audit) |
| `avo_repo/` (`.git`) | the lineage as git commits + tags `avo-v{N}` |
| `<eval_dir>/director_validation.json` | the official, independently-verified result (the contract output) |
| `<eval_dir>/final_patch.diff` | cumulative `avo-v0 → best` patch |

> The artifact **names** under `avo_state/` match GEAK-AVO so the ported tools
> (`avo_lineage*.py`) and this doc apply to both. The native loop's official
> result is `director_validation.json` (PerfSkills convention); the `geak-avo`
> fallback additionally writes its own `final_report.json` (canonical GEAK shape).

## What "committed" means

A version enters the lineage only when it is **correct** AND its
**independently-verified** benchmark speedup ≥ the running best (within epsilon),
above the anti-lazy floor. Intermediate attempts that regress or fail correctness
are recorded in `attempts.jsonl` but are **not** part of the lineage — the paper's
40 committed versions came from 500+ internal attempts.

## Reading scores for the contract

- `best_id` in `lineage.json` is the winning version (e.g. `v21` → tag `avo-v21`).
- Per-version `score.speedup` is the **verified** geomean vs the baseline (`avo-v0`).
- The final patch handed to the Director must be cumulative: `git diff avo-v0
  avo-v{best}` (baseline → best), NOT an incremental step patch.

## Rich commit messages

Each `avo-v{N}` commit/tag carries speedup / latency / strategy (+ full entry) /
rationale / per-shape, so `git -C avo_repo log` / `git show avo-v{N}` is a
self-describing audit trail. `scripts/avo_lineage_report.py` renders the whole
lineage as a table (md/csv).
