# Roofline

Use the [invocation contract](../CONTRACT.md#direct-module-calls) to assess a bound
or search priority. Provide explicit task/case evidence and a separate analysis output;
a scoped input view preserves its original ownership and whole-workload shares.
The report informs Reasoning; it does not force a search or establish E2E gain.

**Input** `$IN` = profile's out (`selection_targets.json` for shares, `tasks/<name>/` for
runnable unit tests).
**Output** `$OUT/roofline.json`: per-kernel `roofline_pct`, `attainable_speedup` and
`bound_type`, plus the **ranking key `expected_e2e_gain_pct`**.

## Implementations

| Directory | Notes |
| --- | --- |
| `kernel-roofline/` | default. External archive, vendored unmodified. Three modes: `tl-only` (TraceLens prediction, no GPUs), `rocprof` (measured counters, requires rocprof-compute ≥ 3.6.0), `combination` (reconciles both legs in the time domain). Integrity enforced by `MANIFEST.kernel-roofline.json` |
| `stub-roofline/` | a minimal worked example of swapping this slot |

Agent-shaped. The mode comes from `params.roofline_mode`, defaulting to `tl-only` — no
GPUs, and sufficient for ranking. Set `rocprof` for measured counters, or `combination`
to reconcile the two legs.

## Ranking

Rank by potential whole-workload savings, retaining unknown bounds as unknown:

```
expected_e2e_gain_pct = pct_gpu_time × (1 − 1 / attainable_speedup)
```

## Artifacts → `result.json`

The slot's `finalize.py` performs this mapping and is the stage's verdict (it belongs to
the slot, not to any implementation, and the driver runs it after every session —
[driver protocol](../../engine/core/README.md#execution-shapes)):

```bash
python3 "$WORKLOAD_ROOT/stages/roofline/finalize.py"
```

What it accepts is written only there: read the file, or run it early with the command
above, rather than looking for a second copy of its thresholds in this document. What
follows is its interface and the reasoning behind it, not a restatement of its rules.

It requires the skill to leave **`rows.json`** in `$OUT` — the raw per-case table, with
fields `kernel`, `regime`, `case`, `measured_us`, `roofline_us` (or whichever column
`--roof-key` names), and optionally `case_weight` and `bound`. That is `kernel-roofline`'s
own `fold.py` input format. Each implementation also supplies **`folded.json`**, a map
keyed by `kernel/regime` whose records contain `pct`, `speedup`, `suspect`, `coverage`,
and `n`. These two files describe the same case set and roof column. A replacement can
produce them with its own tools; the finalizer never invokes implementation code.

For the vendored `kernel-roofline` skill, complete its analysis, then run its existing
folding tool. Set `ROOF_KEY` from `params.roof_key` when configured:

```bash
python3 "$SKILL_ROOT/scripts/fold.py" --rows "$OUT/rows.json" \
    --roof-key "${ROOF_KEY:-roofline_us}" --out "$OUT/folded.json" \
    --label roof > "$OUT/fold.log" 2>&1
```

Stop on a failed fold and keep its log. Regenerate the fold whenever its rows or roof
column change; an earlier fold is not evidence for a new analysis. This call belongs to
the skill's work, before the finalizer runs.

What `finalize.py` does:

1. Reads the supplied fold. Folding stays in the time domain: sum numerator and
   denominator separately and divide once, with weights normalized within a regime.
   The implementation performs that work; the finalizer does not duplicate its sum.
2. Maps task-keyed or leaf-keyed rows to the declared task boundary and its selected
   whole-workload share. It checks that every positive-mass task case has an analysis
   row before distributing the share across regimes by measured time. Incomplete model
   coverage, explicit inadmissibility and suspect rows leave whole-task bounds unknown;
   native subset folds remain diagnostics only.
3. Writes `roofline.json` (all kernels, descending by `expected_e2e_gain_pct`) and
   `result.json`.

Artifacts: `roofline.json`, `rows.json`, `folded.json`, and `fold.log` when the
implementation provides a diagnostic log.

Every delivered task remains in the output. Completed analysis with an explicitly
unsupported model may finish with unknown bounds; missing cases, stale qualification
or unexplained absent models block the stage. Tasks with unknown expected gain retain
`task_plan.json` order for execution, clearly labelled as a fallback rather than a
predicted performance ranking. A partial ranked subset cannot imply a known total gain.

## Three rules that must not be broken

- **An efficiency above 1 falsifies the denominator; it does not mean 100% saturation.**
  Such rows are marked `suspect`, must **not** be clamped to 100%, and no speedup may be
  derived from them. `finalize.py` sets their `attainable_speedup` and
  `expected_e2e_gain_pct` to null — listed, but excluded from the ranking.
- **`bound_type` is a model classification, not proof of a measured bottleneck.** Low
  efficiency on its own does not establish a bandwidth, occupancy or issue limit.
- **Different models can use different denominators.** Reconcile their definitions
  before comparing efficiencies or applying a derating factor; do not derate an already
  adjusted bound twice.
