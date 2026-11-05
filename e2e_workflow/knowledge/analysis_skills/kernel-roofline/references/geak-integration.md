# GEAK Profiler integration

Read this when GEAK invokes the skill through `ANALYSIS_SKILL_DIR` after producing its Top-N.
Default to `tl-only`; honor an explicit mode. Analysis is advisory: preserve `profile_topN.json`,
its measured `pct_gpu_time` and candidate order. Missing analysis must not block profiling.

## Inputs and execution

Start with the finalized Top-N and the matching trace, workload and device provenance. Reuse an
existing trace when possible; analysis of it requires no new GPU measurement. Resolve latest upstream
TraceLens `main` and freeze its full commit as described in [installation.md](installation.md).
An existing report from a different commit may be retained as historical context, not relabeled as
the current analysis.

Follow [tl-only.md](tl-only.md) for existing-trace native reporting with `--include_call_stack`.
Keep native outputs under `profile/round_<N>/results/tl-only/`. Package selected complete rows from
`unified_perf_summary.csv` using `scripts/extract_tl_row.py` and follow
[unified-summary.md](unified-summary.md) to associate Top-N kernels with operations and stacks.
Record unmapped or ambiguous entries instead of choosing a model by a device-name substring.
One operation may own several kernels: its full modeled work and roofline belong to that scope.

For separately requested UT measurements, use [runner.md](runner.md) and `scripts/tl_only.py`;
retain its `rows.json`, native reports and measurement manifest. Missing operation identity may
be enriched following [ut-context.md](ut-context.md), using only actual UT inputs and invoked source.
Shape, dtype, stride, input/output roles and quantization require per-field evidence. A profile
with no available UT cannot be completed by inventing a UT or model arguments. Model-only analysis
must explicitly leave measured metrics unavailable.

Do not initiate unrelated benchmarking to fill gaps in an advisory profile step. Preserve supported
rows and report the missing metadata, native-model limitation, peak reference or setup problem.
Explicit `rocprof` work uses its independent results directory and reference contract.

## Handoff

Write `profile/round_<N>/profile_roofline.json` and a readable `profile_roofline.md`. Return the
JSON path in the existing `profile_roofline_json` field. The JSON is an index of evidence, not the
retired roofline calculator's schema. Use this structure (paths relative to this manifest or absolute):

```json
{
  "schema": "geak.kernel_roofline.v1",
  "skill": "kernel-roofline",
  "mode": "tl-only",
  "engine": "TraceLens native",
  "status": "unavailable",
  "profile_topN_json": "profile_topN.json",
  "reports": [],
  "coverage": [],
  "limitations": []
}
```

- `status`: `available`, `partial` or `unavailable`, based on the requested Top-N/case coverage.
- Each `reports` entry links `rows_json` when produced by `tl_only.py`,
  `unified_perf_summary_csv`, selected `row_packets`, and `provenance_json` as applicable; use
  `null` or omit inapplicable paths. The provenance must record whether a GPU was measured,
  whether this analysis reused a trace, measurement scope, tool commit, architecture hash and
  source, device/partition and runtime. For existing-trace CLI reports, save this provenance
  alongside the CSV rather than fabricate a runner measurement manifest.
- Each `coverage` entry identifies a requested Top-N rank/kernel and case, its `status`, matched
  report/row references, and a `reason` for missing or ambiguous coverage. Keep every requested
  entry, including ones absent from the native report. Preserve per-case coverage separately
  from kernel coverage. Shared operation references must not imply per-subkernel rooflines.
- `limitations` records model assumptions, missing evidence and setup/report failures. If even
  an unavailable manifest cannot be written, return `profile_roofline_json: ""` and the Top-N.

The Markdown summarizes native results, scope, coverage and provenance and links raw artifacts.
Preserve statistic suffixes and native units. For `rocprof`, identify that engine and link its
counter artifacts instead; if both modes are requested, keep separate manifests with each
mode's own engine and coverage and link them from the summary. Never combine their efficiencies.

## Architect consumption

Follow report and row links before interpreting a value. Native `Pct Roofline` is already a
percentage; `distance_to_reference` is not promised attainable speedup. Do not synthesize legacy
`attainable_speedup`, `expected_e2e_gain_pct` or `headroom_class` from these fields. Keep absent
metrics unavailable, preserve values above 100%, and investigate suspect scope/reference choices.
Neither high nor low efficiency proves a specific optimization opportunity or bottleneck.
Keep measured `pct_gpu_time` as the ranking basis; use this analysis to support explicit hypotheses
and validation plans. Only measured A/B results establish realized improvement.
