---
title: Roofline analysis ownership and local hardware planning
kind: technique
gens: [gfx950, gfx942]
updated: 2026-11-05
---

# Roofline analysis in GEAK

## Engine and evidence ownership

GEAK's native `kernel-roofline` analysis skill owns roofline reporting and its associated full-counter
measurement lifecycle. It is discovered through the e2e analysis-skill index. Default to `tl-only`;
use `rocprof` only when requested. If the native skill is absent from a checkout, report unavailable
and continue the existing profile. Do not substitute the legacy calculator or a local model.

Existing-trace TraceLens analysis is offline and does not need a GPU lease. New UT or counter runs
must use GEAK's outer GPU lease and source guard, preserve the measured window and dispatch checks,
and record the frozen calibration. Do not change the GPU selection or reacquire a lock inside a lease.

Preserve native output and `geak.kernel_roofline.v1` report links, mode, engine, coverage and limitations.
TraceLens references come from its selected arch JSON (path, SHA-256, source and resolved precision key);
rocprof uses its frozen calibration CSV. Neither is interchangeable with a SKU datasheet table.
Keep measured Top-N order and `pct_gpu_time`. Distance to a reference is not attainable speedup;
model classification is not proof of a measured bottleneck. Retain values above 100% and investigate
scope, calibration, cache and timing assumptions. Missing coverage stays unavailable.

## Local planning and diagnostic tools

[`../hardware/data/sku.json`](../hardware/data/sku.json) records specification and explicitly labelled
reference values for hardware planning. Read the actual product and dtype; missing precision is unknown.
Generated [CDNA4](../hardware/cdna4_mi350/peak_tables.md) and
[CDNA3](../hardware/cdna3_mi300/peak_tables.md) tables describe those rows.

`kernel_workflow/scripts/kernel_tools/hw_budget.py` and `calc_perf.py` estimate FLOPs, bytes,
resource feasibility and model floors. Their output is labelled `analysis_role: planning`,
`engine: GEAK local analytic model`, `may_gate: false`, including when supplied measured inputs.
Use them for explicit planning/custom analysis; never insert their output into native reports.
`mem_bw_probe.py` can answer a separately requested access-pattern question, but does not replace
TraceLens's frozen reference. Measurements must use the same scope as the model being discussed.

Local thresholds in [`../hardware/data/thresholds.json`](../hardware/data/thresholds.json) are diagnostic
hints, not a universal classifier. Busy cycles, lane occupancy, relative counts and bandwidth have
different denominators. Missing absolute utilization is null, not a normalized raw-counter share.
ATT, ISA, layout and static occupancy remain useful supplementary diagnostics.

## Dependencies and execution environment

Use the native skill's installation instructions and freeze the resolved TraceLens commit. Existing
trace reporting does not require ATT or rocprof. Keep the workload in its framework environment;
constrain rocprof-compute analyzer dependencies for its installed version (3.4 requires `pandas<3`).
Do not install unconstrained profiler requirements into a shared workload/report environment.
Forward `ROOFLINE_MIBENCH_CACHE` with an explicit readable host/container path when using frozen CSVs.

GEAK's oracle, same-window A/B measurement and acceptance thresholds remain the decision authority.
