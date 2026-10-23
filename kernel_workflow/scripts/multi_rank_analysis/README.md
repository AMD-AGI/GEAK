# `multi_rank_analysis` — generic multi-GPU-rank analysis library

This is a **library**, not a Skill: it has no operator-specific analysis doctrine. Invalid contracts
raise explicit exceptions; partial trace/rank data is returned with missing/error metadata so the
caller can degrade deliberately. Operator-specific analysis Skills (see
`kernel_workflow/knowledge/analysis_skills/`) are built ON TOP of this library; this library must
never import or reference a specific kernel, model, or backend. If you are tempted to add a
MegaMoE/AITER/MORI/FlyDSL name anywhere in this directory, stop — that belongs in a Skill's
`category_map` or prompt, not here.

## Why this exists

Before this package, "gather a float from every rank, compute mean/max, and merge into a report" was
implemented once, privately, inside one distributed MoE unit test — a handful of module-private
gather/reduce/timing helpers, none of them exported, none of them named for reuse. (Their identifiers
are deliberately not reproduced here: this tree is walked by capability-eval engineers, and a private
name is an address into a tree they are not supposed to read. The pattern is what transfers.) A repo-wide grep at design time confirmed **zero** other reusable
multi-rank aggregation utilities existed anywhere in GEAK or in the AITER checkouts. Any future distributed UT/benchmark that Kernel Workflow drives would otherwise have
re-derived the same all_gather/mean/max plumbing from scratch.

## Two independent halves

- **`aggregate.reduce_scalar` / `all_ranks_true` / `time_distributed`** — require a live
  `torch.distributed` process group (torch imported lazily). Call these **inside** a running
  distributed harness to produce per-rank numbers, generalizing the module-private
  gather / reduce / graph-timing pattern described above.
- **`aggregate.merge_rank_records` / `classify_case_speedup`** — pure Python, **no torch import at
  all**. Call these **offline**, after per-rank records have already been gathered (by the harness's
  own `dist.all_gather_object`, or read back from a JSON report) — including from unit tests that
  have no GPU.
- **`bundle`** — validates `geak-analysis-bundle-v2` and normalizes existing UT reports, including
  AITER's `cases[].ranks[]`, into one portable Profile→analysis artifact contract.
- **`build_bundle.py`** — deterministic CLI joining a rank report with case artifacts, route
  comparisons, hardware context, measurement tracks, and controlled experiments.
- **`trace_categories.load_category_map` / `bucket_trace_events`** — turn a raw Chrome/Perfetto trace
  (`torch.profiler.export_chrome_trace` output; kernel events carry only a `name` string, no category
  field) into per-category time sums, using a category map the CALLER supplies. This module ships
  only a generic 3-category example (`default_category_map.json`: gemm/communication/elementwise);
  an operator-specific map (e.g. a MoE kernel's `stage1`/`stage2`/`combine`/`quantize` split) belongs
  in that operator's analysis Skill directory, not here. Trace replay count is explicit and all
  duration/overlap outputs are normalized per replay.
- **`intervals`** — computes category active-time union, physical pairwise overlap, and the longest
  weighted path through a caller-supplied dependency DAG.
- **`experiments`** — validates controlled-experiment invariants, correctness, repetitions,
  changed components, provenance, and overlap before comparing non-additive variants.
- **`evidence`** — resolves every complete measurement track against typed evidence, metric, and
  provenance catalogs; unresolved references invalidate completion.
- **`hardware`** — validates portable target device/topology/runtime/capability context so Skills can
  interpret counters and candidate applicability without hardcoding an implementation.
- **`instruction_analysis`** — parses rocprofv3 ATT per-instruction Hitcount/Latency/Stall/Idle,
  source mapping, and category summaries; ATT remains sampled scope and is never treated as E2E time.
- **`provenance`** — requires collector/tool/command/timestamp/scope/repetitions/raw-artifact,
  perturbation, units, confidence and cross-check metadata for traces, ATT, routes and software counters.
  The CLI also records resolved path, SHA-256 and byte count for every direct input artifact.
- **`schema.build_report`** — assembles the neutral top-level envelope
  (`schema_version: "geak-multirank-analysis-v2"`, typed `status`/`primary_metric`, workload,
  expected world size, evidence catalog, `cases[]`,
  optional `route_comparisons[]`, optional `secondary_comparator_role`).
- **`runner.py`** — deterministic CLI that merges rank-record JSON and optional per-rank traces into
  the neutral report; operator Skills consume this mechanical output rather than redoing arithmetic
  from prose.

## Ground truth this schema was generalized from

The one existing multi-rank analysis artifact in this repo
(`artifacts/analysis/mega_moe_v2_analysis.json`, schema `geak-megamoe-analysis-v1`) has this shape,
which every field name in `aggregate.py`/`schema.py` was chosen to match generically:

```text
{schema_version, status, primary_metric, mori_role, cases[], route_comparisons[]}
cases[].benchmark.timing.<metric>_ms = {rank_mean, rank_max, rank_mean_runs, rank_max_runs,
                                         rank_mean_span_pct, rank_max_span_pct}
cases[].profile.categories.<cat> = {kernel_names, per_rank_ms, rank_mean_ms, rank_max_ms,
                                     rank_min_ms, rank_mean_share_pct, rank_tail_spread_pct}
route_comparisons[].profile_category_delta.<cat> = {rank_max_delta_ms, rank_max_delta_pct}
```

Mapping to this package:
- `<metric>_ms` block → `aggregate.merge_rank_records(records, metric_paths, repetitions=...)` one
  entry per `metric_paths` item (`rank_mean`, `rank_max`, `rank_min`, `rank_tail_spread_pct`, plus
  `*_runs`/`*_span_pct` when `repetitions` is given).
- `categories.<cat>` block → `trace_categories.bucket_trace_events` (per rank) merged across ranks via
  `aggregate.merge_rank_records` with metric paths `"per_category_ms.<cat>"`.
- `mori_role` (a MegaMoE-specific "secondary comparator" note) → generalized to
  `schema.build_report(..., secondary_comparator_role=...)`, optional and free-text.
- The artifact itself is **not** regenerated by this library — it remains the frozen record of one
  specific run (see `artifacts/README.md`).

## Non-goals

- No GPU-group leasing, idle-gating, or process teardown — that is
  `kernel_workflow/scripts/gpu_lease.py` / `gpu_lock.sh`'s job; this package only ever reads env/output
  those already produced.
- No opinion on what counts as a "good" bottleneck or which optimization direction to pick — that is
  an advisory Skill's job (`kernel_workflow/knowledge/analysis_skills/<skill>/SKILL.md`).
- No parsing of a single-rank profile's kernel Top-N — that is
  `e2e_workflow/scripts/parse_profile.py`'s job for the single-server e2e case; this package is the
  distributed/multi-rank complement.
