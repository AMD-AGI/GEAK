# profile_engineer — mode=mega

Read after `roles/profile_engineer.md`. In mode=mega, each `##` section below replaces the section with the same
heading in `profile_engineer.md`; a section whose heading is not in `profile_engineer.md` adds a mode=mega procedure.
Everything not listed here is used as written in `profile_engineer.md`.

## Inputs
`WORKSPACE` (canonical current-best), `EVAL_DIR`, `SKILL_DIR`, `GPU_ID`, the COMMANDMENT path, and
(for reprofile) the PREVIOUS metrics to diff against, plus `ROUND`. Optionally `INCREMENTAL_RESUME`.

For **PHASE=mega_analysis** you are additionally given the output paths for the MegaMoE bench's own
instrumentation — `ANALYSIS_JSON_OUT` (per-rank `--json-output`), `ANALYSIS_XGMI_OUT`
(`--xgmi-output`), `ANALYSIS_COMBINE_WAIT_OUT` (`--combine-wait-output`) — and `COMBINE_WAIT_ENV`
(`AITER_MEGAMOE_COMBINE_WAIT_STATS=1`, prepend it to the combine-wait run so the fused arrival-wait
timer is armed).

**FAST PATH — if `INCREMENTAL_RESUME` is set** (a resumed deep wave; PHASE=baseline): the bottleneck was
already classified in a prior wave. Do NOT re-run the full baseline profile from scratch — read the prior
`EVAL_DIR/baseline_metrics.json` (or the latest `round_N_metrics.json` under STATE) and return the same
schema with the cached `bottleneck` / metrics. Re-profile fully only if no prior metrics exist. This
keeps the per-wave fixed cost low so the burst spends its budget on optimization rounds. (When
`INCREMENTAL_RESUME` is absent — default/fast/first deep burst — do the full baseline profile below.)

Read `SKILL_DIR/knowledge/profiling_guide.md` and `amd_instinct.md` first. **Identify the actual
accelerator on this box** (`amd_instinct.md` §0: `rocminfo` for the gfx arch + CU count, `rocm-smi
--showproductname` for the card) and record it (gfx942/CDNA3 vs gfx950/CDNA4, CU count, HBM peak) in
your metrics — the roofline ceiling and grid-sizing advice downstream depend on the real card, not an
assumed MI300X.

## Steps
1. From `EVAL_DIR/COMMANDMENT.md` get the PROFILE command and parse hint.
2. Clear cache in `WORKSPACE`, then execute the already lease-wrapped PROFILE entry verbatim
   (for reprofile, change only its output directory). Never wrap it in another `gpu_lock.sh`.
   This warms up, then profiles with the best available profiler (rocprof-compute → omniperf →
   rocprof → benchmark-only) and writes a report.
   If the report contains a `!!! PROFILER FAILED` block, work the fault-tolerance ladder in
   `profiling_guide.md` ("Profiler failed?"): use `<tool> --help` to find the renamed flag, re-run once
   with the named env override, then degrade deliberately — and record which tool actually ran + why in
   `profiler_used` / your summary. Do not accept a silent degrade.
3. Read the report. Extract what's available: VALU/VMEM/LDS utilization, effective HBM bandwidth,
   active vs total cycles, dependency/issue wait, L1/L2 hit rate, coalescing %, branch divergence,
   active threads/instr, VGPR/SGPR usage, scratch bytes, **and the per-kernel dispatch breakdown
   (how many distinct kernels launch per call and their % of time)** — the dispatch count is a key
   geomean signal.
4. Classify the bottleneck using the decision tree in `profiling_guide.md`:
   compute-bound / memory-bound / latency-bound / lds-bound / balanced. ALSO flag **overhead-bound**
   when per-case latencies are similar across very different problem sizes, or dispatch count > 1
   with small kernels — this points at host/dispatch overhead (see `geomean_levers.md`).
   - **Do not call it memory-bound on a small AI alone** — only when HBM/VMEM utilization is actually
     high. A small AI with low HBM util is latency-bound (`profiling_guide.md` → decision tree note).
   - **When latency-bound, name the sub-case in your opportunities**: dependency-wait dominant (C1,
     shorten the serial chain) vs issue-wait dominant (C2, raise occupancy / GPU fill). They have
     opposite fixes, so the TechLead needs the sub-case, not just "latency". Re-read this split after
     any tile / `num_stages` / `num_warps` change — it is a property of the config, not the source.
   - **Run the cheap peak/fill sanity-checks** (`profiling_guide.md` → "Cheap checks…") before trusting
     the label: any roofline efficiency > 100% is a mis-calibrated peak (use HBM%/F32), and
     `CTAs = Grid/Workgroup < CU count` means the GPU is not even filled — call that out first.
5. Write `EVAL_DIR/baseline_metrics.json` (or `round_N_metrics.json`) and
   `EVAL_DIR/profiling_summary.md` (or `round_N_shift_analysis.md`). For reprofile, include a
   BEFORE→AFTER shift section explaining why the bottleneck moved and what to target next.

If no profiler is available, fall back to benchmark-only + the per-case table + dispatch count from
`rocprof --stats` if present; still classify as best you can and SAY the profiler was unavailable.

## Return JSON
```json
{
  "bottleneck": "compute|memory|latency|lds|balanced|overhead",
  "profiler_used": "rocprof-compute|omniperf|rocprof|benchmark-only",
  "device": "detected card, e.g. 'MI300X / gfx942 / CDNA3, 304 CU, ~5.3 TB/s'",
  "dispatch_count": 0,
  "key_metrics": {"valu_pct": 0.0, "vmem_pct": 0.0, "lds_pct": 0.0, "hbm_gbps": 0.0,
                  "l2_hit_pct": 0.0, "vgpr": 0, "scratch_bytes": 0},
  "top_kernels": [{"name": "...", "pct_of_total": 0.0}],
  "top_opportunities": ["ranked, specific, tied to a metric or per-case number"],
  "summary_path": "<path to the md>",
  "shift_note": "for reprofile: BEFORE→AFTER and what to target next (empty for baseline)"
}
```

---

## PHASE=mega_analysis — fused-kernel native analysis (do NOT run rocprof)

For a fused MegaMoE megakernel the generic rocprof roofline above is **not just unhelpful, it is
actively wrong**. Under co-resident fusion the standalone `stage1` and `stage2_combine` timers both
RISE while rank-max end-to-end FALLS — combine is folded into the persistent megakernel as a third
ticketed queue and its all-rank barrier is deleted, so a per-stage timer reads a bigger number even
though total time dropped. A roofline that concludes "both stages got slower ⇒ regression" is exactly
backwards (see `GEAK_TASK.md` on fused-timer semantics, and the bench's own combine-wait docstring:
"do NOT read [a flat 0] as an overlap win"). So in this phase you **do not run rocprof**. You run the
bench's OWN instrumentation and classify from it.

### Steps
1. From `EVAL_DIR/COMMANDMENT.md` get the mega bench command (`bench_mega_moe_v2.py`, the
   `8192_uniform` production shape) and its env preamble. Clear cache in `WORKSPACE`.
2. Run the bench **three ways** through the existing lease wrapper (never add another `gpu_lock.sh`),
   writing to the given output paths:
   - **rank records**: add `--json-output ANALYSIS_JSON_OUT`. Parse `cases[0].ranks[*].timing_ms`
     (`e2e`, `stage1`, `stage2_combine`) — the **rank-MAX** across ranks is the straggler and the
     metric that matters, NOT rank-mean. Also read the printed `[RESULT] … mega_e2e=<mean>/<max>ms
     speedup=<pct>%` line (the second number is rank-max) and `[PERF-GUARD] PASS|FAIL speedup=..%
     minimum=..%` if present.
   - **XGMI amplification**: add `--xgmi-output ANALYSIS_XGMI_OUT` (requires `--mega-only`). Parse
     `derived.counter_to_logical_useful_amplification` — fabric bytes moved ÷ logical-useful bytes.
     >1 means the interconnect carried more than the payload (a real traffic lever a roofline cannot
     see); ~1 means payload-bound.
   - **combine-wait**: prepend `COMBINE_WAIT_ENV` and add `--combine-wait-output
     ANALYSIS_COMBINE_WAIT_OUT`. Parse `fused_arrival_wait.rank_max_of_p95_us` and
     `rank_max_of_total_p95_us` (the comparable-to-pre-fusion headline). If `fused_arrival_wait.enabled`
     is false or the numbers are 0, say so — an empty timer under fusion is NOT an overlap win.
3. Verify the path marker: every rank must print `[megamoe] path=MEGA`. Count them. If any rank prints
   `SCATTERED`, the kernel fell back and every number is from the wrong path — set `path_marker` to
   `SCATTERED` or `MIXED` and make that your top finding.
4. Classify `bottleneck` from **rank-max e2e + XGMI amplification + combine-wait p95**, never from
   summed per-stage timers:
   - straggler-bound (large `combine_wait_total_p95_us` relative to e2e ⇒ rank imbalance / late
     arrivals; the fix is scheduling/skew, not more compute),
   - fabric-bound (`xgmi_amplification` ≫ 1 ⇒ redundant XGMI traffic; the fix is payload/quant/route),
   - compute-bound (guard floor missed with low amplification and low wait),
   - overlap-headroom (e2e above the sum of the non-overlapped critical pieces ⇒ room to co-schedule).
5. Write `EVAL_DIR/mega_analysis/summary.md` with the parsed numbers and, in `fusion_note`, the single
   most valuable NEXT fusion/overlap move (e.g. "combine-wait p95 is 18% of e2e and concentrated on 2
   ranks → overlap combine with GEMM2 tail via the SITE-x skew knob", or "amplification 1.0, guard
   met → topology is payload-bound, pursue launch-count not overlap"). This replaces the roofline's
   per-stage-regression call. Keep `bottleneck`/`top_opportunities` populated so the planner degrades
   gracefully.

### Return JSON (PHASE=mega_analysis)
```json
{
  "bottleneck": "straggler|fabric|compute|overlap-headroom|balanced|overhead",
  "top_opportunities": ["ranked, each tied to a parsed number (rank-max ms, amplification, wait p95)"],
  "dispatch_count": 2,
  "device": "detected card",
  "path_marker": "MEGA|SCATTERED|MIXED|unknown",
  "path_marker_count": 8,
  "world_size": 8,
  "rank_max_ms": 0.0,
  "stage1_rank_max_ms": 0.0,
  "stage2_combine_rank_max_ms": 0.0,
  "speedup_pct": null,
  "perf_guard_floor": null,
  "perf_guard_pass": false,
  "xgmi_amplification": null,
  "combine_wait_p95_us": null,
  "combine_wait_total_p95_us": null,
  "per_rank_straggler": [{"rank": 0, "e2e_ms": 0.0, "stage1_ms": 0.0, "stage2_combine_ms": 0.0}],
  "fusion_note": "the single most valuable next fusion/overlap move, tied to a parsed number",
  "summary_path": "<path to mega_analysis/summary.md>"
}
```
Report a field as `null` when its run did not produce it; never invent a number. An empty combine-wait
under fusion, or `--mega-only` making `speedup_pct` NaN/`null`, is DATA — say it, do not paper over it.

---

## PHASE=mega_baseline — measured per-stage baseline (generic mega, Expert Skills off)

Called only when the input `MEGA_MEASUREMENT_GUIDE` is present. Follow that file's §1, not the
PHASE=mega_analysis section above: time the frozen baseline per stage (stage timers + per-kernel
trace split, rank-max), write `EVAL_DIR/mega_baseline_stages.json`, and return the PROFILE schema
with measured values only.
## PHASE=analyze_profile — Analysis Engineer — Optional deterministic Step-2 evidence pipeline
You execute one explicitly enabled profile-analysis Skill after generic Profile has already
succeeded. You never modify kernels, choose implementation directions, or overwrite generic profile
classification.

### Inputs

`WORKSPACE`, `EVAL_DIR`, `ROUND`, `COMMANDMENT`, `PROFILE_SUMMARY`, `ANALYSIS_SKILL`,
`ANALYSIS_SKILL_DIR`, and `SKILL_DIR`.

### Required behavior

1. If `ANALYSIS_SKILL_DIR` is missing or has no `SKILL.md`, return `status=degraded` and preserve the
   generic Profile result.
2. Read `ANALYSIS_SKILL_DIR/SKILL.md`.
3. Discover only artifacts produced by the immutable harness/Profile commands. Do not invent values,
   evidence references, completion labels, hardware ceilings, or provenance.
4. Execute the checked-in bundle builder, generic runner, domain analyzer, and output validator
   exactly as specified by the Skill. Never reproduce numerical arithmetic manually.
5. A measurement track is `complete` only when the generic runner resolves every artifact, metric,
   and provenance reference against its evidence catalog.
6. Run the Skill output validator. Return `status=ready` only when it succeeds.
7. On any missing artifact, collector, or command failure, write a short failure note and return
   `status=degraded`. Never fail or replace the already-successful generic Profile call.

### Mega route-report pre-step (mode=mega / label `mega:analyze` only)

When `PROFILE_SUMMARY` is a mega fused-kernel summary (route/paired evidence, not a single-op
profile), the bench does not emit a rocprofv3 report — this is a **Tier-B** (no-rocprof) route-causality
pass. Before the skill chain, shape the latencies the mega bench already wrote under `EVAL_DIR` into the
minimal `geak-megamoe-analysis-v1` report the analyzer accepts:

1. Use `PROFILE_SUMMARY.candidate_evidence` as the ONLY input. The orchestrator builds it from this
   round's independent Verify and binds it to one identity (`candidate_id`, `head`, `attempt_id`): the
   paired guard readings `[{guard,base,cand}]`, `frozen_per_case` (the `BASELINE_PER_CASE` rows),
   `per_case`, `denominator_mismatch`, and `component_breakdown`/`hot_path_counters` as
   `--component-timings`. Do NOT gather other files from `EVAL_DIR`, do NOT pair two baseline routes
   (uniform vs skew) as base/candidate, and do NOT re-run GPU work or invent numbers. Stamp the
   candidate id and HEAD into the report. If `candidate_evidence` is absent, skip this pre-step and
   fall through to step 2 of Required behavior with a short note (never fail).
2. Run `python3 SKILL_DIR/scripts/multi_rank_analysis/build_mega_route_report.py --paired <…>
   --baseline-per-case <…> [--component-timings <…>] --tokens-per-rank <target N> --output
   EVAL_DIR/mega_route_report.json`. This is NEW infra, NOT part of the SHA-pinned analysis bundle.
3. Feed that report as the `--report` input to the Skill's `analyze.py` (in place of a rocprof report),
   then run its `validate_output.py` exactly as in Required behavior. Attribution is capped
   `awaiting_measurement`, confidence `low` (and `incomplete_reasons` is stamped when the bench lumped
   `stage2_combine`); surface it at that declared confidence — never as resolved root cause.

### Return JSON

```json
{
  "status": "ready|degraded",
  "analysis_skill": "skill slug",
  "analysis_schema_version": "schema version or empty",
  "analysis_status": "awaiting_measurement|evidence_complete|unavailable",
  "analysis_json": "validated JSON path or empty",
  "failure_note": "empty on success; concise cause on degradation"
}
```
