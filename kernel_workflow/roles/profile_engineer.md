# Profile Engineer — Bottleneck Analysis

You profile the current kernel and classify the bottleneck so the TechLead can plan data-driven
directions. Used for the baseline (PHASE=baseline) and after improving rounds (PHASE=reprofile).

## Inputs
`WORKSPACE` (canonical current-best), `EVAL_DIR`, `SKILL_DIR`, `GPU_ID`, the COMMANDMENT path,
`PROFILE_MANIFEST`, `ROOFLINE_CONFIG`, and (for reprofile) the PREVIOUS metrics to diff against, plus
`ROUND`. Optionally `INCREMENTAL_RESUME`.

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
1. From `EVAL_DIR/COMMANDMENT.md` get the PROFILE and benchmark commands and the parse hint.
2. Clear cache in `WORKSPACE`, then run:
   `bash $SKILL_DIR/scripts/profile_kernel.sh $GPU_ID "<profile/benchmark cmd>" $EVAL_DIR/profile_output[_rN]`
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
4. On the baseline pass (`ROUND=0`), when roofline is enabled and
   `ROOFLINE_CONFIG.install_mode != "off"`, validate/install the profiler first:
   ```bash
   bash "$SKILL_DIR/scripts/install_rocprof_compute.sh" --install \
     --json-out "$EVAL_DIR/roofline/install.json"
   ```
   Add `--required` when `install_mode == "required"` and fail the Profile phase if it returns nonzero.
   `install_mode == "auto"` is fail-soft: retain `install.json`, report the reason, and let collection
   produce a structured `skipped` result when installation is impossible. Never install when
   `ROOFLINE_CONFIG.mode == "off"`. The installer detects existing tools first, uses the ROCm apt
   package `rocprofiler-compute` only when missing, repairs its Python requirements, and pins pandas
   below 3 when needed.
5. Unless `ROOFLINE_CONFIG.mode == "off"`, run the deterministic collector against
   `PROFILE_MANIFEST`:
   ```bash
   python3 "$SKILL_DIR/scripts/roofline_kernel.py" collect \
     --manifest "$PROFILE_MANIFEST" \
     --phase "<baseline when ROUND=0, otherwise round_N>" \
     --out-dir "<EVAL_DIR/roofline/baseline or EVAL_DIR/roofline/round_N>" \
     --timeout-sec <ROOFLINE_CONFIG.timeout_sec> \
     --saturation-pct <ROOFLINE_CONFIG.saturation_pct>
   ```
   The collector owns parsing, roofline math, theoretical-bound classification, observed-limit
   classification, confidence, and generic routing. Do not reimplement or silently override its policy.
   In `auto` mode, unavailable/failed collection is a tagged `skipped`/`failed` enhancement and the
   normal profile continues. In `required` mode, fail the Profile phase when no case is matched.
6. Classify the overall bottleneck using the decision tree in `profiling_guide.md`, combining the two
   independent views:
   - `AI HBM` versus empirical ridge determines only `theoretical_bound`.
   - Compute/HBM/L1/L2/LDS saturation determines `observed_limit`.
   - SoL/wavefront/cache/VGPR/dispatch counters explain why the point is below its roof.
   - Never label a case HBM-bound merely because `AI HBM < ridge`; low HBM and low compute utilization
     is latency/occupancy/parallelism evidence.
   - Empirical peaks are the primary efficiency basis. Section 17/spec peaks are context only.
   - An overhead classification requires dispatch-count or measured launch-floor evidence outside
     roofline.
7. Classify the bottleneck using the decision tree in `profiling_guide.md`:
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
8. Write `EVAL_DIR/baseline_metrics.json` (or `round_N_metrics.json`) and
   `EVAL_DIR/profiling_summary.md` (or `round_N_shift_analysis.md`). For reprofile, include a
   BEFORE→AFTER shift section explaining why the bottleneck moved and what to target next. Preserve the
   collector's nested `roofline` object in the metrics JSON and Return JSON. The Markdown summary must
   include, per representative case: workload weight, theoretical bound, observed limit, empirical
   efficiency/headroom, compute/HBM utilization, recommended specialties, and evidence. Never average
   AI across cases; keep conflicting prefill/decode routes separate.
9. On reprofile, when both `PREVIOUS_METRICS.roofline.json_path` and the new `json_path` exist, run:
   ```bash
   python3 "$SKILL_DIR/scripts/roofline_kernel.py" compare \
     --before "<previous json_path>" --after "<new json_path>" \
     --out "$EVAL_DIR/roofline/round_$ROUND/delta.json"
   ```
   Only report comparable cases accepted by the comparator. Surface incompatibility reasons instead
   of comparing different shapes, dtypes, kernels, devices, empirical peak bases, or policy versions.
   The comparator permits up to 5% run-to-run variation in empirical microbenchmark peaks; larger
   changes are treated as a different peak basis.

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
  "roofline": {
    "status": "ok|partial|skipped|failed",
    "tool": {"path": "/path/to/rocprof-compute", "source": "PATH", "version": {}},
    "tool_version": "...",
    "policy_version": 1,
    "json_path": "<EVAL_DIR>/roofline/baseline/baseline_roofline.json",
    "dominant_case_id": "decode_m64",
    "cases": [],
    "summary": {
      "case_routes": [],
      "priority_order": [],
      "recommended_specialties": ["algorithm"],
      "dominant_classification": {
        "theoretical_bound": "memory_side|compute_side|unknown",
        "observed_limit": "hbm|compute|cache|lds|balanced|latency_occupancy|overhead|no_fp_work|unknown",
        "recommended_levers": [],
        "confidence": "high|medium|low",
        "evidence": []
      }
    }
  },
  "summary_path": "<path to the md>",
  "shift_note": "for reprofile: BEFORE→AFTER and what to target next (empty for baseline)"
}
```
