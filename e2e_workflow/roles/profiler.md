# Profiler

Produce canonical `profile_topN.json` / `.md` and workload attribution with `scripts/parse_profile.py`.
Use the same serving workload/config as the throughput benchmark; do not optimize. Read
`SKILL_DIR/knowledge/profile_parse.md`, backend guidance, and the contract in `scripts/adapters/atom.sh`
when using ATOM. Read `EVAL_DIR/env_report.json` for actual trace-source availability.

## PHASE=baseline / PHASE=reprofile

Inputs: `EVAL_DIR`, `MODEL_PATH`, `BACKEND`, `GPU_ID`, serving TP/GPU invariant, `WORKLOAD`, `ROUND`,
`OVERLAY_PYTHONPATH`, `EXTRA_SERVER_ARGS`, `EXTRA_ENV`, `SKILL_DIR`, and optional TraceLens paths
`TRACELENS_ANALYSIS_MD`, `TRACELENS_KERNEL_CANDIDATES_JSON`, `TRACELENS_REPORT_JSON`,
`TRACELENS_TRACE_FILE`. The supplied overlay/flags/env are the current accepted stack.

### Capture contract

- Keep ISL/OSL/concurrency, TP/GPU and caller-owned client protocol unchanged. Warm the server, then
  profile a bounded load window containing the prefill ramp and steady decode. Shape-hidden decode
  under graph replay is expected; missing dimensions alone do not justify recapture.
- `bench_e2e.sh` sizes the window from workload/chunk size and TPOT. Preserve supplied bounds:
  `PROFILE_NUM_STEPS`, `PROFILE_NUM_PROMPTS`, `PROFILE_WINDOW_SEC[_MAX]`, ATOM's corresponding limits,
  `PROFILE_MAX_ITERS` and `PROFILE_NUM_STEPS_MAX`. `PROFILE_WARMUP_SEC=0` retains the prefill ramp.
  If evidence shows a missed serving phase, adjust within the declared budget; do not run an unbounded
  doubling loop. Distinguish analytically estimated coverage from measured step annotations.
- When available, torch traces provide names/shapes and rocprofv3 refines device durations. If torch
  capture is unavailable, use an available device trace; if neither yields usable evidence, report
  failed capture rather than inventing Top-N rows.
- `total_gpu_time_ms` is summed device duration, not serving throughput. Preserve source, raw timing
  and missingness. For TP>1, check collective kernels in the full trace; absence requires explanation/
  capture repair, not a fabricated communication share.
- Use `bench_e2e.sh` and its owned teardown. Allow profiler finalization within the run deadline;
  multi-rank CSV flush can take up to about 25 minutes. Poll artifacts in short calls. Do not kill
  processes by name or assume a slow flush is a hung server.

### Baseline TraceLens reuse

Only for PHASE=baseline / ROUND 0, reuse a non-empty existing `TRACELENS_ANALYSIS_MD` when its
workload/config match this baseline and usable profiler evidence accompanies it. Reprofile always
captures the current stack. Read machine `hot_kernels[]` from candidates/report JSON and map name,
GPU share, call count, duration, shapes/dtypes, class/editability and source/launcher hints into the
canonical schema. Convert units explicitly (`total_ms=duration_us/1000`); preserve unknown fields.

Select the top-level rank0 worker serving trace, never `capture_traces/` warmup or the async engine
trace. Annotate rows from profiler evidence; never hand-write `entity_kind`:
```bash
TLT=$(ls -1 "$TRACELENS_TRACE_FILE"/*rank0*.pt.trace.json.gz 2>/dev/null | head -1)
[ -z "$TLT" ] && TLT=$(ls -1 "$TRACELENS_TRACE_FILE"/*.pt.trace.json.gz \
  "$TRACELENS_TRACE_FILE"/*.json.gz "$TRACELENS_TRACE_FILE"/*.json 2>/dev/null | head -1)
python3 "$EVAL_DIR/parse_profile.py" \
  --annotate "$EVAL_DIR/profile/round_${ROUND}/profile_topN.json" --torch-trace "$TLT" \
  --annotate-out "$EVAL_DIR/profile/round_${ROUND}/profile_topN.json"
python3 "$EVAL_DIR/parse_profile.py" --torch-trace "$TLT" \
  --top 25 --out "$EVAL_DIR/profile/round_${ROUND}/profile_topN_tracelens" \
  --isl <isl> --osl <osl> --conc <conc> \
  --workload-out "$EVAL_DIR/profile/round_${ROUND}/profile_workload.json"
```
Run this only when a usable trace was found; otherwise use normal collection. Reconcile parser
shapes/dtypes with prior text, preferring direct trace evidence and reporting disagreements. Preserve
annotated `entity_kind`, `device_kernel`, `profile_parent` and split device-child shares. Keep prior
ranking as a documented view, not authority over conflicting evidence. Set source `tracelens+trace`.

### Normal collection and parsing

```bash
BACKEND="$BACKEND" OUT_DIR="$EVAL_DIR/profile/round_${ROUND}" GPU="$SERVING_GPU" TP="$SERVING_TP" MODEL="$MODEL_PATH" \
ISL=<isl> OSL=<osl> CONC=<conc> REPEATS=1 PROFILE=1 \
OVERLAY_PYTHONPATH="$OVERLAY_PYTHONPATH" EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS" EXTRA_ENV="$EXTRA_ENV" \
  bash "$EVAL_DIR/bench_e2e.sh" 2>&1 | tee "$EVAL_DIR/logs/profile_r${ROUND}.log"
```
Select a finalized worker trace. ATOM requires `atom_profile_manifest.json` with `status:complete`
and all expected ranks complete before choosing rank0; do not silently accept partial rank coverage.
```bash
PDIR="$EVAL_DIR/profile/round_${ROUND}/profile"
if [ "$BACKEND" = atom ]; then
  python3 - "$PDIR/atom_profile_manifest.json" <<'PYCODE'
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("status") == "complete", d
assert len(d.get("completed_ranks", [])) >= int(d.get("expected_ranks", 1)), d
PYCODE
  TRACE=$(find "$PDIR/rank_0" "$PDIR/dp0_tp0" -type f -name '*.trace.json.gz' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
else
  TRACE=$(ls -t "$PDIR"/*rank0*.json.gz "$PDIR"/*rank0*.json 2>/dev/null | head -1)
  [ -n "$TRACE" ] || TRACE=$(ls -t "$PDIR"/*.json.gz "$PDIR"/*.json 2>/dev/null | head -1)
fi
python3 "$EVAL_DIR/parse_profile.py" \
  ${TRACE:+--torch-trace "$TRACE"} ${ROCPROF_DIR:+--rocprof-dir "$ROCPROF_DIR"} \
  --isl <isl> --osl <osl> --conc <conc> \
  ${CHUNK:+--prefill-chunk "$CHUNK"} ${CAPTURE_SIZES:+--capture-sizes "$CAPTURE_SIZES"} \
  --top 25 --out "$EVAL_DIR/profile/round_${ROUND}/profile_topN" \
  --workload-out "$EVAL_DIR/profile/round_${ROUND}/profile_workload.json"
```
Require at least one usable trace source before parsing. Derive CHUNK/capture sizes from live config/
logs. `gpu_user_annotation` step spans permit measured phase/call attribution when present;
`est_shape`/`est_calls` remain analytic estimates. Do not call them measured or recapture merely
because a build lacks step annotations. Report `weight_source:regime_prior` when shapes are unavailable.

Resolve top `other` entries from installed source and note classification evidence. Check per-call
distributions when available: distinguish collective wait, one-time warmup/outliers and genuine
prefill/decode mixtures. Preserve raw shares; any adjusted view must explain its basis and renormalize
ALL rows against the adjusted total. Do not label guessed discounts as measurements or erase real
collective workload cost. Without samples, record a qualitative caveat only.

After finalizing Top-N, if `ANALYSIS_SKILL_DIR` is a non-empty path that EXISTS, read its `SKILL.md`
and produce the requested analysis artifact. Otherwise skip and return `profile_roofline_json:""`.
Analysis may enrich but never edit Top-N shares. On analysis failure, retain valid Top-N and report
the limitation. Reprofile reports before/after changes with evidence.

Return JSON:
```json
{
  "round": 0,
  "profile_topN_json": "<EVAL_DIR>/profile/round_0/profile_topN.json",
  "profile_workload_json": "<EVAL_DIR>/profile/round_0/profile_workload.json",
  "profile_topN_md": "<EVAL_DIR>/profile/round_0/profile_topN.md",
  "profile_roofline_json": "<EVAL_DIR>/profile/round_0/profile_roofline.json  (\"\" if no analysis skill ran)",
  "source": "torch-trace|merged|tracelens|tracelens+trace",
  "total_gpu_time_ms": 0.0,
  "top_kernels": [
    {"rank": 1, "short_name": "...", "classification": "...", "pct_gpu_time": 0.0,
     "entity_kind": "<copied from the profile_topN.json row>", "device_kernel": "<copied from the profile_topN.json row>",
     "calls": 0, "avg_us": 0.0, "shapes": [], "editable": true, "regime_note": "prefill|decode|both"}
  ],
  "shift_note": "for reprofile: how the bottleneck moved vs previous round",
  "notes": "resolved 'other' entries, rocprof availability, anything unusual"
}
```
