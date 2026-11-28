# Profiler — multi-node serving

Read this only when `MN_STATE` is in your inputs. It changes the role file's steps as follows.

**Inputs.** `TRACELENS_TRACE_FILES` (may be absent): space-separated rank0 traces, one per engine pod;
`analysis.md` is merged over all of them.

**Step 0, when `TRACELENS_TRACE_FILES` is non-empty.** The merged `analysis.md` spans every engine, so
annotate against ALL the traces at once (one engine's trace alone marks the other engine's kernels
`unresolved`). Run this instead of the `--annotate` block:
```bash
TLARGS=(); for t in $TRACELENS_TRACE_FILES; do TLARGS+=(--torch-trace "$t"); done
python3 "$EVAL_DIR/parse_profile.py" \
  --annotate "$EVAL_DIR/profile/round_${ROUND}/profile_topN.json" "${TLARGS[@]}" \
  --annotate-out "$EVAL_DIR/profile/round_${ROUND}/profile_topN.json"
```
Run the additional trace-analysis pass once per trace in the list (`--out .../profile_topN_tracelens_<i>`,
i = its position) and reconcile each kernel against the trace it appears in. Do NOT write
`profile_topN_by_role` / `profile_workload_by_role` from this prior: the per-engine tables come from
GEAK's own reprofile (step 3 below).

**Step 3, picking the trace.** Every rank's trace is under `$PDIR`; `mn_traces.py trace-gate` names the one to
parse (`mn_traces.py` is the multinode-traces skill: `$SKILL_DIR/knowledge/multinode_skills/multinode-traces/SKILL.md`).
In the non-ATOM branch, replace the `ls -t` line with:
```bash
TRACE=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("parsed_trace") or "")' \
  "$PDIR/ranks/mn_trace_report.json" 2>/dev/null)
[ -n "$TRACE" ] || TRACE=$(ls -t "$PDIR"/*.json.gz "$PDIR"/*.json 2>/dev/null | head -1)
```

## PD (`SERVING_TOPOLOGY=pd_disagg`)

**Step 1, with `BACKEND=sglang` or `atom`: the same leg also probes sampling points**, which is where the two
engines' weights come from (step 3). Plan them from the BASELINE leg's engine logs, then add three
variables to the step 1 command:
```bash
KX="$SKILL_DIR/knowledge/kernel_extraction_skills/kernel-profile"
BL_LOG() { python3 -c 'import json,sys;print(next(e["log"] for e in json.load(open(sys.argv[1]))["pd"]["engines"] if e["role"]==sys.argv[2]))' "$EVAL_DIR/baseline/mn_launch.json" "$1"; }
PTS="$EVAL_DIR/profile/round_${ROUND}/points"
IDS=$(python3 "$SKILL_DIR/knowledge/multinode_skills/multinode-traces/scripts/mn_traces.py" pd-points --skill-root "$KX" --out-dir "$PTS" \
        --engine-log prefill="$(BL_LOG prefill)" --engine-log decode="$(BL_LOG decode)" \
      | sed -n 's/^PROFILE_PROBE_IDS=//p')
# add to the bench_e2e.sh env:
#   SKILL_ROOT="$KX" PROFILE_PROBE_POINTS="$PTS/sampling_points.json" PROFILE_PROBE_IDS="$IDS" PROFILE_PROBE_WITH_WINDOW=1
```
If `pd-points` exits 2, run the leg without the three variables and say why in `notes`; step 3 then
weighs on AgentX's analytic step counts, as one node does without a serving log.

**Step 3: when `$PDIR/ranks/mn_trace_roles.json` EXISTS, parse EACH ENGINE, then merge** (instead of
the single parse). That file (from `mn_traces.py trace-gate`) names one TP-0 trace per engine, each in its own
subdirectory; both engines number ranks from TP-0, so `ls -t | head -1` would pick one at random.
```bash
ROLES="$PDIR/ranks/mn_trace_roles.json"
if [ -s "$ROLES" ]; then
  RD="$EVAL_DIR/profile/round_${ROUND}"; MERGE=()
  # Each engine's front-end log, as mn.py launch recorded it.
  LOG_OF() { python3 -c 'import json,sys;print(next(e["log"] for e in json.load(open(sys.argv[1]))["pd"]["engines"] if e["role"]==sys.argv[2]))' "$RD/mn_launch.json" "$1"; }
  for R in $(python3 -c 'import json,sys;print(" ".join(e["role"] for e in json.load(open(sys.argv[1]))["roles"] if e["trace"]))' "$ROLES"); do
    T=$(python3 -c 'import json,sys;print(next(e["trace"] for e in json.load(open(sys.argv[1]))["roles"] if e["role"]==sys.argv[2]))' "$ROLES" "$R")
    python3 "$EVAL_DIR/parse_profile.py" --torch-trace "$T" \
      --isl <isl> --osl <osl> --conc <conc> \
      ${CHUNK:+--prefill-chunk "$CHUNK"} ${CAPTURE_SIZES:+--capture-sizes "$CAPTURE_SIZES"} \
      --top 25 --out "$RD/profile_topN_$R" --workload-out "$RD/profile_workload_$R.json"
    MERGE+=(--engine "$R=$RD/profile_topN_$R.json" --workload "$R=$RD/profile_workload_$R.json")
  done
  # Probed points (step 1): each point's TP-0 trace -> census -> one coverage ledger.
  KX="$SKILL_DIR/knowledge/kernel_extraction_skills/kernel-profile"; PTS="$RD/points"; WEIGH=()
  if [ -s "$PTS/sampling_points.json" ]; then
    CENSUS=()
    for ID in $(python3 -c 'import json,sys;print(" ".join(p["point_id"] for p in json.load(open(sys.argv[1]))["points"]))' "$PTS/sampling_points.json"); do
      J="$PDIR/$ID.journal.json"
      [ "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$J" 2>/dev/null)" = ok ] || continue
      T=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["roles"][0]["trace"])' "$PDIR/$ID/ranks/mn_trace_roles.json")
      python3 "$EVAL_DIR/parse_profile.py" --torch-trace "$T" --top 200 \
        --isl <isl> --osl <osl> --conc <conc> --out "$PTS/census_$ID"
      CENSUS+=(--census "$ID=$PTS/census_$ID.json")
    done
    python3 "$KX/scripts/coverage_ledger.py" --points "$PTS/sampling_points.json" "${CENSUS[@]}" \
      --out "$PTS/coverage.json" && WEIGH=(--coverage "$PTS/coverage.json")
  fi
  python3 "$SKILL_DIR/knowledge/multinode_skills/multinode-traces/scripts/mn_traces.py" topn-merge \
    "${MERGE[@]}" --top 25 --out "$RD/profile_topN" "${WEIGH[@]}" --roles "$ROLES" \
    --workload-out "$RD/profile_workload.json"
fi
```
With probed points, each engine is weighed by its GPU-time mass over the baseline leg: every point's
per-step kernel time × the steps that engine's baseline log counts at that shape, summed, × its ranks
(`engine_weights.rule = gpu_time_mass`, the rule AgentX uses). As on one node, a point that does not
verify is recaptured once (`bench_e2e.sh` keeps the first try as `<id>.try1`); one still not `ok` gets
no census, and the merge weighs on with coverage_ledger's warning that its steps contributed no mass
(`engine_weights.warnings`, also in the .md). Copy every such warning into `notes`: that engine's weight
is low by those steps. Without probed points (vLLM, or `pd-points` failed) each engine is weighed as one
node is without a serving log: its window's mean GPU time per step (`step_ledger`) × parse_profile's
`serving.analytic_calls` for its phase (CONC×⌈ISL/chunk⌉ prefill steps, OSL decode steps) × its ranks
(`engine_weights.rule = analytic_calls`). That count assumes every decode step runs CONC requests; copy
its warning into `notes`. It is only as right as the `--isl/--osl/--conc` and `CHUNK` each engine was
parsed with. If the merge exits 2 (a phase with no probed mass at all, an engine without step
annotations or `serving.analytic_calls`, no rank counts), the profile FAILED — report the reason it
printed; never merge the engines yourself. Put the merged table's `bottleneck_engine`, each
`per_engine[].weight_pct`, `engine_weights.rule` (with `gpu_time_mass`: `inputs.mass_us`; with
`analytic_calls`: `inputs.per_step_ms` and `inputs.calls_per_wave`; every warning) in `notes`.

`$RD/profile_workload.json` is your `profile_workload_json`: one patch runs on both engines, so the
Kernel Extractor weighs a head on both engines' cases at once. If one engine produced no trace, say it
is unprofiled (the merge exits 2 on one engine) — never infer its table from the other. `CHUNK` /
`CAPTURE_SIZES` are per engine: read them from that engine's own server log (`$(LOG_OF prefill)` /
`$(LOG_OF decode)` above).

**Return JSON:** add the per-engine artifacts beside `profile_topN_json`:
```json
  "profile_topN_by_role": {"prefill": "<...>/profile_topN_prefill.json",
                           "decode": "<...>/profile_topN_decode.json"},
  "profile_workload_by_role": {"prefill": "<...>/profile_workload_prefill.json",
                               "decode": "<...>/profile_workload_decode.json"},
```
