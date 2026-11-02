#!/usr/bin/env bash
# applyback: layer the overlay -> pass smoke -> re-measure e2e with the overlay -> hand
# the A/B to the slot adapter.
set -uo pipefail

: "${OUT:?OUT}"; : "${RUN_ROOT:?RUN_ROOT}"; : "${SKILL_ROOT:?SKILL_ROOT}"
: "${GEAK_ROOT:?GEAK_ROOT}"; : "${WORKLOAD_ROOT:?WORKLOAD_ROOT}"
SCRIPTS="$WORKLOAD_ROOT/scripts"
PARAMS="${PARAMS:-$OUT/params.json}"
SLOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

p() {
  python3 -c "
import json,os
d=json.load(open('$PARAMS')) if os.path.exists('$PARAMS') else {}
v=d
for k in '$1'.split('.'):
    v = v.get(k) if isinstance(v, dict) else None
print('$2' if v is None else (str(v).lower() if isinstance(v,bool) else v))"
}

OPTIMIZED="${OPTIMIZED:-$RUN_ROOT/optimize/out/optimized.json}"
TASKS_DIR="${TASKS_DIR:-$RUN_ROOT/profile/out/tasks}"
BASELINE="${BASELINE:-$RUN_ROOT/baseline/out/baseline.json}"
REPEATS="$(p repeats 3)"
SMOKE_REQUIRED="$(p smoke_required true)"
DURATION="$(p duration_s '')"

for f in "$OPTIMIZED" "$BASELINE"; do
  [ -f "$f" ] || { python3 "$SCRIPTS/emit_result.py" --status blocked \
      --notes "missing upstream artifact $f — run the corresponding stage first"; exit 0; }
done

# --- 1. Layer the overlay.
OVERLAY="$OUT/overlay"
python3 "$SLOT/build_overlay.py" --optimized "$OPTIMIZED" --tasks-dir "$TASKS_DIR" \
    --overlay "$OVERLAY" --manifest "$OUT/applied.json" 2>&1 | tee "$OUT/overlay.log"
BRC=${PIPESTATUS[0]}
if [ "$BRC" = 3 ]; then
  python3 "$SCRIPTS/emit_result.py" --status blocked \
    --notes "there is no kernel to apply back (the optimize stage produced nothing that
passed cold re-verification). This is not a fault: every stage ran, and the conclusion is
that no usable optimization has been found yet."
  exit 0
fi
[ "$BRC" = 0 ] || { python3 "$SCRIPTS/emit_result.py" --status fail \
    --notes "building the overlay failed; see overlay.log"; exit 0; }

# --- 2. Smoke. Missing pieces mean blocked; never produce a throughput number that in
#        fact measured something else.
SMOKE="$GEAK_ROOT/e2e_workflow/scripts/agentx_smoke.sh"
if [ "$SMOKE_REQUIRED" = "true" ]; then
  if [ ! -x "$SMOKE" ] && [ ! -f "$SMOKE" ]; then
    python3 "$SCRIPTS/emit_result.py" --status blocked --notes "cannot find $SMOKE"; exit 0
  fi
  if ! bash "$SMOKE" > "$OUT/smoke.log" 2>&1; then
    python3 "$SCRIPTS/emit_result.py" --status blocked \
      --notes "agentx smoke did not pass — aiperf/InferenceX are most likely absent from
this machine (they are on neither \$PATH nor /opt). See the end of smoke.log. Throughput
measured before smoke passes cannot be used."
    exit 0
  fi
fi

# --- 3. Re-measure e2e with the overlay.
# REUSE_SERVER=0: the overlay has to be loaded by a **freshly started server**; reusing an
# already-running one measures an unpatched stack.
export BENCH_CLIENT=agentx REUSE_SERVER=0 REPEATS="$REPEATS"
export OVERLAY_PYTHONPATH="$OVERLAY"
[ -n "$DURATION" ] && export GEAK_AGENTX_DURATION_S="$DURATION"
SERVING_ENV="$(p serving_env '')"

# A multi-config baseline (adopt-ci-artifacts) names representative points; each is
# measured against its own CI number. Otherwise: the single A/B, unchanged.
REP="$(python3 -c "
import json
b=json.load(open('$BASELINE'))
print(' '.join(b.get('representative') or []) if b.get('multi_config') else '')")"

if [ -z "$REP" ]; then
  E2E_OUT="$OUT/e2e"
  mkdir -p "$E2E_OUT"
  export OUT_DIR="$E2E_OUT"
  [ -n "$SERVING_ENV" ] && [ -f "$SERVING_ENV" ] && { set -a; . "$SERVING_ENV"; set +a; }
  echo "[applyback] re-measuring e2e with the overlay: repeats=$REPEATS overlay=$OVERLAY"
  bash "$GEAK_ROOT/e2e_workflow/scripts/bench_e2e.sh" 2>&1 | tee "$OUT/bench.log"
  ERC=${PIPESTATUS[0]}
  echo "[applyback] bench_e2e exited $ERC"
  # --- 4. Slot adapter: A/B + result.json.
  python3 "$SLOT/finalize.py" --bench-rc "$ERC" --e2e-dir "$E2E_OUT" --baseline "$BASELINE"
  exit $?
fi

BASE_DIR="$(dirname "$BASELINE")"
for PID in $REP; do
  PDIR="$OUT/points/$PID"
  mkdir -p "$PDIR/e2e"
  cp "$OUT/applied.json" "$PDIR/applied.json"
  ln -sfn "$OVERLAY" "$PDIR/overlay"   # finalize names the overlay among its artifacts
  PB="$BASE_DIR/points/$PID/baseline.json"
  # Each point is its own deployment: CONC/TP from its CI config, and the launch
  # settings that differ between points (KV offload, speculative decoding, ...) from
  # params.point_serving_env.<pid>, else the serving_env.sh the baseline generated from the
  # point's CI recipe, else the shared serving_env. CONC/TP are exported
  # after the env file so the point's own values win.
  PENV="$(p "point_serving_env.$PID" "")"
  # default: the env adopt-ci-artifacts generated from this point's CI recipe
  [ -z "$PENV" ] && [ -f "$BASE_DIR/points/$PID/serving_env.sh" ] && PENV="$BASE_DIR/points/$PID/serving_env.sh"
  [ -z "$PENV" ] && PENV="$SERVING_ENV"
  (
    [ -n "$PENV" ] && [ -f "$PENV" ] && { set -a; . "$PENV"; set +a; }
    eval "$(python3 -c "
import json,shlex
c=json.load(open('$PB'))['config']
print('export CONC=%s TP=%s' % (shlex.quote(str(c['concurrency'])), shlex.quote(str(c['tp']))))")"
    export OUT_DIR="$PDIR/e2e" OUT="$PDIR"
    echo "[applyback] $PID: CONC=$CONC TP=$TP env=${PENV:-<none>} repeats=$REPEATS"
    bash "$GEAK_ROOT/e2e_workflow/scripts/bench_e2e.sh" > "$PDIR/bench.log" 2>&1
    ERC=$?
    echo "[applyback] $PID: bench_e2e exited $ERC"
    python3 "$SLOT/finalize.py" --bench-rc "$ERC" --e2e-dir "$PDIR/e2e" --baseline "$PB"
  )
done

# --- 4. Fold the per-point A/Bs.
python3 "$SLOT/aggregate_points.py" $REP
