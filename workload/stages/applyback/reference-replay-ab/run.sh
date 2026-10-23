#!/usr/bin/env bash
# applyback: build the overlay -> register it as a case of the reference harness ->
# re-run that harness unchanged -> hand the A/B to the slot adapter.
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

blocked() { python3 "$SCRIPTS/emit_result.py" --status blocked --notes "$1"; exit 0; }

OPTIMIZED="${OPTIMIZED:-$RUN_ROOT/optimize/out/optimized.json}"
TASKS_DIR="${TASKS_DIR:-$RUN_ROOT/profile/out/tasks}"
HARNESS="$(p harness_dir '')"
HCMD="$(p harness_cmd '')"
CASE_FILE="$(p case_file '')"
CASE_NAME="$(p case_name geakx_applyback)"
BASE_CASE="$(p base_case baseline)"
OVL_SUB="$(p overlay_subdir '')"
REF_RUN="$(p reference_run '')"
REF_KEY="$(p reference_key output_tok_per_s)"
TAG_PREFIX="$(p run_tag_prefix wl_geakx_ab)"
REPEATS="$(p repeats 2)"
DURATION="$(p duration_s 3600)"
CACHE_ROOT="$(p cache_root '')"
STOCK_TAG="$(p stock_run_tag "${TAG_PREFIX}_stock")"

[ -f "$OPTIMIZED" ] || blocked "missing $OPTIMIZED — run the optimize stage first"
for f in "$HARNESS/$HCMD" "$HARNESS/$CASE_FILE" "$REF_RUN"; do
  [ -e "$f" ] || blocked "the reference harness is incomplete on this machine: $f is missing.
Without it the candidate leg cannot be measured the same way as the reference, and a
throughput number measured any other way would not be an A/B."
done
[ -n "$CACHE_ROOT" ] || blocked "cache_root is required: stock and candidate arms need the same per-run fresh compile-cache policy"
python3 "$SKILL_ROOT/summarize.py" --settings "$PARAMS" --preflight \
    > "$OUT/contract_preflight.json" 2>&1 \
    || blocked "reference-replay contract is incomplete; configure actual native validity, overlay receipt and serving-rank proof paths. See contract_preflight.json"

# The serving stack takes the node. Refuse rather than measure against someone else's load.
if curl -fsS --max-time 3 "http://localhost:$(p port 8899)/health" >/dev/null 2>&1; then
  blocked "a server is already listening on this port; applyback needs the node to itself
and the overlay must be loaded by a freshly started server (reusing a running one measures
an unpatched stack)"
fi

# --- 1. Build/import in the configured analysis environment (host python3 by default).
OVERLAY="$OUT/overlay"
python3 - "$PARAMS" "$SLOT/build_overlay.py" --optimized "$OPTIMIZED" --tasks-dir "$TASKS_DIR" \
    --overlay "$OVERLAY" --manifest "$OUT/applied.json" --settings "$PARAMS" \
    --geak-root "$GEAK_ROOT" <<'BUILD_PY' 2>&1 | tee "$OUT/overlay.log"
import json, os, subprocess, sys
settings, builder, *arguments = sys.argv[1:]
params = json.load(open(settings)) if os.path.isfile(settings) else {}
prefix = params.get("overlay_build_command", ["python3"])
if (not isinstance(prefix, list) or not prefix or not isinstance(prefix[0], str)
        or not prefix[0] or any(not isinstance(arg, str) or "\0" in arg for arg in prefix)):
    print("overlay_build_command must be a nonempty argv list", file=sys.stderr)
    raise SystemExit(78)
try:
    # Prefix arguments stay literal; a remote/container launcher is just another argv.
    process = subprocess.run(prefix + [builder] + arguments)
except OSError as exc:
    print("overlay build command unavailable: %s" % exc, file=sys.stderr)
    raise SystemExit(78)
raise SystemExit(process.returncode if process.returncode >= 0 else 128 - process.returncode)
BUILD_PY
BRC=${PIPESTATUS[0]}
if [ "$BRC" = 78 ]; then
  python3 "$SCRIPTS/emit_result.py" --status blocked \
    --notes "overlay build command is unavailable or reported an unmet precondition (exit 78); see overlay.log"
  exit 78
fi
if [ "$BRC" = 3 ]; then
  python3 "$SCRIPTS/emit_result.py" --status blocked \
    --notes "there is no kernel to apply back (nothing from the optimize stage passed cold
re-verification). Every stage ran; the conclusion is that no usable optimization was found."
  exit 0
fi
[ "$BRC" = 0 ] || { python3 "$SCRIPTS/emit_result.py" --status fail \
    --notes "building the overlay failed (exit $BRC); see overlay.log"; exit "$BRC"; }

# --- 2. Re-run the reference harness with independent source/cache roots per arm.
mkdir -p "$OUT/e2e"
STOP="$(p stop_cmd '')"
run_arm() {
  local ARM_CASE="$1" TAG="$2" DIR="$HARNESS/runs/$2"
  if [ -f "$DIR/$TAG.json" ]; then
    echo "[applyback] $TAG already has a result; validating retained evidence"
    return 0
  fi
  local CACHE="$CACHE_ROOT/$TAG"
  if [ -e "$CACHE" ]; then
    echo "[applyback] $CACHE already exists without a completed arm; preserve it and use a fresh run tag" >&2
    return 78
  fi
  mkdir -p "$CACHE/triton" "$CACHE/xdg" "$CACHE/inductor" "$CACHE/torch_ext" || return 78
  ( cd "$HARNESS" && env TRITON_CACHE_DIR="$CACHE/triton" XDG_CACHE_HOME="$CACHE/xdg" \
      TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
      TORCH_EXTENSIONS_DIR="$CACHE/torch_ext" \
      GEAK_SERVING_SOURCE_PROOF_RUN_DIR="$DIR" \
      bash "$HCMD" "$ARM_CASE" "$TAG" "$DURATION" ) \
      > "$OUT/e2e/${TAG}.log" 2>&1
  local RC=$?
  echo "[applyback] $TAG exited $RC"
  if [ -n "$STOP" ] && [ -f "$DIR/server_pid.json" ]; then
    ( cd "$HARNESS" && python3 "$STOP" "$DIR" ) >> "$OUT/e2e/${TAG}.log" 2>&1 || return 1
  fi
  return "$RC"
}

# The historical baseline remains a drift reference. A fresh unchanged control
# is the A/B denominator after a long optimization campaign.
run_arm "$BASE_CASE" "$STOCK_TAG"
RC=$?
[ "$RC" = 78 ] && blocked "fresh stock control needs a new run tag/cache path; existing evidence was preserved"
[ "$RC" = 0 ] || { python3 "$SCRIPTS/emit_result.py" --status fail --notes "fresh stock control failed; see e2e/${STOCK_TAG}.log"; exit 0; }
STOCK_RUN="$HARNESS/runs/$STOCK_TAG/$STOCK_TAG.json"
python3 "$SKILL_ROOT/summarize.py" --settings "$PARAMS" --validate-run "$HARNESS/runs/$STOCK_TAG" \
    --stock --duration-s "$DURATION" > "$OUT/e2e/${STOCK_TAG}.validation.json" 2>&1 \
    || { python3 "$SCRIPTS/emit_result.py" --status fail --notes "fresh stock control failed native validity/duration gates"; exit 0; }
RUN_DIRS=()
for i in $(seq 1 "$REPEATS"); do
  TAG="${TAG_PREFIX}_$(printf '%02d' "$i")"
  DIR="$HARNESS/runs/$TAG"
  ARM_CASE="${CASE_NAME}_${TAG}"
  ARM_SUBDIR="${OVL_SUB%/}/$TAG"
  RUN_DIRS+=("$DIR")
  if [ ! -f "$DIR/$TAG.json" ]; then
    # Real copies matter: native builders resolve __file__ and may write beside
    # their source. A symlink or one shared overlay would share those outputs.
    python3 "$SKILL_ROOT/register_case.py" --harness "$HARNESS" --case-file "$CASE_FILE" \
        --case "$ARM_CASE" --base-case "$BASE_CASE" --overlay "$OVERLAY" \
        --overlay-subdir "$ARM_SUBDIR" --out "$OUT/e2e/${TAG}.case_registration.json" \
        >> "$OUT/overlay.log" 2>&1 \
        || { python3 "$SCRIPTS/emit_result.py" --status blocked --notes "candidate arm requires a fresh per-arm source path; previous overlay/build evidence was preserved"; exit 0; }
  fi
  echo "[applyback] leg $i/$REPEATS: $HCMD $ARM_CASE $TAG $DURATION"
  run_arm "$ARM_CASE" "$TAG"
  RC=$?
  [ "$RC" = 78 ] && blocked "candidate arm needs a new run tag/cache path; existing evidence was preserved"
  [ "$RC" = 0 ] || { python3 "$SCRIPTS/emit_result.py" --status fail --notes "candidate arm failed; see e2e/${TAG}.log"; exit 0; }
  python3 "$SKILL_ROOT/summarize.py" --settings "$PARAMS" --validate-run "$DIR" \
      --duration-s "$DURATION" --overlay "$OVERLAY" --applied "$OUT/applied.json" \
      > "$OUT/e2e/${TAG}.validation.json" 2>&1 \
      || { python3 "$SCRIPTS/emit_result.py" --status fail --notes "candidate arm lacks native validity or actual post-lazy-load serving-rank source proof; see e2e/${TAG}.validation.json"; exit 0; }
done

# --- 4. Summarize + verify the server really loaded the overlay, then the slot adapter.
python3 "$SKILL_ROOT/summarize.py" --settings "$PARAMS" --overlay "$OVERLAY" --applied "$OUT/applied.json" \
    --reference-run "$STOCK_RUN" --historical-reference "$REF_RUN" \
    --expected-repeats "$REPEATS" --duration-s "$DURATION" \
    --reference-key "$REF_KEY" --out-reference "$OUT/reference.json" \
    --out-summary "$OUT/e2e/bench_summary.json" --runs "${RUN_DIRS[@]}" \
    2>&1 | tee "$OUT/summarize.log"
SRC=${PIPESTATUS[0]}
if [ "$SRC" != 0 ]; then
  python3 "$SCRIPTS/emit_result.py" --status fail \
    --notes "the candidate legs did not produce a usable measurement, or the server did not
actually load the overlay. An end-to-end number is not reported in this state, because a
'no gain' here would be indistinguishable from a patch that was never loaded. See
summarize.log."
  exit 0
fi

python3 "$SLOT/finalize.py" --bench-rc 0 --e2e-dir "$OUT/e2e" --baseline "$OUT/reference.json"
