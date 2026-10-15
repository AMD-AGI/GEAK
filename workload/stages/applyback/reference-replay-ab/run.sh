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

[ -f "$OPTIMIZED" ] || blocked "missing $OPTIMIZED — run the optimize stage first"
for f in "$HARNESS/$HCMD" "$HARNESS/$CASE_FILE" "$REF_RUN"; do
  [ -e "$f" ] || blocked "the reference harness is incomplete on this machine: $f is missing.
Without it the candidate leg cannot be measured the same way as the reference, and a
throughput number measured any other way would not be an A/B."
done

# The serving stack takes the node. Refuse rather than measure against someone else's load.
if curl -fsS --max-time 3 "http://localhost:$(p port 8899)/health" >/dev/null 2>&1; then
  blocked "a server is already listening on this port; applyback needs the node to itself
and the overlay must be loaded by a freshly started server (reusing a running one measures
an unpatched stack)"
fi

# --- 1. Build the overlay, and check it shadows **here**, inside the serving container.
OVERLAY="$OUT/overlay"
python3 "$SLOT/build_overlay.py" --optimized "$OPTIMIZED" --tasks-dir "$TASKS_DIR" \
    --overlay "$OVERLAY" --manifest "$OUT/applied.json" 2>&1 | tee "$OUT/overlay.log"
BRC=${PIPESTATUS[0]}
if [ "$BRC" = 3 ]; then
  python3 "$SCRIPTS/emit_result.py" --status blocked \
    --notes "there is no kernel to apply back (nothing from the optimize stage passed cold
re-verification). Every stage ran; the conclusion is that no usable optimization was found."
  exit 0
fi
[ "$BRC" = 0 ] || { python3 "$SCRIPTS/emit_result.py" --status fail \
    --notes "building the overlay failed; see overlay.log"; exit 0; }

# --- 2. Register it as a case of the reference harness. The launcher requires the overlay
#        to live inside its own root, so the overlay is copied there; $OUT keeps the
#        original, and removing the case entry restores the harness exactly.
python3 "$SKILL_ROOT/register_case.py" --harness "$HARNESS" --case-file "$CASE_FILE" \
    --case "$CASE_NAME" --base-case "$BASE_CASE" --overlay "$OVERLAY" \
    --overlay-subdir "$OVL_SUB" --out "$OUT/case_registration.json" 2>&1 | tee -a "$OUT/overlay.log"
[ "${PIPESTATUS[0]}" = 0 ] || { python3 "$SCRIPTS/emit_result.py" --status fail \
    --notes "could not register the overlay as a case of the reference harness; see overlay.log"; exit 0; }

# --- 3. Re-run the reference harness, once per repeat, unchanged.
mkdir -p "$OUT/e2e"
RUN_DIRS=()
for i in $(seq 1 "$REPEATS"); do
  TAG="${TAG_PREFIX}_$(printf '%02d' "$i")"
  echo "[applyback] leg $i/$REPEATS: $HCMD $CASE_NAME $TAG $DURATION"
  ( cd "$HARNESS" && bash "$HCMD" "$CASE_NAME" "$TAG" "$DURATION" ) \
      > "$OUT/e2e/${TAG}.log" 2>&1
  echo "[applyback] leg $i exited ${PIPESTATUS[0]}"
  RUN_DIRS+=("$HARNESS/runs/$TAG")
done

# --- 4. Summarize + verify the server really loaded the overlay, then the slot adapter.
python3 "$SKILL_ROOT/summarize.py" --overlay "$OVERLAY" --reference-run "$REF_RUN" \
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
