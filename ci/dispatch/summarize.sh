#!/usr/bin/env bash
# =============================================================================
# summarize.sh — final aggregation for the fan-out L1 matrix.
#
# The per-model GitHub jobs (strategy.matrix) each ran run_matrix.sh <model> for
# a SHARED RUN_TS, so every model's result already lives at
#   $HF_LOGS/<model>/ci_runs/<RUN_TS>/result.json
# This job runs AFTER all of them (needs: verify, if: always()) and re-judges each
# result into ONE final table + overall verdict. It does NOT submit or wait on any
# SPUR job — it only reads what the matrix produced.
#
# Usage:
#   ci/dispatch/summarize.sh <run_ts> <smoke|verify|probe|MODEL...> [--probe]
#
# Exit: non-zero if ANY model FAILs (so the summary check reflects the overall
# result), 0 if all pass. Missing result.json for a model counts as FAIL.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/../lib.sh"

RUN_TS="${1:?usage: summarize.sh <run_ts> <smoke|verify|probe|MODEL...> [--probe]}"; shift || true
SEL="${1:?need a selector (smoke|verify|probe) or explicit model list}"; shift || true
PROBE=0
MODELS=()
case "$SEL" in
  smoke)  mapfile -t MODELS < <(smoke_models) ;;
  verify) mapfile -t MODELS < <(enrolled_models) ;;
  probe)  mapfile -t MODELS < <(probe_models); PROBE=1 ;;
  *)      MODELS=("$SEL") ;;
esac
while [ $# -gt 0 ]; do
  case "$1" in
    --probe) PROBE=1 ;;
    -*)      die "unknown flag: $1" ;;
    *)       MODELS+=("$1") ;;
  esac; shift
done
[ "${#MODELS[@]}" -gt 0 ] || die "no models selected for '$SEL' (check $MODELS_TSV)"

log "summarize '$SEL' ts=$RUN_TS${PROBE:+ probe=$PROBE} models: ${MODELS[*]}"

FAILS=0
rows=""
SCAN_RECORDS=""   # <model>\x1f<verdict>\x1f<status>\x1f<out_dir> per model, for scan_run.sh
for m in "${MODELS[@]}"; do
  out="$HF_LOGS/$m/ci_runs/$RUN_TS"; tp="$(model_tp "$m")"
  if [ "$PROBE" = "1" ]; then
    if [ -f "$out/probe_ok" ]; then v="PASS"; st="probe_ok"; else v="FAIL"; st="probe_incomplete"; fi
    b=""; f=""; sp=""
  else
    IFS=$'\t' read -r v st b f sp < <(judge_result "$out")
  fi
  [ "$v" = "PASS" ] || FAILS=$((FAILS+1))
  rows+="| \`$m\` | $tp | $v | ${st:-} | ${b:-} | ${f:-} | ${sp:-} |"$'\n'
  SCAN_RECORDS+="$m"$'\x1f'"$v"$'\x1f'"${st:-}"$'\x1f'"$out"$'\n'
  log "$m: $v (status=${st:-} baseline=${b:-} out=$out)"
done

TITLE="L1 FINAL — $SEL"; [ "$PROBE" = "1" ] && TITLE="L1 FINAL PROBE (infra-only) — $SEL"
TABLE="## $TITLE (ts \`$RUN_TS\`)

| model | tp | verdict | status | baseline tok/s | final tok/s | speedup |
|---|--:|:--:|---|--:|--:|--:|
$rows
**Result: $(( ${#MODELS[@]} - FAILS ))/${#MODELS[@]} passed.**"
printf '%s\n' "$TABLE" >&2
[ -n "${GITHUB_STEP_SUMMARY:-}" ] && printf '%s\n' "$TABLE" >> "$GITHUB_STEP_SUMMARY"

# ---- post-run diagnostics: blockers vs benign warnings + where to look ----
# Advisory only (never changes pass/fail). Skipped for probe runs (no e2e logs).
if [ "$PROBE" != "1" ]; then
  DIAG="$(printf '%s' "$SCAN_RECORDS" | bash "$HERE/../monitor/scan_run.sh" 2>/dev/null || true)"
  if [ -n "$DIAG" ]; then
    printf '\n%s\n' "$DIAG" >&2
    [ -n "${GITHUB_STEP_SUMMARY:-}" ] && printf '\n%s\n' "$DIAG" >> "$GITHUB_STEP_SUMMARY"
  fi
fi

if [ "$FAILS" -gt 0 ]; then
  die "$FAILS/${#MODELS[@]} model(s) failed" 1
fi
log "all ${#MODELS[@]} model(s) passed"
exit 0
