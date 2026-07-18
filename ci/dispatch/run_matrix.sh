#!/usr/bin/env bash
# =============================================================================
# run_matrix.sh — L1 orchestrator on the jump box (self-hosted runner).
#
# Submits one SPUR job per selected model (ci/slurm_submit.sh), waits for them
# all, then judges each result.json and writes a pass/fail matrix. Exits
# non-zero if ANY model fails, so a single GitHub check turns red.
#
# Selection:
#   ci/run_matrix.sh smoke            # tier==smoke models (the L1 smoke set)
#   ci/run_matrix.sh verify           # ALL enrolled models (the full L1 matrix)
#   ci/run_matrix.sh probe            # models with local weights, PROBE mode (fast)
#   ci/run_matrix.sh <model> [<model> ...]   # explicit list
#
# Flags:
#   --budget SECONDS   per-model GEAK wall-clock budget (default 1800)
#   --poll SECONDS     poll interval while waiting (default 60)
#   --probe            harness check: real SPUR alloc + docker + GPU + weights
#                      (+Claude), but STOP at the GEAK e2e doorstep. Judges on a
#                      probe_ok marker (no e2e/result.json). Implied by 'probe'.
#   --print            show the sbatch commands and exit (no submission/wait)
#                      — use this to validate wiring without touching the cluster.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/../lib.sh"

SEL="${1:?usage: run_matrix.sh <smoke|verify|probe|MODEL...> [--budget N] [--poll N] [--probe] [--print]}"; shift || true
BUDGET="${PERFSKILLS_E2E_TIMEOUT_S:-1800}"
POLL="${GEAK_MATRIX_POLL_S:-60}"
PRINT="${SPUR_DRYRUN:-0}"
PROBE=0
MODELS=()
case "$SEL" in
  smoke)  mapfile -t MODELS < <(smoke_models) ;;
  verify) mapfile -t MODELS < <(enrolled_models) ;;
  probe)  mapfile -t MODELS < <(probe_models); PROBE=1 ;;   # local-weight models, infra-only
  *)      MODELS=("$SEL") ;;   # first was an explicit model; more may follow
esac
while [ $# -gt 0 ]; do
  case "$1" in
    --budget) BUDGET="${2:?}"; shift ;;
    --poll)   POLL="${2:?}"; shift ;;
    --probe)  PROBE=1 ;;
    --print)  PRINT=1 ;;
    -*)       die "unknown flag: $1" ;;
    *)        MODELS+=("$1") ;;
  esac; shift
done

# Extra sbatch/matrix args shared by --print and real submission.
SUB_EXTRA=()
[ "$PROBE" = "1" ] && SUB_EXTRA+=(--probe)

[ "${#MODELS[@]}" -gt 0 ] || die "no models selected for '$SEL' (check $MODELS_TSV)"
RUN_TS="${RUN_TS:-$(new_ts)}"
log "matrix '$SEL' ts=$RUN_TS budget=${BUDGET}s${PROBE:+ probe=$PROBE} models: ${MODELS[*]}"
[ "$PROBE" = "1" ] && log "PROBE mode: infra-only (real SPUR/docker/GPU/weights, stops at GEAK entry)"

# ---- --print: just show what WOULD be submitted, then stop ----
if [ "$PRINT" = "1" ]; then
  for m in "${MODELS[@]}"; do SPUR_DRYRUN=1 bash "$HERE/slurm_submit.sh" "$m" --budget "$BUDGET" --run-ts "$RUN_TS" "${SUB_EXTRA[@]}" --print; done
  log "print-only: no jobs submitted"
  exit 0
fi

# ---- submit all (tracking ids so we can scancel on cancel/timeout) ----
declare -a J_MODEL J_ID J_OUT
DONE=0
# Optional file the workflow reads as a backstop (see ci-l1-smoke-e2e.yml).
JOBS_FILE="${SPUR_JOBS_FILE:-}"
[ -n "$JOBS_FILE" ] && : > "$JOBS_FILE"

# If this orchestrator is cancelled/killed before the jobs finish (e.g. the
# GitHub job hits timeout-minutes and sends SIGTERM), scancel what we launched
# instead of orphaning GPU jobs on the shared cluster. DONE=1 (set once all
# jobs have left the queue) makes this a no-op on the normal path.
_cancel_submitted() {
  [ "$DONE" = "1" ] && return 0
  local ids=() i
  for i in "${!J_ID[@]}"; do [ -n "${J_ID[$i]:-}" ] && ids+=("${J_ID[$i]}"); done
  [ "${#ids[@]}" -gt 0 ] || return 0
  command -v scancel >/dev/null 2>&1 || return 0
  log "cleanup: scancel ${ids[*]}"
  scancel "${ids[@]}" 2>/dev/null || true
}
trap _cancel_submitted INT TERM EXIT

for m in "${MODELS[@]}"; do
  if line="$(bash "$HERE/slurm_submit.sh" "$m" --budget "$BUDGET" --run-ts "$RUN_TS" "${SUB_EXTRA[@]}")"; then
    jid="${line%%$'\t'*}"; out="${line#*$'\t'}"
    J_MODEL+=("$m"); J_ID+=("$jid"); J_OUT+=("$out")
    [ -n "$JOBS_FILE" ] && echo "$jid" >> "$JOBS_FILE"
    log "submitted $m -> job $jid"
  else
    J_MODEL+=("$m"); J_ID+=(""); J_OUT+=("")
    log "SUBMIT FAILED for $m"
  fi
done

# ---- wait for all submitted jobs to leave the queue ----
# Robust against SPUR's JobHoldMaxRequeue: when the reserved (e.g. amd-primus) pool
# is momentarily full, a job can be requeued and *held* (PENDING, never placed),
# which would otherwise hang this wait forever. So we: (1) auto-release a held job
# ONCE (transient congestion usually clears), and (2) give up on a job stuck
# PENDING (never RUNNING) past SPUR_PEND_TIMEOUT_S — scancel it so it leaves the
# queue and is judged FAIL, instead of blocking the whole matrix.
me="$(whoami)"
PEND_TIMEOUT="${SPUR_PEND_TIMEOUT_S:-1800}"
declare -A RELEASED PEND_SINCE
poll_jobs() {  # return 0 while any of our jobs is still in the queue
  local snap; snap="$(squeue -u "$me" -h -o '%i|%T|%r' 2>/dev/null || true)"
  local any=1 i jid line state reason now; now="$(date +%s)"
  for i in "${!J_ID[@]}"; do
    jid="${J_ID[$i]}"; [ -n "$jid" ] || continue
    line="$(grep -E "^${jid}\|" <<<"$snap" || true)"
    if [ -z "$line" ]; then unset 'PEND_SINCE[$jid]' 2>/dev/null || true; continue; fi
    any=0
    state="$(cut -d'|' -f2 <<<"$line")"; reason="$(cut -d'|' -f3 <<<"$line")"
    case "$state" in
      PENDING)
        if [[ "$reason" == *Hold* || "$reason" == *held* ]] && [ -z "${RELEASED[$jid]:-}" ] \
           && command -v scontrol >/dev/null 2>&1; then
          log "job $jid held ($reason) -> scontrol release (once)"
          scontrol release "$jid" 2>/dev/null || true
          RELEASED[$jid]=1
        fi
        : "${PEND_SINCE[$jid]:=$now}"
        if [ "$PEND_TIMEOUT" -gt 0 ] && [ $(( now - ${PEND_SINCE[$jid]} )) -ge "$PEND_TIMEOUT" ]; then
          log "job $jid stuck PENDING ($reason) > ${PEND_TIMEOUT}s -> scancel (judged FAIL)"
          scancel "$jid" 2>/dev/null || true
        fi
        ;;
      *) unset 'PEND_SINCE[$jid]' 2>/dev/null || true ;;   # RUNNING/COMPLETING/etc.
    esac
  done
  return $any
}
log "waiting for ${#J_ID[@]} job(s) (poll ${POLL}s, pend-timeout ${PEND_TIMEOUT}s) ..."
while poll_jobs; do sleep "$POLL"; done
DONE=1   # jobs are terminal now; the cleanup trap becomes a no-op
log "all jobs left the queue; judging results"

# ---- judge each result.json (same criteria as run_model Step F) ----
judge() {  # judge <out_dir> -> prints "VERDICT<TAB>status<TAB>baseline<TAB>final<TAB>speedup"
  python3 - "$1" <<'PY'
import json, os, sys
out = sys.argv[1]
p = os.path.join(out, "result.json")
def emit(v, s="", b="", f="", sp=""): print(f"{v}\t{s}\t{b}\t{f}\t{sp}")
if not os.path.isfile(p):
    emit("FAIL", "no_result"); raise SystemExit
try:
    d = json.load(open(p))
except Exception:
    emit("FAIL", "bad_result"); raise SystemExit
st = d.get("status", "")
b  = d.get("baseline_throughput_tok_s") or 0
f  = d.get("final_throughput_tok_s") or ""
sp = d.get("throughput_speedup") or ""
try: ok_base = float(b) > 0
except Exception: ok_base = False
verdict = "PASS" if (st in ("ok", "no_gain") and ok_base) else "FAIL"
emit(verdict, st, b, f, sp)
PY
}

FAILS=0
rows=""
for i in "${!J_MODEL[@]}"; do
  m="${J_MODEL[$i]}"; jid="${J_ID[$i]:-?}"; out="${J_OUT[$i]}"; tp="$(model_tp "$m")"
  if [ -z "$out" ]; then
    v="FAIL"; st="submit_failed"; b=""; f=""; sp=""
  elif [ "$PROBE" = "1" ]; then
    # Probe: no e2e/result.json — success is the probe_ok marker from run_local --probe.
    if [ -f "$out/probe_ok" ]; then v="PASS"; st="probe_ok"; else v="FAIL"; st="probe_incomplete"; fi
    b=""; f=""; sp=""
  else
    IFS=$'\t' read -r v st b f sp < <(judge "$out")
  fi
  [ "$v" = "PASS" ] || FAILS=$((FAILS+1))
  rows+="| \`$m\` | $tp | $jid | $v | ${st:-} | ${b:-} | ${f:-} | ${sp:-} |"$'\n'
  log "$m: $v (status=${st:-} baseline=${b:-} job=$jid)"
done

TITLE="L1 matrix — $SEL"; [ "$PROBE" = "1" ] && TITLE="L1 PROBE (infra-only) — $SEL"
TABLE="## $TITLE (ts \`$RUN_TS\`, budget ${BUDGET}s)

| model | tp | job | verdict | status | baseline tok/s | final tok/s | speedup |
|---|--:|--:|:--:|---|--:|--:|--:|
$rows"
printf '%s\n' "$TABLE" >&2
[ -n "${GITHUB_STEP_SUMMARY:-}" ] && printf '%s\n' "$TABLE" >> "$GITHUB_STEP_SUMMARY"

if [ "$FAILS" -gt 0 ]; then
  die "$FAILS/${#J_MODEL[@]} model(s) failed" 1
fi
log "all ${#J_MODEL[@]} model(s) passed"
exit 0
