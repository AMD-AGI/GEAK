#!/usr/bin/env bash
# =============================================================================
# run_matrix.sh — L1 orchestrator on the jump box (self-hosted runner).
#
# Runs one model per node, waits for them all, then judges each result.json and
# writes a pass/fail matrix. Exits non-zero if ANY model fails, so a single
# GitHub check turns red.
#
# TWO DISPATCH MODES (GEAK_DISPATCH, default 'held'):
#
#   held   — run on nodes we ALREADY hold. Nothing is queued, so nothing pends.
#            The cluster is saturated by long-lived holds from other users, so a
#            freshly submitted job can pend indefinitely and the matrix never
#            runs. We keep our own holders (~/spur_hold.sh: batch jobs named
#            "$USER-hold" that sleep forever); this mode finds the steady-idle
#            ones and executes on them via overlapping srun steps. Models are
#            run in waves when there are more models than free nodes.
#
#   sbatch — the original submit-and-wait path (ci/dispatch/slurm_submit.sh),
#            kept verbatim as a fallback for when the cluster is drainable.
#            A pristine copy of all three original scripts also lives in
#            ci/dispatch/legacy_sbatch/.
#
# Selection:
#   ci/dispatch/run_matrix.sh smoke            # tier==smoke models (the L1 smoke set)
#   ci/dispatch/run_matrix.sh verify           # ALL enrolled models (the full L1 matrix)
#   ci/dispatch/run_matrix.sh probe            # models with local weights, PROBE mode (fast)
#   ci/dispatch/run_matrix.sh <model> [<model> ...]   # explicit list
#
# Flags:
#   --budget SECONDS   per-model GEAK wall-clock budget (default: PERFSKILLS_E2E_TIMEOUT_S from ci/config.sh)
#   --poll SECONDS     poll interval while waiting (default: GEAK_MATRIX_POLL_S from ci/config.sh)
#   --dispatch MODE    held (default) | sbatch
#   --holders "J1 J2"  held mode: use exactly these holder job ids (skips idle discovery)
#   --probe            harness check: real alloc + docker + GPU + weights
#                      (+Claude), but STOP at the GEAK e2e doorstep. Judges on a
#                      probe_ok marker (no e2e/result.json). Implied by 'probe'.
#   --print            show what WOULD run and exit (no execution)
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/../lib.sh"

SEL="${1:?usage: run_matrix.sh <smoke|verify|probe|MODEL...> [--budget N] [--poll N] [--dispatch held|sbatch] [--probe] [--print]}"; shift || true
BUDGET="$PERFSKILLS_E2E_TIMEOUT_S"   # defaults live in ci/config.sh
POLL="$GEAK_MATRIX_POLL_S"
LOG_S="$GEAK_MATRIX_LOG_S"
PRINT="$SPUR_DRYRUN"
PROBE=0
DISPATCH="${GEAK_DISPATCH:-held}"
HOLDERS_OVERRIDE=""
MODELS=()
case "$SEL" in
  smoke)  mapfile -t MODELS < <(smoke_models) ;;
  verify) mapfile -t MODELS < <(enrolled_models) ;;
  probe)  mapfile -t MODELS < <(probe_models); PROBE=1 ;;   # local-weight models, infra-only
  *)      MODELS=("$SEL") ;;   # first was an explicit model; more may follow
esac
while [ $# -gt 0 ]; do
  case "$1" in
    --budget)   BUDGET="${2:?}"; shift ;;
    --poll)     POLL="${2:?}"; shift ;;
    --dispatch) DISPATCH="${2:?}"; shift ;;
    --holders)  HOLDERS_OVERRIDE="${2:?}"; shift ;;
    --probe)    PROBE=1 ;;
    --print)    PRINT=1 ;;
    -*)         die "unknown flag: $1" ;;
    *)          MODELS+=("$1") ;;
  esac; shift
done
case "$DISPATCH" in held|sbatch) ;; *) die "unknown --dispatch '$DISPATCH' (want: held|sbatch)" ;; esac

# Extra args shared by --print and real execution.
SUB_EXTRA=()
[ "$PROBE" = "1" ] && SUB_EXTRA+=(--probe)

[ "${#MODELS[@]}" -gt 0 ] || die "no models selected for '$SEL' (check $MODELS_TSV)"
RUN_TS="${RUN_TS:-$(new_ts)}"
log "matrix '$SEL' ts=$RUN_TS dispatch=$DISPATCH budget=${BUDGET}s${PROBE:+ probe=$PROBE} models: ${MODELS[*]}"
[ "$PROBE" = "1" ] && log "PROBE mode: infra-only (real alloc/docker/GPU/weights, stops at GEAK entry)"

# Results of whichever dispatch runs: parallel arrays consumed by the judging
# section at the bottom. J_ID is a job id (sbatch) or "<holder>:<node>" (held).
declare -a J_MODEL J_ID J_OUT
DONE=0

# =============================================================================
# HELD DISPATCH — run on nodes we already own
# =============================================================================

# How long to wait for a node to free up when none is idle, and how often to
# re-check. Unlike a queued job this never blocks another user, so it is safe to
# keep looking indefinitely; the GitHub job's timeout-minutes is the outer bound.
HELD_WAIT_POLL_S="${HELD_WAIT_POLL_S:-120}"
# A model whose node died mid-run (holds do get cancelled) is retried this many
# times on a fresh node, provided it never produced a result.
HELD_RETRIES="${HELD_RETRIES:-1}"

declare -a HELD_PIDS=()

# Kill the srun clients we launched. Killing the client terminates the remote
# step (that is how SLURM steps work), which is what we want on cancellation.
# It does NOT touch the holder jobs themselves — those are long-lived nodes the
# user is deliberately sitting on and must survive any CI outcome.
# The client chain is bash -> held_submit -> timeout -> script -> srun, so
# signalling only the top pid would orphan the srun client (and leave the remote
# step running). Walk the tree depth-first instead.
_kill_tree() {
  local p="$1" c
  for c in $(pgrep -P "$p" 2>/dev/null); do _kill_tree "$c"; done
  kill -TERM "$p" 2>/dev/null || true
}

_cancel_held() {
  [ "$DONE" = "1" ] && return 0
  local p n=0
  for p in "${HELD_PIDS[@]:-}"; do
    [ -n "$p" ] || continue
    _kill_tree "$p"; n=$((n + 1))
  done
  [ "$n" -gt 0 ] && log "cleanup: terminated $n held run(s); holder allocations left untouched"
  return 0
}

dispatch_held() {
  # shellcheck source=/dev/null
  source "$HERE/held_nodes.sh"

  local state; state="$(mktemp -d "${TMPDIR:-/tmp}/geak_matrix_${RUN_TS}_XXXX")"
  trap _cancel_held INT TERM EXIT

  local -a pending=("${MODELS[@]}")
  local -A tries=()
  local wave=0

  while [ "${#pending[@]}" -gt 0 ]; do
    # Fresh discovery every wave: holds come and go, and a node that was busy
    # when we started may be free now (or vice versa).
    local -a pool=()
    if [ -n "$HOLDERS_OVERRIDE" ]; then
      local h
      for h in $HOLDERS_OVERRIDE; do
        local n; n="$(held_running | awk -v j="$h" '$1 == j {print $2}')"
        [ -n "$n" ] || die "holder $h is not RUNNING"
        pool+=("$h $n")
      done
    else
      mapfile -t pool < <(held_idle_nodes)
    fi

    if [ "${#pool[@]}" -eq 0 ]; then
      log "no steady-idle held node available for ${#pending[@]} remaining model(s): ${pending[*]}"
      log "  (holders in use by other work are skipped; re-checking in ${HELD_WAIT_POLL_S}s — grab more with ~/spur_hold.sh start)"
      sleep "$HELD_WAIT_POLL_S"
      continue
    fi

    wave=$((wave + 1))
    local n="${#pool[@]}"; [ "$n" -gt "${#pending[@]}" ] && n="${#pending[@]}"
    log "wave $wave: ${#pool[@]} idle node(s), running $n of ${#pending[@]} remaining model(s)"

    local -a running_models=() running_holders=()
    HELD_PIDS=()
    local i
    for (( i = 0; i < n; i++ )); do
      local m="${pending[$i]}" holder node safe
      holder="${pool[$i]%% *}"; node="${pool[$i]##* }"
      safe="${m//[^A-Za-z0-9_.-]/_}"
      log "  assign $m -> holder $holder ($node)"
      # stdout (the "<holder>:<node>\ttab<out_dir>" handle) is captured; stderr is
      # inherited so each model's progress shows up live in the matrix log.
      bash -c 'h="$1"; r="$2"; shift 2; bash "$@" > "$h"; echo $? > "$r"' \
        _ "$state/$safe.handle" "$state/$safe.rc" \
        "$HERE/held_submit.sh" "$m" --jobid "$holder" --budget "$BUDGET" \
        --run-ts "$RUN_TS" "${SUB_EXTRA[@]}" &
      HELD_PIDS+=("$!")
      running_models+=("$m"); running_holders+=("$holder:$node")
    done

    # Wait for the wave. Each client blocks for its whole run, so this is simply
    # a join; progress is visible in each model's ci_runs/<ts>/held.out.
    log "wave $wave: waiting for ${#running_models[@]} run(s) — live logs under $HF_LOGS/<model>/ci_runs/$RUN_TS/held.out"
    local pid; for pid in "${HELD_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    HELD_PIDS=()

    # Collect this wave's outcomes.
    local -a still_pending=()
    for (( i = 0; i < n; i++ )); do
      local m="${running_models[$i]}" safe rc out
      safe="${m//[^A-Za-z0-9_.-]/_}"
      rc="$(cat "$state/$safe.rc" 2>/dev/null || echo 1)"
      out="$HF_LOGS/$m/ci_runs/$RUN_TS"
      # Retry only when the run left nothing behind AND its node is gone — i.e.
      # the hold was cancelled under us, not a genuine model failure.
      local holder="${running_holders[$i]%%:*}" alive
      alive="$(held_running | awk -v j="$holder" '$1 == j {print $1}')"
      if [ "$rc" != "0" ] && [ ! -f "$out/result.json" ] && [ ! -f "$out/probe_ok" ] \
         && [ -z "$alive" ] && [ "${tries[$m]:-0}" -lt "$HELD_RETRIES" ]; then
        tries[$m]=$(( ${tries[$m]:-0} + 1 ))
        log "  $m: holder $holder vanished mid-run and no result was written -> retry ${tries[$m]}/$HELD_RETRIES on a fresh node"
        still_pending+=("$m")
        continue
      fi
      J_MODEL+=("$m"); J_ID+=("${running_holders[$i]}"); J_OUT+=("$out")
      log "  $m: rc=$rc on ${running_holders[$i]}"
    done

    # Anything not started this wave stays queued, plus this wave's retries.
    local -a next=("${still_pending[@]:-}")
    for (( i = n; i < ${#pending[@]}; i++ )); do next+=("${pending[$i]}"); done
    pending=()
    local x; for x in "${next[@]}"; do [ -n "$x" ] && pending+=("$x"); done
  done

  rm -rf "$state"
  DONE=1
}

# =============================================================================
# SBATCH DISPATCH — the original submit-and-wait path (unchanged)
# =============================================================================

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

dispatch_sbatch() {
  # Optional file the workflow reads as a backstop (see ci-l1-smoke-e2e.yml).
  local JOBS_FILE="${SPUR_JOBS_FILE:-}"
  [ -n "$JOBS_FILE" ] && : > "$JOBS_FILE"
  trap _cancel_submitted INT TERM EXIT

  local m line jid out
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

  # ---- wait for ALL submitted jobs to leave the queue ----
  # Policy: NEVER give up on a job for pending too long. As long as ANY job is still
  # PENDING or RUNNING, keep waiting. A job that crashed or was cancelled (by us on a
  # GitHub-cancel, or by an operator on the cluster) simply leaves the queue; we log
  # that once and keep waiting on the rest — the real pass/fail is decided from each
  # result.json in the judging step below. The ONLY intervention here is a one-time
  # scontrol-release of a *held* requeue (SPUR's JobHoldMaxRequeue on transient
  # congestion); we never scancel on our own. If a job pends for a very long time
  # (>~36h), cancel it by hand on the cluster — it will then leave the queue and the
  # matrix proceeds. The GitHub job's timeout-minutes is the only outer backstop.
  local me; me="$(whoami)"
  declare -A RELEASED LEFT GONE_STREAK
  # A job must be confirmed absent (squeue miss AND scontrol non-active) this many
  # CONSECUTIVE polls before it is declared gone — belt-and-suspenders against a flaky
  # SLURM control plane (stale sacct / dropped squeue rows) falsely failing a live run.
  local GONE_CONFIRM="${GEAK_MATRIX_GONE_CONFIRM:-3}"

  # Best-effort terminal state for a job that has left squeue (COMPLETED/FAILED/
  # CANCELLED/TIMEOUT/...). Empty if sacct is unavailable on this cluster.
  sacct_state() {
    command -v sacct >/dev/null 2>&1 || { echo ""; return; }
    sacct -j "$1" -n -X -o State%25 2>/dev/null | head -1 | awk '{$1=$1};1'
  }

  # Confirm a job's live state via scontrol when it is absent from a squeue snapshot.
  # A JobHoldMaxRequeue job briefly leaves squeue during its requeue->hold transition,
  # so absence alone is NOT proof of a terminal state. Echo 'STATE|REASON' and return 0
  # if the job is still active/held OR if SLURM itself couldn't be reached (conservative:
  # keep waiting); return 1 ONLY on positive proof the job is gone — a terminal JobState
  # or an explicit "Invalid job id" (purged). This distinction matters: an empty scontrol
  # reply from a transient controller-unreachable/timeout MUST NOT be read as "terminal",
  # or a single control-plane blip (which also drops the job from squeue) falsely declares
  # the run failed while it is in fact still RUNNING. Proven on job 50633 (2026-07-27):
  # squeue+scontrol flickered together -> false no_result FAIL on a live job.
  scontrol_active() {
    local out rc st rs
    out="$(scontrol show job "$1" 2>&1)"; rc=$?
    # Genuinely purged: SLURM explicitly says the id is invalid -> gone.
    grep -qiE 'Invalid job id' <<<"$out" && return 1
    # Any other failure/empty reply = controller unreachable/timeout, NOT proof of death.
    if [ "$rc" -ne 0 ] || [ -z "$out" ]; then echo "UNKNOWN|controller_unreachable"; return 0; fi
    st="$(sed -n 's/.*JobState=\([A-Z_]*\).*/\1/p' <<<"$out" | head -1)"
    rs="$(sed -n 's/.*Reason=\([^ ]*\).*/\1/p'      <<<"$out" | head -1)"
    case "$st" in
      PENDING|RUNNING|SUSPENDED|COMPLETING|CONFIGURING|REQUEUED|RESIZING|SIGNALING|STAGE_OUT)
        echo "${st}|${rs}"; return 0 ;;
      *) return 1 ;;
    esac
  }

  poll_jobs() {  # return 0 while ANY of our jobs is still in the queue
    local snap; snap="$(squeue -u "$me" -h -o '%i|%T|%r' 2>/dev/null || true)"
    local any=1 i jid m line state reason fst sc
    local -a status=()
    for i in "${!J_ID[@]}"; do
      jid="${J_ID[$i]}"; m="${J_MODEL[$i]}"
      if [ -z "$jid" ]; then status+=("$m=submit_failed"); continue; fi
      line="$(grep -E "^${jid}\|" <<<"$snap" || true)"
      if [ -z "$line" ]; then
        # Absent from THIS squeue snapshot. That can be a genuine terminal state OR a
        # transient gap (a JobHoldMaxRequeue job briefly leaves squeue during its
        # requeue->hold transition). Confirm with scontrol before concluding it is
        # gone, so a flicker never triggers a false 'terminal' verdict + zombie hold.
        if sc="$(scontrol_active "$jid")"; then
          # still alive/held (or SLURM unreachable) per scontrol -> keep waiting.
          GONE_STREAK[$jid]=0
          state="${sc%%|*}"; reason="${sc#*|}"
        else
          # Positive evidence of terminal/purged this poll. Require GONE_CONFIRM
          # CONSECUTIVE such polls before finalizing, so a transient control-plane flap
          # (which drops the job from BOTH squeue and scontrol) can never fail a live run.
          GONE_STREAK[$jid]=$(( ${GONE_STREAK[$jid]:-0} + 1 ))
          if [ "${GONE_STREAK[$jid]}" -lt "$GONE_CONFIRM" ]; then
            status+=("$m=absent?${GONE_STREAK[$jid]}/${GONE_CONFIRM}($jid)")
            any=0; continue    # not yet confirmed -> keep the matrix waiting
          fi
          # Genuinely terminal (finished / crashed / cancelled / purged). Log ONCE and
          # keep waiting on the others; do NOT stop the matrix on a single job leaving.
          if [ -z "${LEFT[$jid]:-}" ]; then
            LEFT[$jid]=1; fst="$(sacct_state "$jid")"
            log "job $jid ($m) left the queue${fst:+ (state=$fst)} after ${GONE_CONFIRM} confirmations — still waiting on any pending/running jobs; judged from result.json later"
          fi
          status+=("$m=gone($jid)")
          continue
        fi
      else
        GONE_STREAK[$jid]=0
        state="$(cut -d'|' -f2 <<<"$line")"; reason="$(cut -d'|' -f3 <<<"$line")"
      fi
      any=0
      if [ "$state" = PENDING ]; then
        # Auto-release a *held* requeue once; never cancel a plain pending job.
        if [[ "$reason" == *Hold* || "$reason" == *held* ]] && [ -z "${RELEASED[$jid]:-}" ] \
           && command -v scontrol >/dev/null 2>&1; then
          log "job $jid ($m) held ($reason) -> scontrol release (once)"
          scontrol release "$jid" 2>/dev/null || true
          RELEASED[$jid]=1
        fi
        status+=("$m=PENDING:${reason// /_}($jid)")
      else
        status+=("$m=${state}($jid)")
      fi
    done
    # Decouple log cadence from poll cadence: emit the status line only every
    # GEAK_MATRIX_LOG_S, but ALWAYS emit immediately when the state string changes
    # (so transitions are never missed) or on the first poll.
    local now cur="${status[*]}"
    now="$(date +%s)"
    if [ "$cur" != "${LAST_STATUS:-}" ] || [ $(( now - ${LAST_LOG_TS:-0} )) -ge "$LOG_S" ]; then
      log "queue: $cur"
      LAST_STATUS="$cur"; LAST_LOG_TS="$now"
    fi
    return $any
  }
  log "waiting for ${#J_ID[@]} job(s) to finish (poll ${POLL}s, status log every ${LOG_S}s or on state change; PENDING jobs are waited on indefinitely — cancel by hand on the cluster if one pends too long) ..."
  while poll_jobs; do sleep "$POLL"; done

  # Reconcile before we consider the matrix done: a job can re-hold (JobHoldMaxRequeue)
  # in the brief window between our last poll and here, re-appearing in the queue after
  # we judged it gone. scancel any of OUR submitted ids still present so we never orphan
  # a held leftover on the shared cluster. Strictly scoped to ids we launched; a no-op
  # on the normal path (nothing of ours is left in the queue).
  if command -v scancel >/dev/null 2>&1; then
    for jid in "${J_ID[@]}"; do
      [ -n "$jid" ] || continue
      if squeue -h -j "$jid" -o '%i' 2>/dev/null | grep -q .; then
        log "reap: job $jid still in queue after wait loop -> scancel (held/requeued leftover)"
        scancel "$jid" 2>/dev/null || true
      fi
    done
  fi

  DONE=1   # jobs are terminal now; the cleanup trap becomes a no-op
  log "no pending or running jobs left; judging results"
}

# =============================================================================
# --print: show what WOULD run, then stop
# =============================================================================
if [ "$PRINT" = "1" ]; then
  if [ "$DISPATCH" = "sbatch" ]; then
    for m in "${MODELS[@]}"; do SPUR_DRYRUN=1 bash "$HERE/slurm_submit.sh" "$m" --budget "$BUDGET" --run-ts "$RUN_TS" "${SUB_EXTRA[@]}" --print; done
  else
    # shellcheck source=/dev/null
    source "$HERE/held_nodes.sh"
    mapfile -t _pool < <(if [ -n "$HOLDERS_OVERRIDE" ]; then for h in $HOLDERS_OVERRIDE; do held_running | awk -v j="$h" '$1 == j {print $1, $2}'; done; else held_idle_nodes; fi)
    log "held pool: ${#_pool[@]} idle node(s) for ${#MODELS[@]} model(s)"
    i=0
    for m in "${MODELS[@]}"; do
      if [ "$i" -lt "${#_pool[@]}" ]; then
        SPUR_DRYRUN=1 bash "$HERE/held_submit.sh" "$m" --jobid "${_pool[$i]%% *}" --budget "$BUDGET" --run-ts "$RUN_TS" "${SUB_EXTRA[@]}" --print
      else
        log "$m: would wait for a free held node (wave ${i})"
      fi
      i=$((i + 1))
    done
  fi
  log "print-only: nothing executed"
  exit 0
fi

# =============================================================================
# dispatch
# =============================================================================
case "$DISPATCH" in
  held)   dispatch_held ;;
  sbatch) dispatch_sbatch ;;
esac

# ---- judge each result.json (judge_result from lib.sh; same criteria as run_model Step F) ----
FAILS=0
rows=""
SCAN_RECORDS=""   # <model>\t<verdict>\t<status>\t<out_dir> per model, for scan_run.sh
for i in "${!J_MODEL[@]}"; do
  m="${J_MODEL[$i]}"; jid="${J_ID[$i]:-?}"; out="${J_OUT[$i]}"; tp="$(model_tp "$m")"
  if [ -z "$out" ]; then
    v="FAIL"; st="submit_failed"; b=""; f=""; sp=""
  elif [ "$PROBE" = "1" ]; then
    # Probe: no e2e/result.json — success is the probe_ok marker from run_local --probe.
    if [ -f "$out/probe_ok" ]; then v="PASS"; st="probe_ok"; else v="FAIL"; st="probe_incomplete"; fi
    b=""; f=""; sp=""
  else
    IFS=$'\t' read -r v st b f sp < <(judge_result "$out")
  fi
  [ "$v" = "PASS" ] || FAILS=$((FAILS+1))
  rows+="| \`$m\` | $tp | $jid | $v | ${st:-} | ${b:-} | ${f:-} | ${sp:-} |"$'\n'
  SCAN_RECORDS+="$m"$'\x1f'"$v"$'\x1f'"${st:-}"$'\x1f'"$out"$'\n'
  log "$m: $v (status=${st:-} baseline=${b:-} job=$jid)"
done

TITLE="L1 matrix — $SEL"; [ "$PROBE" = "1" ] && TITLE="L1 PROBE (infra-only) — $SEL"
TABLE="## $TITLE (ts \`$RUN_TS\`, dispatch $DISPATCH, budget ${BUDGET}s)

| model | tp | node | verdict | status | baseline tok/s | final tok/s | speedup |
|---|--:|--:|:--:|---|--:|--:|--:|
$rows"
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
  die "$FAILS/${#J_MODEL[@]} model(s) failed" 1
fi
log "all ${#J_MODEL[@]} model(s) passed"
exit 0
