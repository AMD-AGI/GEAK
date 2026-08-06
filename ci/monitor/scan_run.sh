#!/usr/bin/env bash
# =============================================================================
# scan_run.sh — post-run diagnostics summary for the L1 matrix.
#
# NOT the liveness monitor (that's run_monitor.sh, a mid-run watchdog that runs on
# the dispatched GPU host, ON by default in stall mode). This is a lightweight,
# dependency-free POST-mortem: after all jobs finish, run_matrix.sh pipes one
# line per model here and we scan that model's on-disk logs, classifying:
#   * BLOCKER  — a real failure cause (SIGKILL/OOM/GPU HBM exhaustion, vLLM
#                serve init failure, hard timeout, missing/errored result.json).
#   * WARNING  — benign / self-recovered noise worth tracking but NOT fatal
#                (e.g. workflow_parse_error that "recovered from disk artifacts").
# It prints a Markdown section (rollup + per-model bullets) with the ABSOLUTE
# log paths to look at, so the GitHub CI log tells you what broke and where.
#
# Input  (stdin): one record per model, fields separated by US (\x1f) so empty
#                 fields never collapse:
#                   <model>\x1f<verdict>\x1f<status>\x1f<out_dir>
#                 verdict is PASS/FAIL (from run_matrix judging); out_dir is
#                 .../geak_runtime/<model>/ci_runs/<RUN_TS>/.
# Output (stdout): a Markdown "## Run diagnostics" section. Never fails the run
#                 (advisory only); exits 0 regardless.
# =============================================================================
set -uo pipefail

# ---- fingerprints ----------------------------------------------------------
# Real, run-killing causes. Kept general (no model/run-specific assumptions).
BLOCKER_PAT='Segmentation fault|SIGSEGV|core dumped|HIP error|hipError|CUDA error|HSA_STATUS|CUDA out of memory|OutOfMemory|Free memory on device|less than desired GPU memory|No available memory for the (kv )?cache|EngineCore[^\n]*fail|engine core[^\n]*fail|Cannot allocate memory|\bKilled\b|exit code 137|GPU preflight failed|appears wedged|WEDGED|hard timeout|killed by liveness monitor|ModuleNotFoundError'
# Benign / self-recovered lines: track them, but they do NOT cause failure.
WARN_PAT='workflow handoff failed \[workflow_parse_error\]|recovered from disk artifacts|sdk_import_failed|falling back to CLI'

# Pull up to N unique matching lines (trimmed) for a pattern from a file.
scan_file() {  # scan_file <file> <pat> <max>
  local f="$1" pat="$2" max="${3:-3}"
  [ -f "$f" ] || return 0
  grep -aEi "$pat" "$f" 2>/dev/null \
    | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//' \
    | awk '!seen[$0]++' \
    | tail -n "$max"
}

# A friendly one-liner hint for the well-known fatal signatures.
annotate() {  # annotate <line>
  case "$1" in
    *"Free memory on device"*|*"less than desired GPU memory"*|*"No available memory"*)
      echo "  -> GPU HBM exhausted at a vLLM serve launch (likely a prior server's HBM not reclaimed between phases — a perfskills serving-lifecycle leak, not the CI harness).";;
    *"Killed"*|*"exit code 137"*)
      echo "  -> process SIGKILLed (137). If no OOM in dmesg, suspect GPU HBM exhaustion above; confirm with rocm-smi/dmesg on the node.";;
    *"hard timeout"*)
      echo "  -> run exceeded the per-model wall budget (watchdog fired; see timeout.json).";;
  esac
}

emit() { printf '%s\n' "$*"; }

records="$(cat)"
[ -n "$records" ] || { emit "## Run diagnostics"; emit; emit "_No models to scan._"; exit 0; }

n_pass=0 n_fail=0 n_block=0 n_warn=0
body=""
while IFS=$'\x1f' read -r model verdict status out; do
  [ -n "${model:-}" ] || continue
  [ "$verdict" = PASS ] && n_pass=$((n_pass+1)) || n_fail=$((n_fail+1))

  run_log="$out/run.log"; slurm_log="$out/slurm.out"; result="$out/result.json"
  seg="### \`$model\` — $verdict (status=${status:-?})"$'\n'

  # ---- synthesized signals from markers (authoritative, not text-scraped) ----
  blockers=""
  if [ -n "$out" ] && [ ! -f "$result" ]; then
    blockers+="- BLOCKER: no result.json produced (crashed/killed/cancelled before it was written)."$'\n'
  fi
  if [ -f "$out/timeout.json" ]; then
    blockers+="- BLOCKER: hard-timeout watchdog fired -> \`$out/timeout.json\`."$'\n'
  fi
  if [ "${status:-}" = error ]; then
    blockers+="- BLOCKER: result.json status=error."$'\n'
  fi

  # ---- log-scraped blocker evidence (only for FAILED models, to avoid noise) ----
  if [ "$verdict" != PASS ]; then
    while IFS= read -r ln; do
      [ -n "$ln" ] || continue
      blockers+="- BLOCKER: ${ln:0:200}"$'\n'
      note="$(annotate "$ln")"; [ -n "$note" ] && blockers+="$note"$'\n'
    done < <( { scan_file "$slurm_log" "$BLOCKER_PAT" 3; scan_file "$run_log" "$BLOCKER_PAT" 3; } | awk '!s[$0]++' | tail -n 4 )
  fi

  # ---- warnings (benign; scanned for ALL models) ----
  warnings=""
  while IFS= read -r ln; do
    [ -n "$ln" ] || continue
    warnings+="- WARNING: ${ln:0:200}"$'\n'
  done < <( { scan_file "$run_log" "$WARN_PAT" 2; scan_file "$slurm_log" "$WARN_PAT" 2; } | awk '!s[$0]++' | tail -n 2 )
  # Friendly gloss for the one we expect a lot of.
  if printf '%s' "$warnings" | grep -q workflow_parse_error; then
    warnings+="  -> benign: workflow return wasn't in the agent transcript; run_e2e recovered it from disk artifacts. Result is valid."$'\n'
  fi

  [ -n "$blockers" ] && n_block=$((n_block+1))
  [ -n "$warnings" ] && n_warn=$((n_warn+1))

  seg+="- logs: \`$run_log\` · \`$slurm_log\` · \`$result\`"$'\n'
  [ -n "$blockers" ] && seg+="$blockers"
  [ -n "$warnings" ] && seg+="$warnings"
  [ -z "$blockers$warnings" ] && seg+="- clean: no blocker or warning fingerprints."$'\n'
  body+="$seg"$'\n'
done <<< "$records"

emit "## Run diagnostics — issues & where to look"
emit
emit "**${n_pass} passed, ${n_fail} failed** · ${n_block} model(s) with blockers · ${n_warn} with warnings."
emit
printf '%s' "$body"
exit 0
