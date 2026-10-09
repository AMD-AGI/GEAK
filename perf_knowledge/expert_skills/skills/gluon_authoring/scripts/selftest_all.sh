#!/usr/bin/env bash
# selftest_all.sh -- run every OFFLINE selftest the gluon_authoring pack depends on, print one
# summary, exit non-zero on any failure. No GPU, no network, no write outside a private temp dir.
#
# What it runs (each discovered, not listed by hand, so a new tool is covered the day it lands):
#   1. every G/scripts/*.py|*.sh that carries a `--selftest` flag        (the pack's own tools)
#   2. every kernel_workflow/scripts/kernel_tools/* that carries one      (the shared KT tools the
#                                                                          pack's shims exec)
#   3. the 13 core runtime selftests by name (canonical_record close_audit context_contracts
#      context_query harness_stub_env recordctl report_lint round_record run_state served_envelope
#      source_excerpt stage_context toolctl) -- asserted present, so a deleted core tool FAILS here
#      instead of silently dropping out of the discovered set
#   4. G/scheduler/tests/test_*.py, with a PRIVATE GEAK_GPU_LOCK_DIR (never /tmp/team_gpu_locks)
#   5. check_term_index.py --pack G and check_pack_refs.py (every pointer in the pack resolves)
#
# Usage: bash scripts/selftest_all.sh [--list] [--keep-going] [-k <substring>]
#   --list        print what would run, run nothing
#   -k <sub>      run only checks whose label contains <sub>
# Logs: one file per check under the printed temp dir (left in place for reading; this script
# deletes nothing).
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK="$(dirname "$HERE")"
REPO="$(cd "$PACK/../../../.." && pwd)"
KT="$REPO/kernel_workflow/scripts/kernel_tools"
SELF="$(basename "${BASH_SOURCE[0]}")"

LIST=0; FILTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --list) LIST=1 ;;
    --keep-going) ;;  # always keeps going; accepted for symmetry with other runners
    -k) shift; FILTER="${1:-}" ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "selftest_all: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

TMP="$(mktemp -d "${TMPDIR:-/tmp}/gluon_selftest_all.XXXXXX")"
export GEAK_GPU_LOCK_DIR="$TMP/locks"      # hermetic: tests never flock the team namespace
export PYTHONDONTWRITEBYTECODE=1           # never write .pyc into the (possibly shared) pack dir
export GEAK_GPU_BROKER=0
mkdir -p "$GEAK_GPU_LOCK_DIR"

CORE="canonical_record close_audit context_contracts context_query harness_stub_env recordctl
report_lint round_record run_state served_envelope source_excerpt stage_context toolctl"

LABELS=(); CMDS=()
add() { LABELS+=("$1"); CMDS+=("$2"); }

has_selftest() { grep -q -- '--selftest' "$1" 2>/dev/null; }
runner_for() { case "$1" in *.sh) echo "bash" ;; *) echo "python3" ;; esac; }

# 3. core set first, asserted present
for c in $CORE; do
  if [ -f "$HERE/$c.py" ]; then
    add "core:$c" "python3 '$HERE/$c.py' --selftest"
  else
    add "core:$c" "echo 'MISSING core tool scripts/$c.py' >&2; false"
  fi
done

# 1. the pack's own tools (shims carry no --selftest of their own; their KT target runs in 2.)
for f in "$HERE"/*.py "$HERE"/*.sh; do
  [ -f "$f" ] || continue
  b="$(basename "$f")"; stem="${b%.*}"
  [ "$b" = "$SELF" ] && continue
  case " $(echo $CORE) " in *" $stem "*) continue ;; esac
  has_selftest "$f" || continue
  add "pack:$b" "$(runner_for "$b") '$f' --selftest"
done

# 2. shared kernel tools
for f in "$KT"/*.py "$KT"/*.sh; do
  [ -f "$f" ] || continue
  b="$(basename "$f")"
  has_selftest "$f" || continue
  add "kt:$b" "$(runner_for "$b") '$f' --selftest"
done

# 4. scheduler tests (each is a plain script with a __main__ runner)
for f in "$PACK"/scheduler/tests/test_*.py; do
  [ -f "$f" ] || continue
  add "scheduler:$(basename "$f")" "python3 '$f'"
done

# 5. pointer checks
add "refs:check_term_index" "python3 '$HERE/check_term_index.py' --pack '$PACK'"
add "refs:check_pack_refs" "python3 '$HERE/check_pack_refs.py' --pack '$PACK'"

# Checks that mostly SLEEP (wait_for.sh's selftest times real 50-95 s jobs against the stall
# detector) start in the background first and are collected last, so they overlap the rest.
SLOW=" pack:wait_for.sh "

pass=0; fail=0; skipped=0; failed=()
echo "selftest_all: pack=$PACK"
echo "selftest_all: logs=$TMP  (GEAK_GPU_LOCK_DIR=$GEAK_GPU_LOCK_DIR)"
declare -A BGPID=() BGT0=()
for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"; cmd="${CMDS[$i]}"
  [[ "$SLOW" == *" $label "* ]] || continue
  if [ -n "$FILTER" ] && [[ "$label" != *"$FILTER"* ]]; then continue; fi
  [ "$LIST" = 1 ] && continue
  log="$TMP/$(echo "$label" | tr ':/' '__').log"
  ( cd "$PACK" && eval "$cmd" ) >"$log" 2>&1 </dev/null &
  BGPID[$i]=$!; BGT0[$i]=$(date +%s)
done
ORDER=()
for i in "${!LABELS[@]}"; do [[ "$SLOW" == *" ${LABELS[$i]} "* ]] || ORDER+=("$i"); done
for i in "${!LABELS[@]}"; do [[ "$SLOW" == *" ${LABELS[$i]} "* ]] && ORDER+=("$i"); done
for i in "${ORDER[@]}"; do
  label="${LABELS[$i]}"; cmd="${CMDS[$i]}"
  if [ -n "$FILTER" ] && [[ "$label" != *"$FILTER"* ]]; then skipped=$((skipped+1)); continue; fi
  if [ "$LIST" = 1 ]; then printf '  %-44s %s\n' "$label" "$cmd"; continue; fi
  log="$TMP/$(echo "$label" | tr ':/' '__').log"
  if [ -n "${BGPID[$i]:-}" ]; then
    wait "${BGPID[$i]}"; rc=$?; t0="${BGT0[$i]}"
  else
    t0=$(date +%s)
    ( cd "$PACK" && eval "$cmd" ) >"$log" 2>&1 </dev/null
    rc=$?
  fi
  dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then
    pass=$((pass+1)); printf '  ok    %-44s %3ss\n' "$label" "$dt"
  else
    fail=$((fail+1)); failed+=("$label"); printf '  FAIL  %-44s %3ss  rc=%s  log=%s\n' "$label" "$dt" "$rc" "$log"
    tail -n 8 "$log" | sed 's/^/        | /'
  fi
done
[ "$LIST" = 1 ] && exit 0

echo
echo "selftest_all: $pass passed, $fail failed$([ $skipped -gt 0 ] && echo ", $skipped filtered out")"
if [ $fail -gt 0 ]; then
  printf '  failed: %s\n' "${failed[@]}"
  exit 1
fi
exit 0
