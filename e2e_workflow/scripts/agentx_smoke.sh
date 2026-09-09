#!/usr/bin/env bash
# Check that this box can actually run an AgentX trace replay -- before a GEAK
# run spends hours finding out that it cannot.
#
# The failures this catches all share a shape: nothing is wrong until the client
# has already launched and warmed a server, so the error arrives 20+ minutes in,
# on a leg whose result is then missing from the middle of a long run. The worst
# of them is not an error at all -- a stock aiperf with no trace-replay support
# happily benchmarks something else.
#
# Usage:
#   bash e2e_workflow/scripts/agentx_smoke.sh                  # environment only
#   MODEL=/models/Kimi-K3 bash .../agentx_smoke.sh             # also check weights
#   BASE_URL=http://127.0.0.1:8000 bash .../agentx_smoke.sh    # also replay for real
#
# With BASE_URL pointing at a SERVER THAT IS ALREADY RUNNING, the last check
# replays a handful of corpus entries for AGENTX_SMOKE_DURATION_S (default 60)
# and maps the result, which exercises the entire chain end to end. That result
# is deliberately tiny and is stamped non-canonical; it is a wiring test, never a
# measurement.
#
# Exit: 0 all clear (warnings allowed), 1 at least one blocker.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENTS="$HERE/adapters/clients"
PY="${PYTHON_BIN:-python3}"

BLOCKERS=0
WARNINGS=0
pass() { printf '  \033[32mok\033[0m      %s\n' "$1"; }
warn() { printf '  \033[33mwarn\033[0m    %s\n' "$1"; WARNINGS=$((WARNINGS + 1)); }
fail() { printf '  \033[31mBLOCKER\033[0m %s\n' "$1"; BLOCKERS=$((BLOCKERS + 1)); }
fix()  { printf '          -> %s\n' "$1"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }

echo "AgentX trace-replay readiness check"
echo "workflow: $(cd "$HERE/.." && pwd)"

# ── 1. the client binary ─────────────────────────────────────────────────────
head_ "1. aiperf (the trace-replay client)"
AIPERF="${AIPERF_BIN:-aiperf}"
if AIPERF_PATH="$(command -v "$AIPERF" 2>/dev/null)"; then
  pass "found: $AIPERF_PATH"
  _ver="$("$AIPERF" --version 2>&1 | head -1)"
  [ -n "$_ver" ] && pass "version: $_ver"

  _help="$("$AIPERF" profile --help 2>&1)"

  # Match with bash's own pattern operator, never `printf ... | grep -q`. This
  # script runs under `set -o pipefail` and aiperf's help is ~169KB: grep -q
  # exits at the first match while printf is still writing, printf dies of
  # SIGPIPE, and pipefail reports 141 for the pipeline -- turning a successful
  # match into a failed check. A small help text hides it, because printf
  # finishes before grep exits.
  _help_has() { case "$_help" in *"$1"*) return 0 ;; *) return 1 ;; esac; }

  # The decisive check. A stock aiperf cannot replay a trace at all; pointed at
  # a server it will still run SOMETHING and report a throughput, which is how a
  # "successful" run ends up measuring a workload nobody asked for.
  #
  # Corroborate the flag with aiperf's scenario registry, which is what aiperf
  # actually consults at startup and what InferenceX's own setup_aiperf.sh
  # probes. Three outcomes, deliberately distinguished: the registry can confirm
  # the scenario, deny it, or be unreachable. Only a denial is a blocker -- an
  # unreachable registry means we guessed the wrong interpreter, and failing the
  # run on our own bad guess is the error this check exists to avoid.
  _scen="${GEAK_AGENTX_SCENARIO:-inferencex-agentx-mvp}"
  _aipy="$(dirname "$AIPERF_PATH")/python"
  [ -x "$_aipy" ] || _aipy="$(command -v python3 2>/dev/null)"
  _scen_state=unreachable
  if [ -n "$_aipy" ]; then
    case "$("$_aipy" - "$_scen" 2>/dev/null <<'PY'
import sys
try:
    from aiperf.common.scenario import get_scenario
except Exception:
    print("UNREACHABLE"); raise SystemExit(0)
try:
    get_scenario(sys.argv[1]); print("RESOLVED")
except Exception:
    print("DENIED")
PY
)" in
      *RESOLVED*) _scen_state=resolved ;;
      *DENIED*)   _scen_state=denied ;;
    esac
  fi

  if _help_has '--scenario'; then
    pass "supports --scenario (trace replay)"
    case "$_scen_state" in
      resolved)
        pass "scenario '$_scen' resolves in aiperf's registry" ;;
      denied)
        fail "aiperf does not register '$_scen': the run would abort at startup"
        fix "check GEAK_AGENTX_SCENARIO, or that this aiperf ships InferenceX's scenarios" ;;
      *)
        warn "could not reach aiperf's scenario registry to confirm '$_scen'" ;;
    esac
  elif [ "$_scen_state" = resolved ]; then
    pass "scenario '$_scen' resolves in aiperf's registry (trace replay)"
    warn "registry resolves '$_scen' but --scenario is absent from the help text"
  else
    fail "this aiperf has no --scenario: it CANNOT replay the AgentX corpus"
    fix "install the AgentX-capable aiperf, or set AIPERF_BIN to it"
    fix "a stock aiperf will still produce a number -- for a different workload"
  fi
  for flag in --warmup-requests-per-lane --failed-request-threshold; do
    if _help_has "$flag"; then
      pass "supports $flag"
    else
      warn "no $flag; the client passes it and aiperf may reject the invocation"
    fi
  done
else
  fail "aiperf not on PATH (looked for '$AIPERF')"
  fix "install the AgentX-capable aiperf, or set AIPERF_BIN=/path/to/aiperf"
fi

# ── 2. the result mapper ─────────────────────────────────────────────────────
head_ "2. map_aiperf.py (aiperf export -> canonical result)"
MAPPER=""
for cand in \
  "${INFERENCEX_PATH:+${INFERENCEX_PATH}/benchmarks/map_aiperf.py}" \
  "${INFERENCEX_PATH:+${INFERENCEX_PATH}/assets/agentx/map_aiperf.py}" \
  "$CLIENTS/map_aiperf.py"; do
  [ -n "$cand" ] && [ -f "$cand" ] && { MAPPER="$cand"; break; }
done
if [ -z "$MAPPER" ]; then
  fail "no map_aiperf.py anywhere (INFERENCEX_PATH=${INFERENCEX_PATH:-<unset>})"
  fix "restore $CLIENTS/map_aiperf.py (GEAK vendors it; no checkout needed)"
else
  pass "using: $MAPPER"
  case "$MAPPER" in
    "$CLIENTS/"*) pass "vendored copy -- no InferenceX checkout required" ;;
    *) pass "InferenceX checkout wins over the vendored copy, as intended" ;;
  esac
  # Run it, because "the file exists" is not the property that matters.
  _tmp="$(mktemp -d)"
  trap 'rm -rf "$_tmp"' EXIT
  printf '%s' '{"metadata":{"submission_valid":true},"metrics":{"output_token_throughput":{"avg":1.5},"input_token_throughput":{"avg":210.0},"request_count":3}}' \
    > "$_tmp/export.json"
  if AGENTX_NONCANONICAL_REASONS="smoke" \
       "$PY" "$MAPPER" "$_tmp/export.json" "$_tmp/result.json" >/dev/null 2>"$_tmp/err"; then
    _axes="$("$PY" -c '
import json,sys
r=json.load(open(sys.argv[1]))
print(r.get("output_throughput"), r.get("total_token_throughput"), r.get("submission_valid"))
' "$_tmp/result.json" 2>/dev/null)"
    pass "maps an export (output/total/valid = ${_axes})"
    case "$_axes" in
      *" False") pass "client-detected deviations force submission_valid=false" ;;
      *) warn "a stamped deviation did not invalidate the result" ;;
    esac
  else
    fail "the mapper failed on a synthetic export: $(tr -d '\n' < "$_tmp/err" | tail -c 200)"
  fi
fi

# ── 3. the GEAK plumbing the client depends on ───────────────────────────────
head_ "3. GEAK bench plumbing"
for f in "$HERE/bench_e2e.sh" "$HERE/bench_replica.sh" "$HERE/server_teardown.sh" \
         "$CLIENTS/agentx.sh"; do
  [ -f "$f" ] && pass "present: ${f#"$HERE"/}" || fail "missing: $f"
done
if [ -f "$CLIENTS/agentx.sh" ]; then
  # shellcheck source=/dev/null
  if ( . "$CLIENTS/agentx.sh" ) >/dev/null 2>&1; then
    pass "the agentx client adapter sources cleanly"
    for fn in adapter_bench adapter_profile_warmup_s _agentx_duration; do
      # shellcheck source=/dev/null
      if ( . "$CLIENTS/agentx.sh"; declare -F "$fn" >/dev/null ); then
        pass "defines $fn"
      else
        fail "the adapter does not define $fn"
      fi
    done
    # shellcheck source=/dev/null
    _delay="$( . "$CLIENTS/agentx.sh"; MEASUREMENT_PURPOSE=parity adapter_profile_warmup_s )"
    # shellcheck source=/dev/null
    _leg="$( . "$CLIENTS/agentx.sh"; MEASUREMENT_PURPOSE=parity _agentx_duration )"
    if [ "${_delay:-0}" -gt 0 ] 2>/dev/null; then
      pass "profile window opens ${_delay}s into the ${_leg}s canonical leg (steady state)"
    else
      fail "the profile window would open at load start, capturing aiperf's ramp"
    fi
  else
    fail "the agentx client adapter fails to source"
  fi
fi
if [ -f "$HERE/bench_env.sh" ]; then
  fail "$HERE/bench_env.sh exists in the SOURCE tree"
  fix "delete it: that file is written per run, and here it silently applies to EVERY run"
else
  pass "no stray bench_env.sh in the source tree (it is written per run)"
fi

# ── 4. the box ───────────────────────────────────────────────────────────────
head_ "4. host"
command -v "$PY" >/dev/null 2>&1 && pass "python: $("$PY" --version 2>&1)" \
  || fail "no $PY on PATH"
if [ -n "${MODEL:-}" ]; then
  if [ -e "$MODEL" ]; then
    pass "MODEL resolves: $MODEL"
    [ -f "$MODEL/config.json" ] && pass "config.json present" \
      || warn "no config.json under MODEL (a served path may still be fine)"
  else
    fail "MODEL=$MODEL does not exist"
  fi
else
  warn "MODEL unset -- weights not checked (pass MODEL=... to include it)"
fi
if command -v amd-smi >/dev/null 2>&1; then
  _n="$(amd-smi list 2>/dev/null | grep -c GPU || true)"
  [ "${_n:-0}" -gt 0 ] && pass "amd-smi sees ${_n} GPU(s)" || warn "amd-smi found no GPU"
elif command -v rocminfo >/dev/null 2>&1; then
  pass "rocm: $(rocminfo 2>/dev/null | grep -m1 gfx | tr -s ' ')"
else
  warn "neither amd-smi nor rocminfo -- GPU visibility unverified"
fi

# ── 5. end to end, only against a server that already exists ────────────────
head_ "5. live replay"
if [ -z "${BASE_URL:-}" ]; then
  echo "  skipped (set BASE_URL=http://host:port, with a server already running,"
  echo "           to replay a few entries and exercise the whole chain)"
elif [ "$BLOCKERS" -gt 0 ]; then
  echo "  skipped (fix the blockers above first)"
else
  if curl -sf --max-time 10 "${BASE_URL}/v1/models" >/dev/null 2>&1 \
     || curl -sf --max-time 10 "${BASE_URL}/health" >/dev/null 2>&1; then
    pass "server responds at $BASE_URL"
    _out="$(mktemp -d)"
    _dur="${AGENTX_SMOKE_DURATION_S:-60}"
    echo "  replaying ${AGENTX_SMOKE_ENTRIES:-4} entries for ${_dur}s (a wiring test, NOT a measurement)"
    # Drive the real adapter, so this tests the code the run will use.
    (
      # shellcheck source=/dev/null
      . "$CLIENTS/agentx.sh"
      export OUT_DIR="$_out" RESULT_JSONL="$_out/results.jsonl" \
             MODEL="${MODEL:-unknown}" BASE_URL="$BASE_URL" \
             MEASUREMENT_PURPOSE=search \
             GEAK_AGENTX_LOOP_DURATION_S="$_dur" \
             AGENTX_NUM_ENTRIES="${AGENTX_SMOKE_ENTRIES:-4}" \
             AGENTX_WARMUP_REQUESTS_PER_LANE="${AGENTX_SMOKE_WARMUP:-1}" \
             AGENTX_WARMUP_GRACE_PERIOD="${AGENTX_SMOKE_GRACE:-60}" \
             CONC="${AGENTX_SMOKE_CONC:-2}"
      adapter_bench 1 "${AGENTX_SMOKE_CONC:-2}" 0
    ) >"$_out/log" 2>&1
    if [ -s "$_out/results.jsonl" ]; then
      pass "replayed and mapped a result"
      "$PY" -c '
import json,sys
r=json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
print("          output tok/s :", r.get("output_throughput"))
print("          total  tok/s :", r.get("total_token_throughput"), "(the graded axis)")
print("          valid        :", r.get("submission_valid"), r.get("submission_invalid_reasons"))
' "$_out/results.jsonl" 2>/dev/null || true
      pass "the full chain works: client -> replay -> mapper -> result"
    else
      fail "the replay produced no result; last lines of its log:"
      tail -12 "$_out/log" | sed 's/^/          /'
    fi
    rm -rf "$_out"
  else
    fail "no server at $BASE_URL (this check does not launch one)"
    fix "start the server first, or unset BASE_URL to skip the live replay"
  fi
fi

# ── verdict ──────────────────────────────────────────────────────────────────
head_ "verdict"
if [ "$BLOCKERS" -gt 0 ]; then
  echo "  ${BLOCKERS} blocker(s), ${WARNINGS} warning(s) — an AgentX run would fail or measure the wrong thing."
  exit 1
fi
echo "  no blockers, ${WARNINGS} warning(s) — this box can run an AgentX trace replay."
echo
echo "  Launch GEAK on it by passing the workload declaration to the workflow:"
echo '    workload_kind: "agentx_trace_replay"'
echo "  See e2e_workflow/README.md (\"Standalone AgentX\") for the full invocation."
exit 0
