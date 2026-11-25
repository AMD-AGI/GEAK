#!/usr/bin/env bash
# Run rocprofv3 with the guards a measured campaign paid for. Never call rocprofv3 bare.
# (GEAK shared kernel tool.)
#
# GUARD 1 -- A CONFIGURABLE TIMEOUT + DEGRADE. A counter group the box cannot schedule in one pass
# does not fail the same way on every ROCm version: depending on the rocprofv3 release it is
# REPLAYED (multi-pass, just slower), ABORTS the app (SIGABRT, the traceback points at the workload),
# or HANGS while holding the GPU lock. The timeout converts the hang into a legible exit 124, which is
# recorded as a DEGRADED layer (never as zero counters); an abort shows up as rc 134 / a signal. In
# both cases the next move is to bisect the --pmc list (parse_pmc.collect_pmc_group does that).
#   --timeout <s> (default $PMC_TIMEOUT or 150)   --no-timeout (or --timeout 0)
# DO NOT use the timeout for e2e serving runs (e2e_workflow): a serving process legitimately spends a
# long time in startup and in rocprofv3's finalize/flush, so a kernel-sized timeout kills a healthy
# collection there. Use --no-timeout and let the serving harness own its own deadline.
#
# GUARD 2 -- HIP_VISIBLE_DEVICES INSIDE ROCR_VISIBLE_DEVICES. rocprofv3 aborts when BOTH are set
# (HIP is then a logical index into the ROCR list). Only in that case is the pair collapsed: ROCR is
# narrowed to the one device HIP selected and HIP is unset, for this collection only -- the profiled
# device does not move. When only HIP_VISIBLE_DEVICES is set (the normal gpu_lock.sh case) it is left
# alone: unsetting it there would expose every GPU and profile whatever device 0 is.
#
# GUARD 3 -- ALWAYS FILTER BY KERNEL. --kernel-include-regex limits collection to the matching
# kernels, so a run with many helper launches collects less. Unfiltered collection also sweeps up
# torch/CK/cutlass helpers, so the rows you then average are not your kernel's.
#
# GEAK conventions: runs under gpu_lock.sh (re-execs itself under it with --gpu <id>, refuses with
# no lock and no id); the device comes from the lock, never from an inline HIP_VISIBLE_DEVICES; the
# --out dir is moved aside if it exists (never rm'd); the command may be argv after `--` or one
# shell string via --cmd (run as `bash -c`).
#
# Usage:
#   rocprofv3_safe.sh --kernel <regex> --out <dir> --pmc "A B C" [--timeout 150|--no-timeout] [--gpu N] -- <cmd...>
#   rocprofv3_safe.sh --kernel <regex> --out <dir> --kernel-trace [...] --cmd "<shell string>"
#   rocprofv3_safe.sh --kernel <regex> --out <dir> --att|--att-wide [...] -- <cmd...>
#   rocprofv3_safe.sh --selftest
#
# --att needs the rocprof-trace-decoder library (NOT vendored by GEAK): ROCPROF_ATT_LIBRARY_PATH names
# its directory (else the ROCm lib dirs are searched). Absent -> exit 3, state "decoder_absent": the
# ATT layer is DEGRADED, the caller carries on. --att-wide also selects every SIMD (retry for a
# traced CU that caught no waves, i.e. code.json "code": null).
#
# GUARD 4 -- TWO LLVM COPIES IN ONE PROCESS. With torch/Triton loaded, a duplicate libamd_comgr.so.3
# aborts every rocprofv3 mode ("Option ... registered more than once", LLVM ERROR) and the process
# then ignores SIGTERM. The log is watched while the run is live; on that line the run is killed and
# recorded as state llvm_dual_load (exit 7) instead of burning the timeout as "no dispatches".
#
# Exit codes: 0 ok | 124 TIMED OUT, incl. a KILL after the grace period (degraded, never a zero) |
#             2 usage / not under gpu_lock | 3 unavailable dependency/locus | 4 no usable matching
#             artifacts | 5 partial artifacts | 7 llvm_dual_load (environment, see
#             profiling-dependencies.md "Known abort") | other = rocprofv3's (134 / signal = abort ->
#             bisect the group)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
# shellcheck source=_kt_common.sh
source "$HERE/_kt_common.sh"

source "$HERE/rocprofv3_env.sh"

if [ "${1:-}" = "--selftest" ]; then
  fail=0
  t="${TMPDIR:-/tmp}/geak_rocprofv3_safe_selftest.$(id -u)"; mkdir -p "$t/bin"
  cat > "$t/bin/rocprofv3" <<'EOF'
#!/usr/bin/env bash
# fake rocprofv3: report the device env it saw, honour -d, optionally hang
case "${1:-}" in
  --version) echo 'fixture sdk'; exit 0;;
  --help) echo '--att --att-library-path --kernel-trace --pmc'; exit 0;;
esac
out=""; while [ $# -gt 0 ]; do [ "$1" = "-d" ] && out="$2"; [ "$1" = "--" ] && { shift; break; }; shift; done
echo "HIP=${HIP_VISIBLE_DEVICES-unset} ROCR=${ROCR_VISIBLE_DEVICES-unset}" > "$out/env_seen.txt"
printf 'Kernel_Name,Start_Timestamp,End_Timestamp\nk,1,2\n' > "$out/test_kernel_trace.csv"
printf 'Kernel_Name,Counter_Name,Counter_Value\nk,SQ_WAVES,1\n' > "$out/test_counter_collection.csv"
[ -n "${FAKE_HANG:-}" ] && sleep 30
"$@"
EOF
  chmod +x "$t/bin/rocprofv3"
  _run() { PATH="$t/bin:$PATH" GEAK_GPU_LOCK_HELD="${GEAK_GPU_LOCK_HELD:-${HIP_VISIBLE_DEVICES:-0}}" bash "$SELF" "$@" >/dev/null 2>&1; }
  # HIP only (the gpu_lock.sh case): HIP must survive.
  ( unset ROCR_VISIBLE_DEVICES
    HIP_VISIBLE_DEVICES=3 _run --kernel k --out "$t/o1" --kernel-trace --cmd "true" )
  grep -q "HIP=3 ROCR=unset" "$t/o1/env_seen.txt" 2>/dev/null \
    || { echo "FAIL: HIP-only env was altered: $(cat "$t/o1/env_seen.txt" 2>/dev/null)"; fail=1; }
  # HIP inside ROCR: collapse to the same device.
  ( ROCR_VISIBLE_DEVICES=4,6 HIP_VISIBLE_DEVICES=1 GEAK_GPU_LOCK_HELD=1 _run --kernel k --out "$t/o2" --pmc "SQ_WAVES" -- true )
  grep -q "HIP=unset ROCR=6" "$t/o2/env_seen.txt" 2>/dev/null \
    || { echo "FAIL: HIP-in-ROCR not collapsed to ROCR=6: $(cat "$t/o2/env_seen.txt" 2>/dev/null)"; fail=1; }
  # timeout -> 124 + degraded state file
  ( export FAKE_HANG=1; PATH="$t/bin:$PATH" GEAK_GPU_LOCK_HELD=0 HIP_VISIBLE_DEVICES=0 \
      bash "$SELF" --kernel k --out "$t/o3" --pmc "SQ_WAVES" --timeout 1 -- true >/dev/null 2>&1; echo $? > "$t/rc3" )
  [ "$(cat "$t/rc3")" = 124 ] && grep -q '"state":"timeout"' "$t/o3/rocprofv3_safe.json" \
    || { echo "FAIL: timeout did not degrade to 124 + state=timeout"; fail=1; }
  # no lock + no id -> refuse
  ( unset GEAK_GPU_LOCK_HELD; PATH="$t/bin:$PATH" KT_ALLOW_UNLOCKED=0 \
      bash "$SELF" --kernel k --out "$t/o4" --kernel-trace -- true >/dev/null 2>&1; echo $? > "$t/rc4" )
  [ "$(cat "$t/rc4")" = 2 ] || { echo "FAIL: outside gpu_lock with no --gpu must refuse (rc 2)"; fail=1; }
  # --att with no decoder anywhere -> exit 3, decoder_absent (degraded, not failed)
  ( PATH="$t/bin:$PATH" GEAK_GPU_LOCK_HELD=0 HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH= ROCPROF_ATT_LIBRARY_PATH="$t/none" KT_ATT_SYSTEM_DIRS="$t/none" \
      bash "$SELF" --kernel k --out "$t/o6" --att -- true >/dev/null 2>&1; echo $? > "$t/rc6" )
  [ "$(cat "$t/rc6")" = 3 ] && grep -q decoder_absent "$t/o6/rocprofv3_safe.json" \
    || { echo "FAIL: --att without a decoder must degrade with rc 3"; fail=1; }
  # an existing --out is moved aside, not deleted
  mkdir -p "$t/o5"; echo keep > "$t/o5/old.txt"
  HIP_VISIBLE_DEVICES=0 _run --kernel k --out "$t/o5" --kernel-trace -- true
  ls -d "$t"/o5.old_* >/dev/null 2>&1 || { echo "FAIL: existing --out was not moved aside"; fail=1; }
  [ "$fail" = 0 ] && echo "ROCPROFV3_SAFE SELFTEST PASS" || echo "ROCPROFV3_SAFE SELFTEST FAIL"
  exit "$fail"
fi

ORIG_ARGS=("$@")
KERNEL=""; OUT=""; PMC=""; TIMEOUT="${PMC_TIMEOUT:-150}"; MODE="pmc"; CMD=(); GPU=""
while [ $# -gt 0 ]; do
  case "$1" in
    --kernel)       KERNEL="$2"; shift 2;;
    --out)          OUT="$2"; shift 2;;
    --pmc)          PMC="$2"; shift 2;;
    --timeout)      TIMEOUT="$2"; shift 2;;
    --no-timeout)   TIMEOUT=0; shift;;
    --kernel-trace) MODE="trace"; shift;;
    --att)          MODE="att"; shift;;
    --att-wide)     MODE="att-wide"; shift;;
    --gpu|--dev)    GPU="$2"; shift 2;;
    --cmd)          CMD=(bash -c "$2"); shift 2;;
    -h|--help)      cat <<'USAGE'
rocprofv3_safe.sh --kernel <regex> --out <dir> (--pmc "A B" | --kernel-trace | --att | --att-wide)
                  [--timeout 150 | --no-timeout] [--gpu N] (-- <cmd...> | --cmd "<shell string>")
Runs rocprofv3 under gpu_lock.sh with a kernel filter, a timeout and artifact checks; writes
<out>/rocprofv3_safe.json (state, reason, rc) and <out>/rocprofv3.log.
Exit: 0 ok | 124 timed out | 2 usage/lock | 3 dependency/locus | 4 no matching artifacts |
      5 partial | 7 llvm_dual_load (profiling-dependencies.md "Known abort") | 134 abort -> split the group
--att: code.json "code": null after the wide retry = no wave on the traced CU (small grid);
       launch >= 1 WG per CU or set KT_ATT_TARGET_CU.
USAGE
                    exit 0;;
    --) shift; CMD=("$@"); break;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$OUT" ] || { echo "--out is required" >&2; exit 2; }
[ ${#CMD[@]} -gt 0 ] || { echo "no app cmd (--cmd \"<string>\" or -- <cmd...>)" >&2; exit 2; }
if [ -z "$KERNEL" ]; then
  # Refused rather than defaulted: an unfiltered collection collects every launch AND averages in torch/CK
  # helper kernels, so the number it returns is not this kernel's and nothing downstream can tell.
  echo "--kernel <regex> is required (guard 3: unfiltered collection is slower AND contaminated)" >&2
  exit 2
fi
[ "$MODE" != "pmc" ] || [ -n "$PMC" ] || { echo "--pmc \"<counters>\" is required unless --kernel-trace/--att" >&2; exit 2; }
[[ "$TIMEOUT" =~ ^[0-9]+$ ]] || { echo "--timeout must be whole seconds (0 = none)" >&2; exit 2; }

# Validate the configured execution environment before any collection. A missing
# container must never silently redirect profiling to the host.
locus_preflight || exit 3
kt_ensure_lock "$GPU" "$SELF" "${ORIG_ARGS[@]}"
kt_move_aside "$OUT"
mkdir -p "$OUT" || exit 2
OUT="$(cd "$OUT" && pwd -P)"
if ! locus_workdir_shared "$OUT"; then
  echo "[rocprofv3_safe] output is not shared with the execution locus: $OUT" >&2
  exit 3
fi
LOCUS_OUT="$(locus_path "$OUT")"
RV3_UNSET_HIP=0
_rv3_fix_visible_devices || exit $?
export LOCUS_ENV="${LOCUS_ENV:-} HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES"
DEVICE_ENV=()
[ "$RV3_UNSET_HIP" = 0 ] || DEVICE_ENV=(env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES)

# Only this primitive owns rocprofv3 arguments, timeouts and the ATT retry.
ARGS=(--kernel-include-regex "$KERNEL" -d "$LOCUS_OUT" -f csv)
case "$MODE" in
  trace) ARGS=(--kernel-trace "${ARGS[@]}") ;;
  att|att-wide)
    if ! kt_att_prepare locus "$OUT/att_dependencies.json"; then
      python3 - "$OUT" "${KT_ATT_STATUS:-dependency_probe_failed}" <<'PY_STATE'
import json, pathlib, sys
pathlib.Path(sys.argv[1], 'rocprofv3_safe.json').write_text(json.dumps(
    {'state': sys.argv[2], 'rc': 7 if sys.argv[2] == 'llvm_dual_load' else 3, 'reason': sys.argv[2]}))
PY_STATE
      echo "[rocprofv3_safe] ATT unavailable: ${KT_ATT_STATUS:-dependency_probe_failed}" >&2
      [ "${KT_ATT_STATUS:-}" = llvm_dual_load ] && exit 7
      exit 3
    fi
    # --kernel-trace + csv: profiler_artifacts.py checks the decoded dispatch against kernel_trace.csv.
    ARGS=(--att "${KT_ATT_ARGS[@]}" --att-target-cu "${KT_ATT_TARGET_CU:-0}" --att-shader-engine-mask 0xF \
          --kernel-trace "${ARGS[@]}")
    [ "$MODE" = att-wide ] && ARGS+=(--att-simd-select 0xF) ;;
  *)
    # shellcheck disable=SC2206
    ARGS=(--pmc $PMC "${ARGS[@]}") ;;
esac
if [[ "${GEAK_GPU_LEASE_IDS:-${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-}}}" == *,* ]]; then
  ARGS+=(--disable-signal-handlers true)
fi
_rv3_run() {
  if [ "$TIMEOUT" = 0 ]; then
    locus_run "${DEVICE_ENV[@]}" rocprofv3 "${ARGS[@]}" -- "${CMD[@]}"
  else
    # The timeout runs INSIDE the container, so it stops the payload, not only docker exec.
    locus_run timeout -k "${KT_TIMEOUT_KILL_AFTER:-10}" "$TIMEOUT" "${DEVICE_ENV[@]}" rocprofv3 "${ARGS[@]}" -- "${CMD[@]}"
  fi
}
RV3_LOG="$OUT/rocprofv3.log"
RV3_DUAL_LOAD_RE="registered more than once|LLVM ERROR: inconsistency in registered CommandLine"
RV3_DUAL_LOAD=0
# Run with the output teed to $RV3_LOG and watch it: the LLVM dual-load abort leaves the process
# hung (SIGTERM ignored), so stop it as soon as the line appears.
_rv3_watched() {
  _rv3_run > >(tee -a "$RV3_LOG") 2> >(tee -a "$RV3_LOG" >&2) &
  local pid=$! rc
  while kill -0 "$pid" 2>/dev/null; do
    if grep -qE "$RV3_DUAL_LOAD_RE" "$RV3_LOG" 2>/dev/null; then
      RV3_DUAL_LOAD=1
      kt_kill_tree "$pid"
      break
    fi
    sleep 0.5
  done
  wait "$pid"; rc=$?
  sleep 0.2   # let the tee processes flush
  grep -qE "$RV3_DUAL_LOAD_RE" "$RV3_LOG" 2>/dev/null && RV3_DUAL_LOAD=1
  return "$rc"
}
_rv3_inspect() {
  locus_fetch "$LOCUS_OUT" "$(dirname "$OUT")"
  python3 "$HERE/profiler_artifacts.py" --out "$OUT" --mode "$MODE" --kernel "$KERNEL" --pmc "$PMC" \
    > "$OUT/collection_artifacts.json"
}
echo "[rocprofv3_safe] timeout=${TIMEOUT}s kernel='$KERNEL' mode=$MODE" >&2
attempts=1
_rv3_watched
profiler_rc=$?
_rv3_inspect || true
# A successful narrow capture with code:null may have missed all waves on that
# SIMD. Retry exactly once here, shared by profile_kernel and task capture.
if [ "$MODE" = att ] && [ "$profiler_rc" = 0 ] && [ "$RV3_DUAL_LOAD" = 0 ] && \
    python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("reason") != "code_null")' "$OUT/collection_artifacts.json"; then
  kt_move_aside "$OUT"
  mkdir -p "$OUT"
  kt_att_prepare locus "$OUT/att_dependencies.json" || exit 3
  ARGS+=(--att-simd-select 0xF)
  attempts=2
  _rv3_watched
  profiler_rc=$?
  _rv3_inspect || true
fi
# Artifact inspection is independent of the process status: even a complete CSV
# does not turn a failed workload into a successful collection.
python3 - "$OUT" "$profiler_rc" "$attempts" "$KERNEL" "$PMC" "$TIMEOUT" "$RV3_DUAL_LOAD" "$MODE" <<'PY_STATE'
import json, pathlib, sys
out, rc, attempts, kernel, pmc, timeout, dual, mode = sys.argv[1:]
rc, timeout = int(rc), int(timeout)
report = json.loads(pathlib.Path(out, 'collection_artifacts.json').read_text())
state = report['state']
if dual == '1':
    # Environment fault: two libamd_comgr copies in one process. Not a counter-budget abort.
    state, rc = 'llvm_dual_load', 7
    report.update(reason='llvm_dual_load',
                  hint='two libamd_comgr.so.3 copies loaded (torch _rocm_sdk_core + rocprofv3 '
                       '_rocm_sdk_devel); fix the image: kernel_tools/profiling-dependencies.md '
                       '"Known abort"')
elif rc:
    # timeout -k sends KILL when TERM is ignored: 137 after a timeout is still a timeout.
    if rc == 124 or (rc == 137 and timeout > 0):
        state, rc = 'timeout', 124
    else:
        state = 'aborted' if rc < 0 or rc > 128 else 'error'
elif mode == 'att' and report.get('reason') == 'code_null' and int(attempts) > 1:
    report.update(reason='no_wave_on_target_cu',
                  hint='the traced CU ran no wave even with every SIMD selected (small grid); '
                       'launch at least one WG per CU or set KT_ATT_TARGET_CU to a CU that runs one')
result = {**report, 'state': state, 'rc': rc if rc else {'ok': 0, 'partial': 5, 'unavailable': 4}[state],
          'profiler_rc': int(sys.argv[2]), 'attempts': int(attempts), 'kernel': kernel, 'pmc': pmc,
          'timeout_s': timeout}
pathlib.Path(out, 'rocprofv3_safe.json').write_text(json.dumps(result, separators=(',', ':')))
PY_STATE
rc="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["rc"])' "$OUT/rocprofv3_safe.json")"
if [ "$rc" = 7 ]; then
  echo "[rocprofv3_safe] LLVM dual-load abort (two libamd_comgr.so.3 copies): an image problem, not the" \
       "kernel or the counter list. Fix: kernel_tools/profiling-dependencies.md \"Known abort\"." >&2
elif [ "$rc" != 0 ]; then
  echo "[rocprofv3_safe] collection incomplete; see $OUT/rocprofv3_safe.json" >&2
fi
exit "$rc"
