#!/usr/bin/env bash
# tile-programming: rocprof-compute per-round probe (the aggregate "which block is the
# ceiling / which memory level binds / occupancy-limiter" layer of the per-round record).
#
# rocprof-compute is the Nsight-Compute analog: ONE report gives System Speed-of-Light
# (block-level %-of-peak), the memory chart (per-cache hit/BW/latency), and the SPI /
# Workgroup Manager occupancy-limiter -- data that hand-picked `rocprofv3 --pmc` only
# covers partially. Use it per optimization round (not per micro-tweak: it is a MULTI-PASS
# replay, so it re-runs the app once per counter group).
#
# (GEAK shared kernel tool; the Gluon pack's scripts/rocprof_compute_probe.sh is a shim to this file.)
#
# Usage:
#   bash rocprof_compute_probe.sh <name> <out_dir> [--roof] [--full] [--kernel <substr>] [--dev N] -- <app cmd...>
#   bash rocprof_compute_probe.sh <name> <out_dir> [...] --cmd "<shell string>"          # run as bash -c
#   bash rocprof_compute_probe.sh <name> <out_dir> --workload <existing_workload_dir>   # analyze-only
#
# GEAK conventions: collection runs under gpu_lock.sh -- invoke it inside the lock, or pass --dev <id>
# and it re-execs itself under `gpu_lock.sh <id>`; with neither it refuses. The device is the one the
# lock exported (HIP_VISIBLE_DEVICES), never an inline override. A stale workloads/<name> is moved
# aside, never rm'd. Payload timeout: TILE_ROCPROF_PAYLOAD_TIMEOUT_S (default 900, 0 = none -- use 0
# for e2e serving payloads, whose startup + finalize are legitimately long).
#
# GOTCHAS baked in (learned on gfx942/MI325X):
#   * argv-JSON breaks: rocprof-compute re-execs the app command and strips the quotes in a
#     JSON argv -> pass a CONFIG-EMBEDDED wrapper script (no JSON on argv). See rc_bench*.py.
#   * deps: rocprof-compute needs its python reqs (plotext/astunparse/dash/...). If import
#     fails, run:  pip install -r "$(dirname "$(readlink -f "$(command -v rocprof-compute)")")/../libexec/rocprofiler-compute/requirements.txt"
#     (path varies; the tool prints the requirements.txt location on --version).
#   * MI200+ only; roofline (--roof) needs MI200+ and runs micro-benchmarks (extra time). On gfx95x
#     (gfx950 MI350X/MI355X) `--roof-only` needs rocprof-compute >= 3.6.0; and on gfx950 the
#     FETCH_SIZE / TCC_BUBBLE-derived read bytes UNDER-count, so a counter-based memory numerator
#     there is a lower bound (cross-check with parse_pmc's independent routes / an in-shape probe).
#   * cost control: filter to the dominant kernel; keep the app's rep count small.
#   * PMC blind (JIT kernels) / rocprof-compute absent (or 'Unsupported arch', e.g. RDNA4) -> DEGRADE
#     to GEAK's kernel_workflow/scripts/profile_kernel.sh <gpu> <cmd> <out> --pmc [--att]
#     (rocprofv3 counters + ATT) + the static ISA layer. This probe is the aggregate layer, not the only one.
#
# Emits: <out_dir>/rc_analyze.txt (blocks 2/3/6) and <out_dir>/rc_metrics.json (parsed SOL /
# memory / LDS / SPI), consumed by round_report.py.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
# Profiler runs WHERE the kernel runs (locus.sh): when TILE_KERNEL_CONTAINER is set the
# rocprof-compute COLLECTION runs in the container via docker exec. The output root must be shared
# and visible at the SAME path on host and container; locus_workdir_shared verifies both directions
# before collection, so an unshared mount is reported as a locus wiring fault rather than profile or
# analyze failure. Host-side passthrough (unchanged) applies with no container.
# _abspath_cmd0 <token>: echo the token abspathed IF it is a relative path containing '/' that
# exists; else echo it unchanged (absolute paths and bare PATH words pass through). The rule the
# 1d fix relies on -- factored out so --selftest can exercise it without a GPU.
_abspath_cmd0() {
  local t="$1"
  case "$t" in
    /*) printf '%s' "$t" ;;
    */*) if [ -e "$t" ]; then printf '%s/%s' "$(cd "$(dirname "$t")" && pwd)" "$(basename "$t")"; else printf '%s' "$t"; fi ;;
    *)  printf '%s' "$t" ;;
  esac
}

if [ "${1:-}" = "--selftest" ]; then
  fail=0
  tmp="$(mktemp -d)"; ( cd "$tmp" && mkdir -p sub && : > sub/app.sh )
  # a relative path containing '/' that exists -> abspathed
  got="$(cd "$tmp" && _abspath_cmd0 "sub/app.sh")"
  [ "$got" = "$tmp/sub/app.sh" ] || { echo "FAIL: relative path not abspathed ($got)"; fail=1; }
  # a bare word (PATH lookup) -> unchanged
  [ "$(_abspath_cmd0 python3)" = "python3" ] || { echo "FAIL: bare word mangled"; fail=1; }
  # an already-absolute path -> unchanged
  [ "$(_abspath_cmd0 /usr/bin/env)" = "/usr/bin/env" ] || { echo "FAIL: absolute path mangled"; fail=1; }
  # a relative path with '/' that does NOT exist -> left as-is (a container-only path)
  [ "$(_abspath_cmd0 no/such/app)" = "no/such/app" ] || { echo "FAIL: nonexistent rel path should pass through"; fail=1; }
  # GEAK conventions, checked on the shipped text: no inline device pin, no rm, gpu_lock routing.
  grep -qE 'env HIP_VISIBLE_DEVICE[S]=|export HIP_VISIBLE_DEVICE[S]=' "$SELF" \
    && { echo "FAIL: an inline HIP_VISIBLE_DEVICES pin remains"; fail=1; }
  grep -nE '^[^#]*\brm -(r|f|rf)\b' "$SELF" | grep -v 'grep -nE' >/dev/null \
    && { echo "FAIL: an rm remains (move aside instead)"; fail=1; }
  grep -q 'kt_ensure_lock' "$SELF" || { echo "FAIL: not routed through gpu_lock (kt_ensure_lock)"; fail=1; }
  [ "$fail" = 0 ] && echo "ROCPROF_COMPUTE_PROBE SELFTEST PASS" || echo "ROCPROF_COMPUTE_PROBE SELFTEST FAIL"
  exit "$fail"
fi

# shellcheck source=_kt_common.sh
source "$HERE/_kt_common.sh"   # gpu_lock routing, move-aside, and the pack's locus.sh when present

ORIG_ARGS=("$@")
NAME="${1:?usage: rocprof_compute_probe.sh <name> <out_dir> [opts] -- <cmd>}"; shift
OUT="${1:?usage: rocprof_compute_probe.sh <name> <out_dir> [opts] -- <cmd>}"; shift
ROOF="no"; KERNEL=""; DEV_ARG=""; WORKLOAD=""; FULL="no"; CMD=()
while [ $# -gt 0 ]; do
  case "$1" in
    --roof) ROOF="yes"; shift;;
    --full) FULL="yes"; shift;;
    --kernel) KERNEL="$2"; shift 2;;
    --dev) DEV_ARG="$2"; shift 2;;
    --workload) WORKLOAD="$2"; shift 2;;
    --cmd) CMD=(bash -c "$2"); shift 2;;
    --) shift; CMD=("$@"); break;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
# Collection needs the GPU lock; analyze-only (--workload) does not touch the GPU.
[ -n "$WORKLOAD" ] || kt_ensure_lock "$DEV_ARG" "$SELF" "${ORIG_ARGS[@]}"
DEV="${HIP_VISIBLE_DEVICES:-${DEV_ARG:-}}"   # the device gpu_lock.sh exported (recorded, not set)
mkdir -p "$OUT"
# Normalize a RELATIVE app path in CMD[0] to absolute BEFORE the profile stage runs inside
# `( cd "$OUT" ...)`. rocprof-compute re-execs the app command; a relative `./run_app.sh` (or
# `sub/dir/app`) would resolve against $OUT after the cd, not the caller's cwd -> "doesn't point to
# an executable" and the whole SOL layer is silently absent even on a PMC-live box. Rule (general):
# if CMD[0] contains a '/', is not already absolute, and exists here, abspath it; a bare word
# (`python`, `run_app.sh` on PATH) is left alone for PATH lookup. Mirrors the OUT abspath below.
[ ${#CMD[@]} -gt 0 ] && CMD[0]="$(_abspath_cmd0 "${CMD[0]}")"
# Normalize OUT to an absolute path BEFORE the profile stage runs inside `( cd "$OUT" ...)`: the
# `_rc_profile` redirect and every "$OUT/rc_profile.log" reference there would otherwise resolve
# against the already-cd'd cwd (relative $OUT -> $OUT/$OUT/...), which does not exist. That doubled
# path made the host shell emit a spurious `line NN: <path>: No such file or directory` every
# PMC-degraded round, and it displaced the real fall-back diagnostic in capture.sh's `tail -3`.
# `workloads/$NAME` intentionally stays relative (it rides the cd); only the $OUT-prefixed refs move.
OUT="$(cd "$OUT" && pwd)"
if locus_active; then
  : "${TILE_KERNEL_HOST_WORKDIR:=$OUT}"
  : "${TILE_KERNEL_CONTAINER_WORKDIR:=$OUT}"
  if ! locus_workdir_shared "$OUT"; then
    echo "[rocprof_compute_probe] profiler_locus_workdir_unshared: '$OUT' is not shared with the kernel container" >&2
    exit 3
  fi
fi
LOCUS_OUT="$(locus_path "$OUT")"
# --full surfaces the warp-state (7.2) / instr-mix (10) / TA-TD (15) / vL1D (16) / L2 (17)
# sections the decision tree needs, not just the aggregate 2/3/6. round_report.py --parse-rc
# extracts warp_state + instr_mix + coalescing from these extra blocks.
BLOCKS=(2 3 6)
[ "$FULL" = "yes" ] && BLOCKS=(2 3 6 7 10 11 15 16 17)

if ! locus_preflight; then
  echo "[rocprof_compute_probe] profiler_locus_mismatch: configured container is not a usable profiling locus. Do not classify this as a profile/analyze failure." >&2
  exit 3
fi
if locus_active && ! locus_workdir_shared "$OUT"; then
  echo "[rocprof_compute_probe] profiler_locus_workdir_unshared: '$OUT' is not readable and "
  echo "  writable from both host and kernel container. Collection/analyze were not attempted." >&2
  exit 3
fi

# Both checks must ask the LOCUS: the collection below is locus_run, so a host-side probe answers
# for the wrong machine and sends a container round down the "tool absent" fallback while the tool
# is installed one `docker exec` away.
if ! locus_have rocprof-compute; then
  # `${V:+container $V}${V:-host}` expanded BOTH arms when V was set -- ":-" yields the value, not the
  # default -- so a configured container was reported as "container NAMENAME". Keep the variable in
  # exactly one arm: this message's only job is to say WHICH machine was asked, and a mangled name is
  # how you end up debugging the wrong host.
  echo "[rocprof_compute_probe] rocprof-compute NOT found in the kernel's locus (${TILE_KERNEL_CONTAINER:+container }${TILE_KERNEL_CONTAINER:-host}) -> DEGRADE to kernel_workflow/scripts/profile_kernel.sh <gpu> <cmd> <out> --pmc [--att] (rocprofv3 counters + ATT) + static ISA." >&2
  exit 3
fi
if ! locus_run rocprof-compute --version >/dev/null 2>&1; then
  echo "[rocprof_compute_probe] rocprof-compute present but its python deps are missing." >&2
  echo "  Install: pip install -r <libexec>/rocprofiler-compute/requirements.txt  (see 'rocprof-compute --version' output)." >&2
  exit 3
fi

if [ -z "$WORKLOAD" ]; then
  [ ${#CMD[@]} -gt 0 ] || { echo "no app cmd after --" >&2; exit 2; }
  # GOTCHA GUARD (argv-JSON quote-strip): rocprof-compute is a MULTI-PASS profiler -- it re-execs the
  # app command once per counter group, and that re-exec drops the quotes around a JSON object passed
  # on argv. A token like  --cfg {"BLOCK_M":128}  reaches the app as  {BLOCK_M:128}  on the replay,
  # so the app's json.loads() raises and every pass produces "Profiling execution failed / No
  # profiling data" -> the whole SOL layer is silently lost for the round. This is the #1 documented
  # trap (see header) and it fired unguarded in the field. Detect a bare JSON-object argv token and
  # FAIL FAST with the fix, instead of burning a multi-pass replay on a command that cannot survive it.
  for _tok in "${CMD[@]}"; do
    case "$_tok" in
      '{'*'"'*'}'*|'{'*'"'*)
        echo "[rocprof_compute_probe] REFUSING: a raw JSON object is on the app argv: ${_tok:0:60}..." >&2
        echo "  rocprof-compute re-execs the app per counter-group and STRIPS the quotes -> the app's" >&2
        echo "  json.loads() will fail on every pass ('No profiling data'). Pass the config WITHOUT" >&2
        echo "  argv-JSON: (a) a config-EMBEDDED wrapper script (bake the dict into a .py, no --cfg)," >&2
        echo "  or (b) have the driver read --config-file <path> / an env var. Then re-run this probe." >&2
        echo "  (ATT + static ISA are unaffected; only the rocprof-compute SOL layer needs this.)" >&2
        exit 5;;
    esac
  done
  # profile-time -k is a NAME filter (substring) - collect ONLY the kernel of interest so the
  # later analyze -k 0 selects it (analyze's -k is a different, index-based flag).
  KPROF=(); [ -n "$KERNEL" ] && KPROF=(-k "$KERNEL")

  # RELATIVE PATHS ON THE APP ARGV DO NOT SURVIVE THIS TOOL.
  #
  # The collection runs from inside $OUT (below) so `workloads/$NAME` lands on the shared root, and
  # the profiler additionally chdir's into its own output directory between passes. Either one
  # breaks a relative path on the app command, and the failure surfaces as the profiler's generic
  # "Profiling execution failed / No profiling data" -- indistinguishable from a PMC-blind box, and
  # it has been read as one. This is a property of the TOOL, not of any host, so absolutising is
  # unconditional: resolve every argv token that names an existing path while we are still in the
  # caller's cwd. Tokens that are not paths are passed through untouched.
  _ABS_CMD=()
  for _tok in "${CMD[@]}"; do
    if [ -e "$_tok" ]; then
      _ABS_CMD+=("$(cd "$(dirname -- "$_tok")" && pwd)/$(basename -- "$_tok")")
    else
      _ABS_CMD+=("$_tok")
    fi
  done
  CMD=("${_ABS_CMD[@]}")

  # DEVICE SELECTION: ONE MECHANISM, NOT TWO.
  #
  # `HIP_VISIBLE_DEVICES`, exported by gpu_lock.sh around this whole script, already pins the device
  # for the app and every replay pass (a container locus forwards it via LOCUS_ENV). Passing
  # the same constraint a second time through the profiler's own `--device` is redundant, and on at
  # least one toolchain version the redundant flag fails the run with a library-path error while the
  # environment variable alone works. Redundant constraints cannot be verified against each other,
  # so the default is the single mechanism. The flag remains available for a caller that has
  # measured it to be necessary here, and then the choice is recorded rather than assumed.
  RC_DEVICE_ARGS=()
  if [ "${TILE_ROCPROF_DEVICE_FLAG:-0}" = "1" ]; then
    RC_DEVICE_ARGS=(--device "$DEV")
  fi
  echo "device_selection=$([ ${#RC_DEVICE_ARGS[@]} -eq 0 ] && echo HIP_VISIBLE_DEVICES || echo 'HIP_VISIBLE_DEVICES+--device (TILE_ROCPROF_DEVICE_FLAG=1)')" \
    > "$OUT/rc_device_selection.txt"
  # multi-pass replay is occasionally flaky back-to-back ("Profiling execution failed" /
  # "No profiling data") -> settle + a single retry before giving up.
  # run the COLLECTION in the kernel's locus (docker exec when a container is configured); the
  # workdir is $OUT so `workloads/$NAME` lands on the shared working root (bind mount).
  export TILE_KERNEL_CONTAINER_CWD="$LOCUS_OUT"
  # Payload timeout (0 = none): e2e serving payloads must run with 0 -- their startup and the
  # profiler's finalize are legitimately long and a kernel-sized deadline kills a healthy collection.
  _rc_run() {
    if [ "${TILE_ROCPROF_PAYLOAD_TIMEOUT_S:-900}" = "0" ]; then locus_run "$@"
    else locus_timeout "${TILE_ROCPROF_PAYLOAD_TIMEOUT_S:-900}" "$@"; fi
  }
  _rc_profile() {
    _rc_run rocprof-compute profile -n "$NAME" --no-roof "${RC_DEVICE_ARGS[@]}" "${KPROF[@]}" -- "${CMD[@]}" \
      > "$OUT/rc_profile.log" 2>&1
  }
  ( cd "$OUT" && kt_move_aside "workloads/$NAME"
    echo "--- rocprof-compute profile (--no-roof, multi-pass${KERNEL:+, kernel~=$KERNEL}) ---"
    if ! _rc_profile || grep -qiE "Profiling execution failed|No profiling data" "$OUT/rc_profile.log"; then
      echo "--- profile flaky -> settle 3s + single retry ---" | tee -a "$OUT/rc_profile.log" >&2
      sleep 3; kt_move_aside "workloads/$NAME"
      if ! _rc_profile || grep -qiE "Profiling execution failed|No profiling data" "$OUT/rc_profile.log"; then
        if [ -s "$OUT/rc_profile.log" ]; then
          echo "profile failed (after 1 retry); tail of $OUT/rc_profile.log:" >&2; tail -5 "$OUT/rc_profile.log" >&2
        else
          # arm-A F6: under a container locus the log can be written container-side and not fetched,
          # so do not point at a path that may not exist here. Say what actually happened.
          echo "profile failed (after 1 retry) and no readable rc_profile.log on this side -> rocprof-compute likely ran in the container locus (analyze/profile ROCm-version or workdir mismatch). FIX the locus first (gluon_authoring references/method/profile.md ### Execution locus) -- SOL is stable + parallel-safe when wired right and worth collecting; fall back to ATT + static only if this box is genuinely PMC-blind." >&2
        fi
        exit 4
      fi
    fi
    if [ "$ROOF" = "yes" ]; then
      echo "--- rocprof-compute profile (--roof-only, empirical hierarchical roofline) ---"
      # gfx95x: --roof-only needs rocprof-compute >= 3.6.0; older builds fail or mis-roof here.
      _rc_run rocprof-compute profile -n "$NAME" --roof-only "${RC_DEVICE_ARGS[@]}" -- "${CMD[@]}" \
        >> "$OUT/rc_profile.log" 2>&1 || echo "roof-only failed (non-fatal; gfx95x needs rocprof-compute >= 3.6.0); see log" >&2
    fi
  )
  locus_fetch "$LOCUS_OUT/workloads" "$OUT"   # noop host-side / tolerant when bind-mounted
  # COLLECTION ran in the container as root; ANALYZE runs here as the host uid. Over a bind mount
  # that is the same inode with a different caller, and rocprof-compute's analyze WRITES into the
  # workload dir (it re-derives pmc_dispatch_info.csv), so a `dr-xr-x---` root-owned tree fails it
  # with PermissionError -- after the expensive multi-pass replay has already succeeded. Widen the
  # tree from inside the locus, where we are the owner. Best-effort on both sides: a host-side
  # chmod is enough when the run was host-side or the uids already match.
  locus_run chmod -R a+rwX "$LOCUS_OUT/workloads/$NAME" >/dev/null 2>&1 || true
  chmod -R a+rwX "$OUT/workloads/$NAME" >/dev/null 2>&1 || true
  WORKLOAD="$(ls -d "$OUT"/workloads/"$NAME"/*/ 2>/dev/null | head -1)"
fi
[ -n "$WORKLOAD" ] || { echo "no workload dir found under $OUT/workloads/$NAME" >&2; exit 4; }

echo "--- rocprof-compute analyze (blocks ${BLOCKS[*]}) ---"
# rocprof-compute analyze -k/--kernel wants an INTEGER dispatch index (NOT a name substring).
# A name filter belongs at profile time; for analyze we take the top kernel (-k 0) unless an
# integer index was passed.
KF=(-k 0)
if [ -n "$KERNEL" ] && [[ "$KERNEL" =~ ^[0-9]+$ ]]; then KF=(-k "$KERNEL"); fi
if ! rocprof-compute analyze -p "$WORKLOAD" "${KF[@]}" --block "${BLOCKS[@]}" > "$OUT/rc_analyze.txt" 2>&1; then
  # A permission failure and an empty profile both used to exit 4 with one message, so the
  # recurring, FIXABLE one read as "rocprof-compute is broken in this container" and every reader
  # downstream accepted a dark SOL layer as a property of the box. Name which one happened.
  if grep -qi "PermissionError\|Permission denied" "$OUT/rc_analyze.txt"; then
    echo "analyze FAILED on permissions, not on data: the profile succeeded and the counters are" >&2
    echo "  on disk, but $WORKLOAD is not writable by this uid ($(id -u)). analyze writes into the" >&2
    echo "  workload dir, so a root-owned collection blocks it. This script already widens the tree" >&2
    echo "  after collection; if you are seeing this, that chmod could not reach it either. FIX:" >&2
    echo "    chmod -R a+rwX '$OUT/workloads/$NAME'   (as root / inside the container), then re-run" >&2
    echo "    THIS SCRIPT with --workload '$WORKLOAD' -- do NOT hand-run analyze into a new file" >&2
    echo "    name, because rc_metrics.json is only built from the path this script writes." >&2
    echo "  This is a wiring bug to fix, NOT a PMC-blind box: do not fall back to ATT on it." >&2
    exit 5
  fi
  echo "analyze failed with no usable profile (not a permission problem); see $OUT/rc_analyze.txt" >&2
  tail -5 "$OUT/rc_analyze.txt" >&2
  exit 4
fi

echo "--- parse SOL/memory/LDS/SPI -> rc_metrics.json ---"
# parse_rc.py refuses to write a zero-row metrics file and searches rc_analyze* siblings, so a
# by-hand recovery that landed next door is still picked up rather than silently ignored.
python3 "$HERE/parse_rc.py" "$OUT/rc_analyze.txt" --out-json "$OUT/rc_metrics.json" \
  || echo "parse produced no metrics (rc_analyze.txt still available for a human read)" >&2

echo "=== done -> $OUT (rc_analyze.txt, rc_metrics.json). Feed to round_report.py --rc-metrics ==="
