#!/usr/bin/env bash
# capture.sh - one-click evidence capture (joint 1 of the decision pipeline).
#
# preflight (is the kernel PMC-visible?) -> branch:
#   PMC-live   : rocprof-compute --full (SOL/warp-state/mem/instr-mix) + rocprofv3 --att
#                (auto-widen CU/SE, retry on code:null) + dump_ir.sh
#   PMC-blind  : native-ISA degrade (dump_ir.sh .s + note the floor-probe path)
#
# (GEAK shared kernel tool; the Gluon pack's scripts/capture.sh is a shim to this file.)
#
# Usage:
#   bash capture.sh <name> <out_dir> --kernel <substr> --source <kernel.py> [--dev N] [--no-att] -- <app cmd...>
#   bash capture.sh <name> <out_dir> --kernel <substr> --source <kernel.py> [...] --cmd "<shell string>"
#
# GEAK conventions: GPU collection runs under gpu_lock.sh -- invoke capture.sh inside the lock, or pass
# --dev <id> and it re-execs itself under `gpu_lock.sh <id>` (refuses with neither; --isa-only needs no
# lock). The device is the one the lock exported -- no inline HIP_VISIBLE_DEVICES. Stale att/ and kt/
# dirs are moved aside, never rm'd. A `bash -c "<string>"` command is accepted (`bash -lc` is not: a
# login shell re-sources profiles and can drop the profiler's injected environment).
# ATT needs the rocprof-trace-decoder library, which GEAK does not vendor: point
# ROCPROF_ATT_LIBRARY_PATH at its directory. Without it the ATT layer is skipped and recorded as
# DEGRADED (capture.json att_state=decoder_absent); every other layer still runs.
#
# Emits into <out_dir>/: preflight.txt, rc/ (rc_analyze.txt + rc_metrics.json), att/ (ui_output),
# ir/ (.amdgcn), and capture.json (paths + which branch was taken). Feeds normalize.py.
#
# Exit: 0 captured (possibly on a degraded branch, recorded in capture.json) | 2 usage
#       4 locus unusable -- a container is configured but the profiler cannot run in it. Fail-closed
#         on purpose: the caller must NOT treat this as a degraded capture, because no artifact was
#         written and anything collected host-side would not describe the kernel under test.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Shared host<->container ownership: never write root-owned .pyc into the bind-mounted skill dir,
# and make artifacts group-writable so the host uid can edit what the (root) container wrote. See
# gluon_authoring references/method/profile.md ### Execution locus (shared ownership).
export PYTHONDONTWRITEBYTECODE=1
umask 0002
# Profiler runs WHERE the kernel runs: when TILE_KERNEL_CONTAINER is set, locus_run wraps the
# profiler+app in `docker exec` (kernel is in the container) and locus_fetch copies artifacts out;
# host-side passthrough (unchanged) when no container is configured. See the pack's scripts/locus.sh
# (STANDALONE runs only; an EMBEDDED GEAK run is already inside its container and gets passthroughs).
# --selftest: structural checks (no GPU/docker, runs BEFORE sourcing locus.sh so it works pre-compose
# too). Asserts the static-ISA dump is routed through the locus (never a bare host-side
# `bash dump_ir.sh`, which dies ModuleNotFoundError: triton in a container locus) and that env
# forwarding is wired. Keeps the 1a+1b fix from silently regressing.
if [ "${1:-}" = "--help" ]; then
  cat <<'USAGE'
Usage:
  capture.sh --payload profile_payload.json
  capture.sh <name> <out_dir> --kernel <device-kernel-substring> --source <kernel.py>
             [--dev <gpu-id>] [--no-att|--lite|--isa-only] -- <application command...>
  capture.sh ... --cmd "<shell string>"        (run as bash -c "<string>")

Run it under GEAK's gpu_lock.sh, or pass --dev <id> and it locks itself (it refuses with neither).

The source file is verified and hashed in the execution locus. Its identity and the requested
device-kernel substring are recorded in capture.json; callers must recapture after changing source.
`TILE_KERNEL_FILE` supplies the same source identity for pack-managed round runners.
Fleet runs use the JSON payload form; the legacy command form cannot carry a shell-wrapped profiler
payload and will be removed after downstream callers migrate.
USAGE
  exit 0
fi

if [ "${1:-}" = "--selftest" ]; then
  fail=0
  self="$HERE/$(basename "${BASH_SOURCE[0]}")"
  grep -qE '_run_dump_ir|locus_run bash "\$script"' "$self" || { echo "FAIL: dump_ir not routed through locus"; fail=1; }
  if grep -nE '^\s*\(?\s*(TRITON_ALWAYS_COMPILE=1 )?bash "\$HERE/dump_ir.sh"' "$self" >/dev/null; then
    echo "FAIL: a bare host-side 'bash \$HERE/dump_ir.sh' remains"; fail=1
  fi
  grep -q 'export LOCUS_ENV=' "$self" || { echo "FAIL: LOCUS_ENV not exported for env forwarding"; fail=1; }
  # GEAK conventions: the device comes from gpu_lock.sh (no inline pin), collection is routed through
  # the lock, and nothing is rm'd (stale dirs are moved aside).
  grep -qE 'export HIP_VISIBLE_DEVICE[S]=' "$self" && { echo "FAIL: an inline HIP_VISIBLE_DEVICES pin remains"; fail=1; }
  grep -q 'kt_ensure_lock' "$self" || { echo "FAIL: collection not routed through gpu_lock (kt_ensure_lock)"; fail=1; }
  grep -nE '^[^#]*[^a-z_]rm -(r|f|rf) ' "$self" | grep -v 'grep -nE' >/dev/null \
    && { echo "FAIL: an rm remains (move aside instead)"; fail=1; }
  # profiler availability must be asked of the LOCUS, not the host: a host-side `command -v` says
  # PMC-blind while a working rocprofv3 sits in the container, and the round degrades for nothing.
  grep -q 'locus_have rocprofv3' "$self" || { echo "FAIL: rocprofv3 probe not routed through the locus"; fail=1; }
  # the [v] class stops this guard from matching its own pattern line
  if grep -nE 'command -[v] rocprofv3' "$self" >/dev/null; then
    echo "FAIL: a host-side rocprofv3 probe remains (must go through locus_have)"; fail=1
  fi
  # a configured-but-broken locus must ABORT, never fall through and file host data as container data
  grep -qE '^\s*exit 4$' "$self" || { echo "FAIL: locus preflight failure is not fail-closed (no exit)"; fail=1; }

  # capture.json's SOL verdict, run for real against artifacts on disk. `rc_metrics: null` used to
  # carry three different facts at once -- never attempted, crashed, collected-but-unparsed -- and
  # they call for opposite responses, so each is asserted separately rather than grepped for.
  # The emitter is extracted from its own heredoc so the test runs the shipped code, not a copy.
  t="$(mktemp -d)"; mkdir -p "$t/rc"
  sed -n '/^python3 - "\$OUT" "\$NAME" "\$KERNEL" "\$BRANCH" "\$SOURCE" "\$SOURCE_SHA" "\$DEV" "\$COMMAND_SHA" <<.PY.$/,/^PY$/p' "$self" \
    | sed '1d;$d' > "$t/emit.py"
  _sol(){ ( cd "$t" && python3 emit.py "$t" n k b s sh d ch >/dev/null 2>&1 ) \
            && python3 -c "import json,sys; print(json.load(open('$t/capture.json'))['sol_state'])"; }
  [ "$(_sol)" = "not_attempted" ] || { echo "FAIL: no probe at all should read not_attempted, got $(_sol)"; fail=1; }
  printf 'Traceback (most recent call last):\nPermissionError: [Errno 13]\n' > "$t/rc/rc_analyze.txt"
  [ "$(_sol)" = "crashed" ] || { echo "FAIL: a traceback should read crashed, got $(_sol)"; fail=1; }
  printf 'System Speed-of-Light\nDependency-Wait 47\n' > "$t/rc/rc_analyze_fixed.txt"
  [ "$(_sol)" = "collected_unparsed" ] \
    || { echo "FAIL: SOL text beside a crash should read collected_unparsed, got $(_sol)"; fail=1; }
  echo '{}' > "$t/rc/rc_metrics.json"
  [ "$(_sol)" = "parsed" ] || { echo "FAIL: parsed metrics should read parsed, got $(_sol)"; fail=1; }
  # ATT decoder detection degrades rather than fails: an empty search path reports absent.
  ( source "$HERE/_kt_common.sh"; ROCPROF_ATT_LIBRARY_PATH="$t/none" KT_ATT_SYSTEM_DIRS="$t/none" kt_att_decoder_dir ) >/dev/null 2>&1 \
    && { echo "FAIL: ATT decoder reported present in an empty search path"; fail=1; }
  mkdir -p "$t/dec"; : > "$t/dec/librocprof-trace-decoder.so"
  [ "$( source "$HERE/_kt_common.sh"; ROCPROF_ATT_LIBRARY_PATH="$t/dec" kt_att_decoder_dir )" = "$t/dec" ] \
    || { echo "FAIL: ATT decoder in ROCPROF_ATT_LIBRARY_PATH not found"; fail=1; }

  [ "$fail" = 0 ] && echo "CAPTURE SELFTEST PASS" || echo "CAPTURE SELFTEST FAIL"
  exit "$fail"
fi

# shellcheck source=_kt_common.sh
source "$HERE/_kt_common.sh"   # gpu_lock routing, move-aside, ATT decoder probe, pack locus.sh if present
if ! locus_preflight; then
  echo "[capture] LOCUS PREFLIGHT FAILED (${LOCUS_DEGRADE_CAUSE:-?}): the task specifies a container but the profiler cannot run in it -> FIX the locus or escalate (gluon_authoring references/method/triage.md, 'Profiler layer unavailable')." >&2
  # Fail CLOSED (exit 4). Continuing would run every layer below on the HOST and write the results
  # into the same capture.json the container round reads -- host evidence filed under a container
  # kernel, which is worse than no evidence because nothing downstream can tell the difference.
  echo "[capture] REFUSING to continue: host-collected data would be filed as in-container evidence. Fix TILE_KERNEL_CONTAINER (live container + profiler inside it), or unset it to profile the host deliberately." >&2
  exit 4
fi

if [ "${1:-}" = "--payload" ]; then
  PAYLOAD="${2:-}"
  [ -n "$PAYLOAD" ] || { echo "capture: --payload needs a path" >&2; exit 2; }
  shift 2
  [ "$#" = 0 ] || { echo "capture: --payload cannot be combined with legacy arguments" >&2; exit 2; }
  # profile_payload.py is a pack runtime tool (it stays in gluon_authoring/scripts).
  [ -f "$KT_PACK_DIR/scripts/profile_payload.py" ] || {
    echo "[capture] --payload needs the Gluon pack's scripts/profile_payload.py, not found under" \
         "$KT_PACK_DIR (set GEAK_GLUON_PACK_DIR). Use the positional form instead:" \
         "capture.sh <name> <out_dir> --kernel <substr> --source <kernel.py> -- <cmd>" >&2; exit 2; }
  python3 "$KT_PACK_DIR/scripts/profile_payload.py" validate --payload "$PAYLOAD" >/dev/null || {
    echo "[capture] invalid profile payload: $PAYLOAD" >&2; exit 2;
  }
  mapfile -d '' -t PAYLOAD_FIELDS < <(python3 - "$PAYLOAD" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
e, o = p["execution"], p["outputs"]
selector = (p.get("kernel_selector") or {}).get("regex", "")
for value in (p["source"]["path"], p["device"]["logical"], o["root"], selector,
              e["cwd"]["host"], e["cwd"]["container"]):
    sys.stdout.write(str(value) + "\0")
for value in e["argv"]:
    sys.stdout.write(value + "\0")
PY
)
  [ "${#PAYLOAD_FIELDS[@]}" -ge 7 ] || { echo "capture: payload omitted execution argv" >&2; exit 2; }
  export TILE_KERNEL_HOST_WORKDIR="${PAYLOAD_FIELDS[4]}"
  export TILE_KERNEL_CONTAINER_WORKDIR="${PAYLOAD_FIELDS[5]}"
  PAYLOAD_CMD=("${PAYLOAD_FIELDS[@]:6}")
  PAYLOAD_NAME="$(basename "${PAYLOAD_FIELDS[2]}")"
  set -- "$PAYLOAD_NAME" "${PAYLOAD_FIELDS[2]}" --kernel "${PAYLOAD_FIELDS[3]:-$PAYLOAD_NAME}" \
    --source "${PAYLOAD_FIELDS[0]}" --dev "${PAYLOAD_FIELDS[1]}" -- "${PAYLOAD_CMD[@]}"
fi

ORIG_ARGS=("$@")
NAME="${1:?usage: capture.sh <name> <out_dir> --kernel <substr> --source <kernel.py> -- <cmd>}"; shift
OUT="${1:?usage: capture.sh <name> <out_dir> --kernel <substr> --source <kernel.py> -- <cmd>}"; shift
KERNEL=""; SOURCE=""; DEV_ARG=""; WANT_ATT="yes"; LITE="no"; ISA_ONLY="no"; CMD=()
while [ $# -gt 0 ]; do
  case "$1" in
    --kernel) KERNEL="$2"; shift 2;;
    --source) SOURCE="$2"; shift 2;;
    --dev) DEV_ARG="$2"; shift 2;;
    --cmd) CMD=(bash -c "$2"); shift 2;;
    --no-att) WANT_ATT="no"; shift;;
    --lite) LITE="yes"; WANT_ATT="no"; shift;;   # fast: kernel-trace timing + static ISA only
    # --isa-only is the MANDATORY per-round tier: dump the IR and audit the ISA, nothing else. No
    # profiler runs at all, so it costs seconds and cannot hang. A tiered read is only honest if
    # its bottom tier is genuinely cheap enough to run consistently; a mandatory layer that is
    # routinely skipped is not a usable evidence floor.
    --isa-only) ISA_ONLY="yes"; LITE="yes"; WANT_ATT="no"; shift;;
    --) shift; CMD=("$@"); break;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ ${#CMD[@]} -gt 0 ] || { echo "no app cmd after --" >&2; exit 2; }
# Profiled tiers need the GPU lock; --isa-only only compiles and audits (dump_ir takes its own lock
# discipline from the caller), so it may run outside it.
if [ "$ISA_ONLY" != "yes" ]; then
  if [ -n "${PAYLOAD:-}" ]; then
    kt_ensure_lock "$DEV_ARG" "$HERE/capture.sh" --payload "$PAYLOAD"
  else
    kt_ensure_lock "$DEV_ARG" "$HERE/capture.sh" "${ORIG_ARGS[@]}"
  fi
fi
DEV="${HIP_VISIBLE_DEVICES:-${DEV_ARG:-}}"   # the device gpu_lock.sh exported (recorded, not set)
if [ "${CMD[0]}" = "bash" ] || [ "${CMD[0]}" = "sh" ]; then
  [ "${CMD[1]:-}" != "-lc" ] || {
    echo "[capture] REFUSING shell-wrapped payload: profiler replay cannot attach reliably through bash -lc; use --payload with argv + config-file." >&2
    exit 5
  }
fi
for token in "${CMD[@]}"; do
  case "$token" in
    '{'*'}') echo "[capture] REFUSING inline JSON argv; pass an immutable config file through --payload." >&2; exit 5 ;;
  esac
done
[ -n "$SOURCE" ] || SOURCE="${TILE_KERNEL_FILE:-}"
[ -n "$SOURCE" ] || { echo "--source is required" >&2; exit 2; }
mkdir -p "$OUT" "$OUT/rc" "$OUT/att" "$OUT/ir"
OUT="$(cd "$OUT" && pwd -P)"
# Keep host/container path translation separate from the process CWD. Older
# same-path mounts keep working; fleet runtime_env.sh supplies both roots for
# differently-mounted workdirs.
if locus_active; then
  : "${TILE_KERNEL_HOST_WORKDIR:=$OUT}"
  : "${TILE_KERNEL_CONTAINER_WORKDIR:=$OUT}"
  if ! locus_workdir_shared "$OUT"; then
    echo "[capture] profiler_locus_workdir_unshared: '$OUT' is not shared with the kernel container" >&2
    exit 4
  fi
fi
LOCUS_OUT="$(locus_path "$OUT")"
export TILE_KERNEL_CONTAINER_CWD="$LOCUS_OUT"
: "${KERNEL:=$NAME}"

# Under a container locus, docker exec inherits none of the host env: locus_run forwards the device
# gpu_lock.sh exported (HIP_VISIBLE_DEVICES) + the Triton cache/import env into the container (see the
# pack's locus.sh LOCUS_ENV). Without this the profiler + dump_ir run on GPU 0 and against a
# stale/absent cache. A caller that selects a source variant via its own env var appends it upstream.
export LOCUS_ENV="${LOCUS_ENV:-} HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES TRITON_CACHE_DIR TRITON_ALWAYS_COMPILE PYTHONPATH"
LOCUS_SOURCE="$(locus_path "$SOURCE")"
if ! locus_run test -f "$LOCUS_SOURCE"; then
  echo "[capture] source is not readable in the execution locus: $SOURCE" >&2
  exit 2
fi
SOURCE_SHA="$(locus_run sha256sum "$LOCUS_SOURCE" | awk '{print $1}')"
[ -n "$SOURCE_SHA" ] || { echo "[capture] could not hash --source in the execution locus" >&2; exit 2; }
COMMAND_SHA="$(printf '%s\0' "${CMD[@]}" | sha256sum | awk '{print $1}')"

# preflight.txt is the ONLY capture artifact a worker reads as narration (no parser consumes it;
# the machine-readable outputs are capture.json / rc_metrics.json / mfma_eff.txt / asm_audit.txt).
# rocprofv3 --att prints ~100 lines of internal chatter per invocation (simple_timer, "Opened
# result file", "SQLite3 generation", HSA/ldconfig init) that carried ~84% of preflight.txt straight
# into the worker's context every round. Route the raw ATT stream to a side log kept on disk for
# debugging, and fold only the SIGNAL lines (real warnings/errors + trace-quality notices) into
# preflight.txt. CAPTURE_VERBOSE=1 restores the full inline dump.
: "${CAPTURE_VERBOSE:=0}"
ATT_RAW_LOG="$OUT/att/att_run.log"
# fold_att_signal <raw-log-file>: append only worker-relevant lines from a raw rocprofv3 stream to
# preflight.txt (dropping the pure-noise families), unless CAPTURE_VERBOSE=1.
fold_att_signal() {
  local raw="$1"
  [ -f "$raw" ] || return 0
  if [ "$CAPTURE_VERBOSE" = "1" ]; then
    cat "$raw" >>"$OUT/preflight.txt"; return 0
  fi
  # keep: capture-run errors, trace-quality warnings (Wave incomplete / Data Lost), the app's own
  # stdout (e.g. aiter import line, Time_(ms)); drop: rocprofv3 timers/file-opens/sqlite/init noise.
  grep -vE 'simple_timer\.cpp|output_stream\.cpp:[0-9]+\] Opened result file|SQLite3 generation ::|generateRocpd\.cpp|tool\.cpp:[0-9]+\] HSA version|/sbin/ldconfig' "$raw" \
    >>"$OUT/preflight.txt" 2>/dev/null || true
  local raw_n
  raw_n=$(wc -l <"$raw" 2>/dev/null || echo 0)
  echo "[capture] (ATT profiler chatter: $raw_n lines -> filtered to signal only; full log in att/$(basename "$raw"), set CAPTURE_VERBOSE=1 to inline)" >>"$OUT/preflight.txt"
}

echo "=== [capture] preflight: is kernel '$KERNEL' PMC-visible? ===" | tee "$OUT/preflight.txt"
PMC_LIVE="no"
# --isa-only skips the probe entirely. The probe is a full profiled run of the app, so paying for
# it in the tier whose whole purpose is to be too cheap to skip would defeat the tier.
if [ "$ISA_ONLY" = "yes" ]; then
  echo "[capture] --isa-only: skipping the PMC-visibility probe (no profiler runs in this tier)" \
    | tee -a "$OUT/preflight.txt"
# Probe the profiler IN THE LOCUS: the tool has to exist where the kernel runs, and under a
# container locus the host's answer is unrelated to that. See locus.sh locus_have.
elif locus_have rocprofv3; then
  kt_move_aside "$OUT/kt"
  # F12: keep the probe's own stdout/stderr instead of discarding it, so an APP crash (which leaves
  # no trace file) is not later misreported as a wrong --kernel filter. See the no-trace branch below.
  locus_run rocprofv3 --kernel-trace -f csv -d "$LOCUS_OUT/kt" -- "${CMD[@]}" >"$OUT/kt_probe.log" 2>&1 || true
  locus_fetch "$LOCUS_OUT/kt" "$OUT"
  if ! find "$OUT/kt" -name '*kernel_trace.csv' 2>/dev/null | grep -q .; then
    echo "[capture] no kernel_trace file produced -> the app/profiler FAILED before any GPU dispatch (this is NOT a --kernel mismatch). Tail of the probe log:" | tee -a "$OUT/preflight.txt"
    tail -8 "$OUT/kt_probe.log" 2>/dev/null | sed 's/^/    /' | tee -a "$OUT/preflight.txt"
  fi
  # grep -rc always prefixes 'file:count' (xargs grep -c on a single file omits it)
  N=$(grep -rc "$KERNEL" "$OUT/kt" 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
  echo "kernel-trace dispatch count for '$KERNEL' = ${N:-0}" | tee -a "$OUT/preflight.txt"
  if [ "${N:-0}" -gt 0 ]; then
    PMC_LIVE="yes"
  elif ! find "$OUT/kt" -name '*kernel_trace.csv' 2>/dev/null | grep -q .; then
    # (case a) no trace file -> already reported above as an app/profiler failure; do NOT print the
    # candidate-kernel listing, which would falsely blame the --kernel filter.
    :
  else
    # (case c) trace exists but 0 rows match the substring. `--kernel` must match the EMITTED DEVICE
    # name, not the wrapper/op name. List the actual device kernels (torch/reference helpers filtered) and
    # AUTO-RETRY with the top non-helper candidate so a near-miss substring self-corrects.
    echo "[capture] '$KERNEL' matched 0 dispatches -> candidate device kernels (torch/reference helpers excluded); listing:" | tee -a "$OUT/preflight.txt"
    TOPCAND="$(python3 - "$OUT/kt" 2>>"$OUT/preflight.txt" <<'PY'
import csv, glob, os, sys, collections, re
c = collections.Counter()
for f in glob.glob(os.path.join(sys.argv[1], "**", "*kernel_trace.csv"), recursive=True):
    try:
        for r in csv.DictReader(open(f)):
            n = (r.get("Kernel_Name") or r.get("kernel_name") or "").strip()
            if n:
                c[n[:120]] += 1
    except OSError:
        pass
# torch/aten/library helper kernels pollute the trace (reference/oracle calls) - exclude from the pick
def is_helper(n):
    nl = n.lower()
    # plain substrings that never collide with a real Triton/Gluon device-kernel name
    if any(t in nl for t in ("at::", "aten", "elementwise_kernel", "vectorized_elementwise",
                             "reduce_kernel", "cutlass", "gemm_universal", "cublas", "hipblas",
                             "rocblas", "_copy_kernel", "fill_kernel", "cast_kernel")):
        return True
    # 'ck_' (Composable Kernel) MUST be a word-boundary prefix, not a bare substring: a Triton kernel
    # named '..._kernel_BLOCK...' contains the substring 'ck_' (kernel_) and was wrongly flagged as a
    # CK helper -> the auto-retry then SKIPPED the real target and the round silently lost SOL+ATT.
    # Match CK only at a name start or after a non-alnum separator (ck_gemm, _ck_..., /ck_...).
    return re.search(r"(?:^|[^a-z0-9])ck_", nl) is not None
listing = c.most_common(12)
for n, k in listing:
    tag = "  [helper]" if is_helper(n) else ""
    print(f"    {k:6d} x  {n}{tag}", file=sys.stderr)
if not c:
    print("    (no kernel_trace rows parsed - is the app dispatching a GPU kernel?)", file=sys.stderr)
cand = [(n, k) for n, k in c.most_common() if not is_helper(n)]
if cand:
    print(cand[0][0])   # stdout = the top NON-helper device kernel name (for auto-retry)
PY
)"
    if [ -n "$TOPCAND" ] && [ "$TOPCAND" != "$KERNEL" ]; then
      echo "[capture] auto-retry with top non-helper device kernel: '$TOPCAND' (was '$KERNEL')" | tee -a "$OUT/preflight.txt"
      KERNEL="$TOPCAND"
      N=$(grep -rc "$KERNEL" "$OUT/kt" 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
      echo "kernel-trace dispatch count for '$KERNEL' = ${N:-0}" | tee -a "$OUT/preflight.txt"
      [ "${N:-0}" -gt 0 ] && PMC_LIVE="yes"
    fi
  fi
else
  echo "rocprofv3 not found in the kernel's locus (${TILE_KERNEL_CONTAINER:+container $TILE_KERNEL_CONTAINER}${TILE_KERNEL_CONTAINER:-host})" | tee -a "$OUT/preflight.txt"
fi

BRANCH="native-isa"
if [ "$ISA_ONLY" = "yes" ]; then
  BRANCH="isa-only"
  echo "=== [capture] --isa-only -> IR dump + static ISA audit, NO profiler (seconds) ===" | tee -a "$OUT/preflight.txt"
elif [ "$LITE" = "yes" ]; then
  BRANCH="lite"
  echo "=== [capture] --lite -> kernel-trace timing + static ISA only (skip rocprof-compute + ATT) ===" | tee -a "$OUT/preflight.txt"
elif [ "$PMC_LIVE" = "yes" ]; then
  BRANCH="pmc-live"
  echo "=== [capture] PMC-live -> rocprof-compute --full ===" | tee -a "$OUT/preflight.txt"
  bash "$HERE/rocprof_compute_probe.sh" "$NAME" "$OUT/rc" --full --kernel "$KERNEL" -- "${CMD[@]}" \
    2>&1 | tail -3 | tee -a "$OUT/preflight.txt" || echo "[capture] rocprof-compute failed -> degrade to rocprofv3 --pmc" | tee -a "$OUT/preflight.txt"
  ATT_DEC_DIR=""
  if [ "$WANT_ATT" = "yes" ] && locus_have rocprofv3 && ! ATT_DEC_DIR="$(kt_att_decoder_dir)"; then
    ATT_STATE="decoder_absent"
    echo "[capture] ATT layer DEGRADED: rocprof-trace-decoder not found (ROCPROF_ATT_LIBRARY_PATH unset or" \
         "holds no decoder; GEAK does not vendor it). Install it and export ROCPROF_ATT_LIBRARY_PATH=<dir>." \
      | tee -a "$OUT/preflight.txt"
    WANT_ATT="no"
  fi
  if [ "$WANT_ATT" = "yes" ] && locus_have rocprofv3; then
    echo "=== [capture] ATT (auto-widen CU/SE) ===" | tee -a "$OUT/preflight.txt"
    export ROCPROF_ATT_LIBRARY_PATH="$ATT_DEC_DIR"
    kt_move_aside "$OUT/att"; mkdir -p "$OUT/att"
    locus_run rocprofv3 --att --att-target-cu 0 --att-shader-engine-mask 0xF \
      --kernel-include-regex "$KERNEL" -d "$LOCUS_OUT/att" -- "${CMD[@]}" >"$ATT_RAW_LOG" 2>&1 || true
    fold_att_signal "$ATT_RAW_LOG"
    locus_fetch "$LOCUS_OUT/att" "$OUT"
    ATTDIR="$(find "$OUT/att" -maxdepth 2 -type d -name 'ui_output_agent_*' 2>/dev/null | head -1)"
    # retry wider if the traced CU caught no waves (code:null)
    if [ -n "$ATTDIR" ] && grep -q '"code": *null' "$ATTDIR/code.json" 2>/dev/null; then
      echo "[capture] code:null -> retry with wider capture" | tee -a "$OUT/preflight.txt"
      kt_move_aside "$OUT/att"; mkdir -p "$OUT/att"
      locus_run rocprofv3 --att --att-target-cu 0 --att-shader-engine-mask 0xF --att-simd-select 0xF \
        --kernel-include-regex "$KERNEL" -d "$LOCUS_OUT/att" -- "${CMD[@]}" >"$ATT_RAW_LOG" 2>&1 || true
      fold_att_signal "$ATT_RAW_LOG"
      locus_fetch "$LOCUS_OUT/att" "$OUT"
      ATTDIR="$(find "$OUT/att" -maxdepth 2 -type d -name 'ui_output_agent_*' 2>/dev/null | head -1)"
    fi
    if [ -n "$ATTDIR" ]; then
      # fold the inter-MFMA rollup for normalize warpstate (needs a wave + code.json)
      WV="$(ls "$ATTDIR"/se*_wv0.json 2>/dev/null | head -1)"
      [ -n "$WV" ] && python3 "$HERE/mfma_efficiency.py" "$WV" "$ATTDIR/code.json" --ideal-cadence 16 \
        > "$OUT/att/mfma_eff.txt" 2>/dev/null || true
    fi
  fi
else
  echo "=== [capture] PMC-blind -> native-ISA degrade (dump_ir + floor-probe path) ===" | tee -a "$OUT/preflight.txt"
fi

# ISA dump always (static audit works in both branches; best-effort for non-triton apps)
echo "=== [capture] dump_ir.sh (.amdgcn for the static ISA audit) ===" | tee -a "$OUT/preflight.txt"
# AUTO-RETRY ONCE (parity with the ATT code:null retry): a transient dump failure (a stale
# TRITON_CACHE hit that emits no fresh .amdgcn, a JIT temp-file race) is the common case and is
# self-healing -- retry with TRITON_ALWAYS_COMPILE=1 forcing a fresh compile before declaring the
# static layer degraded. A PERSISTENT failure (kernel structurally undumpable) survives the retry
# and still degrades -> caught by post_round's persistent-tool-failure signal. dump_ir.sh:71 already
# names TRITON_ALWAYS_COMPILE=1 as the fix for an empty cache dump; this makes it automatic.
_dump_isa_present() { find "$OUT/ir" -maxdepth 3 \( -name '*_final_isa.s' -o -name '*.amdgcn' -o -name '*.s' \) 2>/dev/null | grep -q .; }
# Run dump_ir.sh IN THE KERNEL'S LOCUS. The static-ISA layer must compile the kernel, so it needs
# triton -- which lives in the container, not on the host. A bare host-side `bash dump_ir.sh` dies
# with ModuleNotFoundError: triton on a container-locus box and the whole A1/A3/B3 static layer
# silently vanishes. Push the script into the container and run it via locus_run (env-forwarded);
# host-side this is an ordinary local run (locus_push/locus_run are passthrough). $OUT is the shared
# working root (bind-mounted / same path in the container), so the outputs land where we read them.
_run_dump_ir() {   # $1 = extra env assignment for the retry (e.g. "TRITON_ALWAYS_COMPILE=1"), $2 = log
  local extra="$1" log="$2" script
  if locus_active; then
    script="${TILE_KERNEL_CONTAINER_CWD:-$LOCUS_OUT}/.dump_ir.$$.sh"
    # A FAILED PUSH IS ITS OWN FINDING, and it used to be reported as the kernel's fault. When the
    # push silently did nothing, `bash <script>` said "No such file or directory", the dump "failed",
    # and the line below blamed a "non-triton app or no cache" -- a diagnosis about the KERNEL for
    # what was a one-flag bug in the copy (a symlinked helper, which `docker cp` rejects without -L).
    # The static layer then read as a property of the op in every artifact downstream. Say which of
    # the two actually broke.
    if ! locus_push "$HERE/dump_ir.sh" "$script"; then
      echo "[capture] dump_ir could not be pushed INTO the locus -- this is a WIRING failure, not a "\
"property of the kernel. Check TILE_KERNEL_CONTAINER_WORKDIR exists in the container." \
        | tee -a "$OUT/preflight.txt" >"$log"
      return 1
    fi
    [ -n "$extra" ] && LOCUS_ENV="$LOCUS_ENV ${extra%%=*}" && export "${extra?}"
    locus_run bash "$script" "${CMD[@]}" --variant "$NAME" --out "$LOCUS_OUT/ir" >"$log" 2>&1 || return 1
    # FETCH THE DUMP BACK OUT. $OUT is often the shared working root -- same path, bind-mounted, so
    # the artifacts are already visible -- and this used to ASSUME that. When $OUT is not shared (an
    # --out under /tmp, which is per-namespace), the dump SUCCEEDS inside the container and the host
    # sees an empty ir/, so capture reports "no .amdgcn/.s dumped" about a dump that worked. Copy the
    # dir out; locus_fetch is a noop host-side and tolerant when the path is already shared.
    _dump_isa_present || locus_fetch "$LOCUS_OUT/ir" "$OUT"
  else
    ( [ -n "$extra" ] && export "${extra?}"; bash "$HERE/dump_ir.sh" "${CMD[@]}" --variant "$NAME" --out "$OUT/ir" ) >"$log" 2>&1
  fi
}
_run_dump_ir "" "$OUT/ir/dump.log" \
  || echo "[capture] dump_ir best-effort (non-triton app, no cache, or see ir/dump.log)" | tee -a "$OUT/preflight.txt"
if ! _dump_isa_present; then
  echo "[capture] dump_ir yielded no ISA -> AUTO-RETRY once (TRITON_ALWAYS_COMPILE=1, fresh compile)" | tee -a "$OUT/preflight.txt"
  _run_dump_ir "TRITON_ALWAYS_COMPILE=1" "$OUT/ir/dump.retry.log" \
    || echo "[capture] dump_ir retry best-effort (see ir/dump.retry.log)" | tee -a "$OUT/preflight.txt"
  _dump_isa_present && echo "[capture] dump_ir retry RECOVERED the ISA layer" | tee -a "$OUT/preflight.txt" \
    || echo "[capture] dump_ir retry still empty -> static layer PERSISTENTLY degraded this round" | tee -a "$OUT/preflight.txt"
fi

# static ISA audit: fold the third evidence layer. Run whichever auditor this skill ships
# (asm_loop_audit.py in gluon/tilelang, isa_loop_audit.py in flydsl) on any dumped .amdgcn/.s.
# For a non-Triton app that yields no dump, static is DEGRADED -> record the concrete upgrade path.
AUDIT="$HERE/asm_loop_audit.py"; [ -f "$AUDIT" ] || AUDIT="$HERE/isa_loop_audit.py"
# PER-KERNEL audit (do NOT let one kernel's failure void the whole static layer). A dump may hold
# several kernels; audit EACH, make the one matching --kernel the PRIMARY ir/asm_audit.txt that
# normalize reads. A failure writes the real stderr into ir/asm_audit_<k>.err (NEVER a 0-byte
# asm_audit.txt) so the exact error is recorded, not swallowed. Dumps may nest under ir/<variant>/
# or ir/<func>/; prefer *_final_isa.s (numeric prefix drifts 21_/22_) over a raw .amdgcn/.s.
mapfile -t ASM_FILES < <(find "$OUT/ir" -maxdepth 3 -name '*_final_isa.s' 2>/dev/null)
[ "${#ASM_FILES[@]}" -gt 0 ] || mapfile -t ASM_FILES < <(find "$OUT/ir" -maxdepth 3 \( -name '*.amdgcn' -o -name '*.s' \) 2>/dev/null)
if [ "${#ASM_FILES[@]}" -gt 0 ] && [ -f "$AUDIT" ]; then
  PRIMARY_TXT=""; N_OK=0; N_FAIL=0
  for asm in "${ASM_FILES[@]}"; do
    tag="$(basename "$(dirname "$asm")")_$(basename "$asm" | sed 's/[^A-Za-z0-9._-]/_/g')"
    txt="$OUT/ir/asm_audit_${tag}.txt"; err="$OUT/ir/asm_audit_${tag}.err"
    # --meta points at the whole ir/ dir (dump_ir.sh drops meta_*.json there): the .s may be
    # nested under ir/<variant>/, so the tool's sibling-dir default would miss it and report
    # LDS UNAVAILABLE. isa_loop_audit.py (flydsl) has no --meta -> only pass it when supported.
    META_ARG=(); grep -q -- "--meta" "$AUDIT" 2>/dev/null && META_ARG=(--meta "$OUT/ir")
    # stdout goes to a .partial that is renamed only on success (never a 0-byte asm_audit.txt), and
    # stderr is kept in memory and written to the .err file only on failure -- no rm needed.
    if audit_err="$(python3 "$AUDIT" "$asm" "${META_ARG[@]}" 2>&1 >"$txt.partial")"; then
      N_OK=$((N_OK+1)); mv -f "$txt.partial" "$txt"
      echo "[capture] static audit OK: ${tag} -> $(basename "$txt")" | tee -a "$OUT/preflight.txt"
      if [ -z "$PRIMARY_TXT" ] || echo "$asm" | grep -qi "$KERNEL"; then
        case "$asm" in *"$KERNEL"*) PRIMARY_TXT="$txt";; *) [ -z "$PRIMARY_TXT" ] && PRIMARY_TXT="$txt";; esac
      fi
    else
      N_FAIL=$((N_FAIL+1)); printf '%s\n' "$audit_err" > "$err"; kt_move_aside "$txt.partial"
      echo "[capture] static audit FAILED: ${tag} -> reason in $(basename "$err"): $(tail -1 "$err" 2>/dev/null)" | tee -a "$OUT/preflight.txt"
    fi
  done
  if [ -n "$PRIMARY_TXT" ]; then
    cp "$PRIMARY_TXT" "$OUT/ir/asm_audit.txt"
    echo "[capture] static primary ($N_OK ok / $N_FAIL failed) -> ir/asm_audit.txt ($(basename "$PRIMARY_TXT"))" | tee -a "$OUT/preflight.txt"
  else
    echo "[capture] static layer DEGRADED: all $N_FAIL kernel audit(s) failed -- see ir/asm_audit_*.err for the exact errors (NOT swallowed)." | tee -a "$OUT/preflight.txt"
  fi
else
  echo "[capture] static layer DEGRADED (no .amdgcn/.s dumped). To fold it: dump_ir.sh on a Triton-compiled kernel, or 'llvm-objdump -d <.co/.hsaco>' the loaded code object, then: normalize.py static --asm-txt <disasm>." | tee -a "$OUT/preflight.txt"
fi

export CAPTURE_ATT_STATE="${ATT_STATE:-}"
python3 - "$OUT" "$NAME" "$KERNEL" "$BRANCH" "$SOURCE" "$SOURCE_SHA" "$DEV" "$COMMAND_SHA" <<'PY'
import json, os, re, sys, glob
out, name, kernel, branch, source, source_sha, device, command_sha = sys.argv[1:9]
attdir = next(iter(glob.glob(os.path.join(out, "att", "ui_output_agent_*"))), None)
rc_analyze = os.path.join(out, "rc", "rc_analyze.txt")
rc_metrics = os.path.join(out, "rc", "rc_metrics.json")

def sol_state():
    """Why the SOL slot holds what it holds.

    A bare `rc_metrics: null` covered three different facts -- never attempted, attempted and
    crashed, collected but never parsed -- and they call for opposite responses: one is a wiring
    bug to fix on the spot, one is the floor this pack is designed to run on. Reading them back
    off one null is how a whole campaign accepted a dark group C as a property of the box.
    Same repair as arm_result.completion, one layer down.
    """
    if os.path.exists(rc_metrics):
        return "parsed", None
    seen = sorted(glob.glob(os.path.join(out, "rc", "rc_analyze*")))
    # Every sibling is read before answering "crashed". A failed analyze leaves the canonical name
    # occupied, so a recovery lands next to it -- and the crash sorts FIRST. Returning on the first
    # traceback would report the layer dead while its numbers sit in the next file along.
    crashed = None
    for p in seen:
        try:
            txt = open(p, errors="replace").read()
        except OSError:
            continue
        if re.search(r"Speed-of-Light|System Speed|Dependency-Wait|SPI Stats", txt, re.I):
            return "collected_unparsed", (f"{os.path.basename(p)} holds the SOL report but no "
                                          f"rc_metrics.json was produced from it -- re-run parse_rc.py")
        if crashed is None and ("PermissionError" in txt
                                or "Traceback (most recent call last)" in txt):
            crashed = (f"{os.path.basename(p)} holds a failure, not a report. On a box that ships "
                       f"rocprof-compute this is a wiring bug to FIX (gluon_authoring "
                       f"references/method/profile.md), not a degrade to accept")
    if crashed:
        return "crashed", crashed
    if seen:
        return "crashed", "an rc_analyze artifact exists but holds no report"
    return "not_attempted", "no rocprof-compute probe was run into this capture"

state, why = sol_state()
cap = {
    "name": name, "kernel": kernel, "branch": branch,
    "source_identity": {
        "path": source,
        "sha256": source_sha,
        "verified_in_locus": True,
    },
    "execution_identity": {
        "logical_device": device,
        "command_sha256": command_sha,
        "container": os.environ.get("TILE_KERNEL_CONTAINER"),
    },
    "rc_analyze": rc_analyze if os.path.exists(rc_analyze) else None,
    "rc_metrics": rc_metrics if os.path.exists(rc_metrics) else None,
    # parsed | collected_unparsed | crashed | not_attempted
    "sol_state": state,
    "sol_note": why,
    "att_dir": attdir,
    # collected | decoder_absent (ROCPROF_ATT_LIBRARY_PATH holds no rocprof-trace-decoder) | not_attempted
    "att_state": ("collected" if attdir else (os.environ.get("CAPTURE_ATT_STATE") or "not_attempted")),
    "att_mfma_eff": os.path.join(out, "att", "mfma_eff.txt") if os.path.exists(os.path.join(out, "att", "mfma_eff.txt")) else None,
    "asm_audit": os.path.join(out, "ir", "asm_audit.txt") if os.path.exists(os.path.join(out, "ir", "asm_audit.txt")) else None,
}
json.dump(cap, open(os.path.join(out, "capture.json"), "w"), indent=2)
print(f"[capture] branch={branch}  sol={state}  att={bool(attdir)}  "
      f"static={bool(cap['asm_audit'])}  -> {out}/capture.json")
if why:
    print(f"[capture]   SOL: {why}")
PY
