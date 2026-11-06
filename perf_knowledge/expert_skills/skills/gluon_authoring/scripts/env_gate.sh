#!/usr/bin/env bash
# env_gate.sh — the CHEAP environment gate: disk / scratch / SOL profiler. Seconds, installs
# nothing, exits non-zero on a hard blocker so a caller (e.g. a GEAK role before its first measurement) can gate on it.
#
# Why this ships INSIDE the pack: a pack is distributed self-contained, and a gate that lives only in
# a repo the container may not have is a gate that gets skipped. In GEAK the repo-level counterpart is
# `geak/bootstrap.py` (check_environment(): DETECT-only -- ROCm, a profiler, a serving backend), and
# the two agree on the profiler rule: a missing profiler, or rocprof-compute refusing the arch
# (RDNA4/gfx1201 "Unsupported arch", kernel_workflow/knowledge/amd_rdna4.md), DEGRADES the profiling
# layers -- it is not a launch blocker. (There is no `bootstrap.sh --preflight`; earlier text that
# pointed at one was wrong.)
#
# In GEAK the environment comes from GEAK (geak/bootstrap.py, kernel_workflow) and this gate is
# optional: run it when a kernel lives in a separate container or the box is suspect.
#
# Usage:
#   bash env_gate.sh                       # host-side gate
#   bash env_gate.sh --docker <id>         # ALSO exercise the container->host boundary (see below)
#   bash env_gate.sh --workdir <dir>       # the working root whose free space matters
#   bash env_gate.sh --selftest            # exercise the decision logic, no GPU needed
#
# Env: TILE_DISK_FLOOR_GB (default 5) · TILE_TMPDIR (fleet runtime contract) · TILE_KERNEL_CONTAINER
#
# Exit: 0 clear to launch · 1 hard blocker (disk / scratch / locus wiring / analyze PERMISSIONS)
#       2 nothing to test here (no GPU visible in the locus)
#       3 DEGRADED but clear to launch: a profiler layer is unavailable (no rocprof-compute and/or no
#         rocprofv3; rocprof-compute unsupported on this arch, e.g. RDNA4; SOL smoke failed for a
#         non-permission reason). Say so in the handoff; ATT/PMC/static floors still apply.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=locus.sh
source "$HERE/locus.sh"

ok(){   printf '  \033[32m[ok]\033[0m %s\n' "$*"; }
info(){ printf '  \033[36m[..]\033[0m %s\n' "$*"; }
warn(){ printf '  \033[33m[!]\033[0m %s\n' "$*"; }

# ---- the three decisions this gate makes, factored out so --selftest can reach them ----------

# Free space at <dir>, in whole GB. Empty when df cannot answer, which callers treat as "unknown"
# rather than as zero -- a gate that blocks on an unreadable df blocks on every unusual mount.
_free_gb(){ df -PBG "$1" 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}'; }

_disk_ok(){ # <dir> <floor_gb> -> 0 clear / 1 under floor
  local f; f="$(_free_gb "$1")"
  [ -z "$f" ] && return 0
  [ "$f" -ge "$2" ]
}

_scratch_ok(){ # <dir> -> 0 writable. Creates a NEW file (what a full / read-only scratch refuses),
  # then renames it onto one fixed probe name, so repeated runs leave one file and nothing is rm'd.
  local probe="$1/.tile_env_gate_probe"
  ( : >"$probe.$$" ) 2>/dev/null && mv -f "$probe.$$" "$probe" 2>/dev/null
}

# Why analyze failed. The whole point of the gate is that these two need opposite responses: a
# permission failure is a fixable wiring bug on a working profiler, and anything else may really
# be a box that cannot give SOL. Collapsing them is how "rocprof-compute is broken in this
# container" got written into nine reports about one chmod.
_classify_analyze(){ # <logfile> -> prints "permission" | "unsupported_arch" | "other"
  if grep -qi "PermissionError\|Permission denied" "$1" 2>/dev/null; then
    printf 'permission'
  elif grep -qi "Unsupported arch\|unsupported architecture\|not supported on this arch" "$1" 2>/dev/null; then
    printf 'unsupported_arch'
  else
    printf 'other'
  fi
}

# rocprof-compute's SoC modules are CDNA-oriented: on RDNA (gfx11xx/gfx12xx, e.g. RDNA4 gfx1201) it
# aborts "Unsupported arch". That is a property of the tool on that box -> DEGRADE, use rocprofv3.
_rc_supported_arch(){ # <gfx> -> 0 when rocprof-compute's SOL smoke is meaningful there
  case "${1:-}" in gfx11*|gfx12*) return 1 ;; *) return 0 ;; esac
}

if [ "${1:-}" = "--selftest" ]; then
  fail=0
  tmp="$(mktemp -d)"
  # a floor of 0 is met by any mount; an absurd floor is met by none
  _disk_ok "$tmp" 0 || { echo "FAIL: floor 0 should always pass"; fail=1; }
  if [ -n "$(_free_gb "$tmp")" ]; then
    _disk_ok "$tmp" 999999999 && { echo "FAIL: absurd floor should not pass"; fail=1; }
  fi
  # an unreadable path yields no number, and an unknown must not read as a blocker
  _disk_ok "/no/such/mount/$$" 5 || { echo "FAIL: unknown free space must not block"; fail=1; }
  # writable vs not
  _scratch_ok "$tmp" || { echo "FAIL: fresh tmpdir should be writable"; fail=1; }
  locus_workdir_shared "$tmp" || {
    echo "FAIL: host workdir should be shared with itself"; fail=1;
  }
  chmod a-w "$tmp"
  if [ "$(id -u)" != 0 ]; then     # root ignores the mode bit, so this half is not a test as root
    _scratch_ok "$tmp" && { echo "FAIL: read-only dir reported writable"; fail=1; }
  fi
  chmod u+w "$tmp"
  # THE distinction the gate exists to draw: the same non-zero exit, two different causes.
  printf 'PermissionError: [Errno 13] pmc_dispatch_info.csv\n' > "$tmp/a.log"
  [ "$(_classify_analyze "$tmp/a.log")" = "permission" ] \
    || { echo "FAIL: PermissionError not classified as permission"; fail=1; }
  printf 'ERROR: no profiling data found in workload\n' > "$tmp/b.log"
  [ "$(_classify_analyze "$tmp/b.log")" = "other" ] \
    || { echo "FAIL: a data failure must not read as a permission failure"; fail=1; }
  printf 'ERROR: Unsupported arch: gfx1201\n' > "$tmp/c.log"
  [ "$(_classify_analyze "$tmp/c.log")" = "unsupported_arch" ] \
    || { echo "FAIL: 'Unsupported arch' must classify as unsupported_arch (a DEGRADE)"; fail=1; }
  _rc_supported_arch gfx1201 && { echo "FAIL: RDNA4 must not run the rocprof-compute SOL smoke"; fail=1; }
  _rc_supported_arch gfx950 || { echo "FAIL: gfx950 must run the rocprof-compute SOL smoke"; fail=1; }
  _rc_supported_arch gfx942 || { echo "FAIL: gfx942 must run the rocprof-compute SOL smoke"; fail=1; }
  # the gate itself never rm's (GEAK convention: move aside / overwrite in place)
  grep -nE '^[^#]*[^a-z_]rm -(r|f|rf) ' "${BASH_SOURCE[0]}" | grep -v 'grep -nE' >/dev/null \
    && { echo "FAIL: an rm remains in env_gate.sh"; fail=1; }
  [ "$fail" = 0 ] && echo "ENV_GATE SELFTEST PASS" || echo "ENV_GATE SELFTEST FAIL"
  exit "$fail"
fi

FLOOR_GB="${TILE_DISK_FLOOR_GB:-5}"
TMPD="${TILE_TMPDIR:-}"
DOCKER="${TILE_KERNEL_CONTAINER:-}"
WORKDIR="$PWD"
while [ $# -gt 0 ]; do
  case "$1" in
    --docker) DOCKER="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown arg: $1 (see --help)" >&2; exit 2 ;;
  esac
done
if [ -z "$TMPD" ]; then
  TMPD="$WORKDIR/.tile-runtime/scratch/env-gate"
fi
mkdir -p "$TMPD"
hard=0
[ -n "$DOCKER" ] && export TILE_KERNEL_CONTAINER="$DOCKER"
if [ -n "$DOCKER" ]; then
  : "${TILE_KERNEL_HOST_WORKDIR:=$WORKDIR}"
  : "${TILE_KERNEL_CONTAINER_WORKDIR:=$WORKDIR}"
fi

echo "=== env gate (disk / scratch / SOL profiler) ==="

# (1) DISK / ENOSPC. An out-of-disk box fails writes mid-sweep, which reads as a silent stall.
for d in "$WORKDIR" "$TMPD" "${TILE_HW_CACHE:-$HOME/.cache/tile-hw}"; do
  [ -d "$d" ] || continue
  if _disk_ok "$d" "$FLOOR_GB"; then
    ok "disk $d: $(_free_gb "$d")G free (floor ${FLOOR_GB}G)"
  else
    warn "DISK: $d has $(_free_gb "$d")G free < floor ${FLOOR_GB}G -> clean up or expand first"
    hard=1
  fi
done

# (2) scratch writability: a full or read-only scratch dir breaks the round scripts silently.
if _scratch_ok "$TMPD"; then
  ok "scratch $TMPD is writable"
else
  warn "SCRATCH: $TMPD is NOT writable -> set TILE_TMPDIR=<a writable dir> and re-export it"
  hard=1
fi

# (2b) hardware-source tier. Delegated to hw_sources.sh, which owns the layer and so knows what
# it needs; this gate only decides how loudly to report it. Runs HERE, with the other static
# asset checks, rather than after the profiler section -- that section exits early when there is
# no rocprof-compute, and a check that a box without a profiler never reaches is not a check.
#
# NOT a hard blocker: every miss has a documented fallback (the distilled references/hardware/
# values, probe-then-trust), and on gfx950 the in-pack databases answer with no calculator and
# no network. It is reported loudly anyway because the cost of finding out late is high -- the
# calculator being absent used to surface at the moment an agent needed an operand layout.
echo
if [ -f "$HERE/hw_sources.sh" ]; then
  if bash "$HERE/hw_sources.sh" preflight; then
    ok "hardware sources: every question class has an answerer"
  else
    warn "HW SOURCES: a tier above needs repair — the [!] lines name who repairs it and how."
    warn "  Not a launch blocker: in-pack databases still answer, and the other tiers have a"
    warn "  documented fallback. It BECOMES one if this kernel needs a cdna1/2/3 or rdna3/4"
    warn "  operand layout, since the calculator is the only source for those."
    warn "  If you are a GEAK engineer (e.g. the deep_engineer), the repair is NOT yours: return blocked_environment"
    warn "  naming the line. Installing into a shared container is the thing the project rule"
    warn "  forbids, and a preflight that only printed the command is how that gets done anyway."
  fi
else
  warn "hw_sources.sh is absent from this pack — the hardware-source tier cannot be checked."
fi
echo

# (3) the SOL layer, END TO END.
#
# The version of this smoke that shipped before ran `rocprof-compute profile` into a mktemp dir and
# checked that pmc_perf.csv appeared. That is structurally blind to the failure it most needs to
# catch: the recurring one happens in ANALYZE, not in profile, and it happens because collection
# runs in the container as root while analyze runs here as the host uid -- a boundary a host-side
# mktemp dir never crosses. A gate that certifies a path nobody takes is worse than no gate, so
# this one profiles in the same locus a round does and then runs analyze from here, exactly as
# rocprof_compute_probe.sh will.
have_rc=0
GFX="$(locus_run rocminfo 2>/dev/null | awk '/^ *Name: *gfx[0-9a-f]+/ && $2 != "gfx000" {print $2; exit}')"
if [ -n "$DOCKER" ] && ! locus_active; then
  warn "LOCUS: container '$DOCKER' is unavailable (locus_unavailable); profile/analyze were not attempted"
  hard=1
elif locus_have rocprof-compute; then
  have_rc=1
fi
if [ "$hard" = 1 ] && [ "$have_rc" = 0 ]; then
  echo
  echo "ENV GATE: FAIL — fix the [!] blocker(s) above BEFORE launching."
  exit 1
fi
if ! locus_have rocprofv3 && [ "$have_rc" = 0 ]; then
  warn "PROFILER: neither rocprofv3 nor rocprof-compute in this locus -> profiling DEGRADED to the"
  warn "  static ISA floor (+ timing). Same rule as geak/bootstrap.py: not a launch blocker."
  echo; echo "ENV GATE: DEGRADED (no profiler) — clear to launch; say so in the handoff"; exit 3
fi
if [ "$have_rc" = 1 ] && [ -n "$GFX" ] && ! _rc_supported_arch "$GFX"; then
  info "SOL smoke skipped: rocprof-compute does not support $GFX (RDNA; 'Unsupported arch'). Use"
  info "  rocprofv3 first (kernel trace; list counters with \`rocprofv3 -L\` / \`rocprofv3-avail list --pmc\`)."
  echo; echo "ENV GATE: DEGRADED (rocprof-compute unsupported on $GFX) — clear to launch"; exit 3
fi
if [ "$have_rc" = 0 ]; then
  info "SOL smoke skipped (rocprof-compute not installed in this locus) -- the rocprofv3 PMC/ATT +"
  info "  static floor is what this box can give; say so in the handoff rather than reporting a clear gate"
  echo
  echo "ENV GATE: DEGRADED (no SOL layer on this box) — clear to launch"; exit 3
fi

# A container profile writes the workload and host-side analyze reads it. Put the smoke below the
# requested working root and prove bidirectional visibility before invoking rocprof-compute; /tmp
# is often not bind-mounted and must not be mislabeled as a profiler failure.
if [ -n "$DOCKER" ]; then
  d="$(mktemp -d "${WORKDIR%/}/.tile_env_gate_XXXXXX" 2>/dev/null)" || {
    warn "LOCUS: cannot create a shared-workdir probe under $WORKDIR"
    echo "ENV GATE: FAIL — cannot verify container/host artifact path."
    exit 1
  }
else
  d="$(mktemp -d "${TMPD%/}/tile_env_gate_XXXXXX")"
fi
chmod a+rwx "$d" 2>/dev/null || true
if [ -n "$DOCKER" ] && ! locus_workdir_shared "$d"; then
  warn "LOCUS: container-host workdir is unshared (locus_workdir_unshared); profile/analyze were not attempted"
  warn "  host path: $d"
  echo
  echo "ENV GATE: FAIL — fix the shared workdir mount before launching."
  exit 1
fi
LOCUS_D="$(locus_path "$d")"
export TILE_KERNEL_CONTAINER_CWD="$LOCUS_D"
cat > "$d/mm.py" <<'PY'
import torch
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    c = a @ b
torch.cuda.synchronize()
PY

if ! locus_run python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
  >/dev/null 2>&1; then
  info "SOL smoke skipped (no visible GPU in this locus) -- not a gate failure"
  echo
  [ "$hard" = 1 ] && { echo "ENV GATE: FAIL"; exit 1; }
  echo "ENV GATE: OK (GPU not visible here)"; exit 2
fi

# --no-roof is essential: without it, profile also runs the empirical-roofline microbenchmarks
# (100 reps per dtype) on top of the multi-pass replay, turning a ~10s smoke into minutes.
degraded=0
if ! (cd "$d" && locus_timeout 300 rocprof-compute profile --name tile_env_gate --no-roof -- python3 "$LOCUS_D/mm.py") \
     >"$d/profile.log" 2>&1; then
  # A profile failure on a verified shared locus is a property of the tool on this box (version,
  # arch, counters) -> DEGRADE the SOL layer, as GEAK's profile_kernel.sh fault-tolerance ladder does.
  warn "SOL: rocprof-compute PROFILE exited non-zero after shared-locus verification ($(_classify_analyze "$d/profile.log")). See $d/profile.log"
  tail -3 "$d/profile.log" >&2
  degraded=1
else
  locus_run chmod -R a+rwX "$LOCUS_D/workloads/tile_env_gate" >/dev/null 2>&1 || true
  chmod -R a+rwX "$d/workloads/tile_env_gate" >/dev/null 2>&1 || true
  wl="$(ls -d "$d"/workloads/tile_env_gate/*/ 2>/dev/null | head -1)"
  if [ -z "$wl" ]; then
    warn "SOL: profile reported success but produced no workload dir under $d/workloads/tile_env_gate"
    hard=1
  # The half the old smoke never ran. Analyze is what fails on a root-owned collection, and it
  # fails AFTER the expensive replay has already succeeded -- so a gate that stops at profile
  # reports green for exactly the box that will go dark on every round.
  elif ! rocprof-compute analyze -p "$wl" -k 0 --block 2 >"$d/analyze.log" 2>&1; then
    if [ "$(_classify_analyze "$d/analyze.log")" = "permission" ]; then
      warn "SOL: profile succeeded and ANALYZE failed on PERMISSIONS -- the collection is owned by"
      warn "  the container's root and analyze writes into the workload dir as uid $(id -u)."
      warn "  This is the failure that silently costs a whole run its group-C evidence."
      warn "  rocprof_compute_probe.sh widens the tree itself; if this gate still fails, the bind"
      warn "  mount or the uid mapping is the problem, not rocprof-compute."
      hard=1
    else
      warn "SOL: analyze failed after shared-locus verification ($(_classify_analyze "$d/analyze.log")). See $d/analyze.log"
      tail -3 "$d/analyze.log" >&2
      degraded=1
    fi
  else
    ok "SOL end-to-end: profile + analyze both clean${DOCKER:+ (collected in container $DOCKER)}"
  fi
fi
# The smoke dir stays under the scratch root (.tile-runtime/scratch/, ignored by GEAK workspaces).

echo
if [ "$hard" = 1 ]; then
  echo "ENV GATE: FAIL — fix the [!] blocker(s) above BEFORE launching."
  exit 1
fi
if [ "$degraded" = 1 ]; then
  echo "ENV GATE: DEGRADED (SOL layer unavailable) — clear to launch; ATT/PMC + static floor apply."
  exit 3
fi
echo "ENV GATE: OK — env is clear to launch."
exit 0
