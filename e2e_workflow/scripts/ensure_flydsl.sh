#!/bin/bash
# ensure_flydsl.sh — version-gated FlyDSL bootstrap for GEAK e2e_workflow.
#
# WHY: GEAK's Setup/preflight probes flydsl via `aiter.ops.flydsl.is_flydsl_available()`.
# On images where aiter lacks the ops.flydsl wrapper (and the top-level `flydsl` pkg is
# absent), that probe fails and flydsl is silently dropped from the backend ladder — so the
# high-ROI int4-MoE rewrite path (`flydsl_rewrite_quantized_moe` + `apply_flydsl_moe_to_vllm`,
# which import `flydsl, kernels.moe_gemm_2stage` from FLYDSL_ROOT, NOT aiter.ops.flydsl) never
# runs. This script makes flydsl available WITHOUT disturbing environments that already have a
# good-enough flydsl (so GEAK is portable across boxes/other workloads).
#
# GATING (pin = a MINIMUM, not an exact requirement):
#   - flydsl+kernels importable AND __version__  >  MIN_VERSION           -> USE it (no build)
#   - flydsl+kernels importable AND __version__ ==  MIN_VERSION:
#         its checkout HEAD is at-or-after PIN (git ancestry)             -> USE it (no build)
#         else (older commit / not a git checkout)                        -> clone+build PIN
#   - __version__ < MIN_VERSION, kernels missing, or not importable       -> clone+build PIN
# A build ONLY ever writes to the isolated container-internal FLYDSL_ROOT — it never pip-installs
# system-wide and never overwrites an ambient flydsl, so a newer flydsl in another env is left alone.
#
# Exit 0 => flydsl usable and env file written. Non-zero => install failed (loud, not silent).
set -uo pipefail

# Resolve GEAK repo root from THIS script's location — no hardcoded personal host paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"      # scripts/ -> e2e_workflow/ -> GEAK

# Our own build (when we DO build) lives INSIDE the container (overlay fs), NOT on a bind-mounted
# host dir: avoids cross-container sharing (the concurrent-build corruption) and keeps personal
# host paths out of the workflow. Override via FLYDSL_ROOT.
ROOT="${FLYDSL_ROOT:-/opt/flydsl/FlyDSL}"
PIN="a35627a2fef0a5a70c63536c4174674223866737"   # known-good for Kimi-K2.6 int4-W4A16 MoE on gfx942/MI300X
MIN_VERSION="${FLYDSL_MIN_VERSION:-0.2.2}"        # semver floor the PIN corresponds to (v0.2.2-12-ga35627a2)
REPO="https://github.com/ROCm/FlyDSL.git"
SHIM_DIR="$GEAK_ROOT/perf_knowledge/expert_skills/skills/apply_flydsl_moe_to_vllm"
ENV_FILE="$ROOT/flydsl_env.sh"                    # stable location preflight/launcher source
JOBS="$(nproc 2>/dev/null || echo 32)"

log(){ echo "[ensure_flydsl] $*"; }

# semver compare: prints -1 / 0 / 1 for ($1 vs $2), comparing leading numeric components only.
ver_cmp() {
  python3 - "$1" "$2" <<'PY' 2>/dev/null || echo 0
import re, sys
def norm(v):
    out=[]
    for p in re.split(r'[.\-+_]', (v or '').strip()):
        if p.isdigit(): out.append(int(p))
        else: break
    return out
a, b = norm(sys.argv[1]), norm(sys.argv[2])
print(0 if a == b else (1 if a > b else -1))
PY
}

# Resolve the EFFECTIVE flydsl (with our $ROOT prepended, as the workflow will actually run).
# Prints TAB-separated: <version> <flydsl_dir> <kernels_ok:0/1> <checkout_root_or_empty>
probe_flydsl() {
  FLYDSL_ROOT="$ROOT" \
  PYTHONPATH="$ROOT:$ROOT/build-fly/python_packages${PYTHONPATH:+:$PYTHONPATH}" \
  python3 - <<'PY' 2>/dev/null
import os
try:
    import flydsl
except Exception:
    print("\x1f\x1f0\x1f"); raise SystemExit(0)
ver = getattr(flydsl, "__version__", "")
fdir = os.path.dirname(os.path.abspath(flydsl.__file__))
try:
    import kernels.moe_gemm_2stage  # noqa: F401
    kok = "1"
except Exception:
    kok = "0"
# checkout root = nearest ancestor of the flydsl pkg that also contains a 'kernels' dir
root, c = "", fdir
for _ in range(6):
    c = os.path.dirname(c)
    if os.path.isdir(os.path.join(c, "kernels")):
        root = c; break
print("%s\x1f%s\x1f%s\x1f%s" % (ver, fdir, kok, root))
PY
}

# Decide whether an already-present flydsl is good enough to USE (return 0) or we must BUILD (1).
# Reads EFF_VER / EFF_DIR / EFF_KOK / EFF_ROOT set by the caller.
decide_use() {
  [ -n "$EFF_VER" ] || { return 1; }               # not importable
  [ "$EFF_KOK" = "1" ] || { log "flydsl present ($EFF_VER) but 'import kernels.moe_gemm_2stage' fails — will build."; return 1; }
  local cmp; cmp="$(ver_cmp "$EFF_VER" "$MIN_VERSION")"
  if [ "$cmp" -gt 0 ]; then
    log "flydsl $EFF_VER > floor $MIN_VERSION (at $EFF_DIR) — using it, no build."
    return 0
  elif [ "$cmp" -eq 0 ]; then
    if [ -n "$EFF_ROOT" ] && [ -d "$EFF_ROOT/.git" ] && \
       git -C "$EFF_ROOT" merge-base --is-ancestor "$PIN" HEAD 2>/dev/null; then
      log "flydsl $EFF_VER == floor and checkout HEAD >= PIN (at $EFF_ROOT) — using it, no build."
      return 0
    fi
    log "flydsl $EFF_VER == floor but commit < PIN or not a git checkout — will build PIN."
    return 1
  else
    log "flydsl $EFF_VER < floor $MIN_VERSION — will build PIN."
    return 1
  fi
}

# Write the env file (stable location $ENV_FILE) pointing FLYDSL_ROOT/PYTHONPATH at the resolved
# tree $1 (our build $ROOT, or an ambient checkout we chose to reuse).
emit_env() {
  local r="${1:-$ROOT}"
  mkdir -p "$ROOT" 2>/dev/null || true            # dir that holds the env file
  local pp="$r"
  [ -d "$r/build-fly/python_packages" ] && pp="$r:$r/build-fly/python_packages"
  cat > "$ENV_FILE" <<EOF
# auto-generated by ensure_flydsl.sh — source this to make FlyDSL importable
export FLYDSL_ROOT="$r"
export PYTHONPATH="$pp\${PYTHONPATH:+:\$PYTHONPATH}"
export FLYDSL_SHIM_DIR="$SHIM_DIR"
export VLLM_USE_FLYDSL_MOE=1
EOF
  log "env file written: $ENV_FILE (FLYDSL_ROOT=$r)"
}

# ---- 1. is a good-enough flydsl already present? (covers our own prior build, and a newer
#         flydsl provided by another workload / a different env — pin is a minimum) ----
IFS=$'\x1f' read -r EFF_VER EFF_DIR EFF_KOK EFF_ROOT < <(probe_flydsl)
if decide_use; then
  emit_env "${EFF_ROOT:-$ROOT}"
  exit 0
fi

# ---- 1b. acquire exclusive build lock ----
# WHY: GEAK Setup/preflight can invoke this script more than once (retry). Without a lock, two
# ensure_flydsl runs build the SAME tree (shared llvm-project + $ROOT/build-fly) in parallel and
# corrupt each other's ninja outputs. A single exclusive flock serializes the heavy build.
LOCK_FILE="${FLYDSL_BUILD_LOCK:-$(dirname "$ROOT")/.flydsl_build.lock}"
mkdir -p "$(dirname "$LOCK_FILE")" 2>/dev/null || true
exec 9>"$LOCK_FILE" || { log "FATAL: cannot open build lock $LOCK_FILE"; exit 7; }
log "acquiring build lock $LOCK_FILE (waits if another build is in progress)..."
flock -x 9 || { log "FATAL: flock on $LOCK_FILE failed"; exit 7; }
log "build lock acquired."

# ---- 1c. re-check under lock: a concurrent builder may have finished while we waited ----
IFS=$'\x1f' read -r EFF_VER EFF_DIR EFF_KOK EFF_ROOT < <(probe_flydsl)
if decide_use; then
  log "flydsl became usable while waiting for lock — reusing, skipping build."
  emit_env "${EFF_ROOT:-$ROOT}"
  exit 0
fi

# ---- 2. clone @ pinned commit (clone-only, NO pip) — always into our isolated $ROOT ----
if [ ! -d "$ROOT/.git" ]; then
  log "cloning $REPO -> $ROOT"
  rm -rf "$ROOT"
  git clone "$REPO" "$ROOT" || { log "FATAL: git clone failed"; exit 2; }
fi
log "pinning checkout to $PIN"
git -C "$ROOT" fetch --all --tags --quiet 2>/dev/null || true
git -C "$ROOT" checkout "$PIN" || { log "FATAL: checkout $PIN failed"; exit 3; }
git -C "$ROOT" submodule update --init --recursive 2>/dev/null || true

# ---- 2b. ROCm hip cmake compat shim ----
# WHY: some ROCm images ship hip-targets*.cmake referencing a libamdhip64.so.<ver> that no
# longer matches the actually-installed runtime (observed: cmake wants 7.2.70201 but the box
# has libamdhip64.so.7.2.53211 — hip runtime was downgraded while the rest of ROCm stayed
# 70201). find_package(hip) in FlyDSL build.sh then fails: 'imported target hip::amdhip64
# references <file> but this file does not exist'. Create a compat symlink from each missing
# referenced name to the real libamdhip64.so (SONAME libamdhip64.so.7 still resolves it).
# Idempotent + guarded: only acts when the referenced file is actually missing.
REAL_HIP="$(readlink -f /opt/rocm/lib/libamdhip64.so 2>/dev/null)"
if [ -n "$REAL_HIP" ] && [ -f "$REAL_HIP" ]; then
  for want in $(grep -hoE 'libamdhip64\.so\.[0-9.]+' /opt/rocm/lib/cmake/hip/hip-targets-*.cmake 2>/dev/null | sort -u); do
    if [ ! -e "/opt/rocm/lib/$want" ]; then
      ln -sf "$REAL_HIP" "/opt/rocm/lib/$want" && log "hip cmake compat symlink: /opt/rocm/lib/$want -> $REAL_HIP"
    fi
  done
fi

# ---- 2c. ensure patchelf ----
# WHY: FlyDSL build.sh's final CopyFlyPythonSources step runs `patchelf --set-rpath` on the
# MLIR runtime .so's. On images without patchelf that step fails with code=127 ('patchelf:
# command not found') AFTER LLVM + all C++ compiled — a late, expensive-to-hit failure.
if ! command -v patchelf >/dev/null 2>&1; then
  log "installing patchelf (required by FlyDSL build.sh CopyFlyPythonSources)..."
  apt-get install -y patchelf >/dev/null 2>&1 || pip install patchelf >/dev/null 2>&1 || \
    log "WARN: patchelf install failed — build.sh CopyFlyPythonSources will fail (code=127)"
fi

# ---- 3. build (heavy build_llvm only if no MLIR_PATH and not already built) ----
if [ ! -d "$ROOT/build-fly/python_packages" ]; then
  if [ -n "${MLIR_PATH:-}" ]; then
    log "MLIR_PATH set ($MLIR_PATH) — skipping build_llvm.sh"
  else
    log "building LLVM/MLIR (build_llvm.sh -j$JOBS) — heavy, one-time..."
    bash "$ROOT/scripts/build_llvm.sh" -j"$JOBS" || { log "FATAL: build_llvm.sh failed"; exit 4; }
  fi
  log "building FlyDSL (build.sh -j$JOBS)..."
  bash "$ROOT/scripts/build.sh" -j"$JOBS" || { log "FATAL: build.sh failed"; exit 5; }
else
  log "build-fly/python_packages present — reusing existing build."
fi

# ---- 4. verify + emit env (our freshly-built $ROOT) ----
IFS=$'\x1f' read -r EFF_VER EFF_DIR EFF_KOK EFF_ROOT < <(probe_flydsl)
if [ -n "$EFF_VER" ] && [ "$EFF_KOK" = "1" ]; then
  log "OK: import flydsl, kernels.moe_gemm_2stage works (flydsl $EFF_VER)"
  emit_env "$ROOT"
  exit 0
else
  log "FATAL: FlyDSL built but 'import flydsl, kernels.moe_gemm_2stage' still fails."
  log "  check $ROOT/build-fly/python_packages and py version (needs py3.12 build)."
  exit 6
fi
