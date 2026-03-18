#!/bin/bash
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install FlyDSL from source: clone, build LLVM/MLIR, build FlyDSL, pip-install.
#
# Usage:
#   bash scripts/install-flydsl.sh                        # clone + full build
#   bash scripts/install-flydsl.sh --skip-llvm             # skip LLVM if already built
#   bash scripts/install-flydsl.sh --branch my-branch      # clone a specific branch
#   bash scripts/install-flydsl.sh -j32                    # override parallelism
#   FLYDSL_DIR=/existing/FlyDSL bash scripts/install-flydsl.sh --skip-llvm
#   MLIR_PATH=/path/to/mlir_install bash scripts/install-flydsl.sh --skip-llvm

set -euo pipefail

FLYDSL_REPO="${FLYDSL_REPO:-https://github.com/ROCm/FlyDSL.git}"
FLYDSL_DIR="${FLYDSL_DIR:-/workspace/FlyDSL}"
FLYDSL_BRANCH=""
SKIP_LLVM=false
SKIP_TESTS=false

# Cap parallel jobs based on available memory to avoid OOM kills during
# LLVM builds (each compilation unit can use ~2 GB of RAM at peak).
_mem_based_jobs() {
  local mem_gb
  mem_gb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null) || mem_gb=0
  if (( mem_gb > 0 )); then
    local safe=$(( mem_gb / 2 ))
    (( safe < 1 )) && safe=1
    echo "$safe"
  else
    echo "$(nproc)"
  fi
}

if [[ -n "${PARALLEL_JOBS:-}" ]]; then
  : # caller explicitly set PARALLEL_JOBS — honour it
else
  _nproc=$(nproc)
  _mem_jobs=$(_mem_based_jobs)
  PARALLEL_JOBS=$(( _nproc < _mem_jobs ? _nproc : _mem_jobs ))
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-llvm)  SKIP_LLVM=true; shift ;;
    --skip-tests) SKIP_TESTS=true; shift ;;
    --branch)     FLYDSL_BRANCH="$2"; shift 2 ;;
    -j[0-9]*)     PARALLEL_JOBS="${1#-j}"; shift ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --skip-llvm       Skip building LLVM/MLIR (use existing MLIR_PATH)"
      echo "  --skip-tests      Skip running tests after install"
      echo "  --branch BRANCH   Git branch/tag to clone (default: repo default branch)"
      echo "  -jN               Parallel build jobs (default: \$(nproc) = $(nproc))"
      echo ""
      echo "Environment:"
      echo "  FLYDSL_REPO   Git clone URL (default: https://github.com/ROCm/FlyDSL.git)"
      echo "  FLYDSL_DIR    FlyDSL source directory (default: /workspace/FlyDSL)"
      echo "  MLIR_PATH     Path to existing MLIR install (skips LLVM build)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------
# Step 0a: Clone FlyDSL if the source directory doesn't exist
# ---------------------------------------------------------------
if [[ ! -d "$FLYDSL_DIR" ]]; then
  echo "FlyDSL source not found at $FLYDSL_DIR — cloning from $FLYDSL_REPO ..."
  clone_args=(git clone --recursive)
  if [[ -n "$FLYDSL_BRANCH" ]]; then
    clone_args+=(--branch "$FLYDSL_BRANCH")
  fi
  clone_args+=("$FLYDSL_REPO" "$FLYDSL_DIR")
  "${clone_args[@]}"
else
  echo "Using existing FlyDSL source at $FLYDSL_DIR"
  if [[ -n "$FLYDSL_BRANCH" ]]; then
    echo "Checking out branch/tag: $FLYDSL_BRANCH"
    git -C "$FLYDSL_DIR" fetch origin
    git -C "$FLYDSL_DIR" checkout "$FLYDSL_BRANCH"
  fi
  git -C "$FLYDSL_DIR" submodule update --init --recursive
fi

echo ""
echo "=============================================="
echo "FlyDSL Installation"
echo "  FLYDSL_DIR:     $FLYDSL_DIR"
echo "  PARALLEL_JOBS:  $PARALLEL_JOBS"
echo "  SKIP_LLVM:      $SKIP_LLVM"
echo "  SKIP_TESTS:     $SKIP_TESTS"
echo "=============================================="

# ---------------------------------------------------------------
# Step 0b: System build tools (cmake, ninja, C++ compiler)
# ---------------------------------------------------------------
echo ""
echo "[0/4] Checking build prerequisites..."

MISSING_PKGS=()
command -v cmake  &>/dev/null || MISSING_PKGS+=(cmake)
command -v ninja  &>/dev/null || MISSING_PKGS+=(ninja-build)
command -v g++    &>/dev/null || MISSING_PKGS+=(g++)

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
  echo "Installing missing system packages: ${MISSING_PKGS[*]}"
  apt-get update -qq
  apt-get install -y -qq "${MISSING_PKGS[@]}"
fi

# Python dependencies needed by the build
pip install nanobind numpy pybind11 --quiet
echo "Prerequisites OK."

# ---------------------------------------------------------------
# Step 1: Build LLVM/MLIR
# ---------------------------------------------------------------
if [[ -n "${MLIR_PATH:-}" && -d "${MLIR_PATH}/lib/cmake/mlir" ]]; then
  echo ""
  echo "[1/4] MLIR_PATH already set and valid: $MLIR_PATH — skipping LLVM build."
  SKIP_LLVM=true
fi

if [[ "$SKIP_LLVM" == false ]]; then
  echo ""
  echo "[1/4] Building LLVM/MLIR (this may take 30+ minutes)..."

  # Clean stale CMake cache to avoid generator-mismatch errors (e.g. Unix
  # Makefiles left over from a failed run, now conflicting with Ninja).
  LLVM_BUILD_DIR="$(cd "$FLYDSL_DIR/.." && pwd)/llvm-project/build-flydsl"
  if [[ -f "$LLVM_BUILD_DIR/CMakeCache.txt" ]]; then
    echo "Removing stale CMake cache in $LLVM_BUILD_DIR ..."
    rm -f "$LLVM_BUILD_DIR/CMakeCache.txt"
    rm -rf "$LLVM_BUILD_DIR/CMakeFiles"
  fi

  bash "$FLYDSL_DIR/scripts/build_llvm.sh" -j"$PARALLEL_JOBS"

  # Auto-detect the install path produced by build_llvm.sh
  LLVM_SRC_DIR="$(cd "$FLYDSL_DIR/.." && pwd)/llvm-project"
  for candidate in \
    "$LLVM_SRC_DIR/mlir_install" \
    "$LLVM_SRC_DIR/build-flydsl/mlir_install"; do
    if [[ -d "$candidate/lib/cmake/mlir" ]]; then
      export MLIR_PATH="$candidate"
      break
    fi
  done

  if [[ -z "${MLIR_PATH:-}" ]]; then
    echo "Error: LLVM build finished but could not locate mlir_install." >&2
    exit 1
  fi
  echo "MLIR_PATH set to: $MLIR_PATH"
else
  echo ""
  echo "[1/4] Skipping LLVM build."
fi

# ---------------------------------------------------------------
# Step 2: Build FlyDSL
# ---------------------------------------------------------------
echo ""
echo "[2/4] Building FlyDSL..."
MLIR_PATH="${MLIR_PATH:-}" bash "$FLYDSL_DIR/scripts/build.sh" -j"$PARALLEL_JOBS"

# ---------------------------------------------------------------
# Step 3: Install (editable / development mode)
# ---------------------------------------------------------------
echo ""
echo "[3/4] Installing FlyDSL in editable mode..."
pip install -e "$FLYDSL_DIR"

# ---------------------------------------------------------------
# Step 4: Verification
# ---------------------------------------------------------------
echo ""
echo "[4/4] Verifying installation..."
if python3 -c "import flydsl; print(f'flydsl {flydsl.__version__} installed successfully')" 2>/dev/null; then
  echo "✅ FlyDSL import OK"
else
  echo "⚠️  'import flydsl' failed — setting PYTHONPATH fallback..."
  BUILD_PKG="$FLYDSL_DIR/build-fly/python_packages"
  export PYTHONPATH="${BUILD_PKG}:${FLYDSL_DIR}:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="${BUILD_PKG}/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH:-}"
  echo "  PYTHONPATH=$PYTHONPATH"
  echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  python3 -c "import flydsl; print(f'flydsl {flydsl.__version__} installed successfully')"
  echo "✅ FlyDSL import OK (via PYTHONPATH)"
fi


echo ""
echo "=============================================="
echo "FlyDSL installation complete!"
echo "=============================================="
