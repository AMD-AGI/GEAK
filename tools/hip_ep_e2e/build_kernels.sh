#!/bin/bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Rebuild hip-ep's kernel library (libcustom_kernels_<arch>.so, the arch the release
# package ships) from the release tag plus an overlay of edited files.
#
#   build_kernels.sh <overlay_dir|-> <out_dir>
#
# overlay_dir mirrors paths under the hip-ep root, all under lib/; "-" builds the
# pristine tag. The tree is reset to the tag before the overlay is applied and again
# after the build, so a build depends only on its overlay; the fresh mtimes make ninja
# rebuild what changed either way. It is edited in place under the lock
# tools/ep_task_runner.py takes for EP builds. Paths come from hipep_env.PATHS;
# HIPEP_TAG is the release tag (default v0.5.3).
set -euo pipefail
OVERLAY=${1:?usage: build_kernels.sh <overlay_dir|-> <out_dir>}
OUT=${2:?usage: build_kernels.sh <overlay_dir|-> <out_dir>}
SRC=${HIPEP_SRC:?HIPEP_SRC is not set: a hip-ep checkout at the release tag}
BUILD=${HIPEP_BUILD:?HIPEP_BUILD is not set: a hip-ep build dir}
THEROCK=${HIPEP_THEROCK:?HIPEP_THEROCK is not set: the TheRock dist the release pins}
PKG=${HIPEP_PKG:?HIPEP_PKG is not set: the hip-ep release package}
TAG=${HIPEP_TAG:-v0.5.3}
shopt -s nullglob
SHIPPED=("$PKG"/lib/libcustom_kernels_*.so)
shopt -u nullglob
[ ${#SHIPPED[@]} -eq 1 ] || { echo "$PKG/lib holds ${#SHIPPED[@]} libcustom_kernels_<arch>.so; expected one" >&2; exit 1; }
TARGET=$(basename "${SHIPPED[0]}" .so)
TARGET=${TARGET#lib}
ARCH=${TARGET#custom_kernels_}
LIB=lib/Runtime/Kernels/lib${TARGET}.so

FILES=()
if [ "$OVERLAY" != "-" ]; then
  OVERLAY=$(cd "$OVERLAY" && pwd)
  while IFS= read -r -d '' f; do FILES+=("${f#"$OVERLAY"/}"); done \
    < <(find "$OVERLAY" -type f -print0 | sort -z)
  for f in "${FILES[@]}"; do
    case "$f" in
      lib/*) ;;
      *) echo "overlay file $f is outside lib/" >&2; exit 1 ;;
    esac
  done
fi
mkdir -p "$OUT"
rm -f "$OUT/lib${TARGET}.so"

TOOLS=$(cd "$(dirname "$0")/.." && pwd)
LOCK=$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import ep_task_runner
print(ep_task_runner._ep_build_lock_path(sys.argv[2]))' "$TOOLS" "$SRC")
exec {LOCK_FD}>"$LOCK"
flock "$LOCK_FD"

if [ ! -d "$SRC/.git" ]; then
  git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$TAG" \
    https://github.com/ROCm/hip-ep "$SRC"
fi
mkdir -p "$BUILD"
if [ ! -f "$BUILD/build.ninja" ]; then
  # Name TheRock's compiler explicitly: with a system ROCm at /opt/rocm (as in the
  # GEAK container), CMake's HIP detection picks /opt/rocm's clang instead, and the
  # library no longer matches the release.
  ROCM_PATH="$THEROCK" HIP_PATH="$THEROCK" cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER="$THEROCK/bin/amdclang++" \
    -DTHEROCK_DIST="$THEROCK" -DHIP_ARCHITECTURES="$ARCH" -DBUILD_MOCK_RUNTIME=OFF \
    -DPython3_EXECUTABLE="$(command -v python3)" >"$BUILD/configure.log" 2>&1 \
    || { echo "configure failed; see $BUILD/configure.log" >&2; exit 1; }
fi
grep -qx 'CMAKE_BUILD_TYPE:STRING=Release' "$BUILD/CMakeCache.txt" \
  || { echo "$BUILD is not a Release build; remove it and rerun" >&2; exit 1; }
HIP_CXX=$(sed -n 's/^set(CMAKE_HIP_COMPILER "\(.*\)")$/\1/p' "$BUILD"/CMakeFiles/*/CMakeHIPCompiler.cmake)
case "$HIP_CXX" in
  "$THEROCK"/*) ;;
  *) echo "$BUILD was configured with HIP compiler '$HIP_CXX', not $THEROCK's; remove it and rerun" >&2
     exit 1 ;;
esac

[ "$(git -C "$SRC" describe --tags --exact-match HEAD 2>/dev/null)" = "$TAG" ] \
  || { echo "$SRC is not at $TAG" >&2; exit 1; }
# Tracked files from HEAD (not from the index), and nothing untracked under lib/.
restore() { git -C "$SRC" reset --quiet --hard HEAD && git -C "$SRC" clean -fdq -- lib; }
restore
if [ ${#FILES[@]} -gt 0 ]; then
  cp -r "$OVERLAY/." "$SRC/"
fi

ninja -C "$BUILD" "$TARGET" >"$OUT/build.log" 2>&1 \
  || { restore; echo "build failed; see $OUT/build.log" >&2; exit 1; }
cp "$BUILD/$LIB" "$OUT/"
# Leave the tree at the tag: the tools read sources from git, but a person may not.
restore
echo "built $OUT/lib${TARGET}.so md5 $(md5sum "$OUT/lib${TARGET}.so" | cut -d' ' -f1)" \
  "(${#FILES[@]} overlay files: ${FILES[*]:-})"
