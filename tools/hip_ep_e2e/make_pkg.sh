#!/bin/bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Make a candidate hip-ep package: the base package with its kernel library
# swapped for a rebuilt one.
#
#   make_pkg.sh <base_pkg> <libcustom_kernels_<arch>.so> <out_pkg>
#
# The copy is hard-linked (about 1.9 GB would otherwise be copied per
# candidate), or copied when the base is on another filesystem. Every file this
# script writes has its link removed first, so the base package is never
# written through. A failed run removes the package it started. Runs only read a package; their HOME
# and temp dir are per package and outside it (hipep_env.pkg_env).
set -euo pipefail
BASE=${1:?usage: make_pkg.sh <base_pkg> <kernels.so> <out_pkg>}
SO=${2:?usage: make_pkg.sh <base_pkg> <kernels.so> <out_pkg>}
OUT=${3:?usage: make_pkg.sh <base_pkg> <kernels.so> <out_pkg>}
LIB=$(basename "$SO")

[ -f "$SO" ] || { echo "no such file: $SO" >&2; exit 1; }
[ -f "$BASE/lib/$LIB" ] || { echo "$BASE has no lib/$LIB: not the arch its kernel library is built for" >&2; exit 1; }
[ -e "$OUT" ] && { echo "$OUT exists; refusing to overwrite" >&2; exit 1; }

mkdir -p "$OUT"
trap 'rc=$?; [ "$rc" -eq 0 ] || rm -rf "$OUT"' EXIT
if ln "$BASE/lib/$LIB" "$OUT/.link_probe" 2>/dev/null; then
  rm "$OUT/.link_probe"
  COPY=(cp -al)
else
  COPY=(cp -a)
fi
for entry in "$BASE"/*; do
  case "$entry" in *.zip) continue ;; esac
  "${COPY[@]}" "$entry" "$OUT/"
done
rm "$OUT/lib/$LIB"
cp "$SO" "$OUT/lib/$LIB"
rm -f "$OUT/PKG_INFO.json"  # linked from the base when the base is itself a candidate
python3 - "$OUT/PKG_INFO.json" "$(cd "$BASE" && pwd)" "$(cd "$(dirname "$SO")" && pwd)/$(basename "$SO")" \
  "$(md5sum "$BASE/lib/$LIB" | cut -d' ' -f1)" "$(md5sum "$OUT/lib/$LIB" | cut -d' ' -f1)" <<'EOF'
import json, sys
path, base, so, base_md5, md5 = sys.argv[1:]
json.dump({"base": base, "kernel_lib_from": so, "base_kernel_lib_md5": base_md5,
           "kernel_lib_md5": md5}, open(path, "w"), indent=2)
EOF
echo "package $OUT: kernel lib md5 $(md5sum "$OUT/lib/$LIB" | cut -d' ' -f1)"
