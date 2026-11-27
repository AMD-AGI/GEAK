#!/bin/bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Make a candidate hip-ep package: the base package with its kernel library
# swapped for a rebuilt one.
#
#   make_pkg.sh <base_pkg> <libcustom_kernels_<arch>.so> <out_pkg>
#
# The copy is hard-linked (about 1.9 GB per candidate otherwise), or copied across
# filesystems. The swapped library's link is removed before it is written, so the base
# is never written through. A failed run removes the package it started.
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
echo "package $OUT: kernel lib md5 $(md5sum "$OUT/lib/$LIB" | cut -d' ' -f1)"
