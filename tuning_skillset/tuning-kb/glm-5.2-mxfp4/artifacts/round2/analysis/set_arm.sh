#!/usr/bin/env bash
# Switch the live sglang tree between measurement arms by copying whole files.
# Usage: set_arm.sh <arm>
#   base   -> patches/base/*            (stock sglang at 2948168546)
#   r1     -> patches/applied/*         (round-1 patches 01+02+03)
#   <dir>  -> /work/patches/arms/<dir>/* overlaid on top of r1
set -euo pipefail
SG=/sgl-workspace/sglang
P1=$SG/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py
P2=$SG/python/sglang/srt/layers/attention/dsa_backend.py
P3=$SG/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
ARM="$1"
case "$ARM" in
  base) SRC=/work/patches/base ;;
  r1)   SRC=/work/patches/applied ;;
  *)    SRC=/work/patches/applied
        OVER=/work/patches/arms/$ARM
        [ -d "$OVER" ] || { echo "no such arm dir: $OVER" >&2; exit 1; } ;;
esac
cp "$SRC/tilelang_kernel.py" "$P1"
cp "$SRC/dsa_backend.py"     "$P2"
cp "$SRC/forward_mla.py"     "$P3"
if [ -n "${OVER:-}" ]; then
  [ -f "$OVER/tilelang_kernel.py" ] && cp "$OVER/tilelang_kernel.py" "$P1"
  [ -f "$OVER/dsa_backend.py" ]     && cp "$OVER/dsa_backend.py"     "$P2"
  [ -f "$OVER/forward_mla.py" ]     && cp "$OVER/forward_mla.py"     "$P3"
fi
find $SG/python -name '__pycache__' -path '*dsa*' -prune -exec rm -rf {} + 2>/dev/null || true
echo "[arm] $ARM"
md5sum "$P1" "$P2" "$P3"
