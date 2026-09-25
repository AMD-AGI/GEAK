#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# gpu_lock.sh must pin PYTORCH_ROCM_ARCH after HIP_VISIBLE_DEVICES and refuse mixed ISAs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$SCRIPT_DIR/gpu_lock.sh"

fail() { echo "FAIL: $*" >&2; exit 1; }

grep -n 'export HIP_VISIBLE_DEVICES=' "$LOCK" | grep -q . || fail "no HIP_VISIBLE_DEVICES pin"
# Every pin site must be followed (same block) by _pin_compile_arch.
python3 - "$LOCK" <<'PY' || fail "HIP_VISIBLE_DEVICES is not followed by _pin_compile_arch"
import pathlib, sys
text = pathlib.Path(sys.argv[1]).read_text()
idx = 0
hits = 0
while True:
    i = text.find("export HIP_VISIBLE_DEVICES=", idx)
    if i < 0:
        break
    hits += 1
    window = text[i:i+180]
    if "_pin_compile_arch" not in window:
        raise SystemExit(f"pin without _pin_compile_arch near: {window!r}")
    idx = i + 1
if hits < 2:
    raise SystemExit(f"expected >=2 HIP_VISIBLE_DEVICES pins, got {hits}")
PY

grep -q 'mixed GPU ISAs' "$LOCK" || fail "mixed-ISA refuse missing"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/rocminfo" <<'EOF'
#!/bin/sh
echo "  Name:                    gfx950"
echo "  Name:                    gfx1201"
EOF
chmod +x "$tmp/rocminfo"

# Source just the helper functions.
# shellcheck disable=SC1090
eval "$(sed -n '/^_rocminfo_gpu_gfx_list()/,/^}/p; /^_pin_compile_arch()/,/^}/p' "$LOCK")"
if PATH="$tmp:$PATH" KERNEL_ENV_KEEP_ARCH=0 _pin_compile_arch 2>/dev/null; then
  fail "mixed gfx950+gfx1201 must be refused"
fi

cat > "$tmp/rocminfo" <<'EOF'
#!/bin/sh
echo "  Name:                    gfx000"
echo "  Name:                    gfx1201"
EOF
out="$(PATH="$tmp:$PATH" bash -c '
  eval "$(sed -n "/^_rocminfo_gpu_gfx_list()/,/^}/p; /^_pin_compile_arch()/,/^}/p" "'"$LOCK"'")"
  KERNEL_ENV_KEEP_ARCH=0
  _pin_compile_arch
  printf "%s" "$PYTORCH_ROCM_ARCH"
')"
[ "$out" = "gfx1201" ] || fail "homogeneous gfx1201 should pin PYTORCH_ROCM_ARCH (got $out)"

echo "PASS: gpu_lock pins compile arch after the selected GPU and refuses mixed ISAs."
