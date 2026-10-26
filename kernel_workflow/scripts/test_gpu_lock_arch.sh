#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# gpu_lock.sh must pin PYTORCH_ROCM_ARCH for the selected ROCR-visible GPU and
# refuse genuinely mixed allocations.
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

cat > "$tmp/rocminfo" <<'EOF'
#!/bin/sh
if [ "${ROCR_VISIBLE_DEVICES:-}" = "0" ]; then
  echo "  Name:                    gfx1201"
else
  echo "  Name:                    gfx1036"
  echo "  Name:                    gfx1201"
fi
EOF
chmod +x "$tmp/rocminfo"
out="$(PATH="$tmp:$PATH" bash -c '
  eval "$(sed -n "/^_rocminfo_gpu_gfx_list()/,/^}/p; /^_pin_compile_arch()/,/^}/p" "'"$LOCK"'")"
  KERNEL_ENV_KEEP_ARCH=0
  GPU_ID=0
  unset ROCR_VISIBLE_DEVICES
  _pin_compile_arch
  printf "%s" "$PYTORCH_ROCM_ARCH"
')"
[ "$out" = "gfx1201" ] \
  || fail "selected physical GPU should be probed through a scoped ROCR mask (got $out)"

# Exercise cache isolation without acquiring a GPU or invoking any compiler.
mkdir "$tmp/first" "$tmp/second"
python3 - "$LOCK" "$tmp" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
lock, root = map(Path, sys.argv[1:])
names = ('TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR', 'XDG_CACHE_HOME', 'AITER_JIT_DIR')
shared = dict.fromkeys(names, str(root / 'shared'))
shared['AITER_ROOT_DIR'] = '/installed/dependency'
body = '''eval "$(sed -n '/^_isolate_workspace_caches()/,/^}/p' "$1")"
_isolate_workspace_caches
python3 -c 'import json,os; print(json.dumps({k: os.environ[k] for k in ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "AITER_JIT_DIR", "AITER_ROOT_DIR")}))'
'''
def read(work, enabled):
    return json.loads(subprocess.check_output(['bash', '-c', body, 'cache-test', str(lock)],
        cwd=work, env={**os.environ, **shared, 'GEAK_ISOLATE_WORKSPACE_CACHES': str(enabled)}))
a, b = read(root / 'first', 1), read(root / 'second', 1)
assert all(a[k] != b[k] and Path(a[k]).is_relative_to(root / 'first')
           and Path(b[k]).is_relative_to(root / 'second') for k in names)
assert a['AITER_ROOT_DIR'] == b['AITER_ROOT_DIR'] == shared['AITER_ROOT_DIR']
assert read(root / 'first', 0) == shared
PY

# Container ordinals must resolve through observed PCI identities, not host order.
python3 - "$LOCK" "$tmp" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
source = Path(sys.argv[1]).read_text()
body = 'declare -A GPU_DEVICE_PATHS=()' + source.split('declare -A GPU_DEVICE_PATHS=()', 1)[1].split('# ---- GPU selection', 1)[0]
body += '\n_gpu_device_path 0; _gpu_device_path 1\n'
def run_raw(mapping, pool='0,1'):
    return subprocess.run(['bash', '-c', 'set -eu\nGPU_SPEC="$1"\n' + body, 'mapping-test', pool],
        env={**os.environ, 'GEAK_PHYSICAL_DEVICE_MAP': mapping}, capture_output=True, text=True)
def run(rows, pool='0,1'):
    return run_raw(json.dumps(rows), pool)
rows = [{'local_ordinal': 0, 'bdf': '0000:AD:00.0', 'physical_id': '0000:ad:00.0@0123456789abcdef'},
        {'local_ordinal': 1, 'bdf': '0000:08:00.0', 'physical_id': '0000:08:00.0@123456789abcdef0'}]
result = run(rows)
assert result.returncode == 0, result.stderr
assert result.stdout.splitlines() == ['/sys/bus/pci/devices/0000:ad:00.0', '/sys/bus/pci/devices/0000:08:00.0']
assert run(rows, '0,2').returncode != 0
assert run([rows[0], rows[0]]).returncode != 0
assert run([rows[0], {**rows[1], 'bdf': rows[0]['bdf'],
                      'physical_id': '0000:ad:00.0@123456789abcdef0'}]).returncode != 0
assert run([{**rows[0], 'bdf': '../drm'}, rows[1]]).returncode != 0
assert run([rows[0], {**rows[1], 'physical_id': rows[0]['physical_id']}]).returncode != 0
assert run('').returncode != 0
assert run_raw('').returncode != 0
assert run([{k: v for k, v in rows[0].items() if k != 'physical_id'}, rows[1]]).returncode != 0
for physical_id in ('device-a', '0000:ff:00.0@0123456789abcdef',
                    '0000:ad:00.0@0000000000000000', '0000:ad:00.0@0123456789ABCDEF'):
    assert run([{**rows[0], 'physical_id': physical_id}, rows[1]]).returncode != 0

# Readable sysfs paths can still fail or return empty data when read.
# Extract only the idle helper and substitute ordinary files; never run the wrapper.
device = Path(sys.argv[2]) / 'device'
device.mkdir()
(device / 'gpu_busy_percent').write_text('0\n')
(device / 'mem_info_vram_used').write_text('0\n')
idle = '_gpu_is_idle() {' + source.split('_gpu_is_idle() {', 1)[1].split('\n}', 1)[0] + '\n}\n'
def idle_with(mock=''):
    shell = 'set -eu\n' + idle + '\n_gpu_device_path() { printf "%s\\n" "$FIXTURE_DEVICE"; }\n' + mock
    shell += '\nif _gpu_is_idle 0; then exit 0; else exit 1; fi\n'
    return subprocess.run(['bash', '-c', shell], capture_output=True, text=True,
        env={**os.environ, 'FIXTURE_DEVICE': str(device), 'GEAK_PHYSICAL_DEVICE_MAP': json.dumps(rows)})
assert idle_with().returncode == 0
assert idle_with('cat() { return 1; }').returncode == 1
for field in ('gpu_busy_percent', 'mem_info_vram_used'):
    mock = 'cat() { case "$1" in */' + field + ') return 0;; *) command cat "$@";; esac; }'
    assert idle_with(mock).returncode == 1

# The single-GPU path must honor the caller's remaining queue allowance too.
# Stub only flock and execute its source line; no descriptor or GPU is acquired.
wait_line = next(line.strip() for line in source.splitlines() if line.strip().startswith('flock -x -w '))
wait_shell = 'set -eu\nGPU_ID=0\nflock() { printf "%s\\n" "$@"; }\n' + wait_line
wait_env = {k: v for k, v in os.environ.items() if k != 'GEAK_GPU_POOL_WAIT'}
for override, expected in (({}, '1200'), ({'GEAK_GPU_POOL_WAIT': '300'}, '300')):
    result = subprocess.run(['bash', '-c', wait_shell], env={**wait_env, **override}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['-x', '-w', expected, '200']
PY

echo "PASS: GPU architecture, observed device mapping, and workspace cache isolation."
