#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# No-GPU checks for ci/preflight/w7900_vllm_smoke.sh: it parses, and it refuses anything but one
# Radeon Pro W7900 before it starts a server, with the refusal in its JSON result.
set -euo pipefail
SMOKE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/ci/preflight/w7900_vllm_smoke.sh"

fail() { echo "FAIL: $*" >&2; exit 1; }

bash -n "$SMOKE" || fail "bash -n $SMOKE"
rc=0
bash "$SMOKE" > /dev/null 2>&1 || rc=$?
[ "$rc" -eq 2 ] || fail "no arguments: exit $rc, want 2 (usage)"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/model"
echo '{}' > "$tmp/model/config.json"
# Any refusal that reaches the server is a failure.
cat > "$tmp/bin/vllm" <<'EOF'
#!/bin/sh
echo "vllm $*" >> "$(dirname "$0")/../vllm.calls"
exit 99
EOF
# The host's rocminfo reports a W7800: gfx1100, but not the W7900.
cat > "$tmp/bin/rocminfo" <<'EOF'
#!/bin/sh
cat <<'ROCMINFO'
*******
Agent 2
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon Pro W7800
  Compute Unit:            70
ROCMINFO
EOF
chmod +x "$tmp/bin/vllm" "$tmp/bin/rocminfo"

identity() {
  printf '{"gfx": "%s", "target": "%s", "marketing_name": "%s", "physical_cu_count": %s, "visible_gpu_agents": %s}' "$@"
}

refuse() {
  local name=$1 json=$2 needle=$3 out="$tmp/$1" rc=0
  env -u GEAK_GPU_ARCH -u GEAK_EXPECTED_TARGET PATH="$tmp/bin:$PATH" GEAK_GPU_IDENTITY_JSON="$json" \
    bash "$SMOKE" "$tmp/model" "$out" > "$out.log" 2>&1 || rc=$?
  [ "$rc" -eq 1 ] || fail "$name: exit $rc, want 1 ($(tail -1 "$out.log"))"
  grep -q 'W7900-VLLM-SMOKE: FAIL' "$out.log" || fail "$name: no FAIL line"
  grep -qF "$needle" "$out.log" || fail "$name: log is missing '$needle' ($(tail -1 "$out.log"))"
  python3 - "$out/w7900_vllm_smoke.json" <<'PY' || fail "$name: JSON result does not record the identity refusal"
import json, sys
result = json.load(open(sys.argv[1]))
assert result["status"] == "fail", result
assert [(c["check"], c["status"]) for c in result["checks"]] == [("identity", "fail")], result
PY
  [ ! -e "$tmp/vllm.calls" ] || fail "$name: vllm was started"
  echo "ok: $name refused"
}

refuse r9700 "$(identity gfx1201 r9700 'AMD Radeon AI PRO R9700' 64 1)" "target='r9700' (want 'w7900')"
refuse w7800 "$(identity gfx1100 unknown 'AMD Radeon Pro W7800' 70 1)" "target='unknown' (want 'w7900')"
refuse mi300x "$(identity gfx942 unknown 'AMD Instinct MI300X' 304 1)" "gfx='gfx942' (want 'gfx1100')"
refuse two-cards "$(identity gfx1100 w7900 'AMD Radeon Pro W7900' 96 2)" "visible_gpu_agents=2 (want 1)"
refuse not-json 'not json' 'identity is not JSON'
# A W7900 identity override on a host whose rocminfo reports a W7800: detect_gpu_arch catches it.
refuse override "$(identity gfx1100 w7900 'AMD Radeon Pro W7900' 96 1)" 'detect_gpu_arch did not resolve W7900'

echo "PASS: w7900_vllm_smoke.sh refuses every non-W7900 identity before starting vLLM."
