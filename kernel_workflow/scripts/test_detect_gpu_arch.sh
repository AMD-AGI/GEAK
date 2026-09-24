#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Allowlist / R9700 fail-closed tests for ci/lib.sh detect_gpu_arch (no GPU required).
set -euo pipefail
CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../ci" && pwd)"
# shellcheck source=/dev/null
source "$CI_DIR/lib.sh"

fail() { echo "FAIL: $*" >&2; exit 1; }

declare -F detect_gpu_arch >/dev/null || fail "lib.sh did not define detect_gpu_arch"
declare -F resolve_image >/dev/null || fail "lib.sh did not define resolve_image"
[ -n "${GEAK_GPU_ARCH_DEFAULT:-}" ] \
  || fail "GEAK_GPU_ARCH_DEFAULT is unset after sourcing lib.sh (want the value from ci/config.sh)"
[ -n "${DOCKER_DEFAULT:-}" ] && [ -f "$DOCKER_DEFAULT" ] \
  || fail "DOCKER_DEFAULT is missing after sourcing lib.sh"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/rocminfo" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod +x "$tmp/rocminfo"

# Prefixes go on the command inside the substitution so they do not leak into this
# shell (including under `set -o posix`) and so they apply to detect_gpu_arch itself.
override() {
  local val="$1"
  PATH="$tmp:/usr/bin:/bin" GEAK_EXPECTED_TARGET= GEAK_GPU_ARCH="$val" detect_gpu_arch
}

got="$(override gfx1201)" || fail "gfx1201 override"
[ "$got" = "gfx1201" ] || fail "gfx1201 must remain architecture-only (got $got)"

got="$(override RDNA4)" || fail "RDNA4 alias"
[ "$got" = "gfx1201" ] || fail "RDNA4 must remain architecture-only (got $got)"

err=""
if err="$(override r9700 2>&1)"; then
  fail "r9700 override without structured product identity must fail closed"
fi
[[ "$err" == *"E_TARGET_EXPECTED_MISMATCH"* ]] \
  || fail "r9700/no-identity stderr missing E_TARGET_EXPECTED_MISMATCH (got: $err)"

got="$(override gfx950)" || fail "gfx950 override"
[ "$got" = "MI355" ] || fail "gfx950 -> MI355 (got $got)"

err=""
if err="$(GEAK_EXPECTED_TARGET=r9700 GEAK_GPU_ARCH=gfx942 detect_gpu_arch 2>&1)"; then
  fail "expected R9700 with gfx942 override must fail closed"
fi
[[ "$err" == *"E_ARCH_EXPECTED_MISMATCH"* ]] \
  || fail "R9700/gfx942 override stderr missing E_ARCH_EXPECTED_MISMATCH (got: $err)"

err=""
if err="$(override gfx1200 2>&1)"; then
  fail "gfx1200 override must be rejected"
fi
[[ "$err" == *"E_ARCH_UNVALIDATED"* ]] || fail "gfx1200 stderr missing E_ARCH_UNVALIDATED (got: $err)"

err=""
if err="$(override not-a-gpu 2>&1)"; then
  fail "unknown GEAK_GPU_ARCH must be rejected"
fi
[[ "$err" == *"E_ARCH_UNSUPPORTED"* ]] || fail "unknown override stderr missing E_ARCH_UNSUPPORTED (got: $err)"

# Install a self-contained rocminfo stub with the same relevant field layout as
# real output. The CPU agent deliberately comes first so the parser must ignore
# gfx000 and select the GPU agent.
install_rocminfo_stub() {
  local gfx="$1" product="${2:-AMD GPU}" cu="${3:-64}"
  cat > "$tmp/rocminfo" <<EOF
#!/bin/sh
cat <<'ROCMINFO'
*******
Agent 1
*******
  Name:                    gfx000
  Uuid:                    CPU-00
  Marketing Name:          AMD Ryzen
  Vendor Name:             CPU
  Feature:                 None specified
*******
Agent 2
*******
  Name:                    $gfx
  Uuid:                    GPU-00
  Marketing Name:          $product
  Compute Unit:            $cu
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH
ROCMINFO
EOF
  chmod +x "$tmp/rocminfo"
}

# Isolation PATH so command -v rocminfo hits the stub, not a host binary.
# /opt/rocm/bin/rocminfo is only used when command -v finds nothing.
detect_from_rocminfo() {
  local gfx="$1" expected="${2:-}" product="${3:-AMD GPU}" cu="${4:-64}"
  install_rocminfo_stub "$gfx" "$product" "$cu"
  PATH="$tmp:/usr/bin:/bin" GEAK_EXPECTED_TARGET="$expected" GEAK_GPU_ARCH= detect_gpu_arch
}

got="$(detect_from_rocminfo gfx1201)" || fail "rocminfo gfx1201"
[ "$got" = "gfx1201" ] || fail "generic rocminfo gfx1201 must stay architecture-only (got $got)"

# Positive expected-target pairing requires exact structured product identity.
got="$(detect_from_rocminfo gfx1201 r9700 'AMD Radeon AI PRO R9700')" \
  || fail "expected R9700 with exact structured identity should succeed"
[ "$got" = "R9700" ] || fail "expected R9700 + gfx1201 -> R9700 (got $got)"

err=""
if err="$(detect_from_rocminfo gfx1201 r9700 'Another gfx1201 Product' 2>&1)"; then
  fail "expected R9700 with a generic gfx1201 product must fail closed"
fi
[[ "$err" == *"E_TARGET_EXPECTED_MISMATCH"* ]] \
  || fail "generic gfx1201 product mismatch missing E_TARGET_EXPECTED_MISMATCH (got: $err)"

err=""
if err="$(detect_from_rocminfo gfx1200 2>&1)"; then
  fail "rocminfo gfx1200 must be rejected"
fi
[[ "$err" == *"E_ARCH_UNVALIDATED"* ]] || fail "rocminfo gfx1200 stderr missing E_ARCH_UNVALIDATED (got: $err)"

got="$(detect_from_rocminfo gfx942)" || fail "rocminfo gfx942"
[ "$got" = "MI300" ] || fail "rocminfo gfx942 -> MI300 (got $got)"

# Expected-target mismatch: a job pinned to R9700 must reject an MI300 node.
err=""
if err="$(detect_from_rocminfo gfx942 r9700 2>&1)"; then
  fail "expected R9700 with rocminfo gfx942 must fail closed"
fi
[[ "$err" == *"E_TARGET_EXPECTED_MISMATCH"* ]] \
  || fail "R9700/gfx942 mismatch stderr missing E_TARGET_EXPECTED_MISMATCH (got: $err)"

# Verbose rocminfo: enough trailing lines that an early-close awk would SIGPIPE
# the writer. Callers (and this test) run under `set -o pipefail`.
install_rocminfo_stub gfx1201 'AMD Radeon AI PRO R9700'
{
  echo 'i=0'
  echo 'while [ "$i" -lt 8000 ]; do echo "  Padding: $i"; i=$((i + 1)); done'
} >> "$tmp/rocminfo"
got="$(PATH="$tmp:/usr/bin:/bin" GEAK_EXPECTED_TARGET= GEAK_GPU_ARCH= detect_gpu_arch)" \
  || fail "verbose rocminfo gfx1201 must not SIGPIPE under pipefail"
[ "$got" = "R9700" ] || fail "verbose rocminfo gfx1201 -> R9700 (got $got)"

# Empty stub: missing gfx while R9700 is expected fails closed; otherwise SPUR default.
cat > "$tmp/rocminfo" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod +x "$tmp/rocminfo"

err=""
if err="$(PATH="$tmp:/usr/bin:/bin" GEAK_EXPECTED_TARGET=r9700 GEAK_GPU_ARCH= detect_gpu_arch 2>&1)"; then
  fail "expected R9700 with empty rocminfo must fail closed"
fi
[[ "$err" == *"E_TARGET_EXPECTED_MISMATCH"* ]] \
  || fail "missing-rocminfo R9700 stderr missing E_TARGET_EXPECTED_MISMATCH (got: $err)"

got="$(PATH="$tmp:/usr/bin:/bin" GEAK_EXPECTED_TARGET= GEAK_GPU_ARCH= detect_gpu_arch)" \
  || fail "missing rocminfo without R9700 expectation should keep the SPUR default"
[ "$got" = "$GEAK_GPU_ARCH_DEFAULT" ] || fail "SPUR default when R9700 is not expected (got $got, default $GEAK_GPU_ARCH_DEFAULT)"

# A generic gfx1201 product must not receive the R9700 vLLM image.
install_rocminfo_stub gfx1201 'Another gfx1201 Product'
err=""
if err="$(PATH="$tmp:/usr/bin:/bin" GEAK_GPU_ARCH=gfx1201 resolve_image vllm 2>&1)"; then
  fail "generic gfx1201 must not resolve the R9700 vLLM image"
fi
[[ "$err" == *"E_IMAGE_UNAVAILABLE"* && "$err" == *"gfx1201"* ]] \
  || fail "generic gfx1201 image failure was not explicit (got: $err)"

# No R9700 SGLang image: resolve_image must fail rather than inherit an MI300 tag.
# resolve_image calls die/exit, so run it in a subshell.
install_rocminfo_stub gfx1201 'AMD Radeon AI PRO R9700'
err=""
if err="$(PATH="$tmp:/usr/bin:/bin" GEAK_GPU_ARCH=R9700 resolve_image sglang 2>&1)"; then
  fail "R9700 SGLang must not fall back to an MI300 image"
fi
[[ "$err" == *"E_IMAGE_UNAVAILABLE"* && "$err" == *"sglang"* && "$err" == *"R9700"* ]] \
  || fail "R9700 SGLang stderr missing fail-closed (got: $err)"

read -r want_r9700 mi300 < <(python3 - "$DOCKER_DEFAULT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])).get("vllm", {})
print(d.get("R9700", ""), d.get("MI300", ""))
PY
) || fail "could not read $DOCKER_DEFAULT"
[ -n "$want_r9700" ] || fail "docker_default.json missing vllm.R9700"
[ -n "$mi300" ] || fail "docker_default.json missing vllm.MI300"
got="$(PATH="$tmp:/usr/bin:/bin" GEAK_GPU_ARCH=R9700 resolve_image vllm)" || fail "R9700 vLLM image"
[ "$got" = "$want_r9700" ] || fail "R9700 vLLM must be the R9700 key, not a fallback (got $got, want $want_r9700)"
[ "$got" != "$mi300" ] || fail "R9700 vLLM resolved to the MI300 image"

echo "PASS: GPU arch overrides, rocminfo parsing, R9700 expectations, and image resolution."
