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
  local gfx="gfx1201" product="Another gfx1201 Product" cu="64"
  case "${val,,}" in
    gfx950|mi355|mi350|mi350x) gfx="gfx950"; product="AMD Instinct MI355X"; cu="256" ;;
    gfx942|gfx90a|mi300|mi300x) gfx="gfx942"; product="AMD Instinct MI300X"; cu="304" ;;
    gfx1200) gfx="gfx1200"; product="AMD Radeon Graphics"; cu="56" ;;
  esac
  cat > "$tmp/rocminfo" <<EOF
#!/bin/sh
cat <<'ROCMINFO'
*******
Agent 1
*******
  Name:                    $gfx
  Marketing Name:          $product
  Compute Unit:            $cu
ROCMINFO
EOF
  chmod +x "$tmp/rocminfo"
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

# An identified ISA without a GEAK image policy is refused, not sent to MI355.
for spec in 'gfx1151|AMD Radeon 8060S Graphics|40' 'gfx1101|AMD Radeon Pro W7700|48' 'gfx1030|AMD Radeon PRO W6800|60'; do
  IFS='|' read -r gfx product cu <<<"$spec"
  err=""
  if err="$(detect_from_rocminfo "$gfx" '' "$product" "$cu" 2>&1)"; then
    fail "$gfx must not resolve to an image bucket (got $err)"
  fi
  [[ "$err" == *"E_ARCH_UNSUPPORTED"* ]] || fail "$gfx stderr missing E_ARCH_UNSUPPORTED (got: $err)"
done

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

# Empty/invalid rocminfo is an identity failure and must never select the SPUR
# default. The default is retained only when no identity helper is available.
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
  || [[ "$err" == *"E_GPU_IDENTITY"* ]] \
  || fail "missing-rocminfo R9700 stderr missing identity failure (got: $err)"

err=""
if err="$(PATH="$tmp:/usr/bin:/bin" GEAK_EXPECTED_TARGET= GEAK_GPU_ARCH= detect_gpu_arch 2>&1)"; then
  fail "an installed but invalid rocminfo probe must fail rather than select the SPUR default"
fi
[[ "$err" == *"E_GPU_IDENTITY"* ]] \
  || fail "invalid rocminfo stderr missing E_GPU_IDENTITY (got: $err)"

got="$(PATH="$tmp:/usr/bin:/bin" GPU_IDENTITY_SCRIPT="$tmp/missing-identity-helper" \
       GEAK_EXPECTED_TARGET= GEAK_GPU_ARCH= detect_gpu_arch)" \
  || fail "an unavailable identity helper should retain the SPUR default"
[ "$got" = "$GEAK_GPU_ARCH_DEFAULT" ] \
  || fail "SPUR default when no probe is available (got $got, default $GEAK_GPU_ARCH_DEFAULT)"

# A typical R9700 workstation may expose a generic Ryzen iGPU. The narrow
# integrated-agent filter must keep the exact R9700 identity.
cat > "$tmp/rocminfo" <<'EOF'
#!/bin/sh
cat <<'ROCMINFO'
*******
Agent 1
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon AI PRO R9700
  Compute Unit:            64
*******
Agent 2
*******
  Name:                    gfx1036
  Marketing Name:          AMD Radeon Graphics
  Compute Unit:            2
ROCMINFO
EOF
chmod +x "$tmp/rocminfo"
got="$(PATH="$tmp:/usr/bin:/bin" GEAK_EXPECTED_TARGET=r9700 GEAK_GPU_ARCH= detect_gpu_arch)" \
  || fail "R9700 plus generic iGPU should resolve"
[ "$got" = "R9700" ] || fail "R9700+iGPU resolved to $got"

# An unrelated discrete product must not be hidden by the iGPU exception.
cat > "$tmp/rocminfo" <<'EOF'
#!/bin/sh
cat <<'ROCMINFO'
*******
Agent 1
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon AI PRO R9700
  Compute Unit:            64
*******
Agent 2
*******
  Name:                    gfx950
  Marketing Name:          AMD Instinct MI355X
  Compute Unit:            256
ROCMINFO
EOF
chmod +x "$tmp/rocminfo"
err=""
if err="$(PATH="$tmp:/usr/bin:/bin" GEAK_EXPECTED_TARGET= GEAK_GPU_ARCH= detect_gpu_arch 2>&1)"; then
  fail "a genuinely mixed discrete host must fail identity"
fi
[[ "$err" == *"mixed identities"* ]] \
  || fail "mixed-discrete stderr missing explicit identity failure (got: $err)"

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

model_cfg="$tmp/docker-model-string.json"
python3 - "$DOCKER_DEFAULT" "$model_cfg" <<'PY'
import json, sys
source, target = sys.argv[1:3]
with open(source, encoding="utf-8") as handle:
    data = json.load(handle)
data["models"] = {"string-pin": "example.invalid/mi-only:latest"}
with open(target, "w", encoding="utf-8") as handle:
    json.dump(data, handle)
PY
got="$(PATH="$tmp:/usr/bin:/bin" DOCKER_DEFAULT="$model_cfg" GEAK_GPU_ARCH=R9700 \
       resolve_image vllm string-pin)" || fail "R9700 should skip a plain-string per-model pin"
[ "$got" = "$want_r9700" ] \
  || fail "R9700 inherited a plain-string model pin instead of vllm.R9700 (got $got)"

# W7900 / gfx1100: only the exact product names resolve, and nothing on gfx1100
# falls through to the SPUR default.
for product in 'AMD Radeon Pro W7900' 'AMD Radeon PRO W7900 Dual Slot'; do
  got="$(detect_from_rocminfo gfx1100 '' "$product" 96)" || fail "rocminfo $product"
  [ "$got" = "W7900" ] || fail "$product -> W7900 (got $got)"
  got="$(detect_from_rocminfo gfx1100 w7900 "$product" 96)" || fail "expected W7900 + $product"
  [ "$got" = "W7900" ] || fail "expected W7900 + $product -> W7900 (got $got)"
done

for product in 'AMD Radeon Pro W7800' 'AMD Radeon Pro W7800 48GB' 'AMD Radeon RX 7900 XTX' ''; do
  err=""
  if err="$(detect_from_rocminfo gfx1100 '' "$product" 96 2>&1)"; then
    fail "gfx1100 '$product' must be refused (got $err)"
  fi
  [[ "$err" == *"E_ARCH_UNVALIDATED"* && "$err" != *"$GEAK_GPU_ARCH_DEFAULT"$'\n'* ]] \
    || fail "gfx1100 '$product' stderr missing E_ARCH_UNVALIDATED (got: $err)"
  err=""
  if err="$(detect_from_rocminfo gfx1100 w7900 "$product" 96 2>&1)"; then
    fail "expected W7900 with gfx1100 '$product' must fail closed"
  fi
  [[ "$err" == *"E_TARGET_EXPECTED_MISMATCH"* ]] \
    || fail "expected W7900 + '$product' stderr missing E_TARGET_EXPECTED_MISMATCH (got: $err)"
done

err=""
if err="$(detect_from_rocminfo gfx1201 w7900 'AMD Radeon AI PRO R9700' 2>&1)"; then
  fail "expected W7900 on an R9700 must fail closed"
fi
[[ "$err" == *"E_TARGET_EXPECTED_MISMATCH"* ]] \
  || fail "W7900/R9700 stderr missing E_TARGET_EXPECTED_MISMATCH (got: $err)"
err=""
if err="$(detect_from_rocminfo gfx1100 r9700 'AMD Radeon Pro W7900' 96 2>&1)"; then
  fail "expected R9700 on a W7900 must fail closed"
fi
[[ "$err" == *"E_TARGET_EXPECTED_MISMATCH"* ]] \
  || fail "R9700/W7900 stderr missing E_TARGET_EXPECTED_MISMATCH (got: $err)"
err=""
if err="$(detect_from_rocminfo gfx1201 '' 'AMD Radeon Pro W7900' 96 2>&1)"; then
  fail "a W7900 name on gfx1201 must fail identity"
fi
[[ "$err" == *"requires gfx1100"* ]] || fail "W7900 on gfx1201 stderr missing ISA failure (got: $err)"

# Overrides: GEAK_GPU_ARCH=W7900 is a product request, gfx1100 is ISA-only.
install_rocminfo_stub gfx1100 'AMD Radeon Pro W7900 Dual Slot' 96
for val in W7900 gfx1100; do
  got="$(PATH="$tmp:/usr/bin:/bin" GEAK_EXPECTED_TARGET= GEAK_GPU_ARCH="$val" detect_gpu_arch)" \
    || fail "$val override on a W7900"
  [ "$got" = "W7900" ] || fail "$val override on a W7900 -> W7900 (got $got)"
done
install_rocminfo_stub gfx1100 'AMD Radeon Pro W7800' 70
err=""
if err="$(PATH="$tmp:/usr/bin:/bin" GEAK_EXPECTED_TARGET= GEAK_GPU_ARCH=W7900 detect_gpu_arch 2>&1)"; then
  fail "W7900 override on a W7800 must fail closed"
fi
[[ "$err" == *"E_TARGET_EXPECTED_MISMATCH"* ]] \
  || fail "W7900 override/W7800 stderr missing E_TARGET_EXPECTED_MISMATCH (got: $err)"
err=""
if err="$(GEAK_EXPECTED_TARGET=w7900 GEAK_GPU_ARCH=gfx942 detect_gpu_arch 2>&1)"; then
  fail "expected W7900 with gfx942 override must fail closed"
fi
[[ "$err" == *"E_ARCH_EXPECTED_MISMATCH"* ]] \
  || fail "W7900/gfx942 override stderr missing E_ARCH_EXPECTED_MISMATCH (got: $err)"
for val in W7900 gfx1100; do
  err=""
  if err="$(PATH="$tmp:/usr/bin:/bin" GPU_IDENTITY_SCRIPT="$tmp/missing-identity-helper" \
            GEAK_EXPECTED_TARGET= GEAK_GPU_ARCH="$val" detect_gpu_arch 2>&1)"; then
    fail "$val override without a probe must fail closed (got $err)"
  fi
  [[ "$err" == *"E_TARGET_EXPECTED_MISMATCH"* || "$err" == *"E_ARCH_UNVALIDATED"* ]] \
    || fail "$val override without a probe stderr missing refusal (got: $err)"
done
err=""
if err="$(PATH="$tmp:/usr/bin:/bin" GPU_IDENTITY_SCRIPT="$tmp/missing-identity-helper" \
          GEAK_EXPECTED_TARGET=w7900 GEAK_GPU_ARCH= detect_gpu_arch 2>&1)"; then
  fail "expected W7900 without a probe must not select the SPUR default"
fi
[[ "$err" == *"E_TARGET_EXPECTED_MISMATCH"* ]] \
  || fail "W7900/no-probe stderr missing E_TARGET_EXPECTED_MISMATCH (got: $err)"

err=""
if err="$(PATH="$tmp:/usr/bin:/bin" GEAK_EXPECTED_TARGET=w7800 GEAK_GPU_ARCH= detect_gpu_arch 2>&1)"; then
  fail "GEAK_EXPECTED_TARGET=w7800 must be rejected"
fi
[[ "$err" == *"E_TARGET_UNSUPPORTED"* ]] || fail "w7800 target stderr missing E_TARGET_UNSUPPORTED (got: $err)"

# W7900 vLLM resolves only through its own vllm.W7900 key, which pins the rocm/vllm tag the
# W7900 was validated with. SGLang has no W7900 image and fails closed instead of inheriting MI300.
install_rocminfo_stub gfx1100 'AMD Radeon Pro W7900 Dual Slot' 96
want_w7900="$(python3 - "$DOCKER_DEFAULT" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("vllm", {}).get("W7900", ""))
PY
)" || fail "could not read $DOCKER_DEFAULT"
[ "$want_w7900" = "rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0" ] \
  || fail "vllm.W7900 is not the rocm/vllm tag the W7900 was validated with (got $want_w7900)"
got="$(PATH="$tmp:/usr/bin:/bin" GEAK_GPU_ARCH=W7900 resolve_image vllm)" || fail "W7900 vLLM image"
[ "$got" = "$want_w7900" ] || fail "W7900 vLLM must be the W7900 key (got $got, want $want_w7900)"
[ "$got" != "$mi300" ] || fail "W7900 vLLM resolved to the MI300 image"
err=""
if err="$(PATH="$tmp:/usr/bin:/bin" GEAK_GPU_ARCH=W7900 resolve_image sglang 2>&1)"; then
  fail "W7900 sglang must not resolve an image (got $err)"
fi
[[ "$err" == *"E_IMAGE_UNAVAILABLE"* && "$err" == *"sglang"* && "$err" == *"W7900"* ]] \
  || fail "W7900 sglang image failure was not explicit (got: $err)"
no_w7900_cfg="$tmp/docker-no-w7900.json"
python3 - "$DOCKER_DEFAULT" "$no_w7900_cfg" <<'PY'
import json, sys
source, target = sys.argv[1:3]
with open(source, encoding="utf-8") as handle:
    data = json.load(handle)
data["vllm"].pop("W7900", None)
data["vllm"]["default"] = "example.invalid/mi-default:latest"
with open(target, "w", encoding="utf-8") as handle:
    json.dump(data, handle)
PY
err=""
if err="$(PATH="$tmp:/usr/bin:/bin" DOCKER_DEFAULT="$no_w7900_cfg" GEAK_GPU_ARCH=W7900 \
          resolve_image vllm 2>&1)"; then
  fail "W7900 vllm without a W7900 key must not resolve an image (got $err)"
fi
[[ "$err" == *"E_IMAGE_UNAVAILABLE"* && "$err" == *"W7900"* ]] \
  || fail "W7900 vllm failure without a W7900 key was not explicit (got: $err)"
w7900_cfg="$tmp/docker-w7900.json"
python3 - "$DOCKER_DEFAULT" "$w7900_cfg" <<'PY'
import json, sys
source, target = sys.argv[1:3]
with open(source, encoding="utf-8") as handle:
    data = json.load(handle)
data["vllm"]["W7900"] = "example.invalid/w7900-vllm:pinned"
data["models"] = {"string-pin": "example.invalid/mi-only:latest"}
with open(target, "w", encoding="utf-8") as handle:
    json.dump(data, handle)
PY
got="$(PATH="$tmp:/usr/bin:/bin" DOCKER_DEFAULT="$w7900_cfg" GEAK_GPU_ARCH=W7900 resolve_image vllm)" \
  || fail "explicit vllm.W7900 entry should resolve"
[ "$got" = "example.invalid/w7900-vllm:pinned" ] || fail "W7900 picked $got instead of vllm.W7900"
got="$(PATH="$tmp:/usr/bin:/bin" DOCKER_DEFAULT="$w7900_cfg" GEAK_GPU_ARCH=W7900 \
       resolve_image vllm string-pin)" || fail "W7900 should skip a plain-string per-model pin"
[ "$got" = "example.invalid/w7900-vllm:pinned" ] \
  || fail "W7900 inherited a plain-string model pin instead of vllm.W7900 (got $got)"

echo "PASS: GPU arch overrides, rocminfo parsing, R9700 and W7900 expectations, and image resolution."
