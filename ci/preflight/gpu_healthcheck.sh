#!/usr/bin/env bash
# GPU preflight probe — run INSIDE the framework container (same image + --device
# flags as the real run) to confirm the AMD GPU is actually usable BEFORE we
# commit hours to a run. Fails LOUDLY and fast; the caller also caps us with an
# outer `timeout` so a wedged driver can't make this hang forever.
#
# Checks, in order (first failure exits non-zero with a clear message):
#   1. rocminfo — the ROCm runtime sees at least one gfx GPU agent.
#   2. torch    — torch.cuda.is_available() + device_count>=1 + a tiny matmul on
#                 GPU 0 that actually round-trips through the device.
#
# This is a normal usability probe, NOT a driver-wedge forensic tool: it just
# answers "can we run work on this GPU right now?".
set -uo pipefail

fail() {
  echo "::error::gpu_healthcheck: $*" >&2
  echo "GPU-HEALTHCHECK: FAIL — $*" >&2
  exit 1
}

echo ">>> gpu_healthcheck: rocminfo (ROCm sees a gfx GPU agent) ..."
# Capture rocminfo output first, THEN grep it. Piping straight into `grep -q`
# makes grep close the pipe on its first match, so rocminfo (which prints a lot)
# dies with SIGPIPE (exit 141); under `set -o pipefail` that non-zero would sink
# the whole pipeline and spuriously report "no gfx agent" even on a healthy GPU.
ROCMINFO_OUT="$(rocminfo 2>/dev/null || true)"
if ! grep -qE 'gfx[0-9]' <<<"$ROCMINFO_OUT"; then
  fail "rocminfo found no gfx GPU agent (ROCm/KFD not usable in this container)"
fi

echo ">>> gpu_healthcheck: torch CUDA/HIP available + tiny matmul on GPU 0 ..."
python3 - <<'PY' || fail "torch GPU probe failed (see error above)"
import sys
try:
    import torch
except Exception as e:
    print(f"import torch failed: {e}", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("torch.cuda.is_available() == False", file=sys.stderr)
    sys.exit(1)
if torch.cuda.device_count() < 1:
    print("torch.cuda.device_count() < 1", file=sys.stderr)
    sys.exit(1)
try:
    x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize()
    assert tuple(y.shape) == (256, 256)
except Exception as e:
    print(f"GPU matmul failed: {e}", file=sys.stderr)
    sys.exit(1)
print(f"torch {torch.__version__} OK — ndev={torch.cuda.device_count()} dev0={torch.cuda.get_device_name(0)}")
PY

echo "GPU-HEALTHCHECK: OK"
