#!/usr/bin/env bash
# Assert the container is a stack the reference number can be compared against, and record what it
# actually is. Run INSIDE the container, once, before measuring anything.
#
#   ./preflight.sh
#
# The source session records sglang 0.5.17 and ROCm 7.2.0 and nothing else -- no image tag, no aiter commit. So this
# script does two jobs: it fails on what *is* recorded, and prints what is not, for FINDINGS.md. A
# silent version skew invalidates every number measured afterwards; that is how the sibling Gemma
# experiment lost 4.4% of its baseline to an unnoticed Triton bump.
#
# Overridable to run a deliberate off-reference stack, e.g. EXPECT_FW=0.5.99 ./preflight.sh --
# announced loudly, because a number from another stack cannot be quoted against 1279.949 without
# saying so.
set -uo pipefail

EXPECT_FW="${EXPECT_FW:-0.5.17}"
EXPECT_GPUS="${EXPECT_GPUS:-8}"
MODEL="${MODEL:-/shared_nfs/hyperloom/models/Qwen3.8-2.4T-A95B-Quark-MXFP4}"
EXPECT_SHARDS="${EXPECT_SHARDS:-213}"
EXPECT_ROCM="${EXPECT_ROCM:-7.2.0}"

if [ "$EXPECT_FW" != "0.5.17" ]; then
    echo "[preflight] *** OFF-REFERENCE STACK REQUESTED: sglang $EXPECT_FW"
    echo "[preflight] *** 1279.949 tok/s was measured on sglang 0.5.17."
    echo "[preflight] *** Label any result from this stack accordingly."
fi

FAIL=0
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { printf '  BAD   %s\n' "$1"; FAIL=1; }
note() { printf '  note  %s\n' "$1"; }

echo "[preflight] stack"
# The recorded reference is a release number; installed builds carry a suffix. Match on the release
# prefix and print the full string for the record.
FW="$(python3 -c 'from importlib.metadata import version; print(version("sglang"))' 2>/dev/null)"
case "$FW" in
    "$EXPECT_FW"|"$EXPECT_FW".*|"$EXPECT_FW"+*) ok "sglang = $FW" ;;
    "") bad "sglang not installed" ;;
    *)  bad "sglang: expected $EXPECT_FW, got $FW" ;;
esac

ROCM="$(cut -d- -f1 /opt/rocm/.info/version 2>/dev/null)"
if [ "$ROCM" = "$EXPECT_ROCM" ]; then ok "rocm = $ROCM"; else bad "rocm: expected $EXPECT_ROCM, got ${ROCM:-none}"; fi

# Not recorded by the session, so not assertable -- but printed, because these are what a future
# unexplained delta will turn out to have been.
note "torch   = $(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null)"
note "triton  = $(python3 -c 'import triton;print(triton.__version__)' 2>/dev/null)"
note "python  = $(python3 -V 2>&1 | awk '{print $2}') at $(command -v python3)"
note "aiter sha = $(git -C /sgl-workspace/aiter rev-parse HEAD 2>/dev/null || echo unknown)"
note "sglang sha = $(git -C /sgl-workspace/sglang rev-parse HEAD 2>/dev/null || echo 'unknown (not a source checkout)')"
note "sglang at = $(python3 -c 'import sglang,os;print(os.path.dirname(sglang.__file__))' 2>/dev/null)"
note "host    = $(hostname)"

# The frozen config turns aiter on explicitly. If aiter cannot be imported the server either refuses
# to start or silently falls back, and either way the number is not the reference configuration.
if python3 -c 'import aiter' 2>/dev/null; then
    ok "aiter importable ($(python3 -c 'import aiter,os;print(os.path.dirname(aiter.__file__))' 2>/dev/null))"
else
    bad "aiter not importable -- the aiter settings in the frozen config cannot be honoured"
fi

echo "[preflight] devices"
NGPU="$(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null)"
if [ "$NGPU" = "$EXPECT_GPUS" ]; then ok "visible GPUs = $NGPU"; else bad "visible GPUs: expected $EXPECT_GPUS, got ${NGPU:-0}"; fi
ARCH="$(python3 -c 'import torch;print(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])' 2>/dev/null)"
if [ "$ARCH" = "gfx950" ]; then ok "arch = $ARCH (MI355X)"; else bad "arch: expected gfx950, got ${ARCH:-none}"; fi

# A neighbour holding VRAM on a device this experiment needs is not a nuisance, it is a failed server
# start (or worse, a quietly degraded one). rocm-smi sees the whole node; only the mapped devices are
# ours, so this reports rather than fails, and names what to go and check.
BUSY="$(rocm-smi --showmemuse --csv 2>/dev/null | awk -F, 'NR>1 && $2+0 > 2 {c++} END {print c+0}')"
if [ "${BUSY:-0}" -gt 0 ]; then
    note "$BUSY device(s) on this NODE hold >2% VRAM -- confirm none of them is one of ours:"
    rocm-smi --showmemuse 2>/dev/null | sed 's/^/        /' | head -20
else
    ok "no device on this node holds VRAM"
fi

echo "[preflight] model"
if [ -f "$MODEL/config.json" ]; then
    ok "config.json present at $MODEL"
    N="$(ls "$MODEL"/*.safetensors 2>/dev/null | wc -l)"
    if [ "$N" = "$EXPECT_SHARDS" ]; then ok "safetensors shards = $N"; else bad "safetensors shards: expected $EXPECT_SHARDS, got $N"; fi
else
    bad "no model at $MODEL"
fi

echo
echo "[preflight] reference to reproduce: 1279.949 tok/s output throughput"
echo "            three rounds 0.66% apart; the session ran a 0.5.17.dev20260812 build and this tag is the 0.5.17 release, so record the version the container actually reports"
if [ "$FAIL" = 0 ]; then
    echo "[preflight] PASS -- record the notes above in FINDINGS.md, then measure the baseline"
else
    echo "[preflight] FAIL -- do not measure until this is resolved; a number from a different stack" >&2
    echo "            is not comparable to 1279.949 and cannot be quoted against it." >&2
fi
exit "$FAIL"
