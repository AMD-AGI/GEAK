#!/usr/bin/env bash
# Assert the container is a stack the reference number can be compared against, and record what it
# actually is. Run INSIDE the container, once, before measuring anything.
#
#   ./preflight.sh
#
# The source session records sglang 0.5.15.post1 and ROCm 7.2.0 and nothing else -- no aiter commit.
# The image tag came out of band and is pinned in scripts/start_container.sh; it reports the build in
# full as 0.5.15.post1.dev20260723+g6c9fd0adc5, which the release-prefix match below accepts. So this
# script does two jobs: it fails on what *is* recorded, and prints what is not, for FINDINGS.md. A
# silent version skew invalidates every number measured afterwards; that is how the sibling Gemma
# experiment lost 4.4% of its baseline to an unnoticed Triton bump.
#
# Overridable to run a deliberate off-reference stack, e.g. EXPECT_FW=0.5.15.99 ./preflight.sh --
# announced loudly, because a number from another stack cannot be quoted against 804.190 without
# saying so.
set -uo pipefail

EXPECT_FW="${EXPECT_FW:-0.5.15.post1}"
EXPECT_GPUS="${EXPECT_GPUS:-8}"
MODEL="${MODEL:-/shared_nfs/hyperloom/models/Kimi-K3}"
EXPECT_SHARDS="${EXPECT_SHARDS:-96}"
EXPECT_ROCM="${EXPECT_ROCM:-7.2.0}"

if [ "$EXPECT_FW" != "0.5.15.post1" ]; then
    echo "[preflight] *** OFF-REFERENCE STACK REQUESTED: sglang $EXPECT_FW"
    echo "[preflight] *** 804.190 tok/s was measured on sglang 0.5.15.post1."
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

# The version string alone does not tell you whether this tree can serve the model: the released
# 0.5.15.post1 and the K3 build 0.5.15.post1.dev20260723 share the prefix the check above matches on,
# and only the second registers the architecture. Asserting the registry entry is what separates
# "right version number" from "right tree", and it is cheap compared with discovering it 25 minutes
# into a weight load. Import needs the GPUs: sglang resolves the arch through rocminfo at import, so
# outside a GPU-mapped container every model module fails and this reports a false negative.
echo "[preflight] model support"
ARCHOK="$(python3 -c '
from sglang.srt.models.registry import ModelRegistry
print("KimiK3ForConditionalGeneration" in ModelRegistry.get_supported_archs())' 2>/dev/null)"
if [ "$ARCHOK" = "True" ]; then
    ok "KimiK3ForConditionalGeneration registered"
else
    bad "KimiK3ForConditionalGeneration NOT in the model registry -- this tree cannot serve Kimi-K3;"
    note "     use the K3 image named in scripts/start_container.sh"
fi
# Read the parser choices out of argparse rather than grepping `launch_server --help`: under
# `set -o pipefail`, piping a 2196-line help text into `grep -q` makes grep exit at the first match,
# python die of SIGPIPE, and the whole pipeline report failure on a build that is in fact fine.
PARSERS="$(python3 -c '
import argparse
from sglang.srt.server_args import ServerArgs
ap = argparse.ArgumentParser(); ServerArgs.add_cli_args(ap)
for f in ("--reasoning-parser", "--tool-call-parser"):
    a = next((x for x in ap._actions if f in x.option_strings), None)
    print(f, "yes" if a is not None and (a.choices is None or "kimi_k3" in a.choices) else "no")' 2>/dev/null)"
for f in --reasoning-parser --tool-call-parser; do
    case "$PARSERS" in
        *"$f yes"*) ok "$f accepts kimi_k3" ;;
        *) bad "$f rejects kimi_k3 -- the frozen config cannot be honoured on this build" ;;
    esac
done

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
echo "[preflight] reference to reproduce: 804.190 tok/s output throughput"
echo "            three rounds 0.23% apart; 1.56 TB of weights over 96 shards, so a first load is slow and a failed one is expensive"
if [ "$FAIL" = 0 ]; then
    echo "[preflight] PASS -- record the notes above in FINDINGS.md, then measure the baseline"
else
    echo "[preflight] FAIL -- do not measure until this is resolved; a number from a different stack" >&2
    echo "            is not comparable to 804.190 and cannot be quoted against it." >&2
fi
exit "$FAIL"
