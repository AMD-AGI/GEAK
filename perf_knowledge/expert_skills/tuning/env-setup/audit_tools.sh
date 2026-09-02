#!/usr/bin/env bash
# Inventory the tuning tools present in the current environment.
# Run inside the container you intend to tune in. Read-only; changes nothing.
#
#   docker exec <container> bash /path/to/audit_tools.sh
#
# Every line is  STATUS  NAME  DETAIL. Grep for MISSING to get your work list.

say() { printf '%-8s %-26s %s\n' "$1" "$2" "$3"; }

have_bin() {   # name  binary  [version-cmd]
  local p; p=$(command -v "$2" 2>/dev/null)
  if [ -n "$p" ]; then say OK "$1" "$p"; else say MISSING "$1" "no '$2' on PATH"; fi
}

have_py() {    # name  import  expr
  local out
  if out=$(python3 -c "import $2; print($3)" 2>&1); then
    say OK "$1" "$out"
  else
    say MISSING "$1" "$(printf '%s' "$out" | tail -1 | cut -c1-90)"
  fi
}

echo "=== platform ==="
say INFO rocm-version   "$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
say INFO gpu-arch       "$(rocminfo 2>/dev/null | awk '/gfx[0-9]/{print $2; exit}' || echo 'rocminfo failed - GPUs not passed in?')"
say INFO visible-gpus   "${HIP_VISIBLE_DEVICES:-<all>}"

echo
echo "=== bench clients (tuners that race candidates) ==="
have_bin  hipblaslt-bench   hipblaslt-bench
have_bin  ckProfiler        ckProfiler
have_bin  rocblas-bench     rocblas-bench

echo
echo "=== python tuning stacks ==="
have_py   torch             torch  "torch.__version__"
have_py   torch-tunableop   torch  "'yes' if hasattr(torch.cuda,'tunable') else 'no'"
have_py   triton            triton "triton.__version__"
have_py   triton-autotune   triton "'yes' if hasattr(triton,'autotune') else 'no'"
have_py   aiter             aiter  "getattr(aiter,'__version__','present')"
have_py   flydsl            flydsl "getattr(flydsl,'__version__','present')"
have_py   vllm              vllm   "vllm.__version__"
have_py   sglang            sglang "sglang.__version__"

echo
echo "=== aiter tuner entry points ==="
AITER=$(python3 -c "import aiter,os;print(os.path.dirname(aiter.__file__))" 2>/dev/null)
if [ -n "$AITER" ]; then
  ROOT=$(dirname "$AITER")
  say INFO aiter-root "$ROOT"
  for f in gradlib/gemm_tuner.py csrc/ck_gemm_a8w8/gemm_a8w8_tune.py \
           csrc/ck_gemm_a4w4/gemm_a4w4_tune.py aiter/configs; do
    [ -e "$ROOT/$f" ] && say OK "$(basename "$f")" "$ROOT/$f" || say MISSING "$(basename "$f")" "$ROOT/$f"
  done
else
  say MISSING aiter-tuners "aiter not importable"
fi

echo
echo "=== aiter kernelName dispatch (blockscale GEMM tuning is undeployable without it) ==="
# aiter 7136b240e, 2026-05-21, "blockscale gemm: dispatch by kernelName" (#3075), first in v0.1.15.
# Below this the serving wrapper reads only `libtype`, so a tuned row selects a LIBRARY and not the
# instance you tuned. Measured cost of not checking: -6.48% e2e where +23.88% was available.
KNFIX=7136b240e
if [ -n "${AITER:-}" ]; then
  W="$AITER/ops/gemm_op_a8w8.py"
  if [ -f "$W" ]; then
    # Test the op SIGNATURE, not just "does kernelName appear somewhere in the wrapper".
    # On the broken build the asm branch forwards kernelName while ck/cktile do not, so any
    # looser grep returns a false OK -- verified against both images.
    python3 - "$W" <<'PY'
import re, sys
src = open(sys.argv[1], errors="ignore").read()
def say(s, n, d): print("%-8s %-26s %s" % (s, n, d))
def sig(name):
    m = re.search(r"^def %s\((.*?)\)\s*(->|:)" % re.escape(name), src, re.M | re.S)
    return m.group(1) if m else None
def body(name):
    m = re.search(r"^def %s\(.*?\n(.*?)(?=^def [a-zA-Z])" % re.escape(name), src, re.M | re.S)
    return m.group(1) if m else ""

broken = False
for op in ("gemm_a8w8_blockscale_bpreshuffle_cktile", "gemm_a8w8_blockscale_bpreshuffle_ck"):
    tag = "sig-" + op.rsplit("_", 1)[-1]
    s = sig(op)
    if s is None:
        say("INFO", tag, "not found: " + op)
    elif "kernelName" in s:
        say("OK", tag, op + " accepts kernelName")
    else:
        broken = True
        say("BROKEN", tag, op + " takes NO kernelName -- it cannot be told which instance to run")

b = body("gemm_a8w8_blockscale_bpreshuffle")
m = re.search(r"gemm_a8w8_blockscale_bpreshuffle_cktile\((.*?)\)", b, re.S)
if m is None:
    say("INFO", "wrapper-forwards", "no cktile branch found in the serving wrapper")
elif "kernelName" in m.group(1):
    say("OK", "wrapper-forwards", "cktile branch forwards kernelName")
else:
    broken = True
    say("BROKEN", "wrapper-forwards", "cktile branch drops kernelName -- picks a LIBRARY, not your tuned instance")

if broken:
    say("ACTION", "update-aiter",
        "predates 7136b240e (#3075, 2026-05-21); need >= v0.1.15. Here only libtype=asm rows "
        "deploy, and a ck/cktile table REGRESSES e2e (measured -6.48% where +23.88% was available)")
PY
  else
    say INFO kernelName-dispatch "no $W (wheel install? check your own op's signature by hand)"
  fi
  # Confirmation when the git tree has the commit. Old clones often do not contain it at all,
  # so absence proves nothing and must not be read as OK.
  if [ -d "$ROOT/.git" ]; then
    D=$(git -C "$ROOT" describe --tags HEAD 2>/dev/null || echo unknown)
    if git -C "$ROOT" cat-file -e "${KNFIX}^{commit}" 2>/dev/null; then
      if git -C "$ROOT" merge-base --is-ancestor "$KNFIX" HEAD 2>/dev/null; then
        say OK kernelName-ancestry "HEAD contains $KNFIX ($D)"
      else
        say BROKEN kernelName-ancestry "HEAD PREDATES $KNFIX ($D) -- update aiter"
      fi
    else
      say INFO kernelName-ancestry "$KNFIX not in this clone ($D) -- inconclusive, trust the signature check above"
    fi
  fi
fi

echo
echo "=== build prerequisites (needed only if you must build a client) ==="
for pkg in cmake git; do have_bin "$pkg" "$pkg"; done
for h in /usr/include/gtest/gtest.h /usr/include/gmock/gmock.h /usr/include/boost/filesystem.hpp; do
  [ -f "$h" ] && say OK "$(basename "$h")" "$h" || say MISSING "$(basename "$h")" "$h"
done
