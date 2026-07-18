#!/usr/bin/env bash
# Host-side GPU-wedge pre-check. Runs BEFORE any rocminfo/rocm-smi/torch probe.
#
# WHY: when the AMD GPU/KFD driver wedges, tasks block in UNINTERRUPTIBLE sleep
# (process state "D") inside the amdgpu/kfd/dma_fence kernel path. A D-state task
# cannot be killed — not by SIGTERM, not by SIGKILL, not by `timeout` — so if we
# naively run rocminfo/torch against a wedged driver, OUR probe hangs forever too
# (exactly what bit us: a stuck `rocminfo` the 120s preflight timeout couldn't
# reap, hanging the whole job toward its multi-hour wall-clock cap).
#
# This check touches NO GPU — it only reads /proc. It samples twice a few seconds
# apart and flags PIDs that remain in D state across BOTH samples (SUSTAINED, so a
# healthy task briefly in dma_fence_wait isn't misread as a wedge), then hard-fails
# if any such PID is a GPU tool (rocminfo/rocm-smi/amd-smi) or is parked in the
# amdgpu/kfd/dma_fence kernel path.
#
# Exit 0 = clear;  Exit 3 = wedged (offenders printed to stderr).  Never hangs.
set -uo pipefail

GAP_S="${GEAK_DSTATE_SAMPLE_GAP_S:-3}"
TOOLS_RE='^(rocminfo|rocm-smi|rocmsmi|amd-smi|amdsmi|rocm_smi)$'
PATH_RE='amdgpu|kfd|dma_fence|fence_wait'

# Emit "<pid>\t<comm>\t<wchan>" for every process currently in D (uninterruptible)
# state. comm may contain spaces/parens, so state = first char AFTER the last ")".
_dstate_snapshot() {
  local p state comm wchan
  for p in /proc/[0-9]*; do
    state=$(sed -E 's/^.*\) //' "$p/stat" 2>/dev/null | cut -d' ' -f1) || continue
    [ "$state" = "D" ] || continue
    comm=$(tr -d '\0' < "$p/comm" 2>/dev/null)
    wchan=$(tr -d '\0' < "$p/wchan" 2>/dev/null)
    printf '%s\t%s\t%s\n' "${p#/proc/}" "$comm" "${wchan:-?}"
  done
}

s1="$(_dstate_snapshot)"
[ -z "$s1" ] && { echo "GPU-DSTATE: clear (no D-state processes)"; exit 0; }
sleep "$GAP_S"
s2="$(_dstate_snapshot)"
[ -z "$s2" ] && { echo "GPU-DSTATE: clear (D-state cleared within ${GAP_S}s)"; exit 0; }

# PIDs in D in BOTH snapshots = sustained (not a momentary I/O blip).
persistent="$(comm -12 \
  <(printf '%s\n' "$s1" | cut -f1 | sort -u) \
  <(printf '%s\n' "$s2" | cut -f1 | sort -u))"
[ -z "$persistent" ] && { echo "GPU-DSTATE: clear (no sustained D-state)"; exit 0; }

# Classify the sustained set using the 2nd snapshot's comm/wchan.
offenders=""
while IFS= read -r pid; do
  [ -n "$pid" ] || continue
  line="$(printf '%s\n' "$s2" | awk -F'\t' -v p="$pid" '$1==p{print; exit}')"
  comm="$(printf '%s' "$line" | cut -f2)"
  wchan="$(printf '%s' "$line" | cut -f3)"
  if printf '%s' "$comm" | grep -qE "$TOOLS_RE" || printf '%s' "$wchan" | grep -qE "$PATH_RE"; then
    offenders="${offenders}  pid=$pid comm=$comm wchan=$wchan"$'\n'
  fi
done <<< "$persistent"

if [ -n "$offenders" ]; then
  echo "GPU-DSTATE: WEDGED — process(es) stuck (uninterruptible) in the GPU driver:" >&2
  printf '%s' "$offenders" >&2
  echo "  => KFD/amdgpu is hung; rocminfo/torch would hang too. Needs a GPU reset or reboot." >&2
  exit 3
fi

echo "GPU-DSTATE: clear (sustained D-state present but not GPU-related)"
exit 0
