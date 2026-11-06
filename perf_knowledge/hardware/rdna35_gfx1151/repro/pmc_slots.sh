#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# How many hardware counters does rocprofv3 accept in ONE pass on this part?
#
# Backs ../pitfalls.md, which reports this cumulative set
# getting three deep and rejecting the fourth with "error code 38: Request
# exceeds the capabilities of the hardware to collect"; this script has to
# actually demonstrate that, so it sweeps 1..5 rather than bisecting.
#
# What it deliberately does NOT claim:
#   * that three of ANYTHING is safe. The sweep is cumulative over ONE ordered
#     set, so the result describes that set. A metric named in a `pmc:` line is
#     not necessarily one hardware counter -- some expand into several -- so a
#     different trio can exceed the same budget. Probe the set you want.
#
# A rejected set does not make rocprofv3 exit: the workload keeps spinning on the
# CPU until something kills it, so every pass runs under a timeout (PMC_TIMEOUT,
# default 60 s) and its wall time is printed. Accepted passes take a couple of
# seconds; rejected ones take the full timeout. An accepted pass also prints each
# counter's summed value, so a counter that reads zero is visible.
#
# Exit status: 0 if the sweep ran, 1 if it could not run at all (no rocprofv3,
# no GPU, missing workload). A counter set that fails is a RESULT, not an error.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

command -v rocprofv3 >/dev/null 2>&1 || { echo "error: rocprofv3 not on PATH" >&2; exit 1; }
[ -f "$SCRIPT_DIR/pmc_work.py" ] || { echo "error: pmc_work.py not next to this script" >&2; exit 1; }

PY="${PYTHON:-python3}"
"$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
  || { echo "error: no usable GPU for ${PY}" >&2; exit 1; }

# One cumulative set, cheapest first. Not a claim that these are atomic.
ALL="SQ_WAVES GRBM_GUI_ACTIVE FETCH_SIZE WRITE_SIZE GL2C_HIT"
TIMEOUT="${PMC_TIMEOUT:-60}"   # a rejected set never exits on its own; see the header
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# Every gfx agent, not the first match: rocminfo also lists generic targets and other GPUs.
GFX="$(rocminfo 2>/dev/null | awk '/^ *Name: *gfx/{print $2}' | sort -u | paste -sd' ')"
echo "counter-budget sweep on ${GFX:-unknown gfx}"
for n in 1 2 3 4 5; do
  C="$(echo "$ALL" | cut -d' ' -f"1-$n")"
  rm -rf "$OUT/run"; mkdir -p "$OUT/run"
  printf 'pmc: %s\n' "$C" > "$OUT/in.txt"
  t0=$SECONDS
  log="$(timeout "$TIMEOUT" rocprofv3 -i "$OUT/in.txt" -d "$OUT/run" --output-format csv -- "$PY" pmc_work.py 2>&1)"
  rc=$?
  dt=$((SECONDS - t0))
  if printf '%s' "$log" | grep -qiE 'exceeds the capabilities|error code 38|aborted'; then
    printf '  n=%d  FAIL (counter budget exceeded)  %4ds rc=%d  [%s]\n' "$n" "$dt" "$rc" "$C"
  elif printf '%s' "$log" | grep -q '^ok'; then
    # OK means the counters were collected, not only that the workload ran: name each one with
    # its summed value, so a counter that reads zero is visible.
    vals="$("$PY" - "$OUT/run" <<'PY'
import csv, glob, sys
tot = {}
for f in glob.glob(sys.argv[1] + "/**/*counter_collection.csv", recursive=True):
    for r in csv.DictReader(open(f)):
        tot[r["Counter_Name"]] = tot.get(r["Counter_Name"], 0.0) + float(r["Counter_Value"])
print(" ".join("%s=%g" % kv for kv in sorted(tot.items())) or "NO-COUNTER-DATA")
PY
)"
    printf '  n=%d  OK                              %4ds rc=%d  [%s]  %s\n' "$n" "$dt" "$rc" "$C" "$vals"
  else
    printf '  n=%d  ????  %4ds rc=%d  %s\n' "$n" "$dt" "$rc" "$(printf '%s' "$log" | tail -1 | cut -c1-70)"
  fi
done
echo "note: a FAIL above is a measurement, not a script error."
