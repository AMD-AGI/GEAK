#!/bin/bash
# Batch driver for the `rocprof` leg: one profiled command per (task, case, arm).
#
# Copy this next to your own case_runner.py and edit the CONFIG block. It is a
# template rather than a library because the only part that varies is the command
# being profiled, and hiding that behind flags would hide the one thing you must
# read before trusting the output.
#
# Three things here are load-bearing; all three cost a whole leg when dropped.
#
#   1. `</dev/null` on the profiled command. rocprof-compute -- and torch under it --
#      reads stdin, and stdin inside this loop is the case list. Without it the first
#      profiled run swallows the rest of cases.txt and the loop exits after ONE case
#      while still printing "ALL DONE". Check the file count, not the banner.
#      The `3< cases.txt` / `read <&3` pair is the second half of that fix.
#   2. ONE frozen denominator for every run in the leg. rocprof-compute's own
#      microbench drifts 17% between cases, which invalidates the whole comparison
#      table. See rocprof.md section 2.
#   3. Both arms through the identical path. combination.md section 6: the
#      two-implementation natural experiment is the only construction that adjudicates
#      which engine is wrong, and the measured speedup is the answer key.
#
# The `_${arm}` suffix in the output name is what `join_legs.py --arm` /
# `compare_legs.py --arm` look for. Keep it.

set -u

# ---------------- CONFIG ----------------
WORKDIR=${WORKDIR:-$PWD}                      # holds case_runner.py and cases.txt
SKILL=${SKILL:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=${OUT:-results/rocprof/leg}
FROZEN_CSV=${FROZEN_CSV:-$WORKDIR/frozen/roofline.csv}   # see rocprof.md section 2
HBM_PEAK=${HBM_PEAK:-6857}                    # GB/s, independently measured
ARMS=${ARMS:-"baseline candidate"}
ITERS=${ITERS:-1}; WARMUP=${WARMUP:-3}
# If you carry the gfx950 HBM-read fix as a PATH shim, point at it:
HBMFIX_BIN=${HBMFIX_BIN:-}
# ----------------------------------------

cd "$WORKDIR" || exit 1
RR=$SKILL/scripts/run_roofline.py
[ -f "$FROZEN_CSV" ] || { echo "no frozen roofline.csv at $FROZEN_CSV -- refusing to"\
  " run a leg whose denominator drifts between cases (rocprof.md section 2)"; exit 1; }
export ROOFLINE_MIBENCH_CACHE=$FROZEN_CSV
[ -n "$HBMFIX_BIN" ] && export PATH=$HBMFIX_BIN:$PATH
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
mkdir -p "$OUT"

run() {  # task case arm
  local t=$1 c=$2 arm=$3 n="$1_$2_$3"
  if [ -f "$OUT/${n}_roofline.txt" ]; then echo "[skip] $n"; return; fi
  echo "=== $(date +%H:%M:%S) $n"
  timeout "${PER_CASE_TIMEOUT:-2400}" python3 "$RR" --workdir "$WORKDIR" \
    --cmd "python3 case_runner.py --task $t --case $c --arm $arm --iters $ITERS --warmup $WARMUP" \
    --name "$n" --output "$OUT" --peaks both --hbm-peak-const "$HBM_PEAK" \
    >"$OUT/${n}.log" 2>&1 </dev/null
  echo "  rc=$? $(grep -ao 'SUCCESS in [0-9.]*s' "$OUT/${n}.log" | tail -1)"
}

n_cases=0
while read -r t c <&3; do
  [ -z "${t:-}" ] && continue
  case "$t" in \#*) continue;; esac
  n_cases=$((n_cases + 1))
  for arm in $ARMS; do run "$t" "$c" "$arm"; done
done 3< cases.txt

# Count, do not trust the banner: a leg truncated by the stdin trap above still
# reaches this line.
echo "ALL DONE $(date) -- $n_cases case(s) requested, \
$(ls "$OUT"/*_roofline.txt 2>/dev/null | wc -l) roofline file(s) on disk"
