#!/usr/bin/env bash
# Re-launch tune_one.py past any candidate that faults the GPU.
OUT=$1; M=$2; N=$3; K=$4
: > "$OUT"
start=0
for attempt in $(seq 1 400); do
  python3 /work/analysis/tune_one.py "$OUT" "$M" "$N" "$K" "$start" >/dev/null 2>&1
  if grep -q '"kind": "done"' "$OUT"; then break; fi
  last=$(grep -o '"kind": "probe", "idx": [0-9]*' "$OUT" | tail -1 | grep -o '[0-9]*$')
  [ -z "$last" ] && break
  start=$((last+1))
done
