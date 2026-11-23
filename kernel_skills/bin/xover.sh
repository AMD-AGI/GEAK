#!/usr/bin/env bash
# Cross-over comparison of two candidate workspaces X and Y: with two GPUs each pair runs X on g1 and Y on
# g2 at the same time, then swapped; with one GPU the order alternates (XY, YX). Bias and drift cancel in the mean. Each side is a full
# paired measurement against the baseline (bin/ab.sh + bin/score.py).
# Usage: xover.sh <RUN|cmd.env> <ws_base|-> <ws_X> <ws_Y> <out_dir> [pairs=2] [reps=1]
#   GPUs: the first two ids of GPU in cmd.env (e.g. pool:1:0,1 -> 0 and 1). ws_base "-" = in-process harness.
# Prints JSON {delta_mean (Y - X), deltas, x, y}; logs in out_dir/p<k>_<side>/.
set -euo pipefail
ENVF="${1:?RUN}"; [[ -d "$ENVF" ]] && ENVF="$ENVF/cmd.env"; ENVF="$(readlink -f "$ENVF")"
BASE="$2"; X="$(cd "$3" && pwd)"; Y="$(cd "$4" && pwd)"; OUT="${5:?out}"; PAIRS="${6:-2}"; REPS="${7:-1}"
HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source "$ENVF"
IFS=, read -r G1 G2 _ <<<"${GPU##*:}"   # one GPU only: the pair runs X then Y, the next pair Y then X
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
side() {  # side <label> <ws> <gpu>
  AB_GPU=$3 bash "$HERE/ab.sh" "$ENVF" "$BASE" "$2" "$OUT/$1" "$REPS" >"$OUT/$1.log" 2>&1 || echo "rc=$?" >>"$OUT/$1/errors"
  python3 "$HERE/score.py" "$OUT/$1" ${LAT_REGEX:+"$LAT_REGEX"} >"$OUT/$1.json" 2>/dev/null || true
}
for k in $(seq 1 "$PAIRS"); do
  if (( k % 2 )); then gx=$G1; gy=$G2; else gx=$G2; gy=$G1; fi
  if [[ -n "${G2:-}" ]]; then side "p${k}_x" "$X" "$gx" & side "p${k}_y" "$Y" "$gy" & wait
  elif (( k % 2 )); then side "p${k}_x" "$X" "$G1"; side "p${k}_y" "$Y" "$G1"
  else side "p${k}_y" "$Y" "$G1"; side "p${k}_x" "$X" "$G1"; fi
done
python3 - "$OUT" "$PAIRS" <<'PY'
import json, sys, statistics
out, n = sys.argv[1], int(sys.argv[2])
get = lambda f: json.load(open(f"{out}/{f}.json")).get("speedup", 0.0)
x = [get(f"p{k}_x") for k in range(1, n + 1)]; y = [get(f"p{k}_y") for k in range(1, n + 1)]
ok = all(v > 0 for v in x + y)
d = [b - a for a, b in zip(x, y)]
print(json.dumps({"ok": ok, "delta_mean": round(statistics.mean(d), 4) if ok else None,
                  "deltas": [round(v, 4) for v in d], "x": x, "y": y}))
PY
