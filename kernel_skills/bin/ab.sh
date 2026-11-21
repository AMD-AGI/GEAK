#!/usr/bin/env bash
# Paired A/B timing: run BENCH_CMD from RUN/cmd.env alternately in workspace A and B (ABBA order)
# under ONE GPU lease, so both sides see the same GPU, clocks and drift.
# Usage: ab.sh <RUN|cmd.env> <ws_A|-> <ws_B> <out_dir> [reps=3]   (RUN = dir holding cmd.env)
#   ws_A = "-" times B only (for harnesses that already time the baseline in-process).
#   Logs: out_dir/{A,B}_<rep>.log; failures are appended to out_dir/errors (rc 86 = invalid source).
set -euo pipefail
ENVF="${1:?RUN}"; [[ -d "$ENVF" ]] && ENVF="$ENVF/cmd.env"; ENVF="$(readlink -f "$ENVF")"; OUT="${4:?out_dir}"; REPS="${5:-3}"
A="-"; [[ "$2" != "-" ]] && A="$(cd "$2" && pwd)"
B="$(cd "${3:?ws_B}" && pwd)"
KW="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../kernel_workflow" && pwd)"
# shellcheck disable=SC1091
source "$ENVF"                 # BENCH_CMD, GPU
GPU="${AB_GPU:-$GPU}"           # xover.sh pins each side to one GPU
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
export A B OUT REPS BENCH_CMD

cd "$B"
bash "$KW/scripts/gpu_lock.sh" "$GPU" bash -c '
for i in $(seq 1 "$REPS"); do
  if (( i % 2 )); then order="A B"; else order="B A"; fi
  for s in $order; do
    w="${!s}"; [[ "$w" == "-" ]] && continue
    ( cd "$w" && TORCH_EXTENSIONS_DIR="$w/.torch_ext" bash -c "$BENCH_CMD" ) \
      > "$OUT/${s}_$i.log" 2>&1 || echo "rc=$? side=$s rep=$i" >> "$OUT/errors"
  done
done'
echo "$OUT"
