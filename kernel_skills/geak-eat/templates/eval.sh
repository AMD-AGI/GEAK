#!/bin/bash
# Official GEAK eval for an EAT run. The last line is "SCORE: <x>": the primary speedup over the frozen
# baseline, or 0 on any failure. Usage: bash eat/eval.sh [--ws <dir>] [--reps N]
#   --ws: score the EDITABLE paths of another workspace (an agent's private copy); default = this repo.
# The candidate is never run in place: a fresh copy of the pristine task gets only the EDITABLE paths of
# the workspace, so harness, oracle and baseline edits have no effect. Every eval has its own directory
# and GPU lease, so several agents may evaluate at once.
set -uo pipefail
cd "$(dirname "$0")/.." && source eat/env.sh && source eat/cmd.env
WS="$EAT_ROOT"; REPS="${EVAL_REPS:-3}"
while [ $# -gt 0 ]; do case "$1" in --ws) WS="$(cd "$2" && pwd)"; shift 2;; --reps) REPS="$2"; shift 2;; *) shift;; esac; done
OUT="$EAT_WORK/evals/$(date +%Y%m%d_%H%M%S)_$$"; mkdir -p "$OUT"
tidy() { cp "$OUT"/cand/run/ut_report*.json "$OUT"/ 2>/dev/null; [ "${EVAL_KEEP_TREES:-0}" = 1 ] || rm -rf "$OUT/cand" "$OUT/base"; }
fail() { tidy; echo "EVAL FAIL: $*  (logs: $OUT)"; echo "SCORE: 0"; exit 0; }
MAT="$GEAK_KW/scripts/materialize_workspace.sh"

bash "$MAT" --src "$EAT_WORK/orig" --dst "$OUT/cand" --shared-root "$EAT_WORK/_shared" --link-aiter >/dev/null || fail materialize
for p in $EDITABLE; do
  [ -e "$WS/$p" ] || fail "missing editable path $p in $WS"
  rm -rf "${OUT:?}/cand/$p" && mkdir -p "$(dirname "$OUT/cand/$p")" && cp -a "$WS/$p" "$OUT/cand/$p"
done
find "$OUT/cand" \( -name .torch_ext -o -name __pycache__ -o -name build \) -prune -exec rm -rf {} + 2>/dev/null

run() { (cd "$OUT/cand" && timeout "${EVAL_TIMEOUT:-3600}" bash "$GEAK_KW/scripts/gpu_lock.sh" "$GPU" bash -c "$1") > "$OUT/$2.log" 2>&1; }
if [ -n "${CORRECTNESS_CMD:-}" ]; then   # empty when BENCH_CMD is a UT that gates correctness itself
  run "$CORRECTNESS_CMD" correctness; rc=$?
  [ $rc -eq 86 ] && fail "invalid source binding (exit 86)"
  [ $rc -ne 0 ] && { tail -20 "$OUT/correctness.log"; fail "correctness (exit $rc)"; }
fi
if [ -n "${HOLDOUT_CMD:-}" ]; then run "$HOLDOUT_CMD" holdout || fail "held-out correctness"; fi

A=-
if [ "$PAIRING" = interleave ]; then
  bash "$MAT" --src "$EAT_WORK/orig" --dst "$OUT/base" --shared-root "$EAT_WORK/_shared" --link-aiter >/dev/null
  A="$OUT/base"
fi
bash "$GEAK_SK/bin/ab.sh" "$EAT_ROOT/eat/cmd.env" "$A" "$OUT/cand" "$OUT/ab" "$REPS" >"$OUT/ab.log" 2>&1 \
  || echo "ab.sh rc=$? (GPU lease timeout GEAK_GPU_RUN_TIMEOUT=${GEAK_GPU_RUN_TIMEOUT:-900}s or lease failure; see $OUT/ab.log)" >> "$OUT/ab/errors"
[ -s "$OUT/ab/errors" ] && { cat "$OUT/ab/errors"; tail -n 15 "$OUT"/ab/B_*.log; fail "benchmark run failed"; }
R=$(python3 "$GEAK_SK/bin/score.py" "$OUT/ab" ${LAT_REGEX:+"$LAT_REGEX"}) || fail "unscorable benchmark output"
echo "$R" > "$OUT/score.json"; tidy
python3 -c "import json,sys; r=json.load(open(sys.argv[1]))
for c in r.get('per_case', []): print(f\"  {c['name']}: {c['base_ms']:.4f} -> {c['cand_ms']:.4f} ms  ({c['base_ms']/c['cand_ms']:.3f}x)\")
print(f\"speedup {r['speedup']}  spread {r['spread']}  reps {r['reps']}  noise band {sys.argv[2]}  (logs: {sys.argv[3]})\")
print(f\"SCORE: {r['speedup']}\")" "$OUT/score.json" "${NOISE:-?}" "$OUT"
