#!/usr/bin/env bash
# Combined apply-back A/B: ONE base server vs ONE server with every authored fusion stacked.
#
# Each side is one bench_e2e.sh warm_server lifecycle (one full discarded warm-up round, then
# AB_REPEATS timed rounds), and gsm8k runs on that same live server through POST_BENCH_HOOK
# before teardown. fusion_combined_decide.py then compares the two sides.
#
#   OUT_DIR=<dir> CAND_OVERLAY=<combined loader dir> \
#   [BASE_OVERLAY=<current accepted overlay>] [EXPECT_BANNERS="tag1 tag2 ..."] \
#   bash fusion_combined_ab.sh
#
# Everything else bench_e2e.sh reads (EVAL_DIR, BACKEND, TP, GPU, MODEL, ISL, OSL, CONC,
# EXTRA_SERVER_ARGS, EXTRA_ENV, the bench protocol, SERVING_GPU_LOCK, ...) is passed through
# from the caller's environment unchanged, so both sides use the run's serving config.
# Writes $OUT_DIR/{base,cand}/ and $OUT_DIR/combined_decision.json; exits 0 once a decision
# is written, 2 when a side produced no throughput summary.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OUT_DIR:?OUT_DIR is required}"
: "${CAND_OVERLAY:?CAND_OVERLAY is required}"
BENCH="${EVAL_DIR:+$EVAL_DIR/bench_e2e.sh}"
[ -n "$BENCH" ] && [ -f "$BENCH" ] || BENCH="$HERE/bench_e2e.sh"
GSM8K="${GSM8K_EVAL_SCRIPT:-$HERE/gsm8k_eval.py}"
DECIDE="$HERE/fusion_combined_decide.py"
AB_REPEATS="${AB_REPEATS:-3}"
GSM8K_LIMIT="${GSM8K_LIMIT:-200}"
GSM8K_CONCURRENCY="${GSM8K_CONCURRENCY:-${CONC:-4}}"
GSM8K_THINKING="${GSM8K_THINKING:-false}"
mkdir -p "$OUT_DIR"

_think_flag=""
[ "$GSM8K_THINKING" = "true" ] || _think_flag="--no-thinking"
# Expanded now; BASE_URL/MODEL/OUT_DIR are expanded by bench_e2e.sh when the hook runs.
HOOK="python3 $(printf %q "$GSM8K") --base-url \"\$BASE_URL/v1\" --model \"\$MODEL\" \
--limit $GSM8K_LIMIT --max-tokens 4096 --seed 0 --concurrency $GSM8K_CONCURRENCY $_think_flag \
--out \"\$OUT_DIR/gsm8k.json\" > \"\$OUT_DIR/gsm8k.log\" 2>&1"

run_side() {
  local side="$1" overlay="$2"
  echo ">>> combined A/B: $side (overlay='${overlay:-<none>}') $(date -Is)"
  OUT_DIR="$OUT_DIR/$side" OVERLAY_PYTHONPATH="$overlay" \
    GEAK_REPEAT_MODE=warm_server REPEATS="$AB_REPEATS" \
    MEASUREMENT_PURPOSE="${MEASUREMENT_PURPOSE:-search}" \
    POST_BENCH_HOOK="$HOOK" \
    bash "$BENCH" > "$OUT_DIR/$side.log" 2>&1
  echo ">>> combined A/B: $side rc=$? $(grep -m1 '^E2E_SUMMARY' "$OUT_DIR/$side.log") $(date -Is)"
}

run_side base "${BASE_OVERLAY:-}"
run_side cand "$CAND_OVERLAY"

for side in base cand; do
  if [ ! -s "$OUT_DIR/$side/bench_summary.json" ]; then
    echo "!!! combined A/B: $side produced no bench_summary.json (see $OUT_DIR/$side.log)" >&2
    exit 2
  fi
done

_banners=()
for tag in ${EXPECT_BANNERS:-}; do _banners+=(--expect-banner "$tag"); done
python3 "$DECIDE" \
  --base-summary "$OUT_DIR/base/bench_summary.json" \
  --cand-summary "$OUT_DIR/cand/bench_summary.json" \
  --base-gsm8k "$OUT_DIR/base/gsm8k.json" \
  --cand-gsm8k "$OUT_DIR/cand/gsm8k.json" \
  --cand-server-log "$OUT_DIR/cand/server.log" \
  --tp "${TP:-1}" --noise-band-pct "${NOISE_BAND_PCT:-0.5}" \
  "${_banners[@]}" \
  --out "$OUT_DIR/combined_decision.json"
