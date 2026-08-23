#!/usr/bin/env bash
# Run the serving benchmark against an already-running server. INSIDE the container.
#
#   ./run_bench.sh                 # the fixed reference workload
#   TAG=my_change ./run_bench.sh   # label the output directory
#
# These arguments are the measurement contract: they reproduce the reference number. Change any of
# them and your result is not comparable to it. Values come from the harness config of the reference
# round (../reference/results/baseline_measure/config.yaml).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${BENCH:-$HERE/../bench}"

MODEL="${MODEL:-/shared_nfs/hyperloom/models/gpt-oss-120b}"
PORT="${PORT:-43112}"
CONC="${CONC:-64}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-192}"
NUM_WARMUPS="${NUM_WARMUPS:-8}"
TAG="${TAG:-run}"
OUT="${OUT:-$HERE/../results/${TAG}_$(date +%Y%m%d_%H%M%S)}"

if ! curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[bench] no healthy server on port ${PORT} -- start it first" >&2; exit 1
fi

mkdir -p "$OUT"
export BENCH_TRUST_REMOTE_CODE=1
export HF_HUB_TRUST_REMOTE_CODE=1

cd "$BENCH"
python3 benchmark_serving.py \
    --backend vllm \
    --base-url "http://0.0.0.0:${PORT}" \
    --endpoint /v1/completions \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio 1.0 \
    --random-prefix-len 0 \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONC" \
    --num-warmups "$NUM_WARMUPS" \
    --seed 0 \
    --ignore-eos \
    --trust-remote-code \
    --percentile-metrics 'ttft,tpot,itl,e2el' \
    --metric-percentiles '90,99,99.9' \
    --save-result \
    --result-dir "$OUT" \
    --result-filename inferencex_result.json \
    2>&1 | tee "$OUT/bench_stdout.log"

echo
echo "baseline for this configuration: 4201.601 tok/s output throughput"
echo "  reference mean TTFT 3684.0 ms, mean TPOT 11.63 ms -- check those too"
echo "  the same config's other reference round gave 4204.183 tok/s (0.06% apart);"
echo "  read ../BASELINE.md before treating a small delta as a real change."
echo "  NOT YET MEASURED HERE. The figure is the config ceiling of the source session"
echo "  (three rounds 4204.183/4201.601/4168.010, 0.86% apart). The untouched"
echo "  configuration measured 1451.311 -- --page-size 64 is worth +183% on its own."
echo "result -> $OUT/inferencex_result.json"
