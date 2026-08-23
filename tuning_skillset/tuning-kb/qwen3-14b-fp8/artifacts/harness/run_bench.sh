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

MODEL="${MODEL:-/shared_nfs/hyperloom/models/Qwen3-14B-FP8}"
PORT="${PORT:-43102}"
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
echo "baseline for this configuration: 1537.725 tok/s output throughput"
echo "  measured here 2026-08-20 on crsuse2-m2m-115, n=4, spread 0.50%"
echo "  the source session recorded 1501.458 tok/s for the same configuration; the local"
echo "  figure runs +2.42% against it, which is larger than the spread, so it is the one to compare against."
echo "  reference mean TTFT 8677.2 ms, mean TPOT 34.17 ms."
echo "  read ../BASELINE.md before treating a small delta as a real change."
echo "result -> $OUT/inferencex_result.json"
