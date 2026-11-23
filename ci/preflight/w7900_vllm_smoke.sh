#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# W7900 runtime smoke (opt-in; not part of CI L0). Run it INSIDE the W7900 vLLM image with only
# the target card exposed (--device /dev/kfd --device /dev/dri/renderD<N>), after
# gpu_healthcheck.sh passes, and cap it with an outer `timeout`. It needs a W7900, a local
# model and a few minutes.
#
#   usage: ci/preflight/w7900_vllm_smoke.sh <model-dir> <out-dir>
#
# Checks, in order (the first failure exits non-zero with a clear message):
#   1. identity — scripts/gpu_identity.py reports gfx1100, target w7900, 96 CUs and one visible
#                 agent (GEAK_GPU_IDENTITY_JSON replaces the probe, as in run_e2e.py), and
#                 ci/lib.sh detect_gpu_arch with GEAK_EXPECTED_TARGET=w7900 prints W7900.
#   2. serve    — `vllm serve <model> --dtype float16` answers /health and lists the model.
#   3. handoff  — the card's power_dpm_force_performance_level, READ from sysfs, goes into a
#                 W7900 handoff as gpu_perf_level; run_e2e.py --dry-run must select vllm and map
#                 the level and the identity unchanged. This script never writes the level.
#   4. bench    — `vllm bench serve` (random prompts) completes every request with output.
#
# Every step has its own cap (W7900_SMOKE_STEP_TIMEOUT, _SERVE_TIMEOUT, _BENCH_TIMEOUT, in
# seconds). Writes <out-dir>/w7900_vllm_smoke.json ({status, message, checks}) and the server,
# dry-run and bench logs beside it.
set -uo pipefail

[ $# -eq 2 ] || { echo "usage: $0 <model-dir> <out-dir>" >&2; exit 2; }
MODEL=$1
OUT=$2
GEAK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT=${W7900_SMOKE_PORT:-8011}
PROMPTS=${W7900_SMOKE_PROMPTS:-32}
CONCURRENCY=${W7900_SMOKE_CONCURRENCY:-8}
INPUT_LEN=${W7900_SMOKE_INPUT_LEN:-128}
OUTPUT_LEN=${W7900_SMOKE_OUTPUT_LEN:-32}
STEP_TIMEOUT=${W7900_SMOKE_STEP_TIMEOUT:-120}
SERVE_TIMEOUT=${W7900_SMOKE_SERVE_TIMEOUT:-900}
BENCH_TIMEOUT=${W7900_SMOKE_BENCH_TIMEOUT:-600}

mkdir -p "$OUT"
RESULT="$OUT/w7900_vllm_smoke.json"
CHECKS="$OUT/checks.jsonl"
: > "$CHECKS"
SERVER_PID=""

# record <check> <status> <json-object>: one line per check, merged into $RESULT at the end.
record() {
  python3 -c 'import json, sys; print(json.dumps({"check": sys.argv[1], "status": sys.argv[2], **json.loads(sys.argv[3])}))' \
    "$1" "$2" "$3" >> "$CHECKS"
}

# The server runs in its own session (setsid), so the TERM reaches its engine processes too.
stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  kill -TERM -- "-$SERVER_PID" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
}
trap stop_server EXIT
trap 'stop_server; exit 143' TERM INT

finish() {
  local status=$1 message=${2:-}
  stop_server
  python3 - "$CHECKS" "$RESULT" "$status" "$message" <<'PY'
import json, sys
checks_path, result_path, status, message = sys.argv[1:5]
with open(checks_path) as fh:
    checks = [json.loads(line) for line in fh if line.strip()]
with open(result_path, "w") as fh:
    json.dump({"status": status, "message": message, "checks": checks}, fh, indent=2)
    fh.write("\n")
PY
  if [ "$status" = ok ]; then
    echo "W7900-VLLM-SMOKE: OK ($RESULT)"
    exit 0
  fi
  echo "::error::w7900_vllm_smoke: $message" >&2
  echo "W7900-VLLM-SMOKE: FAIL — $message" >&2
  exit 1
}

# ---- 1. identity -------------------------------------------------------------------------------
echo ">>> w7900_vllm_smoke: identity ..."
if [ -n "${GEAK_GPU_IDENTITY_JSON:-}" ]; then
  IDENTITY=$GEAK_GPU_IDENTITY_JSON
  IDENTITY_SOURCE=GEAK_GPU_IDENTITY_JSON
elif ! IDENTITY="$(timeout --kill-after=10 "$STEP_TIMEOUT" python3 "$GEAK_ROOT/scripts/gpu_identity.py" 2>&1)"; then
  record identity fail "$(python3 -c 'import json, sys; print(json.dumps({"error": sys.argv[1]}))' "$IDENTITY")"
  finish fail "gpu_identity.py failed: $IDENTITY"
else
  IDENTITY_SOURCE=rocminfo
fi
if ! WHY="$(python3 - "$IDENTITY" <<'PY'
import json, sys
try:
    got = json.loads(sys.argv[1])
except ValueError:
    print("identity is not JSON")
    sys.exit(1)
want = {"gfx": "gfx1100", "target": "w7900", "physical_cu_count": 96, "visible_gpu_agents": 1}
bad = ["%s=%r (want %r)" % (k, got.get(k), v) for k, v in want.items() if got.get(k) != v]
print("; ".join(bad))
sys.exit(1 if bad else 0)
PY
)"; then
  record identity fail "$(python3 -c 'import json, sys; print(json.dumps({"source": sys.argv[1], "error": sys.argv[2]}))' "$IDENTITY_SOURCE" "$WHY")"
  finish fail "not a single Radeon Pro W7900: $WHY"
fi
if ! ARCH="$(GEAK_EXPECTED_TARGET=w7900 timeout --kill-after=10 "$STEP_TIMEOUT" \
      bash -c 'source "$1/ci/lib.sh" && detect_gpu_arch' _ "$GEAK_ROOT" 2> "$OUT/detect_gpu_arch.err")" \
    || [ "$ARCH" != "W7900" ]; then
  WHY="${ARCH:-$(tail -1 "$OUT/detect_gpu_arch.err")}"
  record identity fail "$(python3 -c 'import json, sys; print(json.dumps({"source": sys.argv[1], "detect_gpu_arch": sys.argv[2]}))' "$IDENTITY_SOURCE" "$WHY")"
  finish fail "detect_gpu_arch did not resolve W7900: $WHY"
fi
record identity ok "{\"source\": \"$IDENTITY_SOURCE\", \"identity\": $IDENTITY, \"detect_gpu_arch\": \"$ARCH\"}"

[ -f "$MODEL/config.json" ] || finish fail "no config.json under model dir $MODEL"
VERSIONS="$(python3 -c 'import json, torch, vllm; print(json.dumps({"vllm": vllm.__version__, "torch": torch.__version__, "hip": torch.version.hip}))' 2>/dev/null || echo '{}')"

# ---- 2. serve ----------------------------------------------------------------------------------
echo ">>> w7900_vllm_smoke: vllm serve $MODEL --dtype float16 ..."
setsid vllm serve "$MODEL" --dtype float16 --host 127.0.0.1 --port "$PORT" \
  --max-model-len 2048 --gpu-memory-utilization 0.85 > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
START=$SECONDS
until curl -sf --max-time 10 "http://127.0.0.1:$PORT/health" > /dev/null; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    SERVER_PID=""
    record serve fail '{"error": "server exited during startup; see server.log"}'
    finish fail "vllm serve exited during startup (see $OUT/server.log)"
  fi
  if [ $((SECONDS - START)) -ge "$SERVE_TIMEOUT" ]; then
    record serve fail "{\"error\": \"not healthy after ${SERVE_TIMEOUT}s\"}"
    finish fail "vllm serve was not healthy after ${SERVE_TIMEOUT}s (see $OUT/server.log)"
  fi
  sleep 5
done
if ! curl -sf --max-time 30 "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json" \
    || ! python3 -c 'import json, sys; ids = [m["id"] for m in json.load(open(sys.argv[1]))["data"]]; sys.exit(0 if sys.argv[2] in ids else 1)' \
      "$OUT/models.json" "$MODEL"; then
  record serve fail '{"error": "/v1/models does not list the model"}'
  finish fail "/v1/models does not list $MODEL"
fi
record serve ok "{\"dtype\": \"float16\", \"startup_s\": $((SECONDS - START)), \"versions\": $VERSIONS}"

# ---- 3. handoff --------------------------------------------------------------------------------
# Read with the card awake (the server holds it); a runtime-suspended card answers EBUSY.
echo ">>> w7900_vllm_smoke: gpu_perf_level handoff (run_e2e.py --dry-run) ..."
LEVEL="$(python3 - <<'PY'
import errno, glob, os, time
nodes = sorted(glob.glob("/dev/dri/renderD*"))
level = ""
if len(nodes) == 1:
    path = "/sys/class/drm/%s/device/power_dpm_force_performance_level" % os.path.basename(nodes[0])
    for _ in range(20):
        try:
            with open(path) as fh:
                level = fh.read().strip()
            break
        except OSError as error:
            if error.errno != errno.EBUSY:
                break
            time.sleep(0.1)
print(level)
PY
)"
python3 - "$OUT/handoff.json" "$MODEL" "$OUT" "$LEVEL" "$IDENTITY" "$INPUT_LEN" "$OUTPUT_LEN" "$CONCURRENCY" <<'PY'
import json, sys
path, model, out, level, identity, isl, osl, conc = sys.argv[1:9]
handoff = {
    "schema_version": 2, "model_path": model, "framework": "vllm", "tp": 1, "gpu_ids": "0",
    "expected_gfx": "gfx1100", "expected_target": "w7900",
    "expected_device_name": json.loads(identity)["marketing_name"], "expected_physical_cu_count": 96,
    "workload": {"isl": int(isl), "osl": int(osl), "conc": int(conc)}, "exp_root": out + "/geak",
}
if level:
    handoff["gpu_perf_level"] = level
with open(path, "w") as fh:
    json.dump(handoff, fh, indent=2)
PY
if ! timeout --kill-after=10 "$STEP_TIMEOUT" python3 "$GEAK_ROOT/interface/run_e2e.py" \
    "$OUT/handoff.json" "$OUT/dryrun_result.json" --dry-run > "$OUT/dryrun.json" 2> "$OUT/dryrun.err"; then
  record handoff fail "{\"gpu_perf_level\": \"$LEVEL\", \"error\": \"run_e2e.py --dry-run failed; see dryrun.err\"}"
  finish fail "run_e2e.py --dry-run rejected the W7900 handoff (see $OUT/dryrun.err)"
fi
if ! WHY="$(python3 - "$OUT/dryrun.json" "$LEVEL" <<'PY'
import json, sys
mapped = json.load(open(sys.argv[1]))["mapped_args"]
want = {"backend": "vllm", "expected_gfx": "gfx1100", "expected_target": "w7900",
        "expected_physical_cu_count": 96, "gpu_perf_level": sys.argv[2].lower() or None}
bad = ["%s=%r (want %r)" % (k, mapped.get(k), v) for k, v in want.items() if mapped.get(k) != v]
print("; ".join(bad))
sys.exit(1 if bad else 0)
PY
)"; then
  record handoff fail "$(python3 -c 'import json, sys; print(json.dumps({"gpu_perf_level": sys.argv[1], "error": sys.argv[2]}))' "$LEVEL" "$WHY")"
  finish fail "run_e2e.py mapped the W7900 handoff wrongly: $WHY"
fi
record handoff ok "{\"gpu_perf_level\": \"${LEVEL:-unreadable}\", \"mapped\": true}"

# ---- 4. bench ----------------------------------------------------------------------------------
echo ">>> w7900_vllm_smoke: vllm bench serve ($PROMPTS prompts, concurrency $CONCURRENCY) ..."
if ! timeout --kill-after=30 "$BENCH_TIMEOUT" vllm bench serve --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$PROMPTS" --max-concurrency "$CONCURRENCY" --request-rate inf --ignore-eos --seed 0 \
    --save-result --result-dir "$OUT" --result-filename bench.json --disable-tqdm > "$OUT/bench.log" 2>&1; then
  record bench fail '{"error": "vllm bench serve failed or timed out; see bench.log"}'
  finish fail "vllm bench serve failed or timed out (see $OUT/bench.log)"
fi
if ! SUMMARY="$(python3 - "$OUT/bench.json" "$PROMPTS" <<'PY'
import json, sys
bench, prompts = json.load(open(sys.argv[1])), int(sys.argv[2])
keys = ("completed", "failed", "total_input_tokens", "total_output_tokens", "duration",
        "request_throughput", "output_throughput", "mean_ttft_ms", "mean_tpot_ms")
summary = {k: bench[k] for k in keys if k in bench}
print(json.dumps(summary))
ok = bench.get("completed") == prompts and not bench.get("failed") and bench.get("total_output_tokens", 0) > 0
sys.exit(0 if ok else 1)
PY
)"; then
  record bench fail "{\"result\": ${SUMMARY:-null}}"
  finish fail "vllm bench serve did not complete all $PROMPTS prompts: ${SUMMARY:-no result}"
fi
record bench ok "{\"prompts\": $PROMPTS, \"concurrency\": $CONCURRENCY, \"input_len\": $INPUT_LEN, \"output_len\": $OUTPUT_LEN, \"result\": $SUMMARY}"
finish ok
