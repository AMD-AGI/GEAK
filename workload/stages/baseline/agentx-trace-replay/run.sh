#!/usr/bin/env bash
# script-mode entry point (contract §2 A): readiness check -> bench_e2e.sh -> a unified
# baseline.json. See SKILL.md alongside it. Every path comes from $PARAMS; none is
# hardcoded.
set -uo pipefail

PY="${PYTHON_BIN:-python3}"
E2E="$GEAK_ROOT/e2e_workflow/scripts"
emit() { "$PY" "$WORKLOAD_ROOT/scripts/emit_result.py" "$@"; }
param() { "$PY" -c "
import json,sys
p=json.load(open(sys.argv[1]))
v=p.get(sys.argv[2], sys.argv[3] if len(sys.argv)>3 else '')
print('' if v is None else (json.dumps(v) if isinstance(v,(dict,list)) else v))
" "$PARAMS" "$1" "${2-}"; }

BASE_URL="$(param base_url)"
MODEL="$(param model)"
INFERENCEX_PATH="$(param inferencex_path)"
DURATION_S="$(param duration_s 3600)"
REPEATS="$(param repeats 1)"
METRIC="$(param metric p90_intvty_inferencex)"
NOISE_BAND="$(param noise_band 1.045)"
SMOKE_REQUIRED="$(param smoke_required true)"

if [ -z "$BASE_URL" ]; then
  emit --status blocked --notes "params.base_url is not configured. This skill does not bring the service up: start AgentX yourself (TP8/conc10) and put its address in the scenario config, or switch to adopt-measured-receipt to reuse an existing receipt."
  exit 1
fi

# -------- 1. Readiness check. It catches "fails 20 minutes in" and "stock aiperf measured
#             something else entirely".
if [ ! -x "$E2E/agentx_smoke.sh" ] && [ ! -f "$E2E/agentx_smoke.sh" ]; then
  emit --status blocked --notes "cannot find $E2E/agentx_smoke.sh; this skill depends on the agentx support in GEAK e2e_workflow."
  exit 1
fi
{
  echo "=== agentx_smoke.sh  BASE_URL=$BASE_URL  METRIC=$METRIC"
  env ${MODEL:+MODEL="$MODEL"} ${INFERENCEX_PATH:+INFERENCEX_PATH="$INFERENCEX_PATH"} \
      BASE_URL="$BASE_URL" E2E_METRIC="$METRIC" \
      bash "$E2E/agentx_smoke.sh"
  echo "smoke_rc=$?"
} > "$OUT/smoke.log" 2>&1
SMOKE_RC=$(sed -n 's/^smoke_rc=//p' "$OUT/smoke.log" | tail -1)
if [ "${SMOKE_RC:-1}" != "0" ] && [ "$SMOKE_REQUIRED" != "false" ]; then
  emit --status blocked --artifact smoke_log=smoke.log \
    --notes "agentx_smoke.sh reported a blocker (see smoke.log). Common causes: an aiperf that does not support --scenario (the most dangerous one — it silently measures a different workload), a missing map_aiperf.py under INFERENCEX_PATH, or a service that is down or serving the wrong weights. Fix it and rerun this stage."
  exit 1
fi

# -------- 2. Measure. Run against an already warm service; do not start or kill it here.
HOST="$($PY -c "import sys,urllib.parse as u;print(u.urlparse(sys.argv[1]).hostname or '127.0.0.1')" "$BASE_URL")"
PORT="$($PY -c "import sys,urllib.parse as u;print(u.urlparse(sys.argv[1]).port or 8000)" "$BASE_URL")"
BENCH_OUT="$OUT/bench"
mkdir -p "$BENCH_OUT"

EXTRA_ENV_JSON="$(param env)"
ENV_ARGS=()
if [ -n "$EXTRA_ENV_JSON" ]; then
  while IFS= read -r kv; do [ -n "$kv" ] && ENV_ARGS+=("$kv"); done < <(
    "$PY" -c "
import json,sys
for k,v in json.loads(sys.argv[1]).items(): print('%s=%s'%(k,v))
" "$EXTRA_ENV_JSON")
fi

(
  cd "$GEAK_ROOT" || exit 1
  env "${ENV_ARGS[@]}" \
      BENCH_CLIENT=agentx REUSE_SERVER=1 \
      HOST="$HOST" PORT="$PORT" \
      OUT_DIR="$BENCH_OUT" REPEATS="$REPEATS" \
      E2E_METRIC="$METRIC" \
      MEASUREMENT_PURPOSE=validation \
      GEAK_AGENTX_DURATION_S="$DURATION_S" \
      ${MODEL:+MODEL="$MODEL"} ${INFERENCEX_PATH:+INFERENCEX_PATH="$INFERENCEX_PATH"} \
      bash "$E2E/bench_e2e.sh"
) > "$OUT/bench.log" 2>&1
BENCH_RC=$?

SUMMARY="$BENCH_OUT/bench_summary.json"
if [ "$BENCH_RC" != "0" ] || [ ! -f "$SUMMARY" ]; then
  emit --status fail --artifact bench_log=bench.log --artifact smoke_log=smoke.log \
    --notes "bench_e2e.sh exited $BENCH_RC without producing $SUMMARY. See bench.log."
  exit 1
fi

# -------- 3. Translate into the unified baseline.json.
"$PY" - "$SUMMARY" "$OUT" "$NOISE_BAND" "$METRIC" "$BASE_URL" "$DURATION_S" <<'PY'
import hashlib, json, socket, sys
summary, out, noise, metric, base_url, dur = sys.argv[1:7]
d = json.load(open(summary))
tput = d.get("throughput_tok_s_median")
doc = {
    "metric": d.get("metric_basis") or metric,
    "unit": "tok/s",
    "throughput": tput,
    "higher_is_better": True,
    "noise_band": float(noise),
    "config": {k: d[k] for k in ("tp", "ep", "gpus", "concurrency", "model") if k in d},
    "reference": None,
    "measured": tput is not None,
    "provenance": {
        "method": "agentx-trace-replay",
        "bench_summary": summary,
        "bench_summary_sha256": hashlib.sha256(open(summary, "rb").read()).hexdigest(),
        "measurement_mode": d.get("measurement_mode"),
        "reuse_server": True,
        "duration_s": float(dur),
        "base_url": base_url,
        "node": socket.gethostname(),
    },
}
json.dump(doc, open(out + "/baseline.json", "w"), indent=2, ensure_ascii=False)
PY

TPUT="$($PY -c "import json;print(json.load(open('$OUT/baseline.json'))['throughput'])")"
if [ "$TPUT" = "None" ]; then
  emit --status fail --artifact bench_log=bench.log --artifact bench=bench \
    --notes "bench_summary.json contains no throughput_tok_s_median; the scoring axis may not match E2E_METRIC. See bench.log."
  exit 1
fi

emit --status ok \
  --artifact baseline=baseline.json --artifact bench=bench \
  --artifact bench_log=bench.log --artifact smoke_log=smoke.log \
  --metric throughput="$TPUT" --metric noise_band="$NOISE_BAND" \
  --notes "measured against a warm service with REUSE_SERVER=1; applyback's A/B must use the same lifecycle and the same node."
