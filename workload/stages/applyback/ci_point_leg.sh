#!/usr/bin/env bash
# One applyback leg for one CI search-space point, launched exactly as the CI job launched it.
#
#   ci_point_leg.sh <point_dir> <out_dir> [overlay_dir]
#
# <point_dir> is baseline/out/points/<pid>/ of an adopt-ci-artifacts baseline: its
# server_logs/logs/config.yaml is the specification (image, server args, server env, client
# concurrency/duration/env). The server runs in a fresh container of the CI image on this node's
# GPUs, with the overlay handed in as PYTHONPATH -- the only difference from the CI launch. The
# client is the CI's `aiperf profile` argv through the AgentX harness. The result is processed by
# InferenceX's own process_agentic_result, so it has the CI row's schema and keys.
#
# Proof, written before any number counts: <out>/alignment.json (ci_config.py align against the
# running container -- image, every CI arg, every CI env var) and a correctness probe. A leg that
# is not ALIGNED stops there.
#
# Environment (all have site defaults only where one exists; nothing model-specific is here):
#   CI_HF_HUB        HF hub dir holding the CI's model + draft snapshots (mounted, HF_HUB_CACHE)
#   CI_CLIENT_HF_HUB a hub dir this user can WRITE, for the client's tokenizer (default: a private
#                    one under the leg's out dir, symlinking the model snapshot). huggingface_hub
#                    takes a lock under <hub>/.locks even to read, and a shared hub whose .locks
#                    another user (or a root container) created fails the client after the whole
#                    server start-up (2026-11-06: PermissionError, client_exit=1).
#   CI_MODEL_MOUNTS  extra -v specs the hub's symlinks need, space-separated (e.g. /mnt/hf_cache:/mnt/hf_cache:ro)
#   AGENTX_BIN       the AgentX harness CLI (`agentx`), CI version
#   INFERENCEX_PATH  an InferenceX checkout providing infx.results.agentic.process_agentic_result
#   PORT             server port (default 8000)
#   DURATION         override the CI duration (smoke only; a shortened leg is not admitted)
set -uo pipefail
PD=${1:?point dir}; OUT=${2:?out dir}; OV=${3:-}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PINS="$HERE/../_shared/ci_pins.py"
: "${CI_HF_HUB:?CI_HF_HUB}"; : "${AGENTX_BIN:?AGENTX_BIN}"; : "${INFERENCEX_PATH:?INFERENCEX_PATH}"
PORT=${PORT:-8000}
mkdir -p "$OUT/e2e"
PID=$(basename "$PD")
NAME="ci_point_${PID}_$$"
st() { printf '{"point":"%s","phase":"%s","time":"%s"%s}\n' "$PID" "$1" "$(date -u +%FT%TZ)" "${2:+,$2}" | tee "$OUT/status.json"; }

python3 "$HERE/ci_config.py" launch "$PD" > "$OUT/launch.json" || { st failed '"error":"no CI config for this point"'; exit 1; }
# The CI run's pins, from its own records (baseline/measure's rule, stages/_shared/ci_pins.py).
# Refuse before any GPU time: the server is offline, so the hub's refs/main of every repo the CI
# loaded must be the commit the CI loaded -- `main` moves upstream, a commit does not.
python3 "$PINS" pins "$PD" > "$OUT/ci_pins.json"
python3 "$PINS" hub "$PD" "$CI_HF_HUB" 2> "$OUT/pin_check.log" \
  || { st failed '"error":"the hub does not hold the CI revisions (pin_check.log)"'; exit 1; }
MODEL_REV=$(python3 -c "import json; d=json.load(open('$OUT/ci_pins.json')); print(d['revisions'][d['model']])")
J() { python3 -c "import json,sys; v=json.load(open('$OUT/launch.json'))$1; print(v if isinstance(v,(str,int)) else json.dumps(v))"; }
IMAGE=$(J "['image']"); MODEL=$(J "['model']"); CONC=$(J "['conc']")
DUR=$(J "['duration']"); [ -n "${DURATION:-}" ] && DUR=$DURATION
python3 -c "
import json; d=json.load(open('$OUT/launch.json'))
open('$OUT/server.env','w').write(''.join('%s=%s\n' % kv for kv in d['env'].items()))
open('$OUT/server_argv.txt','w').write('\n'.join(d['server_argv']) + '\n')"
mapfile -t SARGV < "$OUT/server_argv.txt"
EXTRA=()
if [ -n "$OV" ]; then
  [ -f "$OV/_overlay_manifest.json" ] || { st failed '"error":"overlay dir has no _overlay_manifest.json"'; exit 1; }
  EXTRA+=(-e "PYTHONPATH=$OV")
fi
MOUNTS=(-v /shared_nfs:/shared_nfs); for m in ${CI_MODEL_MOUNTS:-}; do MOUNTS+=(-v "$m"); done

docker rm -f "$NAME" >/dev/null 2>&1
if ss -ltn | grep -qE ":$PORT "; then st failed "\"error\":\"port $PORT already in use\""; exit 1; fi
st starting_server
docker run -d --name "$NAME" --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --ipc host --network host --shm-size 128g --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable "${MOUNTS[@]}" \
  --env-file "$OUT/server.env" -e HF_HUB_CACHE="$CI_HF_HUB" -e TRITON_CACHE_DIR=/tmp/triton_cache \
  -e XDG_CACHE_HOME=/tmp/xdg_cache -e PYTHONDONTWRITEBYTECODE=1 "${EXTRA[@]}" \
  --entrypoint python3 "$IMAGE" -m sglang.launch_server --model-path "$MODEL" \
  --host 0.0.0.0 --port "$PORT" "${SARGV[@]}" > "$OUT/container.id" \
  || { st failed '"error":"docker run failed"'; exit 1; }
finish() { docker logs "$NAME" > "$OUT/server.log" 2>&1; docker rm -f "$NAME" >/dev/null 2>&1; }
trap finish EXIT

ok=0
for _ in $(seq 1 300); do       # 75 min: a cold page cache plus HiCache admission can take ~40
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ] || { st server_exited; break; }
  curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null && { ok=1; break; }
  sleep 15
done
[ $ok = 1 ] || { st failed '"error":"server not healthy"'; exit 2; }

docker inspect "$NAME" > "$OUT/container_inspect.json"
python3 "$HERE/ci_config.py" align "$PD" "$OUT/container_inspect.json" "$OUT/alignment.json" \
  || { st failed '"error":"the launch does not match the CI config (alignment.json)"'; exit 1; }
curl -s -m 30 "http://127.0.0.1:$PORT/get_server_info" > "$OUT/server_info.json"
curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Answer with just the number.\"}],\"max_tokens\":200,\"temperature\":0}" \
  > "$OUT/sanity.json"
grep -q 391 "$OUT/sanity.json" || { st failed '"error":"correctness probe failed (17*23)"'; exit 1; }
[ -n "$OV" ] && docker logs "$NAME" 2>&1 | grep -E '^\[overlay\]' | sort | uniq -c > "$OUT/overlay_engaged.txt"

st benchmark_running
TMP=$(mktemp -d "/tmp/ci_point_${PID}.XXXX")
CHUB=${CI_CLIENT_HF_HUB:-}
if [ -z "$CHUB" ]; then
  CHUB="$OUT/client_hub"; MDIR="models--${MODEL//\//--}"
  SNAP=$MODEL_REV
  mkdir -p "$CHUB/$MDIR/snapshots" "$CHUB/$MDIR/refs"
  ln -sfn "$CI_HF_HUB/$MDIR/snapshots/$SNAP" "$CHUB/$MDIR/snapshots/$SNAP"
  printf %s "$SNAP" > "$CHUB/$MDIR/refs/main"
fi
(
  eval "$(python3 -c "
import json,shlex; d=json.load(open('$OUT/launch.json'))
print('\n'.join('export %s=%s' % (k, shlex.quote(v)) for k, v in d['client_env'].items()))")"
  export HF_HUB_CACHE="$CHUB" HF_DATASETS_CACHE="$TMP/datasets" AIPERF_DATASET_MMAP_CACHE_DIR="$TMP/mmap"
  cd "$TMP"
  "$AGENTX_BIN" profile --url "http://127.0.0.1:$PORT" --model "$MODEL" --tokenizer "$MODEL" \
    --concurrency "$CONC" --benchmark-duration "$DUR" --tokenizer-trust-remote-code \
    --tokenizer-revision "$MODEL_REV" \
    --server-metrics "http://127.0.0.1:$PORT/metrics" --output-artifact-dir "$OUT/aiperf_artifacts" \
    --public-dataset semianalysis_cc_traces_weka_062126
  echo "client_exit=$?"
) > "$OUT/benchmark.log" 2>&1
CE=$(grep -o 'client_exit=[0-9]*' "$OUT/benchmark.log" | tail -1)
docker logs "$NAME" > "$OUT/server.log" 2>&1
# Prove the pins from this leg's own records: the server loaded the CI's commit of every repo,
# and the client ran the CI's recorded argv (plus --tokenizer-revision, which states the commit
# the CI's `main` resolved to then).
python3 "$PINS" prove "$PD" "$OUT/server.log" "$OUT/benchmark.log" "$OUT/pins.json" \
  || { st failed '"error":"the leg did not hold the CI pins (pins.json)"'; exit 1; }
[ "$CE" = client_exit=0 ] && [ -f "$OUT/aiperf_artifacts/profile_export_aiperf.json" ] \
  || { st failed "\"error\":\"benchmark client failed ($CE)\""; exit 1; }
rm -f "$OUT/aiperf_artifacts/server_metrics_export.json"

# The CI's own result processor, fed the point's own config, so the row is the CI schema.
python3 "$HERE/ci_result.py" "$PD" "$OUT" "$DUR" || { st failed '"error":"result processing failed"'; exit 1; }
st finished
