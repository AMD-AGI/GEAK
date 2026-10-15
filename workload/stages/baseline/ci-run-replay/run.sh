#!/usr/bin/env bash
# script-mode entry point (contract §2 A). Reproduce a named CI run from a fresh container
# and accept the result as the baseline only if it aligns. See SKILL.md.
#
# Order matters: every gate that can refuse without spending GPU time runs first, the
# container is created new for this run, and the proof pass reads the run's own records
# rather than this script's intentions.
set -uo pipefail

PY="${PYTHON_BIN:-python3}"
emit() { "$PY" "$WORKLOAD_ROOT/scripts/emit_result.py" "$@"; }
param() { "$PY" -c "
import json,sys
cur=json.load(open(sys.argv[1]))
for k in sys.argv[2].split('.'):
    cur = cur.get(k) if isinstance(cur, dict) else None
    if cur is None: break
if cur is None: cur = (sys.argv[3] if len(sys.argv)>3 else '')
print('' if cur is None else (json.dumps(cur) if isinstance(cur,(dict,list)) else cur))
" "$PARAMS" "$1" "${2-}"; }

LEDGER="$(param isolation.ledger "$WORKLOAD_ROOT/runs/.benchmark_id_ledger.json")"
IMAGE="$(param pins.image)"
DIGEST="$(param pins.image_digest)"
DURATION="$(param pins.duration_s 3600)"
BID="$(param pins.benchmark_id)"
CNAME="$(param isolation.container_name wl_ci_replay)"
HDIR="$(param harness.dir)"
HCMD="$(param harness.cmd)"
HCASE="$(param harness.case baseline)"
RUN_TAG="$(param harness.run_tag "$BID")"

# -------- 1. Preflight: digest, data/seed, benchmark_id, cache-bust, CPU plan, cold caches.
if ! "$PY" "$SKILL_ROOT/preflight.py" --params "$PARAMS" --out "$OUT" --ledger "$LEDGER" \
        > "$OUT/preflight.log" 2>&1; then
  emit --status blocked --artifact preflight_log=preflight.log --artifact alignment=alignment.json \
    --notes "preflight refused the reproduction: $(sed -n 's/^  - //p' "$OUT/preflight.log" | head -3 | tr '\n' ';') (full list in preflight.log). Nothing was launched and no GPU time was spent."
  exit 1
fi
cat "$OUT/preflight.log"

if [ -z "$HDIR" ] || [ -z "$HCMD" ]; then
  emit --status blocked --artifact alignment=alignment.json \
    --notes "harness.dir / harness.cmd are not configured: this skill drives the site's own CI harness rather than reimplementing the client, so it needs to be told where that harness is."
  exit 1
fi

# -------- 2. Cold caches, created here so their emptiness is this run's fact.
# Exported as variables, and -- when the plan says so -- also bind-mounted at the same
# path inside the container. A harness that exports TRITON_CACHE_DIR itself wins over any
# -e we pass; mounting the path it names is the only way to make its own export land in a
# cold directory. See preflight.cache_plan().
CACHE_ENV=()
while IFS= read -r kv; do
  [ -z "$kv" ] && continue
  case "$kv" in -v*) CACHE_ENV+=(-v "${kv#-v}") ;; *) CACHE_ENV+=(-e "$kv") ;; esac
done < <(
  "$PY" -c "
import json,os,sys
a=json.load(open(sys.argv[1]+'/alignment.json'))
mounts = a.get('cache_mounts') or {}
for var, d in sorted((a.get('cache_vars') or {}).items()):
    os.makedirs(d, exist_ok=True)
    print('%s=%s' % (var, mounts.get(d, d)))
for host, container in sorted(mounts.items()):
    print('-v%s:%s' % (host, container))
" "$OUT")

CPUSET="$("$PY" -c "
import json,sys
a=json.load(open(sys.argv[1]+'/alignment.json'))
print(','.join(str(c) for c in (a.get('cpu_plan') or {}).get('cpus') or []))
" "$OUT")"

# Pins the client must see. The variable names are site-specific, so they come from
# params.env_map rather than being guessed here.
PIN_ENV=()
while IFS= read -r kv; do [ -n "$kv" ] && PIN_ENV+=(-e "$kv"); done < <(
  "$PY" -c "
import json,sys
p=json.load(open(sys.argv[1]))
pins=p.get('pins') or {}
m=(p.get('env_map') or {})
for key, var in m.items():
    v = pins.get(key)
    if v not in (None, ''):
        print('%s=%s' % (var, v))
for k, v in (p.get('env') or {}).items():
    print('%s=%s' % (k, v))
" "$PARAMS")

DOCKER_ARGS=()
while IFS= read -r x; do [ -n "$x" ] && DOCKER_ARGS+=("$x"); done < <(
  "$PY" -c "
import json,sys
d=json.load(open(sys.argv[1])).get('docker') or {}
for v in d.get('volumes') or []: print('-v'); print(v)
for x in d.get('extra_args') or []: print(x)
" "$PARAMS")

# -------- 3. A new container, addressed by digest. Never reuse, never pull by tag.
REF="${IMAGE%%:*}@${DIGEST}"
ARGS="$(param harness.args)"
# params.harness.args is a list of literal arguments in which the bare tokens CASE,
# RUN_TAG and DURATION are substituted. Default: the three positional arguments the
# reference harness takes.
CMD="$("$PY" -c '
import json, shlex, sys
raw, cmd, case, tag, dur = sys.argv[1:6]
args = json.loads(raw) if raw else ["CASE", "RUN_TAG", "DURATION"]
sub = {"CASE": case, "RUN_TAG": tag, "DURATION": dur}
print(" ".join(shlex.quote(str(sub.get(x, x))) for x in [cmd] + list(args)))
' "$ARGS" "$HCMD" "$HCASE" "$RUN_TAG" "$DURATION")"

echo "[ci-run-replay] image   $REF"
echo "[ci-run-replay] cpuset  $CPUSET"
echo "[ci-run-replay] command (cd $HDIR && bash $CMD)"

CID="$(docker run -d --name "$CNAME" --rm \
        --device /dev/kfd --device /dev/dri \
        --network host --ipc host --shm-size 128g \
        --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
        ${CPUSET:+--cpuset-cpus "$CPUSET"} \
        "${CACHE_ENV[@]}" "${PIN_ENV[@]}" "${DOCKER_ARGS[@]}" \
        -w "$HDIR" "$REF" \
        bash -lc "bash $CMD" 2> "$OUT/docker_run.err")"
if [ -z "$CID" ]; then
  emit --status fail --artifact alignment=alignment.json --artifact docker_err=docker_run.err \
    --notes "docker run failed before the server started; see docker_run.err. The container was addressed by digest ($REF), so a missing image here means the digest is not present locally."
  exit 1
fi
echo "[ci-run-replay] container $CID"

docker inspect "$CID" > "$OUT/container_info.json" 2>/dev/null
# Sample the mask the server actually got, from inside the container. A --cpuset-cpus that
# the daemon silently narrowed looks identical from outside.
docker exec "$CID" "$PY" -c "import os,json;print(json.dumps({'cpus':sorted(os.sched_getaffinity(0))}))" \
  > "$OUT/affinity.json" 2>/dev/null || echo '{}' > "$OUT/affinity.json"

docker logs -f "$CID" > "$OUT/replay.log" 2>&1 &
LOGPID=$!
docker wait "$CID" > "$OUT/exit_code.txt" 2>/dev/null
RC="$(cat "$OUT/exit_code.txt" 2>/dev/null || echo 1)"
wait "$LOGPID" 2>/dev/null

RUN_DIR="$(param harness.result_dir "$HDIR/runs/$RUN_TAG")"
if [ ! -d "$RUN_DIR" ]; then
  emit --status fail --artifact replay_log=replay.log --artifact alignment=alignment.json \
    --notes "the harness exited $RC and left no run directory at $RUN_DIR; see replay.log."
  exit 1
fi

# -------- 4. Proof pass: read back what the run recorded, not what this script asked for.
"$PY" "$SKILL_ROOT/verify_alignment.py" --params "$PARAMS" --alignment "$OUT/alignment.json" \
    --out "$OUT" --run-dir "$RUN_DIR" --container-info "$OUT/container_info.json" \
    --affinity "$OUT/affinity.json" --ledger "$LEDGER" \
    $( [ "$(param allow_unproven false)" = "true" ] && echo --allow-unproven ) \
    > "$OUT/verify.log" 2>&1
VRC=$?
cat "$OUT/verify.log"

VERDICT="$("$PY" -c "import json;print(json.load(open('$OUT/alignment_verified.json'))['verdict'])" 2>/dev/null || echo FAILED)"
WHY="$("$PY" -c "import json;d=json.load(open('$OUT/alignment_verified.json'));print('; '.join((d['failures'] or d['unproven'])[:3]))" 2>/dev/null)"
if [ "$VRC" != "0" ]; then
  emit --status blocked --artifact verify_log=verify.log --artifact verified=alignment_verified.json \
    --artifact replay_log=replay.log --artifact alignment=alignment.json \
    --notes "the replay ran but did not prove alignment ($VERDICT): $WHY. An unaligned reproduction must not become the baseline that every later delta is measured against."
  exit 1
fi

# -------- 5. The canonical baseline.json.
"$PY" - "$OUT" "$PARAMS" "$RUN_DIR" <<'PY'
import json, socket, sys
out, params, run_dir = sys.argv[1:4]
p = json.load(open(params)); v = json.load(open(out + "/alignment_verified.json"))
a = json.load(open(out + "/alignment.json")); pins = p.get("pins") or {}; ci = p.get("ci") or {}
doc = {
    "metric": ci.get("metric", "output_token_throughput"),
    "unit": ci.get("unit", "tok/s"),
    "throughput": v["throughput"],
    "higher_is_better": True,
    "noise_band": p.get("noise_band", 1.045),
    "config": p.get("config") or {},
    "reference": {"name": "ci_run_%s" % (ci.get("run_id") or "?"),
                  "value": ci.get("output_throughput"),
                  "drift_pct": v.get("drift_pct"),
                  "url": ci.get("run_url")},
    "measured": v["throughput"] is not None,
    "provenance": {
        "method": "ci-run-replay",
        "image": pins.get("image"), "image_digest": pins.get("image_digest"),
        "model": pins.get("model"), "model_revision": pins.get("model_revision"),
        "seed": pins.get("seed"), "benchmark_id": pins.get("benchmark_id"),
        "cache_bust": pins.get("cache_bust"), "duration_s": pins.get("duration_s"),
        "trace_corpus": pins.get("trace_corpus"),
        "cpu_binding": a.get("cpu_plan"), "cold_cache_dirs": a.get("cache_vars"),
        "fresh_container": True, "run_dir": run_dir,
        "alignment_verdict": v["verdict"], "alignment_checks": v["checks"],
        "node": socket.gethostname(),
    },
}
json.dump(doc, open(out + "/baseline.json", "w"), indent=2, ensure_ascii=False)
PY

TPUT="$("$PY" -c "import json;print(json.load(open('$OUT/baseline.json'))['throughput'])")"
DRIFT="$("$PY" -c "import json;print(json.load(open('$OUT/baseline.json'))['reference']['drift_pct'])")"
emit --status ok \
  --artifact baseline=baseline.json --artifact alignment=alignment.json \
  --artifact verified=alignment_verified.json --artifact replay_log=replay.log \
  --metric throughput="$TPUT" --metric ci_drift_pct="$DRIFT" \
  --metric alignment="$VERDICT" \
  --notes "reproduced CI run $(param ci.run_id) from a fresh container pinned by digest, cold compile caches, physical-core binding, benchmark_id $BID retired to the ledger. Drift vs CI: ${DRIFT}%."
