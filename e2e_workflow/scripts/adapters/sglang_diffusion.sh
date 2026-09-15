# sglang-diffusion adapter for bench_e2e.sh.  Sourced (not executed). Defines the contract functions.
# Reads env exported by the dispatcher: MODEL HOST PORT TP GPU MEM_FRACTION EXTRA_SERVER_ARGS
#   EXTRA_ENV OVERLAY_PYTHONPATH PROFILE PROFILE_DIR BASE_URL RESULT_JSONL LOG ISL OSL CONC SEED
#
# DIFFERENCES FROM THE TEXT-LLM ADAPTERS (all forced by what a diffusion model is):
#   * Metric is img/s, not tok/s. bench_e2e.sh is metric-neutral — it medians whatever
#     `output_throughput` the bench reports — so nothing downstream needs to change.
#   * There is no long-lived server. One in-process DiffGenerator engine lives inside each bench
#     call (scripts/diffusion_bench.py). adapter_launch is therefore a no-op holder process and
#     adapter_health is trivially true. The engine load (~25 s) is inside every bench call; the
#     internal warmup pass keeps it out of the measured number.
#   * No adapter_profile_window: sglang-diffusion's HTTP server exposes no /start_profile. That is
#     fine — a diffusion "steady state" IS the denoise loop of a single request, so bench_e2e.sh's
#     fallback (a PROF=1 bench) captures exactly the right thing via the in-process torch profiler.
#   * ISL/OSL carry the image dimensions (1024/1024 -> 1024x1024). See knowledge/diffusion_sglang.md.

adapter_default_port() { echo 31120; }

# The bench harness and its prompt set live in the sglang_diffusion/ subdir NEXT TO this file, so
# they survive roles/director.md's `cp -r "$SKILL_DIR/scripts/adapters" "$EVAL_DIR/adapters"` — an
# eval dir gets a working adapter without any extra copy step. Override with SGLD_BENCH=<path>.
SGLD_BENCH="${SGLD_BENCH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sglang_diffusion/diffusion_bench.py}"
if [ ! -f "$SGLD_BENCH" ]; then
  echo "!!! sglang_diffusion adapter: bench harness not found at $SGLD_BENCH" >&2
  echo "    (expected adapters/sglang_diffusion/diffusion_bench.py next to this adapter)" >&2
  exit 3
fi

# ---- mandatory config hygiene ------------------------------------------------------------------
# These are NOT tuning choices; without them the measurement is of a different, much slower engine:
#   --model-id FLUX.1-dev : sglang resolves its native pipeline by matching the substring "flux.1"
#       against the model path (registry.py). The local checkout is FLUX-1-dev (dot -> dash), so
#       detection fails and sglang SILENTLY falls back to the diffusers backend — measured 17.2x
#       slower on this box. Any run without this flag is benchmarking the wrong engine.
#   --dit-cpu-offload/--text-encoder-cpu-offload False : sglang defaults BOTH on for any image model
#       regardless of VRAM; FLUX.1-dev is ~24 GB bf16 against 288 GB HBM.
# They go FIRST so a Config-Tuner variant in EXTRA_SERVER_ARGS can still override any of them
# (sglang's argparse takes the last occurrence) — including deliberately testing the offloads.
SGLD_BASE_ARGS=${SGLD_BASE_ARGS:-"--model-id FLUX.1-dev --dit-cpu-offload False --text-encoder-cpu-offload False"}

# ---- measurement mode --------------------------------------------------------------------------
#   serve (default) : a real `sglang serve` process + sglang's own bench_serving driving CONC
#                     concurrent image requests. This is the faithful analogue of "conc=64" and the
#                     ONLY mode in which a data-parallel layout (--dp-size) can show its throughput:
#                     a serial request stream can occupy just one replica at a time, so DP would
#                     measure as a loss no matter how good it is.
#   local           : server-less, one in-process engine, images generated back-to-back. Concurrency
#                     is 1 by construction, so it measures per-image LATENCY (and SP layouts). Used
#                     for profiling (a clean single-request trace) and as a cross-check.
SGLD_MODE=${SGLD_MODE:-serve}

# Ports: sglang-diffusion defaults --master-port to a fixed 30005, so two engines on disjoint GPU
# leases would fight over the same torch.distributed rendezvous. Derive from the first pinned GPU
# (concurrent leases hold disjoint GPU sets) and keep them clear of bench_e2e.sh's own $PORT window.
_sgld_first_gpu() { echo "${GPU%%,*}"; }

# ---- data parallelism: use SGLD_DP_SIZE, NOT --dp-size --------------------------------------------
# UPSTREAM BUG (sglang 0.5.12): `--dp-size` / `--data-parallel-size` / `--dp` all land in the argparse
# dest `data_parallel_size`, but the ServerArgs dataclass field is `dp_size`, and
# ServerArgs.from_dict() copies namespace keys BY FIELD NAME. The value is therefore dropped and the
# server silently runs dp_size=1 (verified: `--dp-size 8` resolved to dp_size=1, sp_degree=8). A
# variant that "tests DP" via the flag is really re-testing SP.
# `sglang serve --config <json>` is the working path: load_config_file()'s dict is merged into
# provided_args keyed by FIELD name, so {"dp_size": N} reaches the field. SGLD_DP_SIZE writes that
# file. In local mode diffusion_bench.py sets the attribute directly for the same reason.
_sgld_config_arg() {
  [ -n "${SGLD_DP_SIZE:-}" ] || { echo ""; return 0; }
  local cfg="${OUT_DIR:-/tmp}/sgld_config.json"
  mkdir -p "$(dirname "$cfg")"
  printf '{"dp_size": %s}\n' "$SGLD_DP_SIZE" > "$cfg"
  echo "--config $cfg"
}
_sgld_port_args() {
  local base=$(( 31000 + $(_sgld_first_gpu) * 200 ))
  echo "--master-port $base --scheduler-port $((base + 60))"
}

# Never set HIP_VISIBLE_DEVICES on this ROCm stack — it can make torch.cuda.is_available() return
# false. GPU pinning for the diffusion engine is ROCR_VISIBLE_DEVICES.
adapter_launch() {
  mkdir -p "$(dirname "$LOG")"
  if [ "$SGLD_MODE" = "serve" ]; then
    # shellcheck disable=SC2086
    env -u HIP_VISIBLE_DEVICES $EXTRA_ENV \
      ROCR_VISIBLE_DEVICES="$GPU" \
      SGLANG_DIFFUSION_TORCH_PROFILER_DIR="$PROFILE_DIR" \
      PYTHONPATH="${OVERLAY_PYTHONPATH:+$OVERLAY_PYTHONPATH:}${PYTHONPATH:-}" \
      sglang serve \
        --model-type diffusion \
        --model-path "$MODEL" \
        --num-gpus "$TP" \
        --host "$HOST" --port "$PORT" \
        $(_sgld_port_args) $(_sgld_config_arg) \
        $SGLD_BASE_ARGS $EXTRA_SERVER_ARGS \
        > "$LOG" 2>&1 &
    SERVER_PID=$!
    return 0
  fi
  {
    echo "[sglang_diffusion] server-less backend: the engine lives inside each bench call."
    echo "[sglang_diffusion] MODEL=$MODEL  num-gpus(TP)=$TP  GPU=$GPU"
    echo "[sglang_diffusion] base args:  $SGLD_BASE_ARGS"
    echo "[sglang_diffusion] extra args: ${EXTRA_SERVER_ARGS:-<none>}"
    echo "[sglang_diffusion] extra env:  ${EXTRA_ENV:-<none>}"
    echo "[sglang_diffusion] overlay:    ${OVERLAY_PYTHONPATH:-<none>}"
  } >> "$LOG" 2>&1
  # bench_e2e.sh requires a non-empty SERVER_PID and reaps it in cleanup(); a holder process keeps
  # that lifecycle contract honest without pretending there is a server.
  sleep 86400 &
  SERVER_PID=$!
}

adapter_health() {
  if [ "$SGLD_MODE" = "serve" ]; then
    curl -sf "${BASE_URL}/health" >/dev/null 2>&1
  else
    [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null
  fi
}

# serve-mode bench: sglang's own diffusion bench_serving drives MAXC concurrent requests. Its metric
# is `output_throughput_ops` (images/s); translate it into bench_e2e.sh's canonical schema.
_sgld_bench_serve() {
  local NUMP="$1" MAXC="$2"
  local raw="${OUT_DIR:-$PWD}/bench_serving_$(date +%s%N).json"
  # shellcheck disable=SC2086
  env -u HIP_VISIBLE_DEVICES \
    python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
      --base-url "$BASE_URL" --model "$MODEL" \
      --dataset "${SGLD_DATASET:-random}" \
      --num-prompts "$NUMP" --max-concurrency "$MAXC" \
      --warmup-requests "${SGLD_WARMUP_REQUESTS:-4}" \
      --width "$ISL" --height "$OSL" \
      --num-inference-steps "${SGLD_NUM_STEPS:-28}" \
      --output-file "$raw" 2>&1 | tee -a "$LOG"
  local rc="${PIPESTATUS[0]}"
  [ -f "$raw" ] || { echo "!!! bench_serving wrote no metrics file" >&2; return 1; }
  RAW="$raw" SINK="$RESULT_JSONL" STEPS="${SGLD_NUM_STEPS:-28}" python3 - <<'PY'
import json, os
m = json.load(open(os.environ["RAW"]))
# latency_* are per-request seconds; there is no TTFT/TPOT for an image, so ttft = whole-request
# latency and tpot = per-denoise-step time, which is what those columns mean here.
lat_ms = float(m.get("latency_median") or 0) * 1000.0
rec = {
    "framework": "sglang_diffusion", "mode": "serve", "throughput_unit": "img/s",
    "output_throughput": float(m.get("output_throughput_ops") or 0),
    "request_throughput": float(m.get("throughput_qps") or 0),
    "completed": m.get("completed_requests"), "failed": m.get("failed_requests"),
    "duration": m.get("duration"),
    "median_ttft_ms": lat_ms, "median_tpot_ms": lat_ms / max(1, int(os.environ["STEPS"])),
    "mean_e2el_ms": float(m.get("latency_mean") or 0) * 1000.0,
    "median_e2el_ms": lat_ms,
    "p99_e2el_ms": float(m.get("latency_p99") or 0) * 1000.0,
    "ms_per_step": lat_ms / max(1, int(os.environ["STEPS"])),
    "raw_metrics": m,
}
if rec["output_throughput"] > 0 and not m.get("failed_requests"):
    with open(os.environ["SINK"], "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[bench] output_throughput={rec['output_throughput']:.4f} img/s "
          f"median_e2el={rec['median_e2el_ms']:.1f}ms completed={rec['completed']}")
else:
    print(f"[bench] REFUSING to record: throughput={rec['output_throughput']} "
          f"failed={m.get('failed_requests')}")
PY
  return "$rc"
}

# adapter_bench NUM_PROMPTS MAX_CONC PROFILE_FLAG
#
# Image count, not prompt count: an "image" here is what a token is there. bench_e2e.sh calls the
# WARMUP round with NUMP == CONC and every TIMED round with NUMP == NUM_PROMPTS (default CONC*5),
# which is how this adapter tells them apart. The timed rounds use SGLD_NUM_ITERATIONS (default
# CONC = 64 images) rather than NUM_PROMPTS, because 320 images per repeat would cost ~17 min for no
# extra precision — the measured run-to-run spread at 64 images is ~0.1%.
#
# The WARMUP round is SKIPPED outright. bench_e2e.sh warms up because a persistent server carries
# JIT/graph state from one bench to the next; here there is no such server — every timed call builds
# its own engine and runs SGLD_WARMUP_CALLS untimed images inside it. An outer warmup round would
# therefore buy nothing and cost a full engine load (minutes at 8 ranks, which each stream the 24 GB
# checkpoint). Skipping it removes ~25% of the wall-clock of every measurement.
adapter_bench() {
  local NUMP="$1" MAXC="$2" PROF="${3:-0}"
  local images

  # serve mode: bench_serving owns warmup (--warmup-requests) and concurrency. Profiling still runs
  # through the LOCAL engine below: the diffusion server exposes no profiler hook, and a clean
  # single-request trace of the denoise loop is what the Profiler actually wants.
  if [ "$SGLD_MODE" = "serve" ] && [ "$PROF" != "1" ]; then
    if [ "$NUMP" -le "$CONC" ] 2>/dev/null; then
      echo "[sglang_diffusion] skipping bench_e2e warmup round (bench_serving warms with" \
           "--warmup-requests ${SGLD_WARMUP_REQUESTS:-4})" | tee -a "$LOG"
      return 0
    fi
    _sgld_bench_serve "${SGLD_NUM_PROMPTS:-$CONC}" "$MAXC"
    return $?
  fi

  # PROF=1 in serve mode: the profile pass builds its OWN in-process engine, so the serve process
  # (which is holding every serving GPU) must go first or the two contend for the same cards and the
  # trace is of a distorted machine. bench_e2e.sh runs the profile step LAST, after the timed
  # repeats, so nothing later needs the server; its cleanup() tolerates an already-dead pid.
  if [ "$SGLD_MODE" = "serve" ] && [ "$PROF" = "1" ] && [ -n "${SERVER_PID:-}" ]; then
    echo "[sglang_diffusion] stopping the serve process before the in-process profile pass" | tee -a "$LOG"
    kill "$SERVER_PID" 2>/dev/null || true
    for _i in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    sleep 5
  fi

  if [ "$NUMP" -le "$CONC" ] 2>/dev/null && [ "$PROF" != "1" ]; then
    echo "[sglang_diffusion] skipping bench_e2e warmup round (server-less: every timed call warms" \
         "its own engine with SGLD_WARMUP_CALLS=${SGLD_WARMUP_CALLS:-3} untimed images)" | tee -a "$LOG"
    return 0
  fi
  images=${SGLD_NUM_ITERATIONS:-$CONC}           # timed / cold round
  local prof_env=()
  if [ "$PROF" = "1" ]; then
    mkdir -p "$PROFILE_DIR"
    prof_env=(PROFILE=1
              SGLANG_DIFFUSION_TORCH_PROFILER_DIR="$PROFILE_DIR"
              SGLANG_TORCH_PROFILER_DIR="$PROFILE_DIR")
    # A profiled image is much slower and the trace grows with every captured step; a couple of
    # images is plenty to rank the denoise loop's kernels.
    images=${SGLD_PROFILE_IMAGES:-2}
  else
    prof_env=(PROFILE=0)
  fi

  local out_dir="${OUT_DIR:-$PWD/e2e_bench_out}"
  # shellcheck disable=SC2086
  env -u HIP_VISIBLE_DEVICES $EXTRA_ENV \
    "${prof_env[@]}" \
    ROCR_VISIBLE_DEVICES="$GPU" \
    MODEL="$MODEL" TP="$TP" CONC="$CONC" ISL="$ISL" OSL="$OSL" \
    SGLD_NUM_ITERATIONS="$images" \
    SGLD_DP_SIZE="${SGLD_DP_SIZE:-}" \
    SGLD_SEED="${SGLD_SEED:-$SEED}" \
    EXTRA_SGLANG_DIFFUSION_ARGS="$SGLD_BASE_ARGS $EXTRA_SERVER_ARGS" \
    OUT_DIR="$out_dir" RESULT_JSONL="$RESULT_JSONL" \
    OVERLAY_PYTHONPATH="$OVERLAY_PYTHONPATH" \
    PYTHONPATH="${OVERLAY_PYTHONPATH:+$OVERLAY_PYTHONPATH:}${PYTHONPATH:-}" \
    python3 "$SGLD_BENCH" 2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}
