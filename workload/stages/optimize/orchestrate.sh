#!/usr/bin/env bash
# **Slot-level** orchestration for optimize: schedule -> cold re-verify -> summarize into
# result.json.
#
# Why it lives in the slot rather than in a skill: these three steps are identical for any
# optimization method, and "one UT per GPU / 48 h / fault tolerance / nothing counts until
# re-verified" is what instruct.md asks of this **stage**, not a property of some method.
# A skill owns only prompt.sh (how to optimize). Hence both skills' run.sh are three lines
# that delegate here.
set -uo pipefail

# $IN is deliberately not required: the upstream is **two** stages, profile and roofline,
# not merely the immediately preceding one, so per contract §2 this reads
# $RUN_ROOT/<stage>/out rather than $IN.
: "${OUT:?OUT}"; : "${RUN_ROOT:?RUN_ROOT}"; : "${SKILL_ROOT:?SKILL_ROOT}"
: "${RUNNER:?RUNNER}"; : "${WORKLOAD_ROOT:?WORKLOAD_ROOT}"
SCRIPTS="$WORKLOAD_ROOT/scripts"
PARAMS="${PARAMS:-$OUT/params.json}"

p() {  # p <json.path> <default>
  python3 -c "
import json,sys
d=json.load(open('$PARAMS')) if __import__('os').path.exists('$PARAMS') else {}
v=d
for k in '$1'.split('.'):
    v = v.get(k) if isinstance(v, dict) else None
print('$2' if v is None else (str(v).lower() if isinstance(v,bool) else v))"
}

WORK_ROOT="$OUT/work"
mkdir -p "$WORK_ROOT"

# Task directories come from the profile stage; the order comes from roofline (descending
# expected end-to-end gain). Both are read through $RUN_ROOT/<stage>/out rather than a
# hardcoded relative path — contract §2.
TASKS_DIR="${TASKS_DIR:-$RUN_ROOT/profile/out/tasks}"
ORDER_FILE="${ORDER_FILE:-$RUN_ROOT/roofline/out/roofline.json}"
if [ ! -d "$TASKS_DIR" ]; then
  python3 "$SCRIPTS/emit_result.py" --status blocked \
    --notes "no task directory at $TASKS_DIR — run the profile stage first"
  exit 0
fi
[ -f "$ORDER_FILE" ] || {
  echo "[optimize] no $ORDER_FILE; running in directory-name order (a degraded ranking, but correctness is unaffected)"
  ORDER_FILE=""
}

TARGET="$(p time_budget_sec 172800)"
CAP="$(p attempt_cap_sec 1500)"
MEASTO="$(p measure_timeout_sec 1800)"
AMIN="$(p attempt_min_sec 600)"
BUDGET="$(p budget 4)"
MIN_IMPROVE="$(p min_improve 0.05)"
KEEP="$(p keep_threshold 1.045)"
STALE="$(p heartbeat_stale_sec 900)"
BADLIM="$(p bad_node_limit 2)"
MAXT="$(p max_tasks 0)"
GPT="$(p gpus_per_task 1)"
# exclude_tasks is a JSON list in the config; the generic reader above would print its
# Python repr, so it is read on its own.
EXCL="$(python3 -c "
import json, os
d = json.load(open('$PARAMS')) if os.path.exists('$PARAMS') else {}
v = d.get('exclude_tasks') or []
print(','.join(v) if isinstance(v, list) else str(v))")"
# Per-seam noise bands: {"task": 1.35}. The band belongs to the seam, not the campaign.
KEEPOV="$(python3 -c "
import json, os
d = json.load(open('$PARAMS')) if os.path.exists('$PARAMS') else {}
v = d.get('keep_threshold_overrides') or {}
print(','.join('%s=%s' % (k, x) for k, x in sorted(v.items())) if isinstance(v, dict) else '')")"
IMAGE="$(p image '')"
REPEATS="$(p verify_repeats 2)"
BACKEND="${OPT_BACKEND:-$( [ -n "$IMAGE" ] && echo docker || echo local )}"

# GPU list: if the config does not give one, use what the machine actually has.
GPUS="$(p gpus '')"
if [ -z "$GPUS" ]; then
  N="$(python3 -c "
import glob
print(len(glob.glob('/dev/dri/renderD*')) or 1)" 2>/dev/null || echo 1)"
  GPUS="$(python3 -c "print(','.join(str(i) for i in range($N)))")"
fi

echo "[optimize] skill=$(basename "$SKILL_ROOT") runner=$(basename "$RUNNER") backend=$BACKEND"
echo "[optimize] gpus=$GPUS gpus_per_task=$GPT target=${TARGET}s cap=${CAP}s keep>=$KEEP tasks_dir=$TASKS_DIR"
[ -n "$EXCL" ] && echo "[optimize] excluded tasks: $EXCL"
[ -n "$KEEPOV" ] && echo "[optimize] per-seam keep thresholds: $KEEPOV"

ARGS=(--work-root "$WORK_ROOT" --tasks-dir "$TASKS_DIR" --skill-root "$SKILL_ROOT"
      --runner "$RUNNER" --gpus "$GPUS" --target "$TARGET" --attempt-cap "$CAP" --attempt-min "$AMIN"
      --keep-threshold "$KEEP" --budget "$BUDGET" --min-improve "$MIN_IMPROVE"
      --stale-sec "$STALE" --bad-node-limit "$BADLIM" --backend "$BACKEND"
      --gpus-per-task "$GPT")
[ -n "$EXCL" ] && ARGS+=(--exclude "$EXCL")
[ -n "$KEEPOV" ] && ARGS+=(--keep-threshold-override "$KEEPOV")
ARGS+=(--measure-timeout "$MEASTO")
[ -n "$ORDER_FILE" ] && ARGS+=(--order-file "$ORDER_FILE")
[ "$MAXT" != "0" ] && [ -n "$MAXT" ] && ARGS+=(--max-tasks "$MAXT")
[ -n "$IMAGE" ] && ARGS+=(--image "$IMAGE")
[ -n "${OPT_MAX_WALL_SEC:-}" ] && ARGS+=(--max-wall-sec "$OPT_MAX_WALL_SEC")

python3 "$SCRIPTS/scheduler.py" run "${ARGS[@]}" 2>&1 | tee -a "$OUT/scheduler.out"
SRC=${PIPESTATUS[0]}
echo "[optimize] scheduler exited $SRC"

# The cold re-verification gate. It runs even when the scheduler exited abnormally —
# results already kept should not be lost because scheduling died.
#
# It needs a GPU, and with --backend docker this process has none: the scheduler only
# starts containers, so the machine it runs on is not required to expose /dev/kfd. The
# gate therefore runs inside one container of the same image, with a single renderD node,
# the way a worker does. Kernels are verified serially in that one container — this is the
# last step of the stage and nothing else wants the card.
VERIFY=(python3 "$SCRIPTS/verify.py" --work-root "$WORK_ROOT" --tasks-dir "$TASKS_DIR"
        --repeats "$REPEATS" --threshold "$KEEP" --out "$OUT/verified.json")
[ -n "$KEEPOV" ] && VERIFY+=(--threshold-override "$KEEPOV")
if [ "$BACKEND" = docker ]; then
  # One slot's worth of GPUs, the same group the scheduler would have given a worker:
  # a collective unit test re-verified on a single card fails for lack of ranks, not
  # for lack of a speedup.
  VDEV=()
  for VGPU in $(python3 -c "
gpus = '$GPUS'.split(',')
print(' '.join(gpus[:max(1, int('$GPT'))]))"); do
    VDEV+=(--device "/dev/dri/renderD$(( 128 + 8 * VGPU ))")   # numbering as scheduler.py
  done
  docker run --rm --name "wl_verify_$$" \
      --device /dev/kfd "${VDEV[@]}" --group-add video --ipc host \
      --security-opt seccomp=unconfined --shm-size 16g -v /shared_nfs:/shared_nfs \
      -e GPU_IDS="$(python3 -c "print(','.join(str(i) for i in range(max(1, int('$GPT')))))")" \
      "$IMAGE" "${VERIFY[@]}" 2>&1 | tee -a "$OUT/verify.out"
else
  "${VERIFY[@]}" 2>&1 | tee -a "$OUT/verify.out"
fi

# Slot adapter: artifacts -> result.json.
python3 "$(dirname "${BASH_SOURCE[0]}")/finalize.py" --scheduler-rc "$SRC"
