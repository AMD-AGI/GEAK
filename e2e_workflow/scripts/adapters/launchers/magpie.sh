#!/usr/bin/env bash
# Magpie server LAUNCHER adapter for bench_e2e.sh.  Sourced (not executed).
# BACKEND-AGNOSTIC: one adapter serves every backend Magpie ships a script for
# (sglang, vllm, ...), because Magpie standardised the server-phase contract:
#   MAGPIE_RUN_PHASE=server  ->  launch server, wait ready, write pid to
#   MAGPIE_SERVER_PID_FILE, disown, exit 0.  Reads MODEL/TP/PORT/RESULT_DIR/
#   SERVER_LOG/PROFILE.  The only per-backend differences follow a REGULAR naming
#   rule, so they are derived from $BACKEND (never hard-coded per backend):
#     * extra server flags var : EXTRA_<BACKEND_UPPER>_ARGS  (EXTRA_SGLANG_ARGS / EXTRA_VLLM_ARGS)
#     * torch-profiler dir  var : <BACKEND_UPPER>_TORCH_PROFILER_DIR
#
# Redefining ONLY adapter_launch makes the served stack BYTE-IDENTICAL to the
# orchestrator's baseline (mem-fraction, --disable-radix-cache / gpu-mem-util,
# --trust-remote-code, *_USE_AITER, firmware-gated HSA_NO_SCRATCH_RECLAIM, ... all
# owned by that one script). The authored-kernel OVERLAY is prepended to
# PYTHONPATH HERE (Magpie's own path never honors OVERLAY_PYTHONPATH), so
# recipe-parity AND overlay application coexist.
#
# Script resolution (general): $MAGPIE_LAUNCH_SCRIPT, else the per-backend
# $MAGPIE_<BACKEND_UPPER>_SCRIPT. Its sibling benchmark_lib.sh / server_cleanup.sh
# must be present next to it. When no script is resolvable it DELEGATES to the
# native backend launch (adapter_launch_native), so a misconfigured run degrades
# instead of failing hard.
#
# bench_e2e.sh contract: sets global SERVER_PID; writes $LOG. Reads env:
#   BACKEND MODEL TP PORT GPU EXTRA_SERVER_ARGS EXTRA_ENV OVERLAY_PYTHONPATH
#   PROFILE PROFILE_DIR LOG OUT_DIR MAX_MODEL_LEN PROFILE_MAX_ITERS
#   PROFILE_DELAY_ITERS.
#
# TWO logs, deliberately: Magpie's script redirects the server with a
# TRUNCATING '> $SERVER_LOG', so anything this adapter appended to the same file
# would be destroyed the moment the server starts, and the script's own set -x
# trace would interleave with server output in whatever survived. The script's
# stdout/stderr therefore goes to magpie_launch.log and $LOG stays exclusively
# the server's, which is also what makes $LOG parseable for the kernel-selection
# fingerprint in result.json.serving_stack.
# adapter_health is inherited from the BACKEND adapter (curl $BASE_URL/health),
# which works regardless of who launched the server, so it is NOT redefined.

adapter_launch() {
  local backend_uc script var_script
  backend_uc="$(printf '%s' "${BACKEND:-sglang}" | tr '[:lower:]' '[:upper:]')"

  # generic path first, then per-backend MAGPIE_<BACKEND>_SCRIPT (indirection).
  script="${MAGPIE_LAUNCH_SCRIPT:-}"
  if [ -z "$script" ]; then
    var_script="MAGPIE_${backend_uc}_SCRIPT"
    script="${!var_script:-}"
  fi

  if [ -z "$script" ] || [ ! -f "$script" ]; then
    echo "!!! magpie launcher: no Magpie script for BACKEND='$BACKEND'" \
         "(set MAGPIE_LAUNCH_SCRIPT or MAGPIE_${backend_uc}_SCRIPT; got '$script');" \
         "falling back to native backend launch." >&2
    if declare -F adapter_launch_native >/dev/null; then
      adapter_launch_native
      return $?
    fi
    echo "!!! magpie launcher: no native launch to fall back to." >&2
    return 2
  fi

  local _out_dir="${OUT_DIR:-${PROFILE_DIR:-$(pwd)}}"
  local _pidfile="$_out_dir/magpie_server.pid"
  local _launchlog="$_out_dir/magpie_launch.log"
  rm -f "$_pidfile" 2>/dev/null || true

  # Per-backend var NAMES (regular rule), passed to the script via env NAME=VALUE.
  local _args_var="EXTRA_${backend_uc}_ARGS"
  local _prof_var="${backend_uc}_TORCH_PROFILER_DIR"

  # The orchestrator's RECORDED launch environment, replayed as the BASE layer.
  # Without it the two servers agree only where their ${X:-default} expansions
  # happen to agree -- true today only because both run in the same image, and
  # false the moment the orchestrator sets anything explicitly (PATH selecting a
  # different venv is the one that would silently serve a different vLLM build
  # entirely). run_e2e.py has already removed the run-scoped names GEAK must own,
  # so everything left here is safe to apply verbatim. NUL-delimited because the
  # recipe records PATH and word-splitting a value would corrupt it.
  local _recipe_env=()
  if [ -n "${RECIPE_ENV_FILE:-}" ] && [ -f "${RECIPE_ENV_FILE}" ]; then
    mapfile -d '' -t _recipe_env < "$RECIPE_ENV_FILE"
    echo ">>> magpie launcher: replaying ${#_recipe_env[@]} recorded env var(s) from the recipe."
  fi

  # The recipe may record EXTRA_<BACKEND>_ARGS -- the flags that decide kernel
  # dispatch (--kv-cache-dtype / --moe-runner-backend / --attention-backend /
  # --quantization / --dtype / --block-size ...). It is the BASE layer, but the
  # launch line below also passes GEAK's own ${_args_var}=... LATER on the same
  # `env`, and `env NAME=VALUE` is LAST-WINS, so the recipe copy would be silently
  # dropped even though RECIPE_ENV_REPLAYED still advertises it. Pull the recipe's
  # value out here and REMOVE it from _recipe_env (otherwise the var is set twice
  # on one env line), then merge it UNDER GEAK's accepted flags at _extra_args
  # init below (recipe first, GEAK covers conflicts by ordering).
  #
  # The recipe env array becomes POSITIONAL operands to the same `env` below, so
  # it carries the identical option/command/mask-injection hazards as EXTRA_ENV
  # (a recipe-recorded `-SCUDA_VISIBLE_DEVICES=7` or a bare word would be an env
  # option/command; an ROCR/HIP/CUDA assignment would steal a card). run_e2e.py
  # already validates the recorded names, but this launcher is also driven
  # directly by tests and could be reused, so filter here too -- keep ONLY strict
  # IDENTIFIER=VALUE tokens and drop GPU masks. The `--` on the env line below is
  # the final backstop.
  local _recipe_extra=""
  if ((${#_recipe_env[@]})); then
    local -a _kept=(); local _kv
    for _kv in "${_recipe_env[@]}"; do
      case "$_kv" in
        "${_args_var}="*) _recipe_extra="${_kv#*=}" ;;
        ROCR_VISIBLE_DEVICES=*|HIP_VISIBLE_DEVICES=*|CUDA_VISIBLE_DEVICES=*)
          echo ">>> magpie launcher: dropping GPU-mask override from recipe env:" \
               "$_kv (GPU pinning is run-scoped)." >&2 ;;
        *)
          if [[ "$_kv" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            _kept+=("$_kv")
          else
            echo ">>> magpie launcher: dropping non-assignment recipe env token" \
                 "(not IDENTIFIER=VALUE; would be an env option/command): $_kv" >&2
          fi ;;
      esac
    done
    if ((${#_kept[@]})); then _recipe_env=("${_kept[@]}"); else _recipe_env=(); fi
  fi

  # EXTRA_ENV carries GEAK's accepted env under test. Split every input line with
  # `read -ra`, which performs shell word splitting but NOT pathname expansion:
  # an accepted value such as `FOO=*` must reach the child literally even when
  # matching files exist in the launcher's cwd. Reading line-by-line is required
  # because a single `read` stops at the first newline. Keep only strict
  # IDENTIFIER=VALUE assignments; GPU masks are run-scoped and are re-asserted
  # separately below.
  local -a _extra_env=() _extra_env_tokens=() _extra_env_line_tokens=()
  local _tok _extra_env_line
  if [ -n "${EXTRA_ENV:-}" ]; then
    while IFS= read -r _extra_env_line; do
      read -ra _extra_env_line_tokens <<< "$_extra_env_line"
      _extra_env_tokens+=("${_extra_env_line_tokens[@]}")
    done <<< "$EXTRA_ENV"
  fi
  for _tok in "${_extra_env_tokens[@]}"; do
    case "$_tok" in
      ROCR_VISIBLE_DEVICES=*|HIP_VISIBLE_DEVICES=*|CUDA_VISIBLE_DEVICES=*)
        echo ">>> magpie launcher: dropping GPU-mask override from EXTRA_ENV:" \
             "$_tok (GPU pinning is run-scoped)." >&2 ;;
      *)
        if [[ "$_tok" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
          _extra_env+=("$_tok")
        else
          echo ">>> magpie launcher: dropping non-assignment EXTRA_ENV token" \
               "(not IDENTIFIER=VALUE; would be an env option/command): $_tok" >&2
        fi ;;
    esac
  done

  # Magpie's vllm script builds its own --profiler-config.* block but sets no step
  # bound, so the host-side event buffer grows for as long as the load runs --
  # the unbounded growth #398 fixed on the native path by passing max_iterations
  # (vllm 0.26+ self-stops the profiler after N worker steps). Its buffer fills
  # FASTER than the native one, too: it asks for with_memory and with_flops, and
  # leaves with_stack at its default, all of which the native adapter declines.
  # The script appends $EXTRA_<BE>_ARGS AFTER its own profiler args, so the bound
  # reaches vllm without editing the recipe's script. ProfilerConfig is strict
  # (extra=forbid) and an unknown key aborts the server, so emit only what this
  # build declares -- the same probe the native adapter runs. It runs under the
  # same recipe + accepted env + overlay PYTHONPATH as the final server: the
  # interpreter/module that answers has to be the exact one that will serve.
  # recipe extras UNDERNEATH GEAK's accepted flags: recipe first so GEAK's
  # EXTRA_SERVER_ARGS overrides any conflicting flag by coming later on the CLI
  # (framework argparse is last-wins); the profiler bound is appended LAST below
  # so it wins over both. This restores the "recipe = BASE layer" contract that
  # last-wins was silently violating (see the extraction above). The separator is
  # inserted only when BOTH layers are non-empty, so neither an empty recipe nor
  # empty GEAK flags leaves a stray leading/trailing space.
  local _extra_args="${EXTRA_SERVER_ARGS:-}" _bound=""
  if [ -n "$_recipe_extra" ]; then
    _extra_args="${_recipe_extra}${_extra_args:+ }${_extra_args}"
  fi
  if [ "${PROFILE:-0}" = "1" ] && [ "$backend_uc" = "VLLM" ]; then
    local _prof_fields
    _prof_fields="$(env -- \
      ${_recipe_env[@]+"${_recipe_env[@]}"} \
      ${_extra_env[@]+"${_extra_env[@]}"} \
      PYTHONPATH="${OVERLAY_PYTHONPATH:+$OVERLAY_PYTHONPATH:}${PYTHONPATH:-}" \
      python3 - <<'PY' 2>/dev/null
names = set()
try:
    import dataclasses
    from vllm.config import ProfilerConfig
    try:
        names |= {f.name for f in dataclasses.fields(ProfilerConfig)}
    except Exception:
        pass
    names |= set(getattr(ProfilerConfig, "model_fields", {}) or {})
    names |= set(getattr(ProfilerConfig, "__annotations__", {}) or {})
    print(" ".join(sorted(names)))
except Exception:
    pass
PY
)"
    case " $_prof_fields " in
      *" max_iterations "*) _bound="$_bound --profiler-config.max_iterations ${PROFILE_MAX_ITERS:-64}" ;;
    esac
    case " $_prof_fields " in
      *" delay_iterations "*) _bound="$_bound --profiler-config.delay_iterations ${PROFILE_DELAY_ITERS:-0}" ;;
    esac
    if [ -n "$_bound" ]; then
      echo ">>> magpie launcher: bounding the profiler buffer:$_bound"
      _extra_args="$_extra_args$_bound"
    else
      echo ">>> magpie launcher: this vllm build declares no profiler step bound;" \
           "the bench time window is the only cap on the trace." >&2
    fi
  fi

  # Map GEAK's env onto Magpie's server-phase env. Ordering IS the precedence
  # policy: recipe replay first, then the accepted env under test, then the
  # run-scoped names GEAK owns -- so a later layer knowingly overrides an
  # earlier one and nothing GEAK sets can be silently displaced by the recipe.
  # Overlay is prepended so the launch_server child imports the patched subtree
  # first. EXTRA_<BE>_ARGS carries the accepted extra flags; Magpie dedupes them
  # against its own DEFAULT_ARGS.
  #
  # GPU pinning has TWO shapes; picking the wrong one steals someone else's card:
  #
  #   * Outer ROCR already set (GEAK CI via run_local.sh: docker
  #     -e ROCR_VISIBLE_DEVICES=4,5,6,7 while /dev/dri is fully passed through).
  #     ROCr has already sliced the physical set and renumbered it 0..N-1, so
  #     $GPU is a LOGICAL index into that slice. Re-writing ROCR=$GPU would
  #     index the FULL physical set (logical 0..3 -> physical 0..3) and land on
  #     cards this job was never given. Keep the inherited ROCR and stack HIP
  #     on top (HIP masks after ROCr renumbering). Re-assert the outer ROCR
  #     AFTER $EXTRA_ENV so an accepted-env leak cannot clobber the mask.
  #
  #   * No outer ROCR (bare Magpie / whole-box Hyperloom). $GPU is PHYSICAL.
  #     Pin with ROCR alone and clear HIP/CUDA: Magpie's script derives the
  #     logical HIP range only while HIP is unset, and a HIP=PHYSICAL overlay
  #     on top of ROCR would index past the renumbered list (ROCR=2 -> device 0,
  #     HIP=2 -> OOB).
  #
  # The discriminator is simply whether ROCR_VISIBLE_DEVICES is already set in
  # THIS shell when the launcher runs — the same signal run_local.sh uses.
  local _outer_rocr="${ROCR_VISIBLE_DEVICES:-}"
  local -a _env_unset=() _gpu_env=()
  if [ -n "$_outer_rocr" ]; then
    echo ">>> magpie launcher: outer ROCR_VISIBLE_DEVICES=$_outer_rocr present;" \
         "pinning with HIP_VISIBLE_DEVICES=$GPU (logical) on top of inherited ROCR."
    _env_unset=(-u CUDA_VISIBLE_DEVICES)
    _gpu_env=(
      ROCR_VISIBLE_DEVICES="$_outer_rocr"
      HIP_VISIBLE_DEVICES="$GPU"
    )
  else
    echo ">>> magpie launcher: no outer ROCR mask; pinning with" \
         "ROCR_VISIBLE_DEVICES=$GPU (physical) and clearing HIP/CUDA."
    _env_unset=(-u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES)
    _gpu_env=(ROCR_VISIBLE_DEVICES="$GPU")
  fi
  # ${_recipe_env[@]+...}: callers under `set -u` (bench/CI drive scripts) may
  # leave these arrays empty; the `+` guard keeps empty expansion safe.
  # `--` terminates env's OWN option parsing: every following token is forced to
  # be an assignment-or-command operand, so no recipe/EXTRA_ENV value beginning
  # with `-` can be reparsed as an env option (belt-and-braces with the
  # allowlists above). `-u` unsets must precede `--`, hence the split.
  env "${_env_unset[@]}" -- \
    ${_recipe_env[@]+"${_recipe_env[@]}"} ${_extra_env[@]+"${_extra_env[@]}"} \
    "${_gpu_env[@]}" \
    PYTHONPATH="${OVERLAY_PYTHONPATH:+$OVERLAY_PYTHONPATH:}${PYTHONPATH:-}" \
    MAGPIE_RUN_PHASE=server \
    MAGPIE_SERVER_PID_FILE="$_pidfile" \
    MODEL="$MODEL" \
    TP="$TP" \
    PORT="$PORT" \
    RESULT_DIR="$_out_dir" \
    SERVER_LOG="$LOG" \
    PROFILE="${PROFILE:-0}" \
    ${MAX_MODEL_LEN:+MAX_MODEL_LEN="$MAX_MODEL_LEN"} \
    ${PROFILE_DIR:+"${_prof_var}=$PROFILE_DIR"} \
    "${_args_var}=${_extra_args}" \
    bash "$script" >> "$_launchlog" 2>&1
  local rc=$?

  if [ "$rc" -ne 0 ]; then
    # Both logs: a script that died before starting the server (missing env,
    # failed download) left nothing in $LOG at all.
    echo "!!! magpie launcher: server-phase script exited $rc. Last launch log:" >&2
    tail -n 40 "$_launchlog" 2>/dev/null || true
    echo "!!! magpie launcher: last server log:" >&2
    tail -n 40 "$LOG" 2>/dev/null || true
    return 2
  fi
  if [ -f "$_pidfile" ]; then
    SERVER_PID="$(cat "$_pidfile" 2>/dev/null)"
  fi
  if [ -z "${SERVER_PID:-}" ]; then
    echo "!!! magpie launcher: no server pid in $_pidfile (server may not have started)." >&2
    tail -n 40 "$_launchlog" 2>/dev/null || true
    tail -n 40 "$LOG" 2>/dev/null || true
    return 2
  fi
  # This pid came from an EXTERNAL script's pid file, so we cannot assume it leads its
  # own group, and a stale file can name a pid that now belongs to someone else
  # entirely (worst case: the caller's orchestrator). Either case must DISABLE group
  # teardown here, at launch, rather than be discovered by a kill that already fired.
  local _mp_pgid _mp_args
  _mp_pgid="$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')"
  _mp_args="$(ps -o args= -p "$SERVER_PID" 2>/dev/null)"
  if [ "$_mp_pgid" != "$SERVER_PID" ]; then
    SERVER_GROUP_UNVERIFIED=1
    echo "!!! magpie launcher: pid $SERVER_PID does not lead its own group (pgid=${_mp_pgid:-?});" \
         "group teardown disabled for this launch." >&2
  fi
  # The "is this actually our server?" test is DERIVED from this run's own config
  # ($BACKEND, $PORT) — never a hard-coded backend list, which would silently
  # mis-judge every future Magpie backend the same way it mis-judges a stale pid.
  case "$_mp_args" in
    *"$BACKEND"*|*"$PORT"*) : ;;
    *)
      SERVER_GROUP_UNVERIFIED=1
      echo "!!! magpie launcher: pid $SERVER_PID matches neither BACKEND='$BACKEND' nor" \
           "PORT='$PORT' (args='$(printf '%s' "$_mp_args" | cut -c1-120)') — stale pid file?" \
           "group teardown disabled for this launch." >&2 ;;
  esac
  export SERVER_GROUP_UNVERIFIED
  echo ">>> magpie launcher: $BACKEND server up (pid $SERVER_PID, pgid=${_mp_pgid:-?}) via $(basename "$script")."
}
