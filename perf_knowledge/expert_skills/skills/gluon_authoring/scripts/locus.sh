#!/usr/bin/env bash
# locus.sh - run profilers WHERE THE KERNEL runs (sourced by capture.sh /
# rocprof_compute_probe.sh / profile_kernel.sh).
#
# Root cause this closes: the candidate JIT-dlopens inside a container while the profiler
# script runs host-side -> the host cannot see the in-container kernel -> every profiler
# silently degrades (sol_pmc_no_kernel_rows / analyze-blind / ATT code:null). See
# ../references/phases/profile.md ## Execution locus and ../references/failure-triage.md
# ## Profiler layer unavailable. This is a FIXABLE mis-config, NOT a blind mode.
#
# Contract (generic; nothing container-specific is baked): the container id comes from the
# env var TILE_KERNEL_CONTAINER (resolve_context.py records it as context.env.kernel_container;
# run_round.sh exports it). When it is set + docker/podman is present + the container is alive,
# profiler collection is wrapped in `docker exec` and artifacts are copied out; otherwise every
# helper is an exact host-side passthrough (unchanged behavior for the no-container case).
#
# API:
#   locus_active            -> rc 0 if a live kernel container is configured, else rc 1
#   locus_run <cmd...>      -> run <cmd> in the kernel container (docker exec) or host-side.
#                              Forwards host env into the container via -e (see LOCUS_ENV below).
#   locus_fetch <cpath> <hpath> -> copy an artifact OUT of the container (noop on host / bind mount)
#   locus_push  <hpath> <cpath> -> copy a file INTO the container (noop host-side / bind mount)
#   locus_preflight         -> verify container alive + rocprofv3/rocprof-compute present in it;
#                              on failure echoes a structured probe line + sets LOCUS_DEGRADE_CAUSE
#
# Env forwarding (the silent-wrong-locus fix): a bare `docker exec` inherits NONE of the host env,
# so HIP_VISIBLE_DEVICES (the app runs on GPU 0), TRITON_CACHE_DIR (the shared-cache bridge), or a
# caller's variant-source selector are all dropped -> the profiler measures the wrong GPU / the stock
# kernel while the log says "variant". locus_run now emits `-e NAME=value` for each NAME in the
# allowlist LOCUS_ENV that is SET in the host env. A structurally-universal default list is built in;
# a caller adds its own (e.g. a variant-root var) with:  LOCUS_ENV="$LOCUS_ENV MY_SRC_ROOT"
# Only NAMES are listed (never values), and only names that are actually set are forwarded, so this
# bakes in nothing container- or kernel-specific.
#
# Smoke: bash locus.sh --selftest  (no real docker needed)

# Structurally-universal env names to forward into the locus. NOT kernel/repo-specific: the GPU
# selector(s), Triton's cache + recompile controls, and the import path. Callers append their own
# names to LOCUS_ENV; they are only forwarded if set in the host env.
: "${LOCUS_ENV:=HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES TRITON_CACHE_DIR TRITON_ALWAYS_COMPILE PYTHONPATH}"

_locus_env_args() {
  # Print `-e NAME=value` tokens (one pair per arg) for every name in LOCUS_ENV that is set in the
  # environment. Names may be space- or comma-separated. Unset names are skipped; an empty LOCUS_ENV
  # emits nothing. Pure function of the environment -> unit-testable with no docker.
  local names name val
  names="${LOCUS_ENV//,/ }"
  for name in $names; do
    if [ -n "${!name+x}" ]; then
      val="${!name}"
      printf '%s\0%s\0' "-e" "$name=$val"
    fi
  done
}

# shellcheck disable=SC2120
_locus_engine() {
  # echo the container engine (docker|podman) if present, else nothing
  if command -v docker >/dev/null 2>&1; then echo docker
  elif command -v podman >/dev/null 2>&1; then echo podman
  fi
}

locus_active() {
  local cid="${TILE_KERNEL_CONTAINER:-}"
  [ -n "$cid" ] || return 1
  local eng; eng="$(_locus_engine)"
  [ -n "$eng" ] || return 1
  # container must be alive (running)
  "$eng" inspect -f '{{.State.Running}}' "$cid" 2>/dev/null | grep -q true
}

locus_run() {
  # run a command in the kernel's locus. Host-side passthrough when no live container.
  # When wrapping in `docker exec`, forward the LOCUS_ENV allowlist via -e so the app sees the
  # host's GPU selector / cache dir / import path / variant selector (see header). Host-side the
  # env is already inherited, so passthrough is unchanged.
  if locus_active; then
    local eng; eng="$(_locus_engine)"
    local -a eargs=()
    local tok
    while IFS= read -r -d '' tok; do eargs+=("$tok"); done < <(_locus_env_args)
    "$eng" exec ${TILE_KERNEL_CONTAINER_WORKDIR:+-w "$TILE_KERNEL_CONTAINER_WORKDIR"} \
      "${eargs[@]}" "$TILE_KERNEL_CONTAINER" "$@"
  else
    "$@"
  fi
}

locus_fetch() {
  # copy <container_path> -> <host_dest>. Noop when host-side. Tolerant when the working
  # root is bind-mounted (the file is already visible) -> ignore a cp failure.
  local cpath="${1:?locus_fetch <container_path> <host_dest>}"
  local hdest="${2:?locus_fetch <container_path> <host_dest>}"
  if locus_active; then
    local eng; eng="$(_locus_engine)"
    "$eng" cp "$TILE_KERNEL_CONTAINER:$cpath" "$hdest" 2>/dev/null || true
  fi
  return 0
}

locus_push() {
  # copy <host_path> -> <container_path> (docker cp IN). Needed to run a host-side helper script
  # (e.g. dump_ir.sh) INSIDE the locus where triton is importable. Noop host-side; tolerant when the
  # working root is bind-mounted (the file is already visible in the container) -> ignore a failure.
  local hpath="${1:?locus_push <host_path> <container_path>}"
  local cpath="${2:?locus_push <host_path> <container_path>}"
  if locus_active; then
    local eng; eng="$(_locus_engine)"
    "$eng" cp "$hpath" "$TILE_KERNEL_CONTAINER:$cpath" 2>/dev/null || true
  fi
  return 0
}

locus_have() {
  # rc 0 if <tool> is runnable IN THE LOCUS. A host-side `command -v rocprofv3` answers a question
  # nobody asked when the kernel runs in a container: it reports the tool missing on a box where the
  # profiler is one `docker exec` away, so the round degrades to PMC-blind with a working profiler
  # sitting right there -- and "PMC-blind" is then read as a property of the kernel. Host-side (no
  # live container) this is exactly `command -v`, so the no-container path is unchanged.
  local tool="${1:?locus_have <tool>}"
  if locus_active; then
    locus_run bash -lc 'command -v "$1" >/dev/null 2>&1' _ "$tool"
  else
    command -v "$tool" >/dev/null 2>&1
  fi
}

locus_preflight() {
  # returns 0 if in-locus profiling is usable (or no container configured = host is the locus);
  # returns 1 and sets LOCUS_DEGRADE_CAUSE + echoes a structured probe line when a container is
  # configured but the profiler cannot run in it (the fix-or-escalate signal).
  LOCUS_DEGRADE_CAUSE=""
  local cid="${TILE_KERNEL_CONTAINER:-}"
  [ -n "$cid" ] && locus_active || { [ -z "$cid" ] && return 0; }
  if [ -n "$cid" ] && ! locus_active; then
    LOCUS_DEGRADE_CAUSE="profiler_locus_mismatch"
    echo "[locus] PROBE-FAIL: TILE_KERNEL_CONTAINER='$cid' set but container not alive / no engine" >&2
    return 1
  fi
  # container alive: confirm at least one profiler tool is present in it
  if ! locus_run bash -lc 'command -v rocprofv3 >/dev/null 2>&1 || command -v rocprof-compute >/dev/null 2>&1'; then
    LOCUS_DEGRADE_CAUSE="profiler_locus_mismatch"
    echo "[locus] PROBE-FAIL: no rocprofv3/rocprof-compute inside container '$cid'" >&2
    return 1
  fi
  return 0
}

_locus_selftest() {
  local fail=0
  # (a) no container -> host passthrough: locus_active false, locus_run runs directly.
  unset TILE_KERNEL_CONTAINER
  if locus_active; then echo "FAIL: locus_active true with no container"; fail=1; fi
  out="$(locus_run echo hello)"; [ "$out" = "hello" ] || { echo "FAIL: passthrough run"; fail=1; }
  locus_fetch /x /y || { echo "FAIL: fetch noop nonzero"; fail=1; }
  locus_preflight || { echo "FAIL: preflight should pass with no container (host is locus)"; fail=1; }
  # (b) container set but engine/container not alive -> preflight records the fixable cause.
  TILE_KERNEL_CONTAINER="tile_locus_no_such_container_$$"
  if locus_active; then echo "FAIL: locus_active true for a bogus container"; fail=1; fi
  out="$(locus_run echo hi 2>/dev/null)"; [ "$out" = "hi" ] || { echo "FAIL: run passthrough when container dead"; fail=1; }
  if locus_preflight 2>/dev/null; then echo "FAIL: preflight should fail for a dead container"; fail=1; fi
  [ "$LOCUS_DEGRADE_CAUSE" = "profiler_locus_mismatch" ] || { echo "FAIL: cause not set ($LOCUS_DEGRADE_CAUSE)"; fail=1; }
  unset TILE_KERNEL_CONTAINER
  # (b2) locus_have host-side == command -v, including for a name with no match.
  locus_have bash || { echo "FAIL: locus_have should find bash host-side"; fail=1; }
  if locus_have tile_no_such_tool_$$; then echo "FAIL: locus_have found a nonexistent tool"; fail=1; fi
  # (c) _locus_env_args: pure-argv env forwarding. No docker needed.
  # count the "-e" tokens emitted (null-delimited pairs -> one "-e" per forwarded name).
  _count_e() { local n=0 tok; while IFS= read -r -d '' tok; do [ "$tok" = "-e" ] && n=$((n+1)); done < <(_locus_env_args); echo "$n"; }
  ( export LOCUS_ENV="FOO_A FOO_B" FOO_A=1 FOO_B=2
    [ "$(_count_e)" = 2 ] ) || { echo "FAIL: env_args should forward 2 set names"; fail=1; }
  ( export LOCUS_ENV="FOO_A FOO_MISSING" FOO_A=1; unset FOO_MISSING
    [ "$(_count_e)" = 1 ] ) || { echo "FAIL: env_args should skip an unset name"; fail=1; }
  ( export LOCUS_ENV=""
    [ "$(_count_e)" = 0 ] ) || { echo "FAIL: empty LOCUS_ENV should forward nothing"; fail=1; }
  ( export LOCUS_ENV="FOO_A,FOO_B" FOO_A=1 FOO_B=2   # comma-separated also accepted
    [ "$(_count_e)" = 2 ] ) || { echo "FAIL: env_args should accept comma separators"; fail=1; }
  # the value token must carry an embedded space intact (NAME=val with spaces)
  ( export LOCUS_ENV="FOO_SP" FOO_SP="a b c"
    got=""; while IFS= read -r -d '' tok; do got="$tok"; done < <(_locus_env_args)
    [ "$got" = "FOO_SP=a b c" ] ) || { echo "FAIL: env_args must preserve spaces in a value"; fail=1; }
  if [ "$fail" = 0 ]; then echo "LOCUS SELFTEST PASS"; else echo "LOCUS SELFTEST FAIL"; fi
  return "$fail"
}

# Only run the smoke when invoked directly (not when sourced).
if [ "${BASH_SOURCE[0]}" = "${0}" ] && [ "${1:-}" = "--selftest" ]; then
  _locus_selftest
fi
