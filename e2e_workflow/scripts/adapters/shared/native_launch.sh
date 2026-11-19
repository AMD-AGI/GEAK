# Launch helpers shared by the native serving adapters (vllm.sh, sglang.sh, atom.sh). Sourced, not executed.
# Lives in a subdirectory so bench_e2e.sh does not list it as a backend.

_NATIVE_LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# _native_server_argv WHO ARGS -> fills the global array _NATIVE_SERVER_ARGV with ARGS split by shell
# quoting rules (split_args.py, as atom.sh and the magpie launcher do). An unquoted $EXTRA_SERVER_ARGS
# keeps quotes as literal characters and glob-expands `*`/`[..]`, so a JSON- or list-valued flag
# (`--speculative-config '{"method":"mtp"}'`) reached the server quoted or rewritten. Fails closed
# (returns 2) when the helper is missing or cannot split the string: a server started from mangled
# flags may come up healthy with a different configuration, so there is no word-splitting fallback.
_native_server_argv() {
  local _who="$1" _helper="$_NATIVE_LAUNCH_DIR/../split_args.py" _out
  _NATIVE_SERVER_ARGV=()
  [ -n "${2//[[:space:]]/}" ] || return 0
  if [ ! -f "$_helper" ]; then
    echo "!!! $_who adapter: $_helper is missing; refusing to launch with unsplit EXTRA_SERVER_ARGS." >&2
    return 2
  fi
  _out="$(mktemp)" || return 2
  if ! python3 "$_helper" "$2" > "$_out"; then
    rm -f "$_out"
    echo "!!! $_who adapter: EXTRA_SERVER_ARGS does not split (see above); refusing to launch." >&2
    return 2
  fi
  mapfile -d '' -t _NATIVE_SERVER_ARGV < "$_out"
  rm -f "$_out"
  if [ "${#_NATIVE_SERVER_ARGV[@]}" -eq 0 ]; then
    echo "!!! $_who adapter: EXTRA_SERVER_ARGS split to no arguments; refusing to launch." >&2
    return 2
  fi
}

# _native_gpu_env GPU [EXTRA_ENV] -> fills the global array _NATIVE_GPU_ENV with the device-visibility
# assignments for a server pinned to GPU. HIP/CUDA alone leave every GPU of the box visible to ROCr,
# and a server whose ROCr saw all of them hit a GPU memory-access fault. So ROCR_VISIBLE_DEVICES is
# narrowed to the same cards and HIP/CUDA renumber them 0..n-1.
#
# GPU keeps the meaning it had under the HIP/CUDA-only pin: physical ids when no ROCr mask is in
# effect, positions in the mask when one is. The mask is the shell's ROCR_VISIBLE_DEVICES, or the one
# EXTRA_ENV sets (EXTRA_ENV is applied after the shell env). A GPU this cannot map -- a non-numeric
# id, a repeat, a position past the mask -- keeps the HIP/CUDA-only pin unchanged.
_native_gpu_env() {
  local _gpu="${1//[[:space:]]/}" _mask="${ROCR_VISIBLE_DEVICES:-}" _t _i _rocr="" _seq="" _seen="," _n=0
  local -a _toks=() _ids=() _slots=()
  _NATIVE_GPU_ENV=(HIP_VISIBLE_DEVICES="$1" CUDA_VISIBLE_DEVICES="$1")
  read -r -a _toks <<< "${2:-}"
  for _t in ${_toks[@]+"${_toks[@]}"}; do
    case "$_t" in ROCR_VISIBLE_DEVICES=*) _mask="${_t#ROCR_VISIBLE_DEVICES=}" ;; esac
  done
  [[ "$_gpu" =~ ^[0-9]+(,[0-9]+)*$ ]] || return 0
  _mask="${_mask//[[:space:]]/}"
  IFS=',' read -r -a _ids <<< "$_gpu"
  [ -z "$_mask" ] || IFS=',' read -r -a _slots <<< "$_mask"
  for _i in "${_ids[@]}"; do
    _i=$((10#$_i))
    case "$_seen" in *",$_i,"*) return 0 ;; esac
    _seen+="$_i,"
    if [ -n "$_mask" ]; then
      [ "$_i" -lt "${#_slots[@]}" ] && [ -n "${_slots[$_i]}" ] || return 0
      _rocr+="${_rocr:+,}${_slots[$_i]}"
    else
      _rocr+="${_rocr:+,}$_i"
    fi
    _seq+="${_seq:+,}$_n"
    _n=$((_n + 1))
  done
  _NATIVE_GPU_ENV=(ROCR_VISIBLE_DEVICES="$_rocr" HIP_VISIBLE_DEVICES="$_seq" CUDA_VISIBLE_DEVICES="$_seq")
}
