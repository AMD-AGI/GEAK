# Shared EXTRA_ENV decoder for native and Magpie launchers. Sourced, not executed.
_GEAK_EXTRA_ENV_PARSER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/extra_env.py"

geak_read_extra_env() {
  if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
    echo "GEAK environment transport requires Bash >= 4.4 (mapfile -d)" >&2
    return 2
  fi
  # A checked temporary file keeps parser failure distinct from an empty result.
  # Process substitution would discard that failure and launch with missing env.
  local _geak_env_file _geak_env_rc
  [ -n "$2" ] || return 0
  _geak_env_file="$(mktemp)" || return 1
  if python3 "$_GEAK_EXTRA_ENV_PARSER" "$2" "${3:-}" > "$_geak_env_file"; then
    mapfile -d '' -t "$1" < "$_geak_env_file"
    _geak_env_rc=$?
  else
    _geak_env_rc=$?
  fi
  rm -f -- "$_geak_env_file"
  return "$_geak_env_rc"
}

geak_read_unset_env() {
  geak_read_extra_env "$1" "${2:-}" --unset
}

geak_env_is_unset() {
  local _geak_name="$1" _geak_operand
  shift
  for _geak_operand in "$@"; do
    [ "$_geak_operand" != "$_geak_name" ] || return 0
  done
  return 1
}
