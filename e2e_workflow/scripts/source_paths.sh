# Sourced by bench_e2e.sh; complete accepted roots survive candidate overlay changes.
if [ -f "$HERE/source_request.json" ]; then
  GEAK_SOURCE_REQUEST="$HERE/source_request.json"
fi
if [ -n "${GEAK_SOURCE_REQUEST:-}" ]; then
  _source_validator="$(dirname "$GEAK_SOURCE_REQUEST")/source_materialization.py"
  OVERLAY_PYTHONPATH="$(python "$_source_validator" --compose-pythonpath \
    "$GEAK_SOURCE_REQUEST" "${OVERLAY_PYTHONPATH:-}" "" "" "")" || return 5
  # Importing a sealed tree must not introduce unlisted bytecode files.
  export GEAK_SOURCE_REQUEST PYTHONDONTWRITEBYTECODE=1
fi
