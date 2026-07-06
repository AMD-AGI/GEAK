#!/usr/bin/env bash
# Shared config + helpers for the GEAK_v4 CI scripts.
# Sourced by the other ci/*.sh scripts; not meant to be run directly.
#
# Paths are DERIVED from this file's location, so the tree just needs to look like:
#   <workspace>/GEAK/ci/*.sh   (this repo)
#   <workspace>/InferenceX     (cloned separately)
#   <workspace>/geak_runtime   (per-model handoff/recipe/tracelens priors + docker_select.log)
# Any of these can be overridden by exporting the matching env var.

CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # <ws>/GEAK/ci
GEAK_ROOT="${GEAK_ROOT:-$(dirname "$CI_DIR")}"                   # <ws>/GEAK
WS="${WS:-$(dirname "$GEAK_ROOT")}"                              # <ws>
INFERENCEX_PATH="${INFERENCEX_PATH:-$WS/InferenceX}"
HF_LOGS="${HF_LOGS:-$WS/geak_runtime}"
CLAUDE_SETUP="${CLAUDE_SETUP:-$CI_DIR/claude_setup.sh}"
MODELS_TSV="${MODELS_TSV:-$CI_DIR/models.tsv}"
DOCKER_SELECT="${DOCKER_SELECT:-$HF_LOGS/docker_select.log}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit "${2:-1}"; }
new_ts() { date -u +%Y%m%dT%H%M%SZ; }

# --- model registry (models.tsv: <model_key>\t<weights_dir>\t<framework>) ---
_model_row() { awk -F'\t' -v k="$1" '$1==k{print; f=1} END{if(!f) exit 3}' "$MODELS_TSV"; }
model_weights()   { _model_row "$1" | awk -F'\t' '{print $2}'; }
model_framework() { _model_row "$1" | awk -F'\t' '{print $3}'; }

# --- pick container image for a framework (docker_select.log) ---
# docker_select.log lines look like:  "<framework> (<arch...>): <image>"
# We treat this box as MI300, so prefer the MI300 line; the vllm image serves both.
# Override entirely with: IMAGE=<repo:tag> run_local.sh ...
resolve_image() {
  local fw="$1"
  if [ -n "${IMAGE:-}" ]; then echo "$IMAGE"; return; fi
  local img
  img=$(awk -F': ' -v f="$fw" '
    { name=$1; sub(/ *\(.*/,"",name) }
    name==f {
      if (index($1,"MI300")) { print $2; exit }   # prefer MI300 line
      if (!c) c=$2                                 # else remember first match
    }
    END { if (c) print c }
  ' "$DOCKER_SELECT")
  [ -n "$img" ] || die "no image for framework=$fw in $DOCKER_SELECT (or pass IMAGE=)"
  echo "$img"
}
