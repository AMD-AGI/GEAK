#!/usr/bin/env bash
# Shared config + helpers for the GEAK_v4 CI scripts.
# Sourced by the other ci/*.sh scripts; not meant to be run directly.
#
# Paths are DERIVED from this file's location, so the tree just needs to look like:
#   <workspace>/GEAK/ci/*.sh   (this repo)
#   <workspace>/InferenceX     (cloned separately)
#   <workspace>/geak_runtime   (per-model handoff/recipe/tracelens priors)
# Any of these can be overridden by exporting the matching env var.

CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # <ws>/GEAK/ci
# All tunables (timeouts / caps / knobs) live in ONE file: ci/config.sh.
# shellcheck source=/dev/null
[ -f "$CI_DIR/config.sh" ] && source "$CI_DIR/config.sh"
GEAK_ROOT="${GEAK_ROOT:-$(dirname "$CI_DIR")}"                   # <ws>/GEAK
WS="${WS:-$(dirname "$GEAK_ROOT")}"                              # <ws>
INFERENCEX_PATH="${INFERENCEX_PATH:-$WS/InferenceX}"
HF_LOGS="${HF_LOGS:-$WS/geak_runtime}"
CLAUDE_SETUP="${CLAUDE_SETUP:-$CI_DIR/preflight/claude_setup.sh}"
MODELS_TSV="${MODELS_TSV:-$CI_DIR/models.tsv}"
# Repo-tracked image map (ci/docker_default.json). Override with DOCKER_DEFAULT=<path>.
DOCKER_DEFAULT="${DOCKER_DEFAULT:-$CI_DIR/docker_default.json}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit "${2:-1}"; }
new_ts() { date -u +%Y%m%dT%H%M%SZ; }

# ---------------------------------------------------------------------------
# Model enrollment registry (models.tsv) + handoff-derived properties
# ---------------------------------------------------------------------------
# models.tsv is the CI *enrollment* list: <model_key>\t<hf_repo>\t<tier>.
# It says WHICH models CI may run and where to fetch their weights.
# The per-run *properties* (framework, tp/GPU count) are NOT duplicated here —
# they are read from each model's handoff.json, the single source of truth.

# -- tsv registry (skip comments/blank lines) --
_tsv_row() {
  awk -F'\t' -v k="$1" '!/^[[:space:]]*#/ && NF && $1==k {print; f=1} END{if(!f) exit 3}' "$MODELS_TSV"
}
model_hf_repo() { _tsv_row "$1" | awk -F'\t' '{print $2}'; }
model_tier()    { _tsv_row "$1" | awk -F'\t' '{print $3}'; }
is_enrolled()   { _tsv_row "$1" >/dev/null 2>&1; }
# All enrolled models (the full l1-ci-full matrix); smoke-tier subset for L1 smoke.
enrolled_models() { awk -F'\t' '!/^[[:space:]]*#/ && NF {print $1}' "$MODELS_TSV"; }
smoke_models()    { awk -F'\t' '!/^[[:space:]]*#/ && NF && $3=="smoke" {print $1}' "$MODELS_TSV"; }

# -- handoff-derived properties (framework + tp = GPU count) --
_handoff_path() { echo "$HF_LOGS/$1/handoff.json"; }
_handoff_get() {  # _handoff_get <model_key> <json_key> <default>
  local h; h="$(_handoff_path "$1")"
  [ -f "$h" ] || { echo "${3:-}"; return; }
  python3 -c 'import json,sys
d=json.load(open(sys.argv[1]))
v=d.get(sys.argv[2], sys.argv[3])
print(v if v not in (None,"") else sys.argv[3])' "$h" "$2" "${3:-}" 2>/dev/null || echo "${3:-}"
}
model_framework() { _handoff_get "$1" framework ""; }
model_tp()        { _handoff_get "$1" tp 1; }        # tensor-parallel = GPUs to allocate

# ---------------------------------------------------------------------------
# Weights resolution (compute-node paths)
# ---------------------------------------------------------------------------
# The handoff's model_path (/wekafs/...) is not mounted here. Weights are picked
# up from a per-model_key CATALOG dir ($HF_MODELS_DIR): each entry is a directory
# OR a symlink into shared NFS (e.g. /home/ethany/hf_models/Qwen-Qwen3-8B ->
# /shared_nfs/huggingface_models/Qwen/Qwen3-8B). Because entries may be symlinks
# into NFS, run_local.sh also bind-mounts $WEIGHTS_EXTRA_MOUNTS so they resolve
# inside the container. Missing models are downloaded here (keyed by model_key).
HF_MODELS_DIR="${HF_MODELS_DIR:-/home/ethany/hf_models}"   # catalog: <model_key> -> weights
WEIGHTS_CACHE="${WEIGHTS_CACHE:-$HF_MODELS_DIR}"           # where downloads land

# Pure resolver (no download): the catalog entry if populated, else the download target.
model_weights() {
  local mk="$1" cand="${HF_MODELS_DIR}/$1"
  if [ -n "${MODEL_PATH:-}" ]; then echo "$MODEL_PATH"; return; fi
  if [ -d "$cand" ] && [ -n "$(ls -A "$cand"/ 2>/dev/null)" ]; then echo "$cand"; return; fi
  echo "$WEIGHTS_CACHE/$mk"   # not present yet — stage_weights() will populate it
}

# True if the catalog already has (non-empty) weights for a model_key — i.e. it can
# run WITHOUT a HuggingFace download. Used to pick the "probe" set (currently the 3
# models whose weights are symlinked in from NFS).
weights_present() {
  local w; w="$(model_weights "$1")"
  [ -d "$w" ] && [ -n "$(ls -A "$w"/ 2>/dev/null)" ]
}
# Enrolled models that already have local weights (the probe/full-verify-available set).
probe_models() { local m; for m in $(enrolled_models); do weights_present "$m" && echo "$m"; done; }

# Ensure weights exist on THIS (compute) node; echo the final path on stdout.
# Downloads from HuggingFace only if absent AND models.tsv gives an hf_repo.
stage_weights() {
  local mk="$1" repo dest
  dest="$(model_weights "$mk")"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then echo "$dest"; return; fi
  # Safety net: CI runs on models with LOCAL weights only. Never auto-download
  # (a stray/misconfigured enrollment shouldn't silently pull 100+GB from HF and
  # burn a GPU allocation). Opt in explicitly with GEAK_ALLOW_DOWNLOAD=1.
  [ "${GEAK_ALLOW_DOWNLOAD:-0}" = "1" ] \
    || die "no local weights for $mk at $dest — refusing to download (set GEAK_ALLOW_DOWNLOAD=1 to allow, or pre-stage under $HF_MODELS_DIR/$mk)"
  repo="$(model_hf_repo "$mk")"
  [ -n "$repo" ] && [ "$repo" != "-" ] \
    || die "no weights for $mk at $dest and no hf_repo in $MODELS_TSV to download (pre-stage them under $WEIGHTS_CACHE/$mk)"
  dest="$WEIGHTS_CACHE/$mk"
  log "staging weights: hf download $repo -> $dest"
  mkdir -p "$dest"
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo" --local-dir "$dest" >&2 || die "hf download failed for $repo"
  else
    python3 -m huggingface_hub.cli.hf download "$repo" --local-dir "$dest" >&2 \
      || die "huggingface_hub download failed for $repo (pip install huggingface_hub)"
  fi
  echo "$dest"
}

# ---------------------------------------------------------------------------
# SPUR / SLURM submission
# ---------------------------------------------------------------------------
# NB: on this cluster there is ONE partition ('amd-spur'); the account/QoS is
# what actually gates scheduling (GPUs are MI300x/MI355x). All the SPUR knobs
# (partition, account candidates/fallback, headroom, probe timings, ...) live in
# ci/config.sh — see there to change them. pick_account() below probes those
# candidates per model and picks one that can place the job now.

# seconds -> SLURM time "H:MM:SS"
fmt_slurm_time() {
  local s="$1"
  printf '%d:%02d:%02d' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

# Quick allocation test for one account/qos: submit a tiny 1-node/<gpus>-GPU
# probe and watch it. The GPU count should match the heaviest real job (tp), so
# the test reflects the actual requirement (a QoS may place a 1-GPU probe yet
# reject an 8-GPU job). Echoes "up" if it reaches RUNNING (or finishes) within
# SPUR_PROBE_WAIT_S -> the QoS can place that shape now; else "full". Cleans up.
_probe_account() {
  local acct="$1" qos="$2" gpus="${3:-1}" out jid state deadline now
  command -v sbatch >/dev/null 2>&1 || { echo up; return; }   # no scheduler here -> don't block
  out="$(sbatch --parsable -A "$acct" -p "$SPUR_PARTITION" --qos "$qos" \
        -J "geak_probe_${acct}" -N1 -G"$gpus" -c1 -t 00:05:00 \
        -o /dev/null -e /dev/null --wrap 'sleep 3' 2>/dev/null)" || { echo full; return; }
  jid="$(grep -oE '[0-9]+' <<<"$out" | tail -1)"
  [ -n "$jid" ] || { echo full; return; }
  now="$(date +%s)"; deadline=$(( now + SPUR_PROBE_WAIT_S ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    state="$(squeue -j "$jid" -h -o '%T' 2>/dev/null | head -1)"
    case "$state" in
      ""|COMPLETED|COMPLETING|RUNNING) scancel "$jid" 2>/dev/null || true; echo up; return ;;
    esac
    sleep "$SPUR_PROBE_POLL_S"
  done
  scancel "$jid" 2>/dev/null || true
  echo full
}

# Choose an account/qos that can place a 1-node/<gpus>-GPU job now. Echoes
# "<account> <qos>". Tries each SPUR_ACCOUNT_CANDIDATES entry in order; if none
# can place the job now, returns SPUR_ACCOUNT_FALLBACK (pend there). Pass the
# heaviest model tp as $1 so the probe matches the real GPU footprint (default 1).
pick_account() {
  local gpus="${1:-1}" pair acct qos res
  for pair in $SPUR_ACCOUNT_CANDIDATES; do
    acct="${pair%%:*}"; qos="${pair##*:}"
    res="$(_probe_account "$acct" "$qos" "$gpus")"
    log "account probe: $acct/$qos (${gpus}xGPU) -> $res"
    [ "$res" = up ] && { echo "$acct $qos"; return 0; }
  done
  pair="$SPUR_ACCOUNT_FALLBACK"; acct="${pair%%:*}"; qos="${pair##*:}"
  log "no candidate can place ${gpus}xGPU now; falling back to $acct/$qos (jobs will pend)"
  echo "$acct $qos"
}

# Detect the GPU arch bucket used to pick a docker_default.json entry.
# Returns MI355 (gfx950 / MI35x) or MI300 (gfx942/gfx90a / MI30x). Honors a
# GEAK_GPU_ARCH override; auto-detects via rocminfo when present; defaults to
# MI355 (this SPUR cluster's nodes are gfx950). resolve_image runs on the
# compute-node host, before the container starts.
detect_gpu_arch() {
  if [ -n "${GEAK_GPU_ARCH:-}" ]; then echo "$GEAK_GPU_ARCH"; return; fi
  local rocminfo_bin gfx
  rocminfo_bin="$(command -v rocminfo 2>/dev/null || true)"
  [ -z "$rocminfo_bin" ] && [ -x /opt/rocm/bin/rocminfo ] && rocminfo_bin=/opt/rocm/bin/rocminfo
  if [ -n "$rocminfo_bin" ]; then
    gfx="$("$rocminfo_bin" 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1)"
  fi
  case "$gfx" in
    gfx950)          echo MI355 ;;
    gfx942|gfx90a)   echo MI300 ;;
    *)               echo "$GEAK_GPU_ARCH_DEFAULT" ;;   # config.sh default (this cluster = gfx950)
  esac
}

# --- pick container image for a framework (docker_default.json) ---
# docker_default.json is a nested dict:  { "<framework>": { "<arch>": "<image>" } }
# Prefer image[framework][arch]; else image[framework].default; else the first
# arch entry for the framework. Override entirely with: IMAGE=<repo:tag>.
resolve_image() {
  local fw="$1"
  if [ -n "${IMAGE:-}" ]; then echo "$IMAGE"; return; fi
  local arch img
  arch="$(detect_gpu_arch)"
  img=$(python3 - "$DOCKER_DEFAULT" "$fw" "$arch" <<'PY'
import json, sys
path, fw, arch = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(path))
except Exception:
    sys.exit(0)
m = d.get(fw)
if not isinstance(m, dict):
    sys.exit(0)
img = m.get(arch) or m.get("default")
if not img:  # last resort: first string image listed for this framework
    img = next((v for v in m.values() if isinstance(v, str)), "")
print(img or "")
PY
)
  [ -n "$img" ] || die "no image for framework=$fw (arch=$arch) in $DOCKER_DEFAULT (or pass IMAGE=)"
  echo "$img"
}
