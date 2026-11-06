#!/usr/bin/env bash
# Runtime environment contract for shared AMD containers.
#
# Only whoever owns the shared container may call `prepare`. GEAK roles (e.g. the deep_engineer)
# receive the emitted JSON and use `verify` / `export`; they never repair the shared container.
#
# OPTIONAL. In GEAK, roles run inside GEAK's own workspace under kernel_workflow/scripts/gpu_lock.sh
# and get their environment from GEAK, so this is not called. This contract (and the container locus
# it feeds, scripts/locus.sh) is only for a kernel that lives in a separate container. `prepare` writes per-run HOME/XDG/TMP/Triton/hw caches under
# <work>/.tile-runtime/ -- GEAK workspace copies exclude `.tile-runtime/` (materialize_workspace.sh)
# and candidate patches ignore it (roles/director.md .gitignore), so these caches never travel with a
# candidate.
set -euo pipefail

SCHEMA="tile.runtime-env/1"
MODE="${1:-}"
shift || true
WORK=""; CONTAINER_WORK=""; RUN_ID=""; VARIANT="default"; OUT=""; CONTRACT=""
ALLOWED_ROOTS=()

usage() {
  cat <<'EOF'
Usage:
  runtime_env.sh prepare --work HOST_ROOT --container-work CONTAINER_ROOT --run-id ID [--variant NAME] [--allowed-host-root ROOT] --out runtime-env.json
  runtime_env.sh verify --contract runtime-env.json
  runtime_env.sh export --contract runtime-env.json
  runtime_env.sh --help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --work) WORK="$2"; shift 2 ;;
    --container-work) CONTAINER_WORK="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --contract) CONTRACT="$2"; shift 2 ;;
    --allowed-host-root) ALLOWED_ROOTS+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "runtime_env: unknown argument $1" >&2; exit 2 ;;
  esac
done

configure_allowed_roots() {
  if [ "${#ALLOWED_ROOTS[@]}" = 0 ] && [ -n "${TILE_HOST_ALLOWED_ROOTS:-}" ]; then
    IFS=: read -r -a ALLOWED_ROOTS <<<"$TILE_HOST_ALLOWED_ROOTS"
  fi
  # A caller that names no policy can only write inside the declared work root.
  # Deployments that share a broader host root set TILE_HOST_ALLOWED_ROOTS or
  # repeat --allowed-host-root; no username or machine path is embedded here.
  [ "${#ALLOWED_ROOTS[@]}" -gt 0 ] || ALLOWED_ROOTS=("$WORK")
  local index
  for index in "${!ALLOWED_ROOTS[@]}"; do
    ALLOWED_ROOTS[$index]="$(realpath -m "${ALLOWED_ROOTS[$index]}")"
  done
}

require_allowed_host_path() {
  local value="$1" root
  for root in "${ALLOWED_ROOTS[@]}"; do
    case "$value" in "$root"|"$root"/*) return 0;; esac
  done
  echo "runtime_env: host path outside configured roots: $value" >&2
  exit 2
}

safe_component() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "runtime_env: run-id and variant must use [A-Za-z0-9._-]" >&2; exit 2;
  }
}

read_contract() {
  [ -n "$CONTRACT" ] || { echo "runtime_env: --contract is required" >&2; exit 2; }
  python3 - "$CONTRACT" "$MODE" <<'PY'
import json, os, sys
path, mode = sys.argv[1:3]
try:
    with open(path) as stream:
        doc = json.load(stream)
except (OSError, ValueError) as exc:
    raise SystemExit(f"runtime_env: unreadable contract: {exc}")
if doc.get("schema") != "tile.runtime-env/1":
    raise SystemExit("runtime_env: unexpected contract schema")
host = doc.get("host") or {}
allowed = doc.get("host_allowed_roots") or []
if not isinstance(allowed, list) or not allowed or not all(isinstance(item, str) for item in allowed):
    raise SystemExit("runtime_env: contract has no configured host_allowed_roots")
paths = doc.get("paths") or {}
for key in ("work_root", "home", "xdg_cache", "tmp", "triton_cache", "tile_hw_cache"):
    value = paths.get(key) if key in paths else host.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"runtime_env: contract has no {key}")
if mode == "verify":
    for value in host.values():
        if isinstance(value, str) and not any(value == root or value.startswith(root.rstrip("/") + "/") for root in allowed):
            raise SystemExit(f"runtime_env: host path is outside contract policy: {value}")
    missing = [p for p in (host["work_root"], host["home"], host["xdg_cache"], host["tmp"],
                           host["triton_cache"], host["tile_hw_cache"]) if not os.path.isdir(p)]
    if missing:
        raise SystemExit("runtime_env: missing host runtime directories: " + ", ".join(missing))
    print(json.dumps({"ok": True, "run_id": doc["run_id"], "variant": doc["variant"]}))
else:
    for key, value in (doc.get("env") or {}).items():
        print(f"export {key}={json.dumps(value)}")
PY
}

case "$MODE" in
  prepare)
    [ -n "$WORK" ] && [ -n "$CONTAINER_WORK" ] && [ -n "$RUN_ID" ] && [ -n "$OUT" ] || {
      usage; exit 2;
    }
    WORK="$(realpath -m "$WORK")"
    OUT="$(realpath -m "$OUT")"
    configure_allowed_roots
    require_allowed_host_path "$WORK"
    require_allowed_host_path "$OUT"
    safe_component "$RUN_ID"; safe_component "$VARIANT"
    ROOT="$WORK/.tile-runtime"
    HOME_DIR="$ROOT/home/$RUN_ID"
    XDG_DIR="$ROOT/cache/$RUN_ID/xdg"
    TMP_DIR="$ROOT/scratch/$RUN_ID"
    TRITON_DIR="$ROOT/cache/$RUN_ID/triton/$VARIANT"
    HW_DIR="$ROOT/cache/$RUN_ID/tile-hw"
    mkdir -p "$HOME_DIR" "$XDG_DIR" "$TMP_DIR" "$TRITON_DIR" "$HW_DIR"
    chmod u+rwx "$HOME_DIR" "$XDG_DIR" "$TMP_DIR" "$TRITON_DIR" "$HW_DIR"
    python3 - "$OUT" "$WORK" "$CONTAINER_WORK" "$RUN_ID" "$VARIANT" "$HOME_DIR" "$XDG_DIR" "$TMP_DIR" "$TRITON_DIR" "$HW_DIR" "${ALLOWED_ROOTS[@]}" <<'PY'
import json, os, sys, tempfile
out, host_root, container_root, run_id, variant, home, xdg, tmp, triton, hw, *allowed_roots = sys.argv[1:]
def container_path(host_path):
    rel = os.path.relpath(host_path, host_root)
    return os.path.normpath(os.path.join(container_root, rel))
paths = {"work_root": host_root, "home": home, "xdg_cache": xdg, "tmp": tmp,
         "triton_cache": triton, "tile_hw_cache": hw}
container = {key: container_path(value) for key, value in paths.items()}
env = {
    "TILE_KERNEL_HOST_WORKDIR": host_root,
    "TILE_KERNEL_CONTAINER_WORKDIR": container_root,
    "HOME": container["home"],
    "XDG_CACHE_HOME": container["xdg_cache"],
    "MPLCONFIGDIR": os.path.join(container["home"], "matplotlib"),
    "TMPDIR": container["tmp"],
    "TMP": container["tmp"],
    "TEMP": container["tmp"],
    "TRITON_CACHE_DIR": container["triton_cache"],
    "TILE_HW_CACHE": container["tile_hw_cache"],
    "PYTHONDONTWRITEBYTECODE": "1",
}
doc = {"schema": "tile.runtime-env/1", "run_id": run_id, "variant": variant,
       "host_allowed_roots": allowed_roots,
       "host": paths, "container": container, "paths": paths, "env": env}
target = os.path.abspath(out)
os.makedirs(os.path.dirname(target), exist_ok=True)
fd, temp = tempfile.mkstemp(prefix=".runtime-env.", suffix=".tmp", dir=os.path.dirname(target))
with os.fdopen(fd, "w") as stream:
    json.dump(doc, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temp, target)
print(json.dumps({"contract": target, "run_id": run_id, "variant": variant}))
PY
    ;;
  verify|export) read_contract ;;
  --help|-h) usage ;;
  *) usage; exit 2 ;;
esac
