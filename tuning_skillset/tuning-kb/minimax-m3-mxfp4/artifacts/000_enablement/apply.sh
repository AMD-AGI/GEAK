#!/usr/bin/env bash
# Apply (or check) the enablement patch this model needs in order to start at all.
#
#   ./apply.sh            # apply, idempotent
#   ./apply.sh --check    # exit 0 if already applied, 1 if not
#   ./apply.sh --revert   # restore the pristine files
#
# This is NOT an optimization. Read README.md in this directory: without it the engine dies during
# full-cudagraph decode capture, before serving a single token. It is part of the frozen baseline.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/minimax_m3_uniform_decode_capture_guard.patch"
PRISTINE="$HERE/pristine"

VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null)"
if [ -z "$VLLM_DIR" ] || [ ! -d "$VLLM_DIR" ]; then
    echo "[enablement] cannot locate the installed vllm package" >&2; exit 2
fi

FILES=(models/minimax_m3/common/indexer.py models/minimax_m3/common/sparse_attention.py)

# The marker the patch introduces. Present in both files once applied.
MARKER='padded_capture_batch'

applied() {
    for f in "${FILES[@]}"; do
        [ -f "$VLLM_DIR/$f" ] || return 2
        grep -q "$MARKER" "$VLLM_DIR/$f" || return 1
    done
    return 0
}

case "${1:-}" in
    --check)
        applied
        rc=$?
        case $rc in
            0) echo "[enablement] applied" ;;
            1) echo "[enablement] NOT applied" ;;
            2) echo "[enablement] target files missing under $VLLM_DIR" >&2 ;;
        esac
        exit $rc
        ;;
    --revert)
        if [ ! -d "$PRISTINE" ]; then
            echo "[enablement] no pristine copies saved; cannot revert" >&2; exit 1
        fi
        for f in "${FILES[@]}"; do
            cp "$PRISTINE/$(basename "$f")" "$VLLM_DIR/$f" || exit 1
            echo "[enablement] restored $f"
        done
        exit 0
        ;;
esac

if applied; then
    echo "[enablement] already applied — nothing to do"
    exit 0
fi

for f in "${FILES[@]}"; do
    if [ ! -f "$VLLM_DIR/$f" ]; then
        echo "[enablement] expected $VLLM_DIR/$f — the MiniMax-M3 source layout is not what this patch targets" >&2
        exit 1
    fi
done

# Keep a pristine copy before touching anything. There is no git checkout in this image, so this is the
# only way to produce an honest diff later, and the only way back.
mkdir -p "$PRISTINE"
for f in "${FILES[@]}"; do
    [ -f "$PRISTINE/$(basename "$f")" ] || cp "$VLLM_DIR/$f" "$PRISTINE/$(basename "$f")"
done

# The patch was authored against paths of the form vllm/models/minimax_m3/common/*.py, so strip the
# leading "vllm/" component and apply from inside the package directory.
if patch -p1 -d "$VLLM_DIR/.." --dry-run < "$PATCH" >/dev/null 2>&1; then
    patch -p1 -d "$VLLM_DIR/.." < "$PATCH" || exit 1
else
    echo "[enablement] patch does not apply cleanly. Dry run says:" >&2
    patch -p1 -d "$VLLM_DIR/.." --dry-run < "$PATCH" >&2
    exit 1
fi

if applied; then
    echo "[enablement] applied to $VLLM_DIR"
    echo "[enablement] pristine copies kept in $PRISTINE"
    exit 0
fi
echo "[enablement] patch reported success but the marker is absent — inspect manually" >&2
exit 1
