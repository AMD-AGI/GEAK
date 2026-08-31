#!/usr/bin/env bash
# Materialize an isolated workspace from SRC into DST (issue #429 / aiter-.so amplification).
#
# WHY: Role prompts used to inline `tar --exclude='*.so'`. Agents sometimes omitted that
# exclude (seen in production verify bash), and even a correct `*.so` pattern is easy to
# regress. This script is the single copy path for director / engineer / verify / integrate.
#
# Contract:
#   - Never dereference symlinks (-h): absolute symlinks (reference_io / shared aiter) stay links.
#   - Never copy build artifacts or nested *.so/*.o (recursive exclude).
#   - Optionally symlink immutable trees (aiter/) onto a shared physical copy.
#   - Does NOT touch kernel_src editability: source trees that are not excluded remain writable files.
#
# Usage:
#   materialize_workspace.sh --src <dir> --dst <dir> [--shared-root <dir>] [--link-aiter]
#                            [--soft-budget-bytes N]  # advisory only; never aborts optimize
#
# Exit 0 on success. Prints a one-line JSON summary to stdout (also appended to
# $DST/../materialize_telemetry.jsonl when DST's parent exists).
set -euo pipefail

SRC=""
DST=""
SHARED_ROOT=""
LINK_AITER=0
SOFT_BUDGET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="${2:?}"; shift 2 ;;
    --dst) DST="${2:?}"; shift 2 ;;
    --shared-root) SHARED_ROOT="${2:?}"; shift 2 ;;
    --link-aiter) LINK_AITER=1; shift ;;
    --soft-budget-bytes) SOFT_BUDGET="${2:?}"; shift 2 ;;
    -h|--help)
      sed -n '1,25p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$SRC" && -d "$SRC" ]] || { echo "materialize_workspace: --src must be an existing directory" >&2; exit 2; }
[[ -n "$DST" ]] || { echo "materialize_workspace: --dst required" >&2; exit 2; }

SRC="$(cd "$SRC" && pwd)"
mkdir -p "$DST"
DST="$(cd "$DST" && pwd)"

# Recursive exclude of nested *.so/*.o: GNU tar needs wildcards-match-slash so patterns
# cross path components. Also drop aiter/jit trees (JIT outputs land here and are the
# multi-GiB amplifier on MXFP4 MoE tasks).
TAR_EXCLUDES=(
  --wildcards --wildcards-match-slash
  --exclude='./.git' --exclude='*/.git'
  --exclude='./build' --exclude='*/build'
  --exclude='./__pycache__' --exclude='*/__pycache__'
  --exclude='./.torch_ext' --exclude='*/.torch_ext'
  --exclude='./.rocprofv3' --exclude='*/.rocprofv3'
  --exclude='./reference_io.pt' --exclude='*/reference_io.pt'
  --exclude='./aiter/jit' --exclude='*/aiter/jit'
  --exclude='./aiter/aiter/jit' --exclude='*/aiter/aiter/jit'
  --exclude='*.so' --exclude='*.o'
  --exclude='./wave*_archive_*' --exclude='*/wave*_archive_*'
)

# Fresh destination contents (caller creates a unique DST; we clear files but keep the dir).
# No `rm -rf` of DST itself — unique out_dirs are required by the lane protocol.
find "$DST" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true

# Critical: do NOT pass -h/--dereference. Symlinks must be preserved as symlinks.
( cd "$SRC" && tar "${TAR_EXCLUDES[@]}" -cf - . ) | ( cd "$DST" && tar -xf - )

# Share the immutable golden when present on SRC (or as a dangling absolute link target).
if [[ -e "$SRC/reference_io.pt" || -L "$SRC/reference_io.pt" ]]; then
  rm -f "$DST/reference_io.pt"
  ln -s "$(readlink -f "$SRC/reference_io.pt" 2>/dev/null || echo "$SRC/reference_io.pt")" \
    "$DST/reference_io.pt" 2>/dev/null \
    || ln -s "$SRC/reference_io.pt" "$DST/reference_io.pt"
fi

# Optional: one physical aiter/ tree per eval (shared-root), symlink into DST.
# Only for immutable vendor trees — never for editable kernel_src/.
if [[ "$LINK_AITER" -eq 1 ]]; then
  for AITER_REL in aiter aiter/aiter; do
    if [[ -d "$SRC/$AITER_REL" && ! -L "$SRC/$AITER_REL" ]]; then
      if [[ -n "$SHARED_ROOT" ]]; then
        SHARED_AITER="$SHARED_ROOT/$AITER_REL"
        mkdir -p "$(dirname "$SHARED_AITER")"
        if [[ ! -e "$SHARED_AITER" ]]; then
          # First materialize seeds the shared copy WITHOUT jit/*.so.
          mkdir -p "$SHARED_AITER"
          ( cd "$SRC/$AITER_REL" && tar \
              --wildcards --wildcards-match-slash \
              --exclude='./jit' --exclude='*/jit' \
              --exclude='*.so' --exclude='*.o' \
              -cf - . ) | ( cd "$SHARED_AITER" && tar -xf - )
        fi
        rm -rf "$DST/$AITER_REL"
        mkdir -p "$(dirname "$DST/$AITER_REL")"
        ln -s "$SHARED_AITER" "$DST/$AITER_REL"
      fi
      break
    fi
  done
fi

bytes_dst=$(du -sb "$DST" 2>/dev/null | awk '{print $1}')
n_so=$(find "$DST" -type f -name '*.so' 2>/dev/null | wc -l | tr -d ' ')
so_bytes=0
if [[ "$n_so" != "0" ]]; then
  so_bytes=$(find "$DST" -type f -name '*.so' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
fi

summary=$(printf '{"ok":true,"src":"%s","dst":"%s","bytes_dst":%s,"n_so":%s,"so_bytes":%s,"link_aiter":%s,"soft_budget_bytes":%s}' \
  "$SRC" "$DST" "${bytes_dst:-0}" "${n_so:-0}" "${so_bytes:-0}" "$LINK_AITER" "$SOFT_BUDGET")
echo "$summary"

parent="$(dirname "$DST")"
if [[ -d "$parent" ]]; then
  echo "$summary" >> "$parent/materialize_telemetry.jsonl" 2>/dev/null || true
fi

# Soft budget is advisory telemetry only — NEVER abort (issue #429 revised policy:
# reclaim pressure must not stop the optimize loop).
if [[ "$SOFT_BUDGET" -gt 0 && "${bytes_dst:-0}" -gt "$SOFT_BUDGET" ]]; then
  echo "materialize_workspace: soft budget exceeded (bytes_dst=$bytes_dst > $SOFT_BUDGET); continuing" >&2
fi

exit 0
