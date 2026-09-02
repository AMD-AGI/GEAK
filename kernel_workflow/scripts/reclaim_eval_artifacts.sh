#!/usr/bin/env bash
# Reclaim heavy, superseded artifacts under an eval dir (issue #429 corrected root cause).
#
# WHY: Engineer/verify/integrate workspaces accumulate aiter JIT *.so copies; wave archives
# compound them. Disk pressure must TRIGGER reclaim, never abort the optimize loop.
#
# Usage:
#   reclaim_eval_artifacts.sh --eval-dir <EVAL_DIR> [--keep-round N] [--force-heavy]
#
# Keeps:
#   - EVAL_DIR/workspace (CANONICAL)
#   - EVAL_DIR/baseline (when present)
#   - patches / metrics / STATE / COMMANDMENT / reports
#   - round_K for K >= keep-round (default: keep only the latest round dir)
# Removes / lightens:
#   - older round_*/engineer_*/workspace trees
#   - verify ws*/ctl_* copies under older (and optionally current) rounds
#   - nested *.so under round_* (never under CANONICAL kernel_src edits — those use .torch_ext)
#   - wave*_archive_* when --force-heavy (GEAK-side cleanup if AKA left them under eval)
#
# Prints one-line JSON summary to stdout; appends to EVAL_DIR/storage_telemetry.jsonl.
set -euo pipefail

EVAL_DIR=""
KEEP_ROUND=-1
FORCE_HEAVY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-dir) EVAL_DIR="${2:?}"; shift 2 ;;
    --keep-round) KEEP_ROUND="${2:?}"; shift 2 ;;
    --force-heavy) FORCE_HEAVY=1; shift ;;
    -h|--help) sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$EVAL_DIR" && -d "$EVAL_DIR" ]] || { echo "reclaim_eval_artifacts: --eval-dir required" >&2; exit 2; }
EVAL_DIR="$(cd "$EVAL_DIR" && pwd)"

bytes_before=$(du -sb "$EVAL_DIR" 2>/dev/null | awk '{print $1}')
removed=0
bytes_reclaimed=0

reclaim_path() {
  local p="$1"
  [[ -e "$p" || -L "$p" ]] || return 0
  local b
  b=$(du -sb "$p" 2>/dev/null | awk '{print $1}')
  rm -rf "$p"
  removed=$((removed + 1))
  bytes_reclaimed=$((bytes_reclaimed + ${b:-0}))
}

# Discover round numbers
mapfile -t ROUND_DIRS < <(find "$EVAL_DIR" -maxdepth 1 -type d -name 'round_*' | sort -V)
if [[ "$KEEP_ROUND" -lt 0 && ${#ROUND_DIRS[@]} -gt 0 ]]; then
  # keep only the highest round_*
  latest="${ROUND_DIRS[-1]}"
  KEEP_ROUND=$(basename "$latest" | sed 's/round_//')
fi

for rd in "${ROUND_DIRS[@]+"${ROUND_DIRS[@]}"}"; do
  [[ -d "$rd" ]] || continue
  rnum=$(basename "$rd" | sed 's/round_//')
  if [[ "$rnum" =~ ^[0-9]+$ ]] && [[ "$rnum" -lt "$KEEP_ROUND" ]]; then
    # Older rounds: drop engineer workspaces + verify clones; keep manifests/patches if any.
    while IFS= read -r -d '' ws; do
      reclaim_path "$ws"
    done < <(find "$rd" -type d \( -name workspace -o -name 'ws' -o -name 'ws2_*' -o -name 'ctl_*' -o -name 'ws_*' \) -print0 2>/dev/null)
    # Any leftover nested .so under the old round
    while IFS= read -r -d '' so; do
      reclaim_path "$so"
    done < <(find "$rd" -type f -name '*.so' -print0 2>/dev/null)
  else
    # Current/kept round: remove verify clones and stray *.so, keep active engineer workspaces
    # unless --force-heavy (disk pressure path).
    while IFS= read -r -d '' ws; do
      reclaim_path "$ws"
    done < <(find "$rd" -type d \( -path '*/verify/*' -o -name 'ws2_*' -o -name 'ctl_*' \) -print0 2>/dev/null)
    while IFS= read -r -d '' so; do
      reclaim_path "$so"
    done < <(find "$rd" -type f -name '*.so' -print0 2>/dev/null)
    if [[ "$FORCE_HEAVY" -eq 1 ]]; then
      while IFS= read -r -d '' ws; do
        # Keep best_patch.diff sibling: only remove .../engineer_*/workspace
        reclaim_path "$ws"
      done < <(find "$rd" -type d -path '*/engineer_*/workspace' -print0 2>/dev/null)
    fi
  fi
done

# validation_workspace.old_* leftovers (director uses mv, not rm)
while IFS= read -r -d '' old; do
  reclaim_path "$old"
done < <(find "$EVAL_DIR" -maxdepth 1 -type d -name 'validation_workspace.old_*' -print0 2>/dev/null)

# Wave archives under eval (if the outer harness left them here)
while IFS= read -r -d '' arch; do
  if [[ "$FORCE_HEAVY" -eq 1 ]]; then
    reclaim_path "$arch"
  else
    # Lighten: strip *.so / aiter/jit from archives but keep structure for debugging
    while IFS= read -r -d '' so; do
      reclaim_path "$so"
    done < <(find "$arch" -type f -name '*.so' -print0 2>/dev/null)
    while IFS= read -r -d '' jit; do
      reclaim_path "$jit"
    done < <(find "$arch" -type d \( -path '*/aiter/jit' -o -path '*/aiter/aiter/jit' \) -print0 2>/dev/null)
  fi
done < <(find "$EVAL_DIR" -maxdepth 1 -type d -name 'wave*_archive_*' -print0 2>/dev/null)

bytes_after=$(du -sb "$EVAL_DIR" 2>/dev/null | awk '{print $1}')
n_so=$(find "$EVAL_DIR" -type f -name '*.so' 2>/dev/null | wc -l | tr -d ' ')

summary=$(printf '{"ok":true,"eval_dir":"%s","bytes_before":%s,"bytes_after":%s,"bytes_reclaimed":%s,"removed_paths":%s,"n_so_remaining":%s,"keep_round":%s,"force_heavy":%s}' \
  "$EVAL_DIR" "${bytes_before:-0}" "${bytes_after:-0}" "${bytes_reclaimed:-0}" "$removed" "${n_so:-0}" "$KEEP_ROUND" "$FORCE_HEAVY")
echo "$summary"
echo "$summary" >> "$EVAL_DIR/storage_telemetry.jsonl" 2>/dev/null || true
exit 0
