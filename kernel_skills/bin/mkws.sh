#!/usr/bin/env bash
# Fresh candidate workspace = RUN/orig + patches, as a git repo with tags `base` and `parent`.
# Usage: mkws.sh <RUN> <dst> [patch.diff ...]
#   Patches are applied in order and committed together as `parent`; edits after that are the
#   candidate's own delta (`git diff --binary parent`). Never edits RUN/orig.
set -euo pipefail
RUN="$(cd "${1:?RUN}" && pwd)"; DST="${2:?dst}"; shift 2
PATCHES=()
for p in "$@"; do
  [[ -f "$p" ]] || { echo "mkws: patch not found: $p" >&2; exit 2; }
  PATCHES+=("$(readlink -f "$p")")
done
KW="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../kernel_workflow" && pwd)"
[[ -e "$DST" && -n "$(ls -A "$DST" 2>/dev/null)" ]] && { echo "mkws: $DST is not empty" >&2; exit 2; }

bash "$KW/scripts/materialize_workspace.sh" --src "$RUN/orig" --dst "$DST" \
  --shared-root "$RUN/_shared" --link-aiter >/dev/null
cd "$DST"
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
G=(git -c user.email=geak@local -c user.name=geak)
git init -q
printf '%s\n' build/ __pycache__/ '*.pyc' '*.so' '*.o' .torch_ext/ .torch_ext.stale_*/ \
  .rocprofv3/ /.geak/ reference_io.pt >> .git/info/exclude
"${G[@]}" add -A && "${G[@]}" commit -qm base --allow-empty && git tag base
for p in "${PATCHES[@]}"; do
  [[ -s "$p" ]] || continue   # an empty patch (e.g. c000) is a no-op
  git apply --whitespace=nowarn "$p" || { echo "mkws: failed to apply $p" >&2; exit 3; }
done
"${G[@]}" add -A && "${G[@]}" commit -qm parent --allow-empty && git tag parent
pwd
