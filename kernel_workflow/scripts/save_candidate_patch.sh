#!/usr/bin/env bash
# Capture only the task's editable source, including new backend/build source files.
# Usage: save_candidate_patch.sh WORKSPACE OUTPUT BASE [EDITABLE_PATH ...]
# Extracted tasks use kernel_src; other tasks must name their declared editable paths.
set -euo pipefail
[[ $# -ge 3 ]] || { echo "usage: $0 WORKSPACE OUTPUT BASE [EDITABLE_PATH ...]" >&2; exit 2; }
workspace=$1
output=$2
base=$3
shift 3
workspace=$(cd "$workspace" && pwd -P)
output=$(realpath -m -- "$output")
cd "$workspace"
[[ "$(git rev-parse --show-toplevel)" == "$workspace" ]] || { echo "isolated candidate repository required" >&2; exit 2; }
if [[ -d kernel_src ]]; then
  paths=(kernel_src)
else
  [[ $# -gt 0 ]] || { echo "name the task's declared editable files/directories" >&2; exit 2; }
  paths=("$@")
fi
for source_path in "${paths[@]}"; do
  case "$source_path" in /*|..|../*|*/../*|*/..) echo "editable paths must be relative to WORKSPACE" >&2; exit 2;; esac
  [[ "$source_path" != /* ]] || { echo "editable paths must be relative to WORKSPACE" >&2; exit 2; }
done
# Intent-to-add makes new .hip/.s/module/build-source files part of the diff. Never
# stage the whole workspace implicitly: measurement, tool and log paths stay outside
# the declared source scope. Existing .gitignore/info/exclude rules still apply.
git --literal-pathspecs add --intent-to-add -- "${paths[@]}"
tmp=$(mktemp "${output}.tmp.XXXXXX")
trap 'rm -f -- "$tmp"' EXIT
git --literal-pathspecs diff --binary --no-color --no-ext-diff "$base" -- "${paths[@]}" > "$tmp"
mv -- "$tmp" "$output"
trap - EXIT
