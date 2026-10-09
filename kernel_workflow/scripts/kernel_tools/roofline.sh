#!/usr/bin/env bash
# GEAK execution adapter for the upstream native kernel-roofline skill.
# Numerical models, reporting, dispatch windows and calibration belong to that skill.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_kt_common.sh"
ORIG_ARGS=("$@")
ENGINE=tl-only GPU="" WORKSPACE="$PWD"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --engine) ENGINE="${2:?missing engine}"; shift 2;;
    --gpu) GPU="${2:?missing gpu}"; shift 2;;
    --workspace) WORKSPACE="${2:?missing workspace}"; shift 2;;
    --) shift; break;;
    --help|-h)
      echo 'Usage: roofline.sh [--engine tl-only|rocprof|measure] [--gpu ID] [--workspace DIR] -- NATIVE_ARGS...'
      echo 'tl-only: offline tl_only.py reporting from an existing measurement (default; no GPU lock).'
      echo 'measure: case_runner_template.py for a separately requested UT measurement.'
      echo 'rocprof: run_roofline.py; measured-window command and frozen ROOFLINE_MIBENCH_CACHE required.'
      echo 'GPU modes run under the GEAK lease/source guard. Set --workspace to the source workspace.'
      echo 'Paths in NATIVE_ARGS must be valid in the execution locus. No automatic engine fallback.'
      exit 0;;
    *) echo "unknown adapter option: $1 (native options follow --)" >&2; exit 2;;
  esac
done
case "$ENGINE" in
  tl-only) SCRIPT=tl_only.py;;
  rocprof) SCRIPT=run_roofline.py;;
  measure) SCRIPT=case_runner_template.py;;
  *) echo "unknown engine: $ENGINE" >&2; exit 2;;
esac
NATIVE="${GEAK_KERNEL_ROOFLINE_DIR:-$KT_REPO/e2e_workflow/knowledge/analysis_skills/kernel-roofline}"
if [ ! -f "$NATIVE/scripts/$SCRIPT" ]; then
  echo "kernel-roofline unavailable in this checkout: $NATIVE. Integrate GEAK's native TraceLens skill first; no local calculator fallback." >&2
  exit 3
fi
# Native --gpu would override the outer lease, possibly selecting an unrelated device.
for arg in "$@"; do
  case "$arg" in --gpu|--gpu=*) echo 'Pass GPU selection to the GEAK adapter before --, not to the native runner.' >&2; exit 2;; esac
done
if [ "$ENGINE" = tl-only ]; then
  [ -z "$GPU" ] || { echo 'tl-only is offline; --gpu is not applicable' >&2; exit 2; }
  # The reporting environment can differ from the workload container, as upstream supports.
  exec "${GEAK_TL_PYTHON:-python3}" "$NATIVE/scripts/$SCRIPT" "$@"
fi
cd "$WORKSPACE"
kt_ensure_lock "$GPU" "$HERE/roofline.sh" "${ORIG_ARGS[@]}"
if [ -n "${TILE_KERNEL_CONTAINER:-}" ] && ! locus_active; then
  echo 'Configured kernel container is unavailable; refusing host fallback.' >&2; exit 3
fi
if [ "$ENGINE" = rocprof ]; then
  [ -n "${ROOFLINE_MIBENCH_CACHE:-}" ] && [ -r "$ROOFLINE_MIBENCH_CACHE" ] || {
    echo 'rocprof requires a readable frozen ROOFLINE_MIBENCH_CACHE; follow native calibration guidance.' >&2; exit 2;
  }
  locus_run test -r "$(locus_path "$ROOFLINE_MIBENCH_CACHE")" || {
    echo 'Frozen calibration is not readable in the execution locus; fix its mount mapping.' >&2; exit 3;
  }
fi
NATIVE_SCRIPT="$(locus_path "$NATIVE/scripts/$SCRIPT")"
locus_run test -r "$NATIVE_SCRIPT" || { echo "Native runner is not mounted in the execution locus: $NATIVE_SCRIPT" >&2; exit 3; }
locus_run "${GEAK_GPU_PYTHON:-python3}" "$NATIVE_SCRIPT" "$@"
