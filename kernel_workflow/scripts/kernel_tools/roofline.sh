#!/usr/bin/env bash
# GEAK execution adapter for an optional, separately installed kernel-roofline skill.
# Numerical models, reporting, dispatch windows and calibration belong to that skill.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_kt_common.sh"
ORIG_ARGS=("$@")
ENGINE=tl-only GPU="" WORKSPACE="$PWD"
HARDWARE="${GEAK_HARDWARE_CONTEXT:-}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --engine) ENGINE="${2:?missing engine}"; shift 2;;
    --gpu) GPU="${2:?missing gpu}"; shift 2;;
    --workspace) WORKSPACE="${2:?missing workspace}"; shift 2;;
    --hardware-context) HARDWARE="${2:?missing hardware context}"; shift 2;;
    --) shift; break;;
    --help|-h)
      echo 'Usage: roofline.sh [--engine tl-only|rocprof|measure] [--gpu ID] [--workspace DIR] -- NATIVE_ARGS...'
      echo 'tl-only: offline tl_only.py reporting from an existing measurement (default; no GPU lock).'
      echo 'measure: case_runner_template.py for a separately requested UT measurement.'
      echo 'rocprof: run_roofline.py; measured-window command and frozen ROOFLINE_MIBENCH_CACHE required.'
      echo 'GPU modes run under the GEAK lease/source guard. Set --workspace to the source workspace.'
      echo 'Paths in NATIVE_ARGS must be valid in the execution locus. No automatic engine fallback.'
      echo '--hardware-context FILE (or GEAK_HARDWARE_CONTEXT) validates product/resource scope; native references are never rewritten.'
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
  echo "kernel-roofline unavailable in this checkout: $NATIVE. Install the external runner and set GEAK_KERNEL_ROOFLINE_DIR; no local calculator fallback." >&2
  exit 3
fi
# Native --gpu would override the outer lease, possibly selecting an unrelated device.
for arg in "$@"; do
  case "$arg" in --gpu|--gpu=*) echo 'Pass GPU selection to the GEAK adapter before --, not to the native runner.' >&2; exit 2;; esac
done
if [ -z "$HARDWARE" ]; then
  for candidate in "$WORKSPACE/hardware_context.json" "$WORKSPACE/../hardware_context.json"; do
    if [ -f "$candidate" ]; then HARDWARE="$candidate"; break; fi
  done
fi
HC="$KT_REPO/scripts/hardware_context.py"
if [ -n "$HARDWARE" ]; then
  HARDWARE="$(realpath "$HARDWARE")"
  export GEAK_HARDWARE_CONTEXT="$HARDWARE"
  "${GEAK_TL_PYTHON:-python3}" "$HC" --check "$HARDWARE" >/dev/null
fi
# Inspect only GEAK's boundary metadata. Pass the native argument vector unchanged.
NATIVE_ARCH="" NATIVE_MEASUREMENT="" NATIVE_OUTPUT=measurement NATIVE_REFERENCE=""
next=""
for arg in "$@"; do
  if [ -n "$next" ]; then
    case "$next" in
      --arch) NATIVE_ARCH="$arg";; --measurement) NATIVE_MEASUREMENT="$arg";;
      --output) NATIVE_OUTPUT="$arg";; --rocprof-reference) NATIVE_REFERENCE="$arg";;
    esac
    next=""; continue
  fi
  case "$arg" in
    --arch|--measurement|--output|--rocprof-reference) next="$arg";;
    --arch=*) NATIVE_ARCH="${arg#*=}";; --measurement=*) NATIVE_MEASUREMENT="${arg#*=}";;
    --output=*) NATIVE_OUTPUT="${arg#*=}";; --rocprof-reference=*) NATIVE_REFERENCE="${arg#*=}";;
  esac
done
if [ "$ENGINE" = tl-only ]; then
  [ -z "$GPU" ] || { echo 'tl-only is offline; --gpu is not applicable' >&2; exit 2; }
  if [ -n "$HARDWARE" ]; then
    [ -n "$NATIVE_ARCH" ] && [ -n "$NATIVE_MEASUREMENT" ] || {
      echo 'hardware-scoped tl-only requires native --arch and --measurement' >&2; exit 2;
    }
    "${GEAK_TL_PYTHON:-python3}" "$HC" --check "$HARDWARE" --native-arch "$NATIVE_ARCH" --reference "$NATIVE_MEASUREMENT" >/dev/null
    if [ -n "$NATIVE_REFERENCE" ]; then
      "${GEAK_TL_PYTHON:-python3}" "$HC" --check "$HARDWARE" --reference "$NATIVE_REFERENCE" >/dev/null
    fi
  fi
  # The reporting environment can differ from the workload container, as upstream supports.
  exec "${GEAK_TL_PYTHON:-python3}" "$NATIVE/scripts/$SCRIPT" "$@"
fi
kt_ensure_lock "$GPU" "$HERE/roofline.sh" "${ORIG_ARGS[@]}"
cd "$WORKSPACE"
if [ -n "${TILE_KERNEL_CONTAINER:-}" ] && ! locus_active; then
  echo 'Configured kernel container is unavailable; refusing host fallback.' >&2; exit 3
fi
if [ -n "$HARDWARE" ]; then
  locus_run "${GEAK_GPU_PYTHON:-python3}" "$(locus_path "$HC")" --check "$(locus_path "$HARDWARE")" --live >/dev/null
fi
if [ "$ENGINE" = rocprof ]; then
  [ -n "${ROOFLINE_MIBENCH_CACHE:-}" ] && [ -r "$ROOFLINE_MIBENCH_CACHE" ] || {
    echo 'rocprof requires a readable frozen ROOFLINE_MIBENCH_CACHE; follow native calibration guidance.' >&2; exit 2;
  }
  locus_run test -r "$(locus_path "$ROOFLINE_MIBENCH_CACHE")" || {
    echo 'Frozen calibration is not readable in the execution locus; fix its mount mapping.' >&2; exit 3;
  }
  if [ -n "$HARDWARE" ]; then
    locus_run "${GEAK_GPU_PYTHON:-python3}" "$(locus_path "$HC")" --check "$(locus_path "$HARDWARE")" --reference "$(locus_path "$ROOFLINE_MIBENCH_CACHE")" >/dev/null
  fi
fi
NATIVE_SCRIPT="$(locus_path "$NATIVE/scripts/$SCRIPT")"
locus_run test -r "$NATIVE_SCRIPT" || { echo "Native runner is not mounted in the execution locus: $NATIVE_SCRIPT" >&2; exit 3; }
locus_run "${GEAK_GPU_PYTHON:-python3}" "$NATIVE_SCRIPT" "$@"
if [ "$ENGINE" = measure ] && [ -n "$HARDWARE" ]; then
  locus_run "${GEAK_GPU_PYTHON:-python3}" "$(locus_path "$HC")" --check "$(locus_path "$HARDWARE")" --live \
    --record-measurement "${KRF_MEASUREMENT_MANIFEST:-$NATIVE_OUTPUT/measurement.json}" >/dev/null
fi
