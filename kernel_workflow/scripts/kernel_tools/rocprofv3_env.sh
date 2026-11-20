#!/usr/bin/env bash
# Shared rocprofv3 device-mask preparation. Sourcing defines the helper; executing
# runs rocprofv3 with otherwise unchanged arguments. Changes stay in this child.
# Collapse HIP-inside-ROCR to a single ROCR device (GUARD 2). Pure function of the environment.
_rv3_fix_visible_devices() {
  [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -n "${HIP_VISIBLE_DEVICES:-}" ] || return 0
  local -a rocr hip selected=()
  local idx
  IFS=',' read -r -a rocr <<< "$ROCR_VISIBLE_DEVICES"
  if [[ ! "$HIP_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[rocprofv3_safe] invalid HIP indices inside ROCR; refusing to widen the GPU allocation" >&2
    return 2
  fi
  IFS=',' read -r -a hip <<< "$HIP_VISIBLE_DEVICES"
  for idx in "${hip[@]}"; do
    idx=$((10#$idx))
    if [ "$idx" -ge "${#rocr[@]}" ] || [ -z "${rocr[$idx]}" ]; then
      echo "[rocprofv3_safe] HIP index $idx is outside ROCR; refusing to widen the GPU allocation" >&2
      return 2
    fi
    selected+=("${rocr[$idx]}")
  done
  export ROCR_VISIBLE_DEVICES="$(IFS=,; echo "${selected[*]}")"
  echo "[rocprofv3_safe] collapsed HIP indices to ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES" >&2
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  RV3_UNSET_HIP=1
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  _rv3_fix_visible_devices || exit $?
  exec rocprofv3 "$@"
fi
