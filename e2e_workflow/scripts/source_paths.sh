# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Sourced only for source-bearing launches; controls-only launches are unchanged.
_GEAK_SOURCE_PATHS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

geak_source_pythonpath() {
  local validator="$_GEAK_SOURCE_PATHS_DIR/source_materialization.py"
  # Installed benchmark bundles carry the validator beside this helper. The
  # repository keeps its public interface module in interface/.
  if [ ! -f "$validator" ]; then
    validator="$_GEAK_SOURCE_PATHS_DIR/../../interface/source_materialization.py"
  fi
  if [ ! -f "$validator" ] || [ -z "${GEAK_SOURCE_REQUEST:-}" ]; then
    echo 'unresolved_baseline_source:missing_source_path_helper_or_request' >&2
    return 2
  fi
  python3 "$validator" --compose-pythonpath "$GEAK_SOURCE_REQUEST" \
    "${OVERLAY_PYTHONPATH:-}" "${1:-}" "${PYTHONPATH:-}" "${GEAK_SOURCE_BOOTSTRAP_PYTHONPATH:-}"
}
