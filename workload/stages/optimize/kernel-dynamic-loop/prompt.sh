#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${PROMPT_OUT:?}" "${TASK_DIR:?}" "${RUN_DIR:?}" "${EVAL_DIR:?}" "${WORK_ROOT:?}"
: "${K:?}" "${ATTEMPT:?}" "${ATTEMPT_CAP:?}" "${GPU_IDS:?}" "${KERNEL_SRC_ALL:?}"
: "${BUDGET:?}" "${MIN_IMPROVE:?}" "${MEASURE_TIMEOUT:?}" "${BEST_SPEEDUP:?}" "${KEEP_THRESHOLD:?}"
METHOD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="${GEAK_ROOT:-$(cd "$METHOD_ROOT/../../../.." && pwd)}"
cat > "$PROMPT_OUT" <<EOF
Use the kernel dynamic loop skill: $METHOD_ROOT/SKILL.md
Read it, choose modules from evidence and perform this bounded attempt.
Use these supplied values when invoking tools; do not assume they are shell exports.

TASK_DIR: $TASK_DIR
KERNEL_SRC_ALL: $KERNEL_SRC_ALL
RUN_DIR: $RUN_DIR
EVAL_DIR: $EVAL_DIR
WORK_ROOT: $WORK_ROOT
Source / native-entry tools: $GEAK_ROOT/workload/engine/campaign
Worker status: $WORK_ROOT/state/$K.status.json
Prior handoff: $RUN_DIR/HANDOFF.md
ATTEMPT: $ATTEMPT
ATTEMPT_CAP (seconds): $ATTEMPT_CAP
MEASURE_TIMEOUT (seconds): $MEASURE_TIMEOUT
GPU_IDS (complete assigned logical set): $GPU_IDS
BUDGET (optimization directions this attempt): $BUDGET
MIN_IMPROVE (relative inner-incumbent gain): $MIN_IMPROVE
BEST_SPEEDUP (worker-accepted, against original): $BEST_SPEEDUP
KEEP_THRESHOLD (outer baseline-relative gate): $KEEP_THRESHOLD

The parent owns KEEP, cumulative credit and cold verification. Read its current
acceptance rule; do not infer an incremental KEEP rule from the baseline-relative gate.
Finish with source and evidence at the paths above and a concise HANDOFF.md.

${EXTRA_STEER:-}
EOF
