#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${PROMPT_OUT:?}" "${TASK_DIR:?}" "${RUN_DIR:?}" "${EVAL_DIR:?}" "${WORK_ROOT:?}"
: "${K:?}" "${ATTEMPT:?}" "${ATTEMPT_CAP:?}" "${GPU_IDS:?}" "${KERNEL_SRC_ALL:?}"
: "${BUDGET:?}" "${MIN_IMPROVE:?}" "${BEST_SPEEDUP:?}" "${KEEP_THRESHOLD:?}" "${MEASURE_TIMEOUT:?}"
METHOD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="${GEAK_ROOT:-$(cd "$METHOD/../../../.." && pwd)}"
SK="$GEAK_ROOT/kernel_skills"
LEAD="$SK/v4/geak-lead/SKILL.md"
[[ -r "$LEAD" ]] || { echo "Missing v4 lead skill: $LEAD" >&2; exit 2; }

cat > "$PROMPT_OUT" <<EOF
Follow $METHOD/SKILL.md, section "Running one attempt", as the lead in $LEAD.
Pass the workload measurement and delivery instructions to the roles. Use these
literal values when invoking tools. The runner need not export them.

METHOD=$METHOD
SK=$SK
KW=$GEAK_ROOT/kernel_workflow
INPUT=$TASK_DIR
RUN=$EVAL_DIR/kernel-skills
GPUS=$GPU_IDS
APPLY=0
KERNEL_SRC_ALL=$KERNEL_SRC_ALL
SOURCE_TREE=$GEAK_ROOT/workload/engine/campaign/source_tree.py
Worker state: $WORK_ROOT/state/$K.status.json
Read the prior handoff and write this attempt's outcome to $RUN_DIR/HANDOFF.md

ATTEMPT=$ATTEMPT
ATTEMPT_CAP=$ATTEMPT_CAP seconds from launch
BUDGET=$BUDGET optimization directions
MIN_IMPROVE=$MIN_IMPROVE relative inner-incumbent gain
MEASURE_TIMEOUT=$MEASURE_TIMEOUT seconds per full native check
BEST_SPEEDUP=$BEST_SPEEDUP against the original
KEEP_THRESHOLD=$KEEP_THRESHOLD against the original
Read the worker's incremental KEEP rule in $GEAK_ROOT/workload/engine/campaign/worker.sh.

${EXTRA_STEER:-}
EOF
