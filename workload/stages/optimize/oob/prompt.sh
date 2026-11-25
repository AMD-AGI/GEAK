#!/usr/bin/env bash
# The worker owns acceptance; the prompt carries the task and its local constraints.
set -euo pipefail
: "${PROMPT_OUT:?}" "${TASK_DIR:?}" "${RUN_DIR:?}" "${EVAL_DIR:?}"
: "${ATTEMPT_CAP:?}" "${GPU_IDS:?}"
GEAK_ROOT="${GEAK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
ENTRY="$GEAK_ROOT/workload/engine/campaign/run_task_entry.py"
cat > "$PROMPT_OUT" <<EOF
Optimize the kernel task at "$TASK_DIR" using native coding tools.
Budget: ${ATTEMPT_CAP}s; GPUs: [$GPU_IDS]. Preserve that visibility; serialize measurements.
Edit only the computation sources declared by meta.json; the harness stays frozen.
Continue from the handoff and measurement evidence in "$RUN_DIR"; use "$EVAL_DIR" for scratch work.
Native checks: python3 "$ENTRY" "$TASK_DIR"
The worker owns full acceptance; use small probes when useful.
Leave the candidate in kernel_src and a short HANDOFF.md at "$RUN_DIR/HANDOFF.md".

${EXTRA_STEER:-}
EOF
