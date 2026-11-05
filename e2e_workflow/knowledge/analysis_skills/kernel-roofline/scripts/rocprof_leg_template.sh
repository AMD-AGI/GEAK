#!/usr/bin/env bash
# One command per task/case/arm. cases.txt columns: task case regime call_weight.
# Set UT_ROOT (optional HARNESS) or ADAPTER. Output directory must be a fresh leg.
set -euo pipefail
SKILL=${SKILL:-$(cd "$(dirname "$0")/.." && pwd)}
WORKDIR=${WORKDIR:-$PWD}
OUT=${OUT:-$WORKDIR/results/rocprof/leg}
CASES=${CASES:-$WORKDIR/cases.txt}
FROZEN_CSV=${FROZEN_CSV:?Set FROZEN_CSV to a verified frozen roofline.csv}
ITERS=${ITERS:-3}
WARMUP=${WARMUP:-5}
CACHE=${CACHE:-cold-read}
EXECUTION=${EXECUTION:-eager}
ARMS=${ARMS:-"baseline candidate"}
export ROOFLINE_MIBENCH_CACHE=$FROZEN_CSV
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
[[ -f "$FROZEN_CSV" ]] || { echo "Missing frozen peak file" >&2; exit 2; }
[[ ! -e "$OUT/requested_cases.json" ]] || { echo "Use a fresh OUT directory" >&2; exit 2; }
if compgen -G "$OUT/*_roofline*" >/dev/null; then
  echo "OUT already contains reports; use a fresh directory" >&2
  exit 2
fi
mkdir -p "$OUT"
python3 - "$CASES" "$OUT/requested_cases.json" <<'PY'
import json,sys
rows=[]
for line in open(sys.argv[1]):
    if not line.strip() or line.lstrip().startswith('#'): continue
    task,case,regime,weight=line.split()
    rows.append(dict(kernel=task,case=case,regime=regime,w=float(weight)))
json.dump({'cases':rows},open(sys.argv[2],'w'),indent=2)
PY
adapter_args=()
if [[ -n "${ADAPTER:-}" ]]; then
  adapter_args=(--adapter "$ADAPTER")
else
  adapter_args=(--ut-root "${UT_ROOT:?Set UT_ROOT or ADAPTER}")
  [[ -z "${HARNESS:-}" ]] || adapter_args+=(--harness "$HARNESS")
fi
cache_args=(--cache "$CACHE")
[[ -z "${LLC_MIB:-}" ]] || cache_args+=(--llc-mib "$LLC_MIB")
[[ -z "${EVICT_MIB:-}" ]] || cache_args+=(--evict-mib "$EVICT_MIB")
failed=0
requested=0
while read -r task case_name regime weight <&3; do
  [[ -n "$task" && "$task" != \#* ]] || continue
  for arm in $ARMS; do
    requested=$((requested+1))
    name="${task}_${case_name}_${arm}"
    command=(python3 "$SKILL/scripts/case_runner_template.py" "${adapter_args[@]}" --task "$task"
      --case "$case_name" --regime "$regime" --case-weight "$weight" --arm "$arm"
      --iters "$ITERS" --warmup "$WARMUP" "${cache_args[@]}" --execution "$EXECUTION"
      --output "$OUT/${name}_measurement")
    printf -v command_text '%q ' "${command[@]}"
    if timeout "${PER_CASE_TIMEOUT:-2400}" python3 "$SKILL/scripts/run_roofline.py" \
      --workdir "$WORKDIR" --cmd "$command_text" --name "$name" --output "$OUT" \
      --expected-calls "$ITERS" --peaks both >"$OUT/${name}.log" 2>&1 </dev/null; then
      echo "OK $name"
    else
      failed=$((failed+1))
      echo "FAILED $name (see $OUT/${name}.log)" >&2
    fi
  done
done 3< "$CASES"
python3 "$SKILL/scripts/collect_results.py" --root "$OUT" --cases "$OUT/requested_cases.json" --output "$OUT/collected"
echo "Requested $requested runs; failed $failed"
[[ "$failed" -eq 0 ]]
