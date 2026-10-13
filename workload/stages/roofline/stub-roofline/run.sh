#!/usr/bin/env bash
set -euo pipefail
: "${OUT:?OUT}"; : "${WORKLOAD_ROOT:?WORKLOAD_ROOT}"
python3 - <<'PY'
import json, os
# The ordering matches what downstream expects: descending expected_e2e_gain_pct. The
# numbers are fabricated, which caveats states.
json.dump({"kernels": [{"kernel": "stub_kernel_a", "pct_gpu_time": 10.0,
                        "attainable_speedup": 2.0, "expected_e2e_gain_pct": 5.0},
                       {"kernel": "stub_kernel_b", "pct_gpu_time": 5.0,
                        "attainable_speedup": 2.0, "expected_e2e_gain_pct": 2.5}],
           "caveats": ["stub implementation: these numbers are hardcoded, come from no "
                       "measurement, and must not be reported"]},
          open(os.path.join(os.environ["OUT"], "roofline.json"), "w"), indent=1)
PY
python3 "$WORKLOAD_ROOT/scripts/emit_result.py" --status ok \
  --artifact roofline=roofline.json --metric kernels=2 \
  --notes "stub implementation, used only to demonstrate that the slot is swappable; the numbers are hardcoded and must not be reported."
