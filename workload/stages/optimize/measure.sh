#!/usr/bin/env bash
# Native frozen-UT measurement shared by the optimize implementations.
set -uo pipefail
: "${TASK_DIR:?TASK_DIR}" "${RUN_DIR:?RUN_DIR}" "${MEASURE_TIMEOUT:?MEASURE_TIMEOUT}"
RUN="$RUN_DIR"

# Default measurement: run the task's own frozen entry and read what it contracts to print.
# The baseline leg is always the **originally installed module**, not your previous change,
# so this speedup is always the cumulative figure against the real baseline.
#
# The entry is launched through engine/campaign/run_task_entry.py rather than with `cd $TASK_DIR &&
# python3 unittest.py`: a task directory contains its own `unittest.py`, which lands on
# sys.path[0] and shadows the standard library package, and the failure surfaces far away
# ("No module named 'unittest.mock'") as an empty measurement rather than as an error.
#
# The task output contracts below are accepted:
#   * `CORRECTNESS: PASS` plus `GEAK_GEOMEAN_SPEEDUP <x>`
#   * a JSON line {"exit": 0, "weighted": <x>}, where exit 0 means every gate passed
#     (correctness, negative controls, tail contracts) and `weighted` is the
#     serving-weighted speedup over the whole shape sweep;
#   * a JSON *document* {"ut_result": "run/ut_result.json", "ok": true, "gates": {...}} --
#     what the AgentX extraction harness emits. It is pretty-printed, so it has to be read
#     as a whole document, and the speedup is not in it at all: it lives in the referenced
#     ut_result.json under gates.timing.aggregate.primary (weighted, falling back to the
#     geomean). `ok` is the conjunction of the six frozen-harness gates.
#
# Accepting only the first two cost a real measurement on 2026-10-19: all six gates passed,
# the sweep returned a serving-weighted 0.95806 over 13/13 buckets, and the attempt was
# recorded as unmeasured. A regression is a result; "no measurement" is a broken harness.
# Reporting the second when the first happened teaches the optimizer nothing and spends
# the budget anyway.
default_measure() {
  local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../engine/campaign" && pwd)" || return 78
  ( timeout "$MEASURE_TIMEOUT" python3 "$here/run_task_entry.py" "$TASK_DIR" unittest.py 201>&- ) \
      > "$RUN/.measure.log" 2>&1
  local measure_rc=$?
  # This is transport metadata for retention, not a second interpretation of the UT.
  if [ -n "${MEASURE_NATIVE_RC_FILE:-}" ]; then
    printf '%s\n' "$measure_rc" > "$MEASURE_NATIVE_RC_FILE" || return 78
  fi
  if [ "$measure_rc" -ne 0 ]; then
    echo "STATUS=0 SPEEDUP= CLEAN= SINGLE=0"
    return
  fi
  local sp ok clean="" single=0
  sp="$(grep -oE 'GEAK_GEOMEAN_SPEEDUP [^[:space:]]+' "$RUN/.measure.log" | tail -1 | awk '{print $2}')"
  ok=0; grep -q '^CORRECTNESS: PASS' "$RUN/.measure.log" && ok=1
  if [ -z "$sp" ]; then
    # A non-whitespace delimiter preserves missing score fields. Plain read would
    # shift the trailing single-pair count (often zero) into an empty speedup.
    IFS='|' read -r ok sp clean single <<<"$(python3 - "$RUN/.measure.log" "$TASK_DIR" <<'PY'
import json, math, os, sys

log, task_dir = sys.argv[1], sys.argv[2]
text = open(log, errors="replace").read()


def positive_score(value, allow_text=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, str) if allow_text else (int, float)):
        return None
    try:
        value = float(value)
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def docs(s):
    """Every JSON object in the log, pretty-printed ones included.

    The harness writes its summary with indent=2, so a line-wise reader sees only "{".
    Scanning with raw_decode from each "{" at the start of a line costs nothing and
    handles both shapes.
    """
    dec = json.JSONDecoder()
    i = 0
    while True:
        i = s.find("{", i)
        if i < 0:
            return
        if i and s[i - 1] not in "\n\r":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(s, i)
        except ValueError:
            i += 1
            continue
        yield obj
        i = end


def aggregate_speedup(path):
    """(primary, clean, single_pair_count) from an agentx_ut_result document.

    `primary` is the harness's own serving-weighted figure. `clean` re-weights the same
    way over the buckets it measured more than once: the harness stops repeating a bucket
    whose first pair falls outside its undecided band, so a single-pair bucket is one
    unrepeated sample and per-pair noise on a collective seam reaches 12%.
    """
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return None, None, 0
    if not isinstance(d, dict):
        return None, None, 0
    gates = d.get("gates") or {}
    timing = (gates.get("timing") or {}) if isinstance(gates, dict) else {}
    if not isinstance(timing, dict):
        return None, None, 0
    agg = timing.get("aggregate")
    if agg is None and not timing:
        # Some frozen tasks keep the shared harness's aggregate at the top level.
        # Read that result; the caller still requires the task's own gate verdict.
        agg = d.get("serving_weighted_speedup")
        if isinstance(agg, dict):
            timing = dict(timing, per_case=d.get("per_case") or [])
    if not isinstance(agg, dict):
        agg = {}
    primary = None
    for key in ("primary", "weighted", "geomean"):
        v = positive_score(agg.get(key))
        if v is not None:
            primary = v
            break
    if primary is None:
        # Interval form: the harness reports two serving aggregates because the prefix
        # history behind the dominant buckets was never recorded, and refuses to pick
        # one. Decide on the low end -- the high end would assert the model the harness
        # just said it cannot support.
        ends = [positive_score(v) for k, v in timing.items()
                if k.startswith("serving_weighted_speedup_")
                and positive_score(v) is not None]
        iv = timing.get("serving_weighted_speedup_interval")
        if isinstance(iv, (list, tuple)):
            ends += [positive_score(v) for v in iv if positive_score(v) is not None]
        if ends:
            primary = min(ends)
    cases = (timing.get("per_case") or [])
    rows = (timing.get("rows") or [])
    reps = {c.get("sig"): c.get("reps") for c in (cases or rows)}
    num = den = 0.0
    single = 0
    if not cases and rows:
        # The interval variant carries no weights, only measured rows. Count how many
        # rest on a single pair so the worker can say so; the aggregate stays the
        # harness's own.
        single = sum(1 for r in rows if (r.get("reps") or 0) < 2)
        return primary, None, single
    for c in (agg.get("per_case") or []):
        if not c.get("included"):
            continue
        w, sp = c.get("weight"), c.get("speedup")
        if not isinstance(w, (int, float)) or not isinstance(sp, (int, float)):
            continue
        if (reps.get(c.get("sig")) or 0) >= 2:
            num += w * sp
            den += w
        else:
            single += 1
    clean = (num / den) if den > 0 else None
    return primary, clean, single


ok, sp, clean, single = 0, "", "", 0
for d in docs(text):
    if not isinstance(d, dict):
        continue
    if "exit" in d:
        # A gate failure is not a measurement: report STATUS=0 and let the worker revert.
        ok = 1 if d.get("exit") == 0 else 0
        w = positive_score(d.get("weighted"), allow_text=True)
        sp = "" if w is None else repr(w)
        break
    if "gates" not in d and d and all(isinstance(v, bool) for v in d.values()):
        # Bare gate map: the pass/fail verdict is the document itself, and the numbers
        # are in the task's own ut_result.json. "UT PASS" in the log is the harness's
        # verdict; require the gates to agree with it.
        ok = 1 if (d.values() and all(d.values()) and "UT PASS" in text) else 0
        for cand in ("ut_result.json", os.path.join("run", "ut_result.json")):
            p = os.path.join(task_dir, cand)
            if os.path.exists(p):
                v, c, single = aggregate_speedup(p)
                if v is not None:
                    sp = repr(v)
                if c is not None:
                    clean = repr(c)
                break
        break
    if "ut_result" in d and "gates" in d:
        gates = d.get("gates") or {}
        # `ok` is the harness's own verdict; require the gate map to agree, so a future
        # gate the summary forgets to fold in cannot pass silently.
        ok = 1 if (d.get("ok") is True and all(v is True for v in gates.values())) else 0
        p = d["ut_result"]
        if not os.path.isabs(p):
            p = os.path.join(task_dir, p)
        v, c, single = aggregate_speedup(p)
        if v is not None:
            sp = repr(v)
        if c is not None:
            clean = repr(c)
        break
clean = positive_score(clean, allow_text=True)
print("|".join(map(str, (ok, sp, "" if clean is None else clean, single))))
PY
)"
  fi
  # A passing gate verdict without a finite positive score is no measurement.
  # Check the legacy text form too, including scientific notation and malformed tokens.
  if ! python3 - "$sp" <<'PY'
import math, sys
try:
    value = float(sys.argv[1])
    valid = math.isfinite(value) and value > 0
except (ValueError, OverflowError):
    valid = False
sys.exit(0 if valid else 1)
PY
  then
    ok=0
    sp=""
  fi
  echo "STATUS=${ok:-0} SPEEDUP=${sp:-} CLEAN=${clean:-} SINGLE=${single:-0}"
}


default_measure
