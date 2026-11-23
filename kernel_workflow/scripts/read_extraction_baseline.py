#!/usr/bin/env python3
"""Read a weighted oracle's extraction timings before a correctness run overwrites them.

Supports inline MOE/AR results and the WOA/HCB report/record sidecars. An unusable
result is a normal outcome: next_run=full tells setup to measure once instead.
"""

import argparse
import json
import math
from pathlib import Path


def _read(path):
    return json.loads(path.read_text())


def _case_id(case):
    return case.get("sig") or case.get("name")


def _positive(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def read_baseline(workspace):
    workspace = Path(workspace).resolve()
    try:
        meta = _read(workspace / "meta.json")
        cases = meta["workload"]["cases"]
        expected = [_case_id(c) for c in cases]
        if not expected or any(not isinstance(s, str) or not s for s in expected):
            raise ValueError("workload has no complete case identifiers")
        if len(set(expected)) != len(expected):
            raise ValueError("workload has duplicate case identifiers")
        result = _read(workspace / "run/ut_result.json")
        timing = result["gates"]["timing"]
        if result.get("exit_code") != 0 or timing.get("ok") is not True:
            raise ValueError("extraction run did not pass timing")
        if result.get("ok") is False or any(
            isinstance(g, dict) and g.get("ok") is False for g in result["gates"].values()
        ):
            raise ValueError("extraction run has a failed gate")

        # Try every declared source: some oracles retain an empty inline list
        # alongside a complete sidecar. Do not guess an unreferenced old file.
        sources = [("run/ut_result.json:gates.timing.per_case", timing.get("per_case"))]
        for key in ("report", "record"):
            ref = timing.get(key)
            if isinstance(ref, str) and ref:
                path = (workspace / ref).resolve()
                if not path.is_relative_to(workspace):
                    continue
                try:
                    report = _read(path)
                except (OSError, ValueError):
                    continue
                rows = report.get("per_case") if isinstance(report, dict) else report
                sources.append((str(path.relative_to(workspace)), rows))

        for source, rows in sources:
            if not isinstance(rows, list) or not rows:
                continue
            by_id = {}
            for row in rows:
                if not isinstance(row, dict):
                    break
                sig = _case_id(row)
                if (not isinstance(sig, str) or sig in by_id
                        or not _positive(row.get("baseline_ms"))
                        or not _positive(row.get("optimized_ms"))):
                    break
                by_id[sig] = row
            else:
                if set(expected) <= by_id.keys():
                    return {"usable": True, "next_run": "correctness", "source": source,
                            "per_case": [by_id[s] for s in expected]}
        raise ValueError("no declared timing source covers every workload case with positive finite timings")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {"usable": False, "next_run": "full", "reason": str(exc)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--out", help="save outside run/ before the oracle overwrites its outputs")
    args = ap.parse_args()
    text = json.dumps(read_baseline(args.workspace), indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
