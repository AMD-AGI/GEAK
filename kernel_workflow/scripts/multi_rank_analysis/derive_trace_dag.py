#!/usr/bin/env python3
"""Build a kernel-level serialized DAG from normalized trace categories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intervals import critical_path


_ORDER = (
    "pre_dispatch_quant",
    "stage1_dispatch_gemm1",
    "stage2",
    "combine",
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    cases = {}
    for case in report["cases"]:
        trace = case.get("trace")
        if not trace:
            continue
        nodes = {
            category: trace["categories"][f"per_category_ms.{category}"]["rank_max"]
            for category in _ORDER
        }
        edges = [
            [_ORDER[index], _ORDER[index + 1]]
            for index in range(len(_ORDER) - 1)
        ]
        cases[case["case_id"]] = {
            "nodes": nodes,
            "edges": edges,
            "critical_path": critical_path(nodes, edges),
            "measured_pairwise_overlap_ms": {
                pair: max(
                    rank["pairwise"][pair]["overlap_ms"]
                    for rank in trace["per_rank_overlap"].values()
                )
                for pair in next(iter(trace["per_rank_overlap"].values()))[
                    "pairwise"
                ]
                if not pair.startswith("__support__")
            },
        }
    _write(
        args.output,
        {
            "schema_version": "geak-trace-dependency-dag-v1",
            "source": str(args.report.resolve()),
            "cases": cases,
            "scope_warning": (
                "Kernel-level trace DAG only; internal readiness edges and "
                "causal wait durations are not represented."
            ),
        },
    )
    print(f"GEAK_TRACE_DAG_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
