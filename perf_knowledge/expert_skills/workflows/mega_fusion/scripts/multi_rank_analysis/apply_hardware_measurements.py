#!/usr/bin/env python3
"""Apply measured TransferBench ceilings to a hardware-context artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from hardware import validate_hardware_context


_RAW_BY_METRIC = {
    "launch_overhead_us": "transferbench_empty.txt",
    "device_memory_gbps": "transferbench_hbm.txt",
    "all_to_all_interconnect_gbps": "transferbench_a2a.txt",
    "pairwise_interconnect_gbps": "transferbench_p2p.txt",
}


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
    parser.add_argument("--base-context", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = json.loads(args.base_context.read_text(encoding="utf-8"))
    measurements = json.loads(args.measurements.read_text(encoding="utf-8"))
    values = measurements["measurements"]
    timestamp = datetime.now(timezone.utc).isoformat()
    for metric, value in values.items():
        context.setdefault("measured", {})[metric] = value
        context.setdefault("provenance", {})[f"measured.{metric}"] = {
            "collector": measurements["tool"],
            "timestamp": timestamp,
            "confidence": "high",
            "raw_artifact": str(
                (
                    args.measurements.parent / _RAW_BY_METRIC[metric]
                ).resolve()
            ),
            "unit": measurements["units"][metric],
        }
    validated = validate_hardware_context(context)
    _write(args.output, validated)
    print(f"GEAK_HARDWARE_CONTEXT_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
