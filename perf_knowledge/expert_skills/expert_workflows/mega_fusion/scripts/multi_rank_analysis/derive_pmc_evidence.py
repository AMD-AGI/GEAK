#!/usr/bin/env python3
"""Summarize rocprofv3 per-dispatch GMI-sector and occupancy counters."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

GMI_COUNTERS = {
    "TCC_EA0_RDREQ_GMI_32B_sum",
    "TCC_EA0_WRREQ_WRITE_GMI_32B_sum",
    "TCC_EA0_WRREQ_ATOMIC_GMI_32B_sum",
}
OCCUPANCY_COUNTERS = {"MeanOccupancyPerCU", "OccupancyPercent"}


def _named_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def _category(kernel_name: str) -> str | None:
    if "megamoe_stage1" in kernel_name:
        return "stage1"
    if "megamoe_stage2" in kernel_name:
        return "stage2"
    if "ep_combine_intranode" in kernel_name:
        return "combine"
    return None


def _summary(values: list[float]) -> dict:
    return {
        "dispatches": len(values),
        "values": values,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values),
    }


def _dispatch_counters(path: Path) -> list[dict]:
    dispatches = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            category = _category(row["Kernel_Name"])
            if category is None:
                continue
            key = (row["Dispatch_Id"], row["Kernel_Name"])
            entry = dispatches.setdefault(
                key,
                {
                    "dispatch_id": int(row["Dispatch_Id"]),
                    "kernel_name": row["Kernel_Name"],
                    "category": category,
                    "start_timestamp": int(row["Start_Timestamp"]),
                    "end_timestamp": int(row["End_Timestamp"]),
                    "counters": {},
                },
            )
            entry["counters"][row["Counter_Name"]] = float(row["Counter_Value"])
    return sorted(dispatches.values(), key=lambda entry: entry["dispatch_id"])


def _summarize_gmi(path: Path) -> dict:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for dispatch in _dispatch_counters(path):
        present = GMI_COUNTERS.intersection(dispatch["counters"])
        if present != GMI_COUNTERS:
            raise ValueError(
                f"{path}: dispatch {dispatch['dispatch_id']} is missing GMI counters"
            )
        sectors = sum(dispatch["counters"][counter] for counter in GMI_COUNTERS)
        by_category[dispatch["category"]].append(
            {
                "dispatch_id": dispatch["dispatch_id"],
                "kernel_name": dispatch["kernel_name"],
                "gmi_32b_sectors": sectors,
                "gmi_sector_bytes": sectors * 32.0,
                "duration_ns": (
                    dispatch["end_timestamp"] - dispatch["start_timestamp"]
                ),
                "counters": dispatch["counters"],
            }
        )
    return {
        category: {
            "gmi_sector_bytes": _summary(
                [entry["gmi_sector_bytes"] for entry in entries]
            ),
            "dispatch_samples": entries,
        }
        for category, entries in sorted(by_category.items())
    }


def _summarize_occupancy(path: Path) -> dict:
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for dispatch in _dispatch_counters(path):
        present = OCCUPANCY_COUNTERS.intersection(dispatch["counters"])
        if present != OCCUPANCY_COUNTERS:
            raise ValueError(
                f"{path}: dispatch {dispatch['dispatch_id']} is missing occupancy counters"
            )
        for counter in OCCUPANCY_COUNTERS:
            values[dispatch["category"]][counter].append(
                dispatch["counters"][counter]
            )
    return {
        category: {
            counter: _summary(counter_values)
            for counter, counter_values in sorted(counters.items())
        }
        for category, counters in sorted(values.items())
    }


def derive(
    gmi_paths: dict[str, Path],
    occupancy_paths: dict[str, Path],
) -> dict:
    gmi = {
        label: {
            "source": str(path.resolve()),
            "stages": _summarize_gmi(path),
        }
        for label, path in sorted(gmi_paths.items())
    }
    occupancy = {
        label: {
            "source": str(path.resolve()),
            "stages": _summarize_occupancy(path),
        }
        for label, path in sorted(occupancy_paths.items())
    }
    controls = {}
    routes = sorted(
        {
            label.removesuffix(".normal").removesuffix(".no_payload")
            for label in gmi
        }
    )
    for route in routes:
        normal = gmi.get(f"{route}.normal")
        no_payload = gmi.get(f"{route}.no_payload")
        if not normal or not no_payload:
            continue
        normal_stage2 = normal["stages"]["stage2"]["gmi_sector_bytes"]["median"]
        control_stage2 = no_payload["stages"]["stage2"][
            "gmi_sector_bytes"
        ]["median"]
        controls[route] = {
            "normal_stage2_gmi_sector_bytes_rank0": normal_stage2,
            "no_payload_stage2_gmi_sector_bytes_rank0": control_stage2,
            "removed_stage2_gmi_sector_bytes_rank0": (
                normal_stage2 - control_stage2
            ),
            "removed_pct": (
                (normal_stage2 - control_stage2) / normal_stage2 * 100.0
                if normal_stage2
                else 0.0
            ),
        }
    return {
        "schema_version": "geak-rocprof-pmc-evidence-v1",
        "status": "complete_for_profiled_rank0_dispatch_scope",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gmi": gmi,
        "occupancy": occupancy,
        "matched_payload_controls": controls,
        "semantics": {
            "gmi_sector_bytes": (
                "32 * (TCC_EA0_RDREQ_GMI_32B_sum + "
                "TCC_EA0_WRREQ_WRITE_GMI_32B_sum + "
                "TCC_EA0_WRREQ_ATOMIC_GMI_32B_sum)"
            ),
            "gmi_scope": (
                "rank0 TCC/EA GMI 32-byte sectors per profiled kernel dispatch; "
                "not protocol-level wire bytes"
            ),
            "occupancy_scope": (
                "rank0 whole-device counter result per profiled dispatch; profiler "
                "perturbs latency and values are summarized with the median"
            ),
        },
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
    parser.add_argument(
        "--gmi", type=_named_path, action="append", default=[], metavar="LABEL=PATH"
    )
    parser.add_argument(
        "--occupancy",
        type=_named_path,
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gmi_paths = dict(args.gmi)
    occupancy_paths = dict(args.occupancy)
    if len(gmi_paths) != len(args.gmi) or len(occupancy_paths) != len(
        args.occupancy
    ):
        raise ValueError("duplicate PMC labels are not allowed")
    if not gmi_paths and not occupancy_paths:
        raise ValueError("at least one --gmi or --occupancy input is required")
    payload = derive(gmi_paths, occupancy_paths)
    _write(args.output, payload)
    print(f"GEAK_PMC_EVIDENCE_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
