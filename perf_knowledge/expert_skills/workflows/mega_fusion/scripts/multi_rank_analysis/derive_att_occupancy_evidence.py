#!/usr/bin/env python3
"""Build scoped residency evidence from rocprofv3 ATT occupancy events."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from .instruction_analysis import read_att_occupancy
except ImportError:
    from instruction_analysis import read_att_occupancy


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return name, Path(raw_path)


def _named_integer(value: str) -> tuple[str, int]:
    name, separator, raw_value = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("expected STAGE=INTEGER")
    try:
        integer = int(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected STAGE=INTEGER") from error
    if integer <= 0:
        raise argparse.ArgumentTypeError("waves per workgroup must be positive")
    return name, integer


def _realtime_span_us(occupancy_path: Path, summary: dict) -> float | None:
    realtime_path = occupancy_path.with_name("realtime.json")
    if not realtime_path.exists():
        return None
    payload = json.loads(realtime_path.read_text(encoding="utf-8"))
    frequency = float(payload.get("metadata", {}).get("frequency") or 0)
    if frequency <= 0:
        return None
    converted = []
    for shader_engine, se_summary in summary["shader_engines"].items():
        points = payload.get(f"SE{shader_engine}")
        if not isinstance(points, list) or len(points) < 2:
            continue
        first, last = points[0], points[-1]
        gfx_delta = float(last[0]) - float(first[0])
        realtime_delta = float(last[1]) - float(first[1])
        if gfx_delta <= 0 or realtime_delta < 0:
            continue
        event_delta = float(se_summary["trace_span_cycles"])
        converted.append(event_delta * realtime_delta / gfx_delta / frequency * 1.0e6)
    return max(converted) if converted else None


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def derive(
    occupancy_paths: dict[str, Path],
    waves_per_workgroup: dict[str, int],
) -> dict:
    samples = {}
    for label, path in sorted(occupancy_paths.items()):
        route, separator, stage = label.partition(".")
        if not separator or not route or not stage:
            raise ValueError(f"{label}: label must be ROUTE.STAGE")
        if stage not in waves_per_workgroup:
            raise ValueError(f"{label}: missing waves-per-workgroup for {stage}")
        summary = read_att_occupancy(path)
        waves = waves_per_workgroup[stage]
        samples[label] = {
            "route": route,
            "stage": stage,
            "waves_per_workgroup": waves,
            "avg_resident_workgroup_equivalent_per_cu": (
                summary["avg_active_waves_per_cu"] / waves
            ),
            "peak_resident_workgroup_equivalent_per_cu": (
                summary["peak_active_waves_per_cu"] / waves
            ),
            "att_event_span_us": _realtime_span_us(path, summary),
            "occupancy": summary,
        }

    comparisons = {}
    stages = sorted({entry["stage"] for entry in samples.values()})
    for stage in stages:
        uniform = samples.get(f"uniform.{stage}")
        skew = samples.get(f"skew.{stage}")
        if not uniform or not skew:
            continue
        uniform_avg = uniform["occupancy"]["avg_active_waves_per_cu"]
        skew_avg = skew["occupancy"]["avg_active_waves_per_cu"]
        comparisons[stage] = {
            "uniform_avg_active_waves_per_cu": uniform_avg,
            "skew_avg_active_waves_per_cu": skew_avg,
            "skew_minus_uniform_active_waves_per_cu": skew_avg - uniform_avg,
            "uniform_att_event_span_us": uniform["att_event_span_us"],
            "skew_att_event_span_us": skew["att_event_span_us"],
        }
    return {
        "schema_version": "geak-att-occupancy-evidence-v1",
        "status": "complete_for_att_sampled_scope",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples": samples,
        "route_comparisons": comparisons,
        "scope": {
            "shader_engines": "SE0 only",
            "sampled_cus": 8,
            "device_cus": 256,
            "coverage_fraction": 8 / 256,
            "instruction_scope": "target CU differs from occupancy event scope",
        },
        "scope_warning": (
            "Resident-workgroup values are wave-count equivalents over sampled SE0; "
            "they are not exact CTA residency or whole-device sustained occupancy. "
            "ATT event spans include profiler perturbation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--occupancy",
        type=_named_path,
        action="append",
        required=True,
        metavar="ROUTE.STAGE=PATH",
    )
    parser.add_argument(
        "--waves-per-workgroup",
        type=_named_integer,
        action="append",
        required=True,
        metavar="STAGE=INTEGER",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    occupancy_paths = dict(args.occupancy)
    waves = dict(args.waves_per_workgroup)
    if len(occupancy_paths) != len(args.occupancy):
        raise ValueError("duplicate occupancy labels are not allowed")
    payload = derive(occupancy_paths, waves)
    _write(args.output, payload)
    print(f"GEAK_ATT_OCCUPANCY_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
