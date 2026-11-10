"""Parse rocprofv3 ATT per-instruction statistics into evidence categories."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping

ATT_COLUMNS = ("Instruction", "Hitcount", "Latency", "Stall", "Idle")
ATT_OCCUPANCY_SCHEMA_VERSION = "geak-att-occupancy-summary-v1"
ATT_OCCUPANCY_V3_FIELDS = (
    "time",
    "cu",
    "simd",
    "wave_id",
    "start",
    "kernel_id",
)

__all__ = [
    "ATT_COLUMNS",
    "ATT_OCCUPANCY_SCHEMA_VERSION",
    "load_instruction_category_map",
    "parse_att_stats_csv",
    "read_att_occupancy",
]


def load_instruction_category_map(path_or_dict) -> dict:
    if isinstance(path_or_dict, Mapping):
        raw = dict(path_or_dict)
    else:
        raw = json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
    compiled = {}
    for category, pattern in raw.items():
        compiled[str(category)] = re.compile(str(pattern))
    return compiled


def _instruction_category(instruction: str, category_map) -> tuple[str, str]:
    opcode = instruction.split()[0].lower() if instruction.strip() else "?"
    for category, pattern in category_map.items():
        if pattern.search(opcode):
            return category, opcode
    return "unclassified", opcode


def _integer(row: Mapping[str, str], field: str, line: int) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"ATT CSV line {line} has invalid {field}") from error
    if value < 0:
        raise ValueError(f"ATT CSV line {line} has negative {field}")
    return value


def parse_att_stats_csv(path: str | Path, category_map) -> dict:
    """Aggregate ATT Hitcount/Latency/Stall/Idle by instruction category."""
    source = Path(path)
    categories = defaultdict(
        lambda: {
            "static_instruction_count": 0,
            "hitcount": 0,
            "latency_cycles": 0,
            "stall_cycles": 0,
            "idle_cycles": 0,
            "opcodes": set(),
        }
    )
    instructions = []
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in ATT_COLUMNS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"ATT CSV is missing columns: {missing}")
        for line, row in enumerate(reader, start=2):
            instruction = (row.get("Instruction") or "").strip()
            if not instruction or instruction.startswith(";"):
                continue
            category, opcode = _instruction_category(instruction, category_map)
            values = {
                "hitcount": _integer(row, "Hitcount", line),
                "latency_cycles": _integer(row, "Latency", line),
                "stall_cycles": _integer(row, "Stall", line),
                "idle_cycles": _integer(row, "Idle", line),
            }
            if values["stall_cycles"] > values["latency_cycles"]:
                raise ValueError(
                    f"ATT CSV line {line} has Stall greater than Latency"
                )
            entry = categories[category]
            entry["static_instruction_count"] += 1
            entry["opcodes"].add(opcode)
            for key, value in values.items():
                entry[key] += value
            instructions.append(
                {
                    "instruction": instruction,
                    "opcode": opcode,
                    "category": category,
                    **values,
                    "stall_idle_cycles": values["stall_cycles"] + values["idle_cycles"],
                    "source": row.get("Source") or "",
                }
            )
    totals = {
        key: sum(entry[key] for entry in categories.values())
        for key in ("hitcount", "latency_cycles", "stall_cycles", "idle_cycles")
    }
    totals["issue_execute_cycles"] = (
        totals["latency_cycles"] - totals["stall_cycles"]
    )
    totals["accounted_cycles"] = (
        totals["latency_cycles"] + totals["idle_cycles"]
    )
    output_categories = {}
    for category, entry in categories.items():
        accounted_cycles = (
            entry["latency_cycles"] + entry["idle_cycles"]
        )
        output_categories[category] = {
            **{key: value for key, value in entry.items() if key != "opcodes"},
            "opcodes": sorted(entry["opcodes"]),
            "issue_execute_cycles": (
                entry["latency_cycles"] - entry["stall_cycles"]
            ),
            "accounted_cycle_share_pct": (
                accounted_cycles / totals["accounted_cycles"] * 100.0
                if totals["accounted_cycles"]
                else 0.0
            ),
            "stall_within_latency_pct": (
                entry["stall_cycles"] / entry["latency_cycles"] * 100.0
                if entry["latency_cycles"]
                else 0.0
            ),
            "idle_within_accounted_pct": (
                entry["idle_cycles"] / accounted_cycles * 100.0
                if accounted_cycles
                else 0.0
            ),
        }
    top = sorted(
        instructions,
        key=lambda item: item["stall_idle_cycles"],
        reverse=True,
    )[:20]
    for value in totals.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("ATT aggregate produced a non-finite value")
    return {
        "source": str(source),
        "categories": output_categories,
        "totals": totals,
        "top_stall_idle_instructions": top,
        "semantics": {
            "hitcount": "instruction issues observed on traced SIMD/waves",
            "latency_cycles": (
                "total latency; ROCm defines this as Stall + Issue on gfx9 "
                "or Stall + Execute on gfx10+"
            ),
            "stall_cycles": "cycles the hardware pipe could not issue due to backpressure",
            "idle_cycles": (
                "gap after the previous instruction caused by arbitration, "
                "register dependencies, or instruction-cache misses"
            ),
            "accounted_cycles": "Latency + Idle; Stall is not added again",
        },
        "scope_warning": (
            "ATT is sampled thread/wave evidence from selected GPU/CU/SIMDs; "
            "it is not whole-device time or cross-GPU traffic."
        ),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _occupancy_fields(payload: Mapping) -> tuple[str, ...]:
    fields = payload.get("occupancy_fields")
    if fields is not None:
        if not isinstance(fields, list) or not all(
            isinstance(field, str) for field in fields
        ):
            raise ValueError("ATT occupancy_fields must be a list of strings")
        required = set(ATT_OCCUPANCY_V3_FIELDS)
        if not required.issubset(fields):
            raise ValueError(
                "ATT occupancy_fields is missing required fields: "
                f"{sorted(required - set(fields))}"
            )
        return tuple(fields)
    if str(payload.get("version")) == "3.0.0":
        return ATT_OCCUPANCY_V3_FIELDS
    raise ValueError(
        "ATT occupancy JSON has an unknown event layout; expected version 3.0.0 "
        "or an occupancy_fields declaration"
    )


def _summarize_shader_engine(
    shader_engine: int,
    raw_events,
    fields: tuple[str, ...],
    *,
    max_waves_per_cu: int,
) -> dict:
    if not isinstance(raw_events, list):
        raise ValueError(f"ATT occupancy shader engine {shader_engine} must be a list")
    field_index = {field: index for index, field in enumerate(fields)}
    events = []
    malformed = []
    for event_index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, list) or len(raw_event) != len(fields):
            malformed.append(
                {
                    "event_index": event_index,
                    "reason": f"expected {len(fields)} values",
                }
            )
            continue
        try:
            event = {
                field: int(raw_event[index])
                for field, index in field_index.items()
            }
        except (TypeError, ValueError):
            malformed.append(
                {"event_index": event_index, "reason": "non-integer event value"}
            )
            continue
        if event["time"] < 0 or event["cu"] < 0 or event["simd"] < 0:
            malformed.append(
                {"event_index": event_index, "reason": "negative event coordinate"}
            )
            continue
        if event["start"] not in (0, 1):
            malformed.append(
                {"event_index": event_index, "reason": "start must be 0 or 1"}
            )
            continue
        events.append(event)

    events.sort(
        key=lambda event: (
            event["time"],
            event["start"],
            event["cu"],
            event["simd"],
            event["wave_id"],
        )
    )
    if not events:
        return {
            "shader_engine": shader_engine,
            "event_count": 0,
            "start_count": 0,
            "end_count": 0,
            "trace_span_cycles": 0,
            "sampled_cu_count": 0,
            "avg_active_waves_per_cu": 0.0,
            "peak_active_waves_per_cu": 0,
            "wave_slot_occupancy_pct": 0.0,
            "per_cu": {},
            "balanced_events": not malformed,
            "malformed_events": malformed,
        }

    cu_ids = sorted({event["cu"] for event in events})
    active_slots: dict[tuple[int, int, int], int] = {}
    active_by_cu = {cu: 0 for cu in cu_ids}
    wave_cycles_by_cu = {cu: 0 for cu in cu_ids}
    peak_by_cu = {cu: 0 for cu in cu_ids}
    start_count = 0
    end_count = 0
    first_time = events[0]["time"]
    previous_time = first_time
    cursor = 0
    while cursor < len(events):
        timestamp = events[cursor]["time"]
        delta = timestamp - previous_time
        if delta < 0:
            raise ValueError("ATT occupancy timestamps are not monotonic after sorting")
        for cu in cu_ids:
            wave_cycles_by_cu[cu] += active_by_cu[cu] * delta
        group_end = cursor
        while group_end < len(events) and events[group_end]["time"] == timestamp:
            group_end += 1
        group = events[cursor:group_end]
        for event in group:
            if event["start"]:
                continue
            end_count += 1
            slot = (event["cu"], event["simd"], event["wave_id"])
            kernel_id = active_slots.get(slot)
            if kernel_id is None:
                malformed.append(
                    {
                        "event_index": cursor,
                        "reason": "wave end without matching start",
                        "slot": list(slot),
                        "time": timestamp,
                    }
                )
                continue
            if kernel_id != event["kernel_id"]:
                malformed.append(
                    {
                        "event_index": cursor,
                        "reason": "wave end kernel_id mismatch",
                        "slot": list(slot),
                        "time": timestamp,
                    }
                )
            del active_slots[slot]
            active_by_cu[event["cu"]] -= 1
        for event in group:
            if not event["start"]:
                continue
            start_count += 1
            slot = (event["cu"], event["simd"], event["wave_id"])
            if slot in active_slots:
                malformed.append(
                    {
                        "event_index": cursor,
                        "reason": "wave start on an active slot",
                        "slot": list(slot),
                        "time": timestamp,
                    }
                )
                continue
            active_slots[slot] = event["kernel_id"]
            active_by_cu[event["cu"]] += 1
            peak_by_cu[event["cu"]] = max(
                peak_by_cu[event["cu"]], active_by_cu[event["cu"]]
            )
        previous_time = timestamp
        cursor = group_end

    for slot, kernel_id in sorted(active_slots.items()):
        malformed.append(
            {
                "reason": "wave start without matching end",
                "slot": list(slot),
                "kernel_id": kernel_id,
            }
        )
    last_time = events[-1]["time"]
    span = last_time - first_time
    per_cu = {}
    averages = []
    for cu in cu_ids:
        average = wave_cycles_by_cu[cu] / span if span else 0.0
        averages.append(average)
        per_cu[str(cu)] = {
            "avg_active_waves": average,
            "peak_active_waves": peak_by_cu[cu],
            "wave_cycles": wave_cycles_by_cu[cu],
        }
    average_per_cu = sum(averages) / len(averages) if averages else 0.0
    peak_per_cu = max(peak_by_cu.values(), default=0)
    return {
        "shader_engine": shader_engine,
        "event_count": len(events),
        "start_count": start_count,
        "end_count": end_count,
        "first_event_cycle": first_time,
        "last_event_cycle": last_time,
        "trace_span_cycles": span,
        "sampled_cu_count": len(cu_ids),
        "avg_active_waves_per_cu": average_per_cu,
        "p50_active_waves_per_cu": _percentile(averages, 0.50),
        "p95_active_waves_per_cu": _percentile(averages, 0.95),
        "peak_active_waves_per_cu": peak_per_cu,
        "wave_slot_occupancy_pct": (
            average_per_cu / max_waves_per_cu * 100.0
            if max_waves_per_cu
            else 0.0
        ),
        "per_cu": per_cu,
        "balanced_events": (
            start_count == end_count and not active_slots and not malformed
        ),
        "malformed_events": malformed,
    }


def read_att_occupancy(
    path: str | Path, *, max_waves_per_cu: int = 32
) -> dict:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("ATT occupancy JSON must be an object")
    if max_waves_per_cu <= 0:
        raise ValueError("max_waves_per_cu must be positive")
    fields = _occupancy_fields(payload)
    shader_engines = {}
    for key, value in payload.items():
        if not str(key).isdigit():
            continue
        shader_engine = int(key)
        shader_engines[str(shader_engine)] = _summarize_shader_engine(
            shader_engine,
            value,
            fields,
            max_waves_per_cu=max_waves_per_cu,
        )
    if not shader_engines:
        raise ValueError("ATT occupancy JSON has no shader-engine event arrays")
    total_wave_cycles = sum(
        sum(
            entry["wave_cycles"]
            for entry in summary["per_cu"].values()
        )
        for summary in shader_engines.values()
    )
    total_cu_cycles = sum(
        summary["trace_span_cycles"] * summary["sampled_cu_count"]
        for summary in shader_engines.values()
    )
    average = total_wave_cycles / total_cu_cycles if total_cu_cycles else 0.0
    return {
        "schema_version": ATT_OCCUPANCY_SCHEMA_VERSION,
        "source": str(source),
        "raw_version": str(payload.get("version") or ""),
        "event_fields": list(fields),
        "event_field_semantics": {
            "time": "shader-clock cycle timestamp",
            "cu": "CU identifier within the sampled shader engine",
            "simd": "SIMD identifier within the CU",
            "wave_id": "wave-slot identifier within the SIMD",
            "start": "1 starts a resident wave; 0 ends it",
            "kernel_id": "ATT symbol identifier mapped by dispatches",
        },
        "dispatches": dict(payload.get("dispatches") or {}),
        "shader_engines": shader_engines,
        "sampled_shader_engine_count": len(shader_engines),
        "sampled_cu_count": sum(
            summary["sampled_cu_count"] for summary in shader_engines.values()
        ),
        "avg_active_waves_per_cu": average,
        "peak_active_waves_per_cu": max(
            (
                summary["peak_active_waves_per_cu"]
                for summary in shader_engines.values()
            ),
            default=0,
        ),
        "wave_slot_occupancy_pct": average / max_waves_per_cu * 100.0,
        "balanced_events": all(
            summary["balanced_events"] for summary in shader_engines.values()
        ),
        "scope_warning": (
            "ATT occupancy covers only selected shader engines and perturbs execution; "
            "it is not whole-device sustained occupancy or unprofiled kernel latency."
        ),
    }
