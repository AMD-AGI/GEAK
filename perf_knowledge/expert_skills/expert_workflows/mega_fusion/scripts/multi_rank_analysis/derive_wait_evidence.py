#!/usr/bin/env python3
"""Derive ATT wait-cycle evidence and static synchronization-site inventory."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected STAGE=PATH")
    return name, Path(raw_path)


def _source_spec(value: str) -> tuple[str, tuple[Path, str | None]]:
    stage, path = _named_path(value)
    raw_path, marker, function_name = str(path).partition("#")
    return stage, (Path(raw_path), function_name if marker else None)


def _wait_kind(instruction: str) -> str:
    counters = [
        name
        for name in ("vmcnt", "lgkmcnt", "expcnt")
        if name in instruction
    ]
    if len(counters) == 1:
        return counters[0]
    if len(counters) > 1:
        return "mixed"
    return "unspecified"


def _att_summary(path: Path) -> dict:
    total_accounted = 0
    total_latency = 0
    total_stall = 0
    total_idle = 0
    wait_by_kind: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "hitcount": 0,
            "latency_cycles": 0,
            "stall_cycles": 0,
            "idle_cycles": 0,
            "static_instruction_count": 0,
        }
    )
    barrier = {
        "hitcount": 0,
        "latency_cycles": 0,
        "stall_cycles": 0,
        "idle_cycles": 0,
        "static_instruction_count": 0,
    }
    wait_instructions = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            instruction = row["Instruction"].strip()
            hitcount = int(row["Hitcount"])
            latency = int(row["Latency"])
            stall = int(row["Stall"])
            idle = int(row["Idle"])
            total_accounted += latency + idle
            total_latency += latency
            total_stall += stall
            total_idle += idle
            if instruction.startswith("s_waitcnt"):
                kind = _wait_kind(instruction)
                bucket = wait_by_kind[kind]
                bucket["hitcount"] += hitcount
                bucket["latency_cycles"] += latency
                bucket["stall_cycles"] += stall
                bucket["idle_cycles"] += idle
                bucket["static_instruction_count"] += 1
                wait_instructions.append(
                    {
                        "vaddr": int(row["Vaddr"]),
                        "instruction": instruction,
                        "kind": kind,
                        "hitcount": hitcount,
                        "latency_cycles": latency,
                        "stall_cycles": stall,
                        "idle_cycles": idle,
                    }
                )
            elif instruction.startswith("s_barrier"):
                barrier["hitcount"] += hitcount
                barrier["latency_cycles"] += latency
                barrier["stall_cycles"] += stall
                barrier["idle_cycles"] += idle
                barrier["static_instruction_count"] += 1

    wait_totals = {
        key: sum(bucket[key] for bucket in wait_by_kind.values())
        for key in (
            "hitcount",
            "latency_cycles",
            "stall_cycles",
            "idle_cycles",
            "static_instruction_count",
        )
    }
    wait_accounted = wait_totals["latency_cycles"] + wait_totals["idle_cycles"]
    barrier_accounted = barrier["latency_cycles"] + barrier["idle_cycles"]
    sync_accounted = wait_accounted + barrier_accounted
    wait_instructions.sort(
        key=lambda item: item["stall_cycles"] + item["idle_cycles"], reverse=True
    )
    return {
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_totals": {
            "accounted_cycles_latency_plus_idle": total_accounted,
            "latency_cycles": total_latency,
            "stall_cycles": total_stall,
            "idle_cycles": total_idle,
        },
        "waitcnt": {
            **wait_totals,
            "accounted_cycle_share_pct": (
                wait_accounted / total_accounted * 100.0 if total_accounted else 0.0
            ),
            "by_counter": dict(wait_by_kind),
        },
        "barrier": {
            **barrier,
            "accounted_cycle_share_pct": (
                barrier_accounted / total_accounted * 100.0
                if total_accounted
                else 0.0
            ),
        },
        "sync_total": {
            "accounted_cycles": sync_accounted,
            "accounted_cycle_share_pct": (
                sync_accounted / total_accounted * 100.0
                if total_accounted
                else 0.0
            ),
        },
        "top_wait_instructions": wait_instructions[:12],
    }


def _call_name(node: ast.Call) -> str:
    parts = []
    current = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _find_function(tree: ast.AST, name: str) -> ast.AST:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one function named {name!r}, found {len(matches)}")
    return matches[0]


def _static_sites(path: Path, function_name: str | None) -> dict:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    scope = _find_function(tree, function_name) if function_name else tree
    lines = text.splitlines()
    recognized_suffixes = (
        ".barrier",
        ".sched_barrier",
        ".s_waitcnt",
        ".int32_wait_until_equals",
        ".int32_wait_until_greater_than",
        ".uint64_wait_until_equals",
        ".uint64_wait_until_greater_than",
    )
    sites = []
    for node in ast.walk(scope):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name == "barrier" or name.endswith(recognized_suffixes):
            if name.endswith("sched_barrier"):
                kind = "scheduler_fence"
            elif name.endswith("s_waitcnt"):
                kind = "explicit_waitcnt"
            elif "wait_until" in name:
                kind = "cross_rank_signal_wait"
            else:
                kind = "workgroup_barrier"
            sites.append(
                {
                    "line": node.lineno,
                    "kind": kind,
                    "call": name,
                    "source": lines[node.lineno - 1].strip(),
                }
            )
    sites.sort(key=lambda item: item["line"])
    counts: dict[str, int] = defaultdict(int)
    for site in sites:
        counts[site["kind"]] += 1
    return {
        "path": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "function_scope": function_name,
        "site_counts": dict(counts),
        "sites": sites,
        "scope_warning": (
            "Static sites include compile-time branches and are not dynamic execution counts."
        ),
    }


def derive(
    att_paths: dict[str, Path],
    source_specs: dict[str, tuple[Path, str | None]],
) -> dict:
    if set(att_paths) != set(source_specs):
        raise ValueError("ATT and source stage sets must match")
    stages = {}
    for stage in sorted(att_paths):
        stages[stage] = {
            "att_sample": _att_summary(att_paths[stage]),
            "static_sync_inventory": _static_sites(*source_specs[stage]),
        }
    ranked = sorted(
        (
            {
                "stage": stage,
                "sync_accounted_cycle_share_pct": evidence["att_sample"][
                    "sync_total"
                ]["accounted_cycle_share_pct"],
                "waitcnt_accounted_cycle_share_pct": evidence["att_sample"][
                    "waitcnt"
                ]["accounted_cycle_share_pct"],
                "barrier_accounted_cycle_share_pct": evidence["att_sample"][
                    "barrier"
                ]["accounted_cycle_share_pct"],
            }
            for stage, evidence in stages.items()
        ),
        key=lambda item: item["sync_accounted_cycle_share_pct"],
        reverse=True,
    )
    return {
        "schema_version": "geak-wait-evidence-v1",
        "status": "complete_for_att_sampled_operator_scope",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "semantics": {
            "accounted_cycles": "ATT Latency + Idle; Stall is included in Latency",
            "scope": "one sampled GPU/CU trace per isolated Stage1, Stage2, and Combine capture",
            "not_equivalent_to": [
                "whole-device elapsed cycles",
                "cross-rank wait time",
                "E2E latency contribution",
            ],
        },
        "stage_ranking": ranked,
        "stages": stages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--att", type=_named_path, action="append", required=True, metavar="STAGE=CSV"
    )
    parser.add_argument(
        "--source",
        type=_source_spec,
        action="append",
        required=True,
        metavar="STAGE=PY[#FUNCTION]",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    att_paths = dict(args.att)
    source_specs = dict(args.source)
    if len(att_paths) != len(args.att) or len(source_specs) != len(args.source):
        raise ValueError("duplicate stage names are not allowed")
    payload = derive(att_paths, source_specs)
    _write(args.output, payload)
    print(f"GEAK_WAIT_EVIDENCE_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
