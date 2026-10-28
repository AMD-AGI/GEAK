#!/usr/bin/env python3
"""Merge normalized benchmark JSON files into a deduplicated implementation registry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


CORPUS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_AXES = CORPUS_DIR / "performance_axes" / "gemm.yaml"
NORMALIZER_PATH = CORPUS_DIR / "_normalize_performance_decisions.py"
NORMALIZER_SPEC = importlib.util.spec_from_file_location(
    "normalize_performance_decisions", NORMALIZER_PATH,
)
NORMALIZER = importlib.util.module_from_spec(NORMALIZER_SPEC)
NORMALIZER_SPEC.loader.exec_module(NORMALIZER)


def stable_id(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "measure_" + hashlib.sha256(text.encode()).hexdigest()[:16]


def load_result(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("unsupported normalized benchmark result")
    if not isinstance(data.get("implementations"), list):
        raise ValueError("implementations must be a list")
    return data


def flatten(path: Path, result: dict[str, Any], axes: dict[str, Any]):
    legacy_run_id = "legacy_" + hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    context = {
        "run_id": result.get("run_id") or legacy_run_id,
        "seed": result.get("seed"),
        "suite": result.get("suite"),
        "math_contract": result.get("math_contract"),
        "shape": result.get("shape"),
        "dtype": result.get("dtype"),
        "baseline": result.get("baseline"),
        "accounting": result.get("accounting"),
        "environment": result.get("environment"),
    }
    comparable_context = {
        key: context[key] for key in (
            "suite", "math_contract", "shape", "dtype", "baseline", "accounting", "environment",
        )
    }
    context_fingerprint = stable_id(comparable_context).replace("measure_", "context_", 1)
    rows = []
    for raw_implementation in result["implementations"]:
        implementation = dict(raw_implementation)
        implementation["performance_decisions"] = NORMALIZER.normalize_implementation(
            implementation, axes,
        )
        identity = {
            "run_id": context["run_id"],
            "seed": context["seed"],
            "context_fingerprint": context_fingerprint,
            "implementation_id": implementation.get("implementation_id"),
        }
        rows.append({
            "measurement_id": stable_id(identity),
            "context_fingerprint": context_fingerprint,
            "source_result": str(path),
            "context": context,
            "implementation": implementation,
        })
    return rows


def merge(paths, axes_path=DEFAULT_AXES):
    axes = NORMALIZER.load_yaml(Path(axes_path))
    NORMALIZER.validate_schema(axes)
    records = {}
    errors = []
    for raw in paths:
        path = Path(raw).resolve()
        try:
            result = load_result(path)
            for row in flatten(path, result, axes):
                existing = records.get(row["measurement_id"])
                if existing and existing["implementation"] != row["implementation"]:
                    errors.append({
                        "path": str(path),
                        "error": f"measurement ID collision: {row['measurement_id']}",
                    })
                    continue
                records[row["measurement_id"]] = row
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    ordered = [records[key] for key in sorted(records)]
    return {
        "schema_version": 1,
        "records": ordered,
        "records_count": len(ordered),
        "errors": sorted(errors, key=lambda row: (row["path"], row["error"])),
        "contract": (
            "Whole-implementation measurements rank configuration bundles only. "
            "They do not assign causal speedup to individual corpus decisions."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--axes", type=Path, default=DEFAULT_AXES)
    args = parser.parse_args(argv)
    result = merge(args.input, args.axes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
