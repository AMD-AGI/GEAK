#!/usr/bin/env python3
"""Build complete, immutable CSV dispatch tables for per-process environment selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shlex
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path


def _parse_table(content: bytes) -> tuple[list[str], list[dict[str, str]]]:
    with io.StringIO(content.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("Missing or duplicate CSV columns")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV row width differs from its header")
    return columns, rows


def _index(rows: list[dict[str, str]], keys: list[str]) -> dict[tuple[str, ...], dict[str, str]]:
    indexed = {}
    for row in rows:
        key = tuple(_dispatch_key(row[column]) for column in keys)
        if any(not value for value in key) or key in indexed:
            raise ValueError(f"Missing or duplicate dispatch key: {key}")
        indexed[key] = row
    return indexed


def _dispatch_key(value: str) -> str:
    """Match pandas' numeric dispatch keys across integer/float CSV spellings."""
    value = value.strip()
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    if not number.is_finite():
        raise ValueError(f"Non-finite dispatch key: {value}")
    return "0" if number == 0 else str(number.normalize())


def build_runtime_csv(
    baseline: Path, tuned: list[Path], output: Path, env_name: str, keys: list[str]
) -> dict:
    """Overlay explicitly keyed tuned rows onto an already resolved baseline table.

    The caller obtains the baseline from the installed library's effective table
    before tuning. Schema conversion and architecture inference belong to that
    library; differing CSV schemas are rejected here.
    """
    if not re.fullmatch(r"AITER_CONFIG_[A-Z0-9_]+", env_name):
        raise ValueError("An AITER_CONFIG environment variable is required")
    output = output.resolve()
    if os.pathsep in str(output):
        raise ValueError("AITER treats ':' as a file-list separator; the bundle path cannot contain it")
    baseline_bytes = baseline.read_bytes()
    columns, base_rows = _parse_table(baseline_bytes)
    if not keys or len(set(keys)) != len(keys) or not set(keys).issubset(columns):
        raise ValueError("Dispatch keys must be distinct columns of the baseline CSV")
    merged = _index(base_rows, keys)
    inputs = {str(baseline.resolve()): hashlib.sha256(baseline_bytes).hexdigest()}
    replacements = {}
    for path in tuned:
        tuned_bytes = path.read_bytes()
        tuned_columns, rows = _parse_table(tuned_bytes)
        if set(tuned_columns) != set(columns):
            raise ValueError(f"Tuned CSV schema differs from the effective baseline: {path}")
        additions = _index(rows, keys)
        if set(replacements).intersection(additions):
            raise ValueError("Multiple tuned files provide the same dispatch key")
        replacements.update(additions)
        inputs[str(path.resolve())] = hashlib.sha256(tuned_bytes).hexdigest()
    if not replacements:
        raise ValueError("At least one tuned dispatch row is required")
    replaced = len(set(merged).intersection(replacements))
    merged.update(replacements)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(merged.values())
    candidate_bytes = stream.getvalue().encode("utf-8")
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in inputs.items()):
        raise ValueError("A CSV input changed while the runtime table was being assembled")
    output.mkdir(parents=True, exist_ok=False)
    candidate_path = output / "candidate.csv"
    baseline_path = output / "baseline.csv"
    for path, content in ((candidate_path, candidate_bytes), (baseline_path, baseline_bytes)):
        path.write_bytes(content)
        path.chmod(0o444)
    result = {
        "schema_version": 1,
        "env_name": env_name,
        "keys": keys,
        "columns": columns,
        "baseline": {"path": str(baseline_path), "sha256": hashlib.sha256(baseline_bytes).hexdigest(), "rows": len(base_rows)},
        "candidate": {"path": str(candidate_path), "sha256": hashlib.sha256(candidate_bytes).hexdigest(), "rows": len(merged)},
        "tuned_rows": len(replacements),
        "replaced_rows": replaced,
        "input_sha256": inputs,
        "baseline_env": shlex.join([f"{env_name}={baseline_path}"]),
        "apply_env": shlex.join([f"{env_name}={candidate_path}"]),
        "live_tree_files": [],
        "cache_invalidation": [],
    }
    (output / "runtime_csv.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def verify_runtime_csv(manifest: Path, eval_dir: Path) -> dict:
    """Verify that both complete tables still belong to the persisted run bundle."""
    result = json.loads(manifest.read_text(encoding="utf-8"))
    if result.get("schema_version") != 1 or not re.fullmatch(r"AITER_CONFIG_[A-Z0-9_]+", result.get("env_name", "")):
        raise ValueError("Invalid runtime CSV manifest")
    root = eval_dir.resolve()
    keys = result.get("keys") or []
    if not keys or len(set(keys)) != len(keys) or not set(keys).issubset(result["columns"]):
        raise ValueError("Invalid runtime dispatch keys")
    if not manifest.resolve().is_relative_to(root):
        raise ValueError("Runtime CSV manifest is outside the persisted run")
    for arm in ("baseline", "candidate"):
        entry = result[arm]
        path = Path(entry["path"]).resolve()
        if not path.is_relative_to(root) or os.pathsep in str(path):
            raise ValueError("Runtime CSV is outside the persisted run or contains a file-list separator")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Runtime {arm} CSV changed")
        columns, rows = _parse_table(content)
        if columns != result["columns"] or len(rows) != entry["rows"]:
            raise ValueError(f"Runtime {arm} CSV schema or row count changed")
        _index(rows, result["keys"])
    return result


def verify_runtime_tuning(tuning: dict, eval_dir: Path, env_map: dict | None = None) -> list[dict]:
    """Bind declared runtime tables to the environment used by the candidate."""
    manifests = tuning.get("runtime_csv_manifests") or []
    if not manifests:
        installed = tuning.get("live_tree_files") or []
        caches = tuning.get("cache_invalidation") or []
        if any("/aiter/configs/" in "/" + str(path).lstrip("/") for path in installed) or any(
            "aiter_configs" in str(command) for command in caches
        ):
            raise ValueError("Accepted AITER tuning requires complete per-process runtime CSVs; installed tables contaminate later baselines")
        return []
    if tuning.get("live_tree_files") or tuning.get("cache_invalidation"):
        raise ValueError("Runtime CSV tuning cannot also mutate installed files or shared caches")
    if env_map is None:
        env_map = dict(token.split("=", 1) for token in shlex.split(tuning.get("apply_env") or "") if "=" in token)
    tables = []
    selectors = set()
    for manifest in manifests:
        table = verify_runtime_csv(Path(manifest), eval_dir)
        name = table["env_name"]
        if name in selectors:
            raise ValueError("Multiple runtime CSV manifests select the same environment variable")
        if env_map.get(name) != table["candidate"]["path"]:
            raise ValueError("Accepted environment does not select the verified runtime CSV")
        selectors.add(name)
        tables.append(table)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-tuning", action="store_true", help="Verify tuning JSON from stdin")
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--tuned", action="append", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-name")
    parser.add_argument("--keys", help="Comma-separated runtime dispatch-key columns")
    args = parser.parse_args()
    if args.verify_tuning:
        if args.eval_dir is None:
            parser.error("--verify-tuning requires --eval-dir")
        print(json.dumps(verify_runtime_tuning(json.load(sys.stdin), args.eval_dir)))
        return
    if not all((args.baseline, args.tuned, args.output, args.env_name, args.keys)):
        parser.error("building requires --baseline, --tuned, --output, --env-name and --keys")
    print(json.dumps(build_runtime_csv(args.baseline, args.tuned, args.output, args.env_name, args.keys.split(","))))


if __name__ == "__main__":
    main()
