#!/usr/bin/env python3
"""Package one native TL CSV row and its per-kernel call-stack context.

This is a presentation adapter, not a roofline calculator. All original CSV
fields survive in native_row, including strings and unavailable metrics.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path


class _ScalarLiterals(ast.NodeTransformer):
    """Unwrap serialized NumPy scalar reprs without executing any calls."""

    scalar_names = {
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64", "bool_",
    }

    def visit_Call(self, node):
        fn = node.func
        if (not isinstance(fn, ast.Attribute) or not isinstance(fn.value, ast.Name)
                or fn.value.id not in {"np", "numpy"} or fn.attr not in self.scalar_names
                or len(node.args) != 1 or node.keywords):
            raise ValueError("unsupported expression in native cell")
        number = ast.literal_eval(node.args[0])
        if not isinstance(number, (int, float, bool)):
            raise ValueError("NumPy scalar payload is not a numeric literal")
        return ast.copy_location(ast.Constant(number), node)


def parse_cell(value):
    """Read supported native structured cells without evaluating code."""
    if not isinstance(value, str):
        return value
    if not value.strip():
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(_ScalarLiterals().visit(ast.parse(value, mode="eval")))
        except (ValueError, SyntaxError, TypeError):
            return value


def stack_chains(value):
    """Expand TL's flat prefix plus nested branch tails into complete chains."""
    value = parse_cell(value)
    if not isinstance(value, list) or not value:
        return []
    prefix = []
    branches = []
    branched = False
    for item in value:
        if isinstance(item, str) and not branched:
            prefix.append(item)
        elif isinstance(item, list) and item and all(isinstance(x, str) for x in item):
            branched = True
            branches.append(prefix + item)
        else:
            return []  # Unknown schema: keep raw data and avoid a guessed match.
    return branches or ([prefix] if prefix else [])


def package_row(row, *, source, row_index, csv_sha256, tl_commit=None):
    required = {"name", "perf_params", "kernel_details_summary"}
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"missing native columns: {', '.join(missing)}")
    details = parse_cell(row["kernel_details_summary"])
    warnings = []
    if not isinstance(details, list) or any(not isinstance(x, dict) for x in details):
        warnings.append("kernel_details_summary is not a supported list of entries; inspect native_row")
        entries = []
    else:
        entries = details
    full = parse_cell(row.get("call_stack_full"))
    chains = stack_chains(full)
    kernels = []
    for index, detail in enumerate(entries):
        name = detail.get("name")
        embedded = parse_cell(detail.get("call_stack"))
        candidates = []
        origin = None
        if isinstance(embedded, list) and embedded and all(isinstance(x, str) for x in embedded):
            candidates = [embedded if embedded[-1] == name else embedded + [name]] if name else [embedded]
            origin = "kernel_details_summary.call_stack"
        elif name:
            candidates = [chain for chain in chains if chain and chain[-1] == name]
            origin = "call_stack_full" if candidates else None
        candidates = [list(x) for x in dict.fromkeys(tuple(x) for x in candidates)]
        status = "matched" if len(candidates) == 1 else "ambiguous" if candidates else "missing"
        kernels.append({
            "detail_index": index,
            "name": name,
            "detail": detail,
            "call_stack": candidates[0] if status == "matched" else None,
            "call_stack_candidates": candidates,
            "call_stack_status": status,
            "call_stack_source": origin,
        })
        if status != "matched":
            warnings.append(f"kernel entry {index} ({name}): {status} call stack")
    if not row["name"]:
        warnings.append("operation name is empty")
    if not row["perf_params"]:
        warnings.append("perf_params is empty; do not infer model coverage from the device name")
    return {
        "source": {"csv": str(source), "row_index": row_index,
                   "csv_sha256": csv_sha256, "tl_commit": tl_commit},
        "name": row["name"],
        "perf_params": parse_cell(row["perf_params"]),
        "kernel_details_summary": details,
        "call_stack_full": full,
        "kernels": kernels,
        "warnings": warnings,
        "native_row": row,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--row-index", required=True, type=int, help="Zero-based data record index")
    parser.add_argument("--tl-commit", help="Exact report-generating TL commit, when known")
    parser.add_argument("--out", type=Path, help="Default: stdout")
    args = parser.parse_args()
    if args.row_index < 0:
        parser.error("--row-index must be nonnegative")
    if args.out and args.out.resolve() == args.csv.resolve():
        parser.error("--out must not overwrite the native CSV")
    csv.field_size_limit(sys.maxsize)
    try:
        digest = hashlib.sha256(args.csv.read_bytes()).hexdigest()
        with args.csv.open(newline="", encoding="utf-8-sig") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                if index == args.row_index:
                    if None in row:
                        raise ValueError("malformed CSV row has extra unnamed cells")
                    result = package_row(row, source=args.csv.resolve(), row_index=index,
                                         csv_sha256=digest, tl_commit=args.tl_commit)
                    break
            else:
                raise ValueError(f"no data record at index {args.row_index}")
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    for warning in result["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
