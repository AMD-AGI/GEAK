#!/usr/bin/env python3
"""Select exact-context measured implementation bundles from a merged registry."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def gfx_base(value):
    return str(value or "").split(":", 1)[0]


def is_correct(implementation):
    verdict = implementation.get("correctness")
    return isinstance(verdict, dict) and verdict.get("pass") is True


def matches(record, args):
    context = record.get("context") or {}
    implementation = record.get("implementation") or {}
    environment = context.get("environment") or {}
    shape = context.get("shape") or {}
    if not record.get("context_fingerprint"):
        return False
    if args.suite and context.get("suite") != args.suite:
        return False
    if args.dtype and context.get("dtype") != args.dtype:
        return False
    if args.baseline and context.get("baseline") != args.baseline:
        return False
    if args.math_contract and context.get("math_contract") != args.math_contract:
        return False
    if args.context_fingerprint and record.get("context_fingerprint") != args.context_fingerprint:
        return False
    if args.gfx and gfx_base(environment.get("gfx")) != gfx_base(args.gfx):
        return False
    for axis in ("m", "n", "k"):
        value = getattr(args, axis)
        if value is not None and shape.get(axis) != value:
            return False
    language = implementation.get("language")
    if args.language and language != args.language:
        return False
    if args.flydsl_version and language == "flydsl":
        if environment.get("flydsl_version") != args.flydsl_version:
            return False
    for argument, field in (
        ("aiter_commit", "aiter_commit"),
        ("torch_version", "torch"),
        ("hip_version", "hip"),
    ):
        expected = getattr(args, argument)
        if expected and environment.get(field) != expected:
            return False
    speedup = implementation.get("speedup_vs_baseline")
    config = implementation.get("config")
    valid_speedup = (
        not isinstance(speedup, bool)
        and isinstance(speedup, (int, float))
        and math.isfinite(float(speedup))
        and speedup > 0
    )
    return (
        implementation.get("status") == "measured"
        and is_correct(implementation)
        and isinstance(config, dict)
        and config.get("identity_complete") is True
        and environment.get("aiter_dirty") is False
        and valid_speedup
    )


def select(registry, args):
    required_context = ("baseline", "aiter_commit", "torch_version", "hip_version")
    missing_query = [
        field for field in required_context
        if not getattr(args, field) and not args.context_fingerprint
    ]
    if missing_query:
        return {
            "schema_version": 1,
            "candidates": [],
            "matched": 0,
            "missing_query_context": missing_query,
            "contract": (
                "Pass --context-fingerprint or the baseline/AITER/Torch/HIP identity. "
                "A single stale registry context is not evidence of compatibility."
            ),
        }
    records = [
        record for record in registry.get("records") or [] if matches(record, args)
    ]
    contexts = sorted({record.get("context_fingerprint") for record in records})
    if len(contexts) > 1 and not args.context_fingerprint:
        return {
            "schema_version": 1,
            "candidates": [],
            "matched": len(records),
            "ambiguous_contexts": contexts,
            "contract": (
                "Multiple baseline/accounting/toolchain contexts match. Pass --context-fingerprint; "
                "speedups with different denominators are not comparable."
            ),
        }
    grouped = {}
    for record in records:
        implementation = record["implementation"]
        grouped.setdefault(implementation["implementation_id"], []).append(record)
    candidates = []
    for implementation_id, repeated in grouped.items():
        speedups = [
            float(row["implementation"]["speedup_vs_baseline"]) for row in repeated
        ]
        representative = dict(repeated[0])
        representative["implementation"] = dict(representative["implementation"])
        representative["implementation"]["measured_speedup"] = {
            "replicates": len(speedups),
            "median": statistics.median(speedups),
            "min": min(speedups),
            "max": max(speedups),
        }
        representative["replicate_measurement_ids"] = sorted(
            row["measurement_id"] for row in repeated
        )
        candidates.append(representative)
    candidates.sort(
        key=lambda row: (
            -float((row.get("implementation") or {}).get("measured_speedup", {}).get("median") or 0),
            str((row.get("implementation") or {}).get("implementation_id") or ""),
        )
    )
    return {
        "schema_version": 1,
        "candidates": candidates[:args.top_k],
        "matched": len(records),
        "distinct_implementations": len(candidates),
        "context_fingerprint": contexts[0] if len(contexts) == 1 else None,
        "contract": (
            "Exact-context whole-implementation bundles. Ranking is a historical prior; "
            "kernel_workflow must revalidate on the target task."
        ),
    }


def select_task(registry, args):
    """`select` for every timed case of a task directory, with its oracle as the suite.

    The suite, operand dtype and cases come from the task's `meta.json`, and the baseline from its
    frozen `baseline_src/`, the way `ingest_workflow_products.py` records them; the caller still
    states the environment identity, as for a single-shape query.
    """
    import sys
    corpus = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(corpus))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _select_candidates as SELECT
    from ingest_workflow_products import baseline_identity

    task = Path(args.task_dir)
    meta = json.loads((task / "meta.json").read_text(encoding="utf-8"))
    base = argparse.Namespace(**vars(args))
    base.suite = args.suite or meta.get("short_name")
    base.dtype = args.dtype or SELECT.workload_dtype(meta)
    base.baseline = args.baseline or baseline_identity(task)
    cases = []
    for case in SELECT.workload_cases(meta):
        if not case["timed"]:
            continue
        query = argparse.Namespace(**vars(base))
        query.m, query.n, query.k = case["m"], case["n"], case["k"]
        cases.append({"sig": case["sig"], "m": case["m"], "n": case["n"], "k": case["k"],
                      **select(registry, query)})
    return {
        "schema_version": 1,
        "task": str(task),
        "suite": base.suite, "dtype": base.dtype, "baseline": base.baseline,
        "cases": cases,
        "contract": (
            "One exact-context query per timed case. A bundle ranked for one case says nothing about "
            "another; kernel_workflow must revalidate on the target task."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--task-dir", type=Path,
                        help="query every timed case of this task (meta.json + baseline_src/)")
    parser.add_argument("--suite")
    parser.add_argument("--language")
    parser.add_argument("--gfx")
    parser.add_argument("--flydsl-version")
    parser.add_argument("--dtype")
    parser.add_argument("--baseline")
    parser.add_argument("--math-contract")
    parser.add_argument("--aiter-commit")
    parser.add_argument("--torch-version")
    parser.add_argument("--hip-version")
    parser.add_argument("--context-fingerprint")
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported implementation registry schema")
    result = select_task(registry, args) if args.task_dir else select(registry, args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
