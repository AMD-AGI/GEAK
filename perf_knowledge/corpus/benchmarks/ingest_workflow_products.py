#!/usr/bin/env python3
"""Turn validated kernel_workflow products into implementation-registry records.

A workflow run leaves one whole implementation measured by the task's own oracle, per case, in
the Director's fresh processes. This writes one normalized benchmark result per (run, case,
Director replicate) in the format `merge_results.py` reads, and merges them into a registry that
`select_implementations.py` queries - the same exact-context whole-implementation bundles the
upstream benchmark adapters produce, now for what the workflow itself built.

Context, so a later query can only match the same measurement:
  suite     - the task's `meta.json` `short_name` (its oracle);
  baseline  - `baseline_src:<sha256 of the frozen baseline files, 16 hex>`;
  shape     - the case's M, N, K; dtype - the task's operand dtype;
  accounting - how the oracle timed the case (CUDA-graph replay or eager);
  environment - gfx, FlyDSL version, AITER commit and cleanliness, Torch and HIP.

A case is `measured` - rankable - only when the run's ranking basis (as the decision-outcome
aggregator reads it) is `accepted`, or `graph_timed_lower_bound` for a case timed as a graph
replay. Every other case is kept as `not_rankable` with its reason, for audit.

    python3 perf_knowledge/corpus/benchmarks/ingest_workflow_products.py \\
        --runs /path/to/exp --output measured_products.json --torch-version 2.9.1 --hip-version 7.2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

CORPUS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CORPUS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _aggregate_decision_outcomes as AGG  # noqa: E402
import _select_candidates as SELECT  # noqa: E402
import merge_results as MERGE  # noqa: E402


def files_sha256(directory: Path) -> str | None:
    """Content hash of every source file directly in `directory`, independent of mtimes."""
    paths = sorted(p for p in directory.glob("*") if p.is_file() and p.suffix in (".py", ".json"))
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def baseline_identity(task_dir: Path) -> str | None:
    digest = files_sha256(task_dir / "baseline_src")
    return f"baseline_src:{digest}" if digest else None


def _first(*paths: Path) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def replicate_speedups(runs_file: Path | None, per_case: list[dict[str, Any]]) -> tuple[dict[str, list[float]], str]:
    """Per-case speedups from the Director's own oracle runs, else the manifest's single row."""
    if runs_file is not None:
        rows = [json.loads(line) for line in runs_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            fingerprint = rows[-1].get("fp")
            out: dict[str, list[float]] = {}
            for row in rows:
                if row.get("fp") != fingerprint:
                    continue
                for name, bucket in (row.get("buckets") or {}).items():
                    speedup = bucket.get("sp")
                    if isinstance(speedup, (int, float)) and speedup > 0:
                        out.setdefault(name, []).append(float(speedup))
            if out:
                return out, "oracle_runs"
    return ({str(row.get("name")): [float(row["speedup"])] for row in per_case
             if isinstance(row.get("speedup"), (int, float)) and row["speedup"] > 0}, "director_per_case")


def product_results(run: Path, manifest_path: Path, context: dict[str, Any],
                    environment_extra: dict[str, str]) -> list[dict[str, Any]]:
    manifest = AGG._load_manifest(manifest_path)
    task = _first(run / "workspace", run / "validation_workspace")
    meta_path = _first(run / "workspace" / "meta.json", run / "validation_workspace" / "meta.json")
    source = _first(run / "validation_workspace" / "kernel_src", run / "workspace" / "kernel_src")
    if task is None or meta_path is None or source is None:
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cases = {case["sig"]: case for case in SELECT.workload_cases(meta) if case["timed"]}
    graph = {str(case.get("name") or case.get("sig")): case.get("graph") is True
             for case in meta.get("cases") or [] if isinstance(case, dict)}
    validation = manifest.get("validation") or {}
    replicates, replicate_basis = replicate_speedups(
        _first(run / "validation_workspace" / "_runs.jsonl"), validation.get("per_case") or [])
    software = manifest.get("software_detail") or {}
    environment = {
        "gfx": manifest.get("gfx"),
        "flydsl_version": manifest.get("flydsl_version"),
        "aiter_commit": manifest.get("aiter_commit"),
        "aiter_dirty": software.get("aiter_dirty"),
        "torch": manifest.get("torch_version") or environment_extra.get("torch"),
        "hip": manifest.get("hip_version") or environment_extra.get("hip")
        or manifest.get("rocm_version"),
    }
    source_hash = files_sha256(source)
    correct = str(validation.get("correctness") or "").lower().startswith("pass")
    basis = context.get("ranking_basis")
    results = []
    for sig, case in cases.items():
        timer = "cuda_event_graph" if graph.get(sig) else "cuda_event"
        if not correct:
            status, reason = "not_rankable", "correctness did not pass"
        elif basis == "accepted" or (basis == "graph_timed_lower_bound" and graph.get(sig)):
            status, reason = "measured", basis
        else:
            status, reason = "not_rankable", f"ranking basis {basis}, timer {timer}"
        for index, speedup in enumerate(replicates.get(sig) or []):
            results.append({
                "schema_version": 1,
                "run_id": f"{context['run_id']}:{sig}:{index}",
                "seed": None,
                "suite": meta.get("short_name") or manifest.get("kernel"),
                "math_contract": (manifest.get("op_spec") or {}).get("math_contract"),
                "shape": {"m": case["m"], "n": case["n"], "k": case["k"]},
                "dtype": SELECT.workload_dtype(meta),
                "baseline": baseline_identity(task),
                "accounting": {"oracle": "task unittest.py, interleaved A/B vs the frozen baseline",
                               "timer": timer, "replicates": replicate_basis},
                "environment": environment,
                "implementations": [{
                    "implementation_id": f"{manifest.get('kernel')}:{source_hash}",
                    "name": manifest.get("kernel"),
                    "language": context.get("observed_language"),
                    "status": status,
                    "rankability": reason,
                    "correctness": {"pass": correct},
                    "speedup_vs_baseline": speedup,
                    "config": {
                        "identity_complete": source_hash is not None,
                        "source_sha256": source_hash,
                        "product_source": str(source),
                        "final_patch": str(run / "final_patch.diff"),
                        "case": sig,
                    },
                }],
            })
    return results


def ingest(run_paths: list[str], results_dir: Path, assumed_languages: dict[str, str] | None = None,
           environment_extra: dict[str, str] | None = None) -> dict[str, Any]:
    aggregated = AGG.aggregate_decision_outcomes(run_paths, assumed_languages)
    contexts = {}
    for key in ("single_ref_direction_results", "single_ref_seed_results", "bundled_direction_results"):
        for row in aggregated[key]:
            contexts.setdefault(row["run"], row["context"])
    manifests, errors = AGG.discover_manifests(run_paths)
    results_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for manifest_path in manifests:
        run = manifest_path.parent.resolve()
        context = contexts.get(str(run))
        if context is None:
            manifest = AGG._load_manifest(manifest_path)
            context = AGG._context(manifest, run)
            context["run_id"] = AGG._run_identity(manifest)
            language = (assumed_languages or {}).get(str(run))
            if context["observed_language"] is None and language:
                context["observed_language"] = language
        for result in product_results(run, manifest_path, context, environment_extra or {}):
            path = results_dir / (result["run_id"].replace(":", "_") + ".json")
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            written.append(path)
    registry = MERGE.merge(written)
    registry["errors"] = sorted(errors + registry["errors"], key=lambda row: (row["path"], row["error"]))
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", action="append", required=True, metavar="PATH")
    parser.add_argument("--output", required=True, type=Path, help="merged registry JSON")
    parser.add_argument("--results-dir", type=Path, help="where the per-case results go "
                        "(default: <output stem>_results next to the output)")
    parser.add_argument("--assume-language", action="append", default=[], metavar="RUN=LANGUAGE")
    parser.add_argument("--torch-version", help="for runs whose manifest did not record Torch")
    parser.add_argument("--hip-version", help="for runs whose manifest did not record HIP")
    args = parser.parse_args(argv)
    assumed = {}
    for spec in args.assume_language:
        path, sep, language = spec.rpartition("=")
        if not sep or not path or not language:
            parser.error(f"--assume-language expects RUN=LANGUAGE, got {spec!r}")
        assumed[str(Path(path).expanduser().resolve())] = language
    results_dir = args.results_dir or args.output.with_name(args.output.stem + "_results")
    extra = {key: value for key, value in (("torch", args.torch_version), ("hip", args.hip_version)) if value}
    registry = ingest(args.runs, results_dir, assumed, extra)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not registry["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
