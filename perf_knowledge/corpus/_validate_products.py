#!/usr/bin/env python3
"""Validate the GEMM corpus against the implementations the workflow actually produced.

A *product* is a workflow run directory (it holds `validation_environment.yaml`) or a plain task
directory (`meta.json` plus `kernel_src/`: a merged seam, a checked-in task). For each product:

- selection: every case of its `meta.json` classified by `_select_candidates.select_workload`, once
  as a planner starts (no traits) and once with the design traits the product's own source shows;
- citations (runs): every card the run cited, where the selector puts it for the product's context,
  and what the directions citing it reached. Citing a card the selector rejects for the product's own
  context is a contradiction between the planner and the corpus;
- coverage: the product's own code - the lines its final patch adds, or all of it when there is no
  patch - scanned with the extractor's rules, per category and per performance axis, against the
  evidence categories and axes of the cards that apply. A category the product writes that no
  applicable card cites is a gap in the corpus for this kind of kernel;
- measured readiness (runs): the run's ranking basis, the metric its direction scores are in and its
  flag reasons, read exactly as `_aggregate_decision_outcomes.py` reads them.

It measures nothing and writes nothing into the corpus; products and their numbers stay outside it.

    python3 perf_knowledge/corpus/_validate_products.py --runs /path/to/exp \\
        --task /path/to/merged_task --gfx gfx950 --json report.json --markdown report.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _aggregate_decision_outcomes as AGG  # noqa: E402
import _extract_impl_facts as EXTRACT  # noqa: E402
import _select_candidates as SELECT  # noqa: E402

_DETECT_PATH = HERE.parent.parent / "kernel_workflow" / "scripts" / "detect_language.py"
_DETECT_SPEC = importlib.util.spec_from_file_location("detect_language_for_products", _DETECT_PATH)
DETECT = importlib.util.module_from_spec(_DETECT_SPEC)
_DETECT_SPEC.loader.exec_module(DETECT)

CATALOG = HERE / "catalog.yaml"
SCALED_DTYPE = re.compile(r"^(?:fp8|int8|int4|fp4|mxfp)")
# What a design has to show in its own source before the trait-gated cards apply to it.
LDS_STAGING_CATEGORIES = frozenset({"lds_alloc", "lds_swizzle", "async_copy"})


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def corpus_index(catalog_path: Path = CATALOG, family_id: str = "gemm") -> dict[str, Any]:
    """Cards with the categories their evidence cites, the category-to-axis map, known shipped IDs."""
    catalog = _load_yaml(catalog_path)
    family = next(f for f in catalog["families"] if f["id"] == family_id)
    root = catalog_path.parent
    categories = {
        record["evidence_id"]: record["category"]
        for record in _load_yaml(root / family["source_evidence"]).get("source_evidence") or []
    }
    shipped = set()
    for key in ("tuned_evidence", "tuned_db_evidence"):
        if family.get(key):
            data = _load_yaml(root / family[key])
            shipped.update(row["config_id"] for row in data.get("tuned_configs") or [])
            shipped.update(row["selection_id"] for row in data.get("backend_selection") or [])
    axes_by_category: dict[str, set[str]] = {}
    for axis in _load_yaml(root / family["performance_axes"]).get("axes") or []:
        for mapping in axis.get("source_mappings") or []:
            for category in mapping.get("categories") or []:
                axes_by_category.setdefault(category, set()).add(axis["id"])
    cards = {}
    for card in _load_yaml(root / family["decisions"]).get("cards") or []:
        cited = list(card.get("source_evidence") or [])
        for solution in card.get("solutions") or []:
            cited.extend(solution.get("source_evidence") or [])
        cards[card["id"]] = {
            "type": card.get("type"),
            "axes": list(card.get("axes") or []),
            "categories": sorted({categories[i] for i in cited if i in categories}),
        }
    return {"cards": cards, "axes_by_category": axes_by_category, "shipped_ids": shipped}


def added_lines(patch_path: Path) -> dict[str, set[int]]:
    """New-file line numbers each file of a unified diff adds, keyed by file basename."""
    added: dict[str, set[int]] = {}
    current = None
    line = 0
    for text in patch_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if text.startswith("+++ "):
            target = text[4:].strip()
            current = None if target == "/dev/null" else Path(target).name
            continue
        hunk = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", text)
        if hunk:
            line = int(hunk.group(1))
            continue
        if current is None or text.startswith("--- "):
            continue
        if text.startswith("+"):
            added.setdefault(current, set()).add(line)
            line += 1
        elif text.startswith(" "):
            line += 1
    return added


def scan_product(source_dir: Path, language: str, patch: Path | None) -> dict[str, Any]:
    """Rule hits over the product's source, and the subset on lines its patch adds."""
    added = added_lines(patch) if patch and patch.is_file() else None
    every: Counter = Counter()
    introduced: Counter = Counter()
    examples: dict[str, str] = {}
    for path in sorted(source_dir.glob("*.py")):
        records = EXTRACT.scan_file(str(path), path.name, language)[0]
        lines = None if added is None else added.get(path.name, set())
        for record in records:
            every[record["category"]] += 1
            if lines is None or record["line"] in lines:
                introduced[record["category"]] += 1
                examples.setdefault(record["category"], f"{path.name}:{record['line']}")
    return {
        "source_dir": str(source_dir),
        "introduced_basis": "patch_added_lines" if added is not None else "whole_source",
        "categories": dict(sorted(every.items())),
        "introduced_categories": dict(sorted(introduced.items())),
        "introduced_examples": dict(sorted(examples.items())),
    }


def infer_traits(categories: dict[str, int]) -> list[str]:
    """Design traits the source demonstrates, in the vocabulary of `decisions/gemm.yaml`."""
    traits = []
    if LDS_STAGING_CATEGORIES & set(categories):
        traits.append("lds_staging")
        if {"split_k", "scheduling"} <= set(categories):
            traits.append("split_k_hot_loop")
    return traits


def operator_family(dtype: str | None) -> str:
    return "scaled_quant_gemm" if dtype and SCALED_DTYPE.match(dtype) else "dense_gemm"


def _applicable(workload: dict[str, Any]) -> set[str]:
    timed = [case for case in workload["cases"] if case.get("timed", True)] or workload["cases"]
    return {card for case in timed for card in case["eligible"]}


def classify_citation(card: str, workload: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    if card.startswith(("cfg_", "sel_")):
        return {"placement": "shipped_config" if card in index["shipped_ids"] else "unknown_shipped_id"}
    if card not in index["cards"]:
        return {"placement": "unknown_card"}
    timed = [case for case in workload["cases"] if case.get("timed", True)] or workload["cases"]
    eligible = [case["sig"] for case in timed if card in case["eligible"]]
    if len(eligible) == len(timed):
        return {"placement": "eligible_every_case"}
    if eligible:
        return {"placement": "eligible_some_cases", "cases": eligible}
    for bucket in ("deferred", "rejected"):
        reasons = sorted({reason for case in timed for row in case[bucket] if row["id"] == card
                          for reason in row["reasons"]})
        if reasons:
            return {"placement": bucket, "reasons": reasons}
    return {"placement": "unclassified"}


def citations(manifest: dict[str, Any], workload: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for row in manifest.get("decision_outcomes") or []:
        if not isinstance(row, dict) or not row.get("decision"):
            continue
        entry = rows.setdefault(str(row["decision"]), {"attempts": 0, "statuses": Counter(), "winner": 0})
        entry["attempts"] += 1
        entry["statuses"][str(row.get("status") or "not_recorded")] += 1
        entry["winner"] += int(bool(row.get("became_winner")))
    cited = {}
    for card, entry in sorted(rows.items()):
        cited[card] = {
            "attempts": entry["attempts"],
            "statuses": dict(sorted(entry["statuses"].items())),
            "became_winner": entry["winner"],
            **classify_citation(card, workload, index),
        }
    return {
        "cited": cited,
        "contradictions": sorted(card for card, row in cited.items()
                                 if row["placement"] in ("rejected", "unknown_card", "unknown_shipped_id")),
    }


def coverage(scan: dict[str, Any], workload: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    applicable = _applicable(workload)
    card_categories = {c for card in applicable for c in index["cards"][card]["categories"]}
    card_axes = {a for card in applicable for a in index["cards"][card]["axes"]}
    introduced = set(scan["introduced_categories"])
    product_axes = {a for c in introduced for a in index["axes_by_category"].get(c, ())}
    return {
        "applicable_cards": sorted(applicable),
        "product_axes": sorted(product_axes),
        "uncovered_categories": sorted(introduced - card_categories),
        "uncovered_axes": sorted(product_axes - card_axes),
        "unmapped_categories": sorted(c for c in introduced if c not in index["axes_by_category"]),
    }


def resolve_language(product: dict[str, Any], declared: str) -> tuple[str | None, str]:
    """The product's language, and how it is known.

    A run's manifest states it when the lane could decide. Otherwise the source is read with the
    lane's own detector; a verdict refused between several declaring languages is confirmed as the
    declared one only when the source declares it among them.
    """
    stated = product.get("language")
    if stated not in (None, "", "not_captured"):
        return str(stated), "manifest" if product["kind"] == "run" else "declared"
    detected = DETECT.detect([str(product["source_dir"])])
    if detected["language"]:
        return detected["language"], "detected"
    if declared in detected.get("ambiguous_between") or []:
        return declared, "declared_among_detected"
    return None, "undecided"


def readiness(manifest_path: Path, manifest: dict[str, Any],
              assumed_languages: dict[str, str] | None = None) -> dict[str, Any]:
    run = manifest_path.parent.resolve()
    aggregated = AGG.aggregate_decision_outcomes([str(manifest_path)], assumed_languages)
    results = aggregated["single_ref_direction_results"] + aggregated["bundled_direction_results"]
    if results:
        context = results[0]["context"]
    else:
        # A run that cited nothing still has a context; resolve it the way the aggregator would.
        context = AGG._context(manifest, run)
        assumed = {str(Path(path).expanduser().resolve()): language
                   for path, language in (assumed_languages or {}).items()}
        if context["observed_language"] is None and str(run) in assumed:
            context["observed_language"] = assumed[str(run)]
            context["observed_language_basis"] = "assumed_by_operator"
    return {
        "observed_language": context["observed_language"],
        "observed_language_basis": context["observed_language_basis"],
        "validation_status": context["validation_status"],
        "ranking_basis": context["ranking_basis"],
        "flag_reasons": context["flag_reasons"],
        "flag_evidence": context["flag_evidence"],
        "graph_timed": context["graph_timed"],
        "metric_kind": context["metric_kind"],
        "measurement_reliable": context["measurement_reliable"],
        "direction_results": len(results),
        "quality_eligible_results": sum(1 for r in results if r["quality_eligible_for_ranking"]),
        "single_ref_winners": sorted({d for r in aggregated["single_ref_direction_results"]
                                      if r["became_winner"] for d in r["decisions"]}),
    }


def _first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def validate_product(product: dict[str, Any], index: dict[str, Any], gfx: str | None,
                     family: str | None, catalog_path: Path = CATALOG, declared_language: str = "flydsl",
                     assumed_languages: dict[str, str] | None = None) -> dict[str, Any]:
    meta = json.loads(product["meta"].read_text(encoding="utf-8"))
    dtype = SELECT.workload_dtype(meta)
    language, language_basis = resolve_language(product, declared_language)
    scan = scan_product(product["source_dir"], language or declared_language, product.get("patch"))
    traits = infer_traits(scan["categories"])
    context = {
        "operator_family": family or operator_family(dtype),
        "target_language": language,
        "gfx": product.get("gfx") or gfx,
        "flydsl_version": product.get("flydsl_version"),
        "dtype": dtype, "regime": None, "traits": [],
    }
    cases = SELECT.workload_cases(meta)
    report: dict[str, Any] = {
        "name": product["name"], "kind": product["kind"], "path": str(product["path"]),
        "context": context, "language_basis": language_basis, "cases": len(cases),
        "timed_cases": sum(1 for case in cases if case["timed"]),
        "design_traits": traits, "scan": scan,
    }
    if not cases:
        report["error"] = "meta.json yields no (M, N, K) case"
        return report
    plain = SELECT.select_workload(catalog_path, context, cases)
    designed = SELECT.select_workload(catalog_path, dict(context, traits=traits), cases)
    report["selection"] = {
        "eligible_in_every_case": designed["eligible_in_every_case"],
        "eligible_in_some_cases": designed["eligible_in_some_cases"],
        "unlocked_by_design_traits": sorted(_applicable(designed) - _applicable(plain)),
        "undefined_traits": designed.get("undefined_traits", []),
    }
    report["coverage"] = coverage(scan, designed, index)
    if product["kind"] == "run":
        manifest = AGG._load_manifest(product["manifest"])
        report["citations"] = citations(manifest, designed, index)
        report["measured_readiness"] = readiness(product["manifest"], manifest, assumed_languages)
    return report


def discover(runs: list[str], tasks: list[str], language: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    products, errors = [], []
    manifests, discovery_errors = AGG.discover_manifests(runs) if runs else ([], [])
    errors.extend(discovery_errors)
    for manifest_path in manifests:
        run = manifest_path.parent
        manifest = AGG._load_manifest(manifest_path)
        meta = _first_existing(run / "workspace" / "meta.json", run / "validation_workspace" / "meta.json")
        source = _first_existing(run / "validation_workspace" / "kernel_src", run / "workspace" / "kernel_src")
        if meta is None or source is None:
            errors.append({"path": str(run), "error": "run has no task meta.json or kernel_src"})
            continue
        products.append({
            "name": run.name, "kind": "run", "path": run, "manifest": manifest_path, "meta": meta,
            "source_dir": source, "patch": run / "final_patch.diff",
            "language": str(manifest.get("observed_language") or language),
            "gfx": manifest.get("gfx"), "flydsl_version": manifest.get("flydsl_version"),
        })
    for spec in tasks:
        path_text, _, task_language = spec.partition("=")
        task = Path(path_text).expanduser().resolve()
        if not (task / "meta.json").is_file() or not (task / "kernel_src").is_dir():
            errors.append({"path": str(task), "error": "task needs meta.json and kernel_src/"})
            continue
        products.append({
            "name": task.name, "kind": "task", "path": task, "meta": task / "meta.json",
            "source_dir": task / "kernel_src", "patch": None, "language": task_language or language,
        })
    return products, errors


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    gaps: Counter = Counter()
    for report in reports:
        gaps.update((report.get("coverage") or {}).get("uncovered_categories") or [])
    runs = [r for r in reports if r["kind"] == "run"]
    return {
        "products": len(reports),
        "runs": len(runs),
        "runs_with_ranking_basis": sum(1 for r in runs if r["measured_readiness"]["ranking_basis"]),
        # No measured query can match a run whose language is unknown, whatever else it earned.
        "runs_without_observed_language": sorted(
            r["name"] for r in runs if not r["measured_readiness"]["observed_language"]),
        "citation_contradictions": {r["name"]: r["citations"]["contradictions"]
                                    for r in runs if r["citations"]["contradictions"]},
        "uncovered_categories_by_products": dict(gaps.most_common()),
    }


def to_markdown(result: dict[str, Any]) -> str:
    out = ["# Corpus validation against workflow products", ""]
    summary = result["summary"]
    out += [f"- products: {summary['products']} ({summary['runs']} workflow runs)",
            f"- runs that may rank candidates: {summary['runs_with_ranking_basis']}",
            f"- citation contradictions: {summary['citation_contradictions'] or 'none'}",
            f"- uncovered categories (products writing them): "
            f"{summary['uncovered_categories_by_products'] or 'none'}", ""]
    for report in result["products"]:
        out += [f"## {report['name']} ({report['kind']})", ""]
        if report.get("error"):
            out += [f"error: {report['error']}", ""]
            continue
        ctx = report["context"]
        sel, cov = report["selection"], report["coverage"]
        out += [
            f"- context: {ctx['operator_family']}, {ctx['target_language']} ({report['language_basis']}), "
            f"{ctx['gfx']}, dtype {ctx['dtype']}; {report['timed_cases']} timed of {report['cases']} cases",
            f"- design traits from source: {report['design_traits'] or 'none'}; cards they unlock: "
            f"{sel['unlocked_by_design_traits'] or 'none'}",
            f"- applicable cards: {len(cov['applicable_cards'])}; in every timed case: "
            f"{len(sel['eligible_in_every_case'])}; in some: {sel['eligible_in_some_cases'] or 'none'}",
            f"- product categories ({report['scan']['introduced_basis']}): "
            f"{report['scan']['introduced_categories']}",
            f"- uncovered categories: {cov['uncovered_categories'] or 'none'}; uncovered axes: "
            f"{cov['uncovered_axes'] or 'none'}",
        ]
        if report["kind"] == "run":
            ready = report["measured_readiness"]
            out.append(
                f"- measured readiness: language {ready['observed_language']} "
                f"({ready['observed_language_basis']}), status {ready['validation_status']}, ranking basis "
                f"{ready['ranking_basis']}, flags {ready['flag_reasons']}, graph-timed "
                f"{ready['graph_timed']}, metric {ready['metric_kind']}, eligible results "
                f"{ready['quality_eligible_results']}/{ready['direction_results']}, single-ref winners "
                f"{ready['single_ref_winners'] or 'none'}")
            out += ["", "| cited card | placement | attempts | statuses | winner |", "|---|---|---|---|---|"]
            for card, row in report["citations"]["cited"].items():
                where = row["placement"] + (f" ({', '.join(row.get('reasons') or row.get('cases') or [])})"
                                            if row.get("reasons") or row.get("cases") else "")
                out.append(f"| {card} | {where} | {row['attempts']} | {row['statuses']} | "
                           f"{row['became_winner']} |")
        out.append("")
    return "\n".join(out)


def validate(runs: list[str], tasks: list[str], gfx: str | None = None, family: str | None = None,
             language: str = "flydsl", catalog_path: Path = CATALOG,
             assumed_languages: dict[str, str] | None = None) -> dict[str, Any]:
    index = corpus_index(catalog_path)
    products, errors = discover(runs, tasks, language)
    reports = [validate_product(p, index, gfx, family, catalog_path, language, assumed_languages)
               for p in products]
    return {"schema_version": 1, "errors": errors, "summary": summarize(reports), "products": reports}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", action="append", default=[], metavar="PATH",
                        help="workflow run directory or experiment root (repeatable)")
    parser.add_argument("--task", action="append", default=[], metavar="DIR[=LANGUAGE]",
                        help="task directory with meta.json and kernel_src/ (repeatable)")
    parser.add_argument("--gfx", help="architecture for task products (runs state their own)")
    parser.add_argument("--operator-family", help="override the dtype-derived family")
    parser.add_argument("--language", default="flydsl",
                        help="declared language: of task products, and the one a run's refused "
                             "language verdict may be confirmed as")
    parser.add_argument("--assume-language", action="append", default=[], metavar="RUN=LANGUAGE",
                        help="passed to the aggregator for runs whose manifest captured no language")
    parser.add_argument("--json", metavar="PATH", help="write the full report as JSON")
    parser.add_argument("--markdown", metavar="PATH", help="write a readable summary")
    args = parser.parse_args(argv)
    if not args.runs and not args.task:
        parser.error("give at least one --runs or --task")
    assumed = {}
    for spec in args.assume_language:
        path, sep, language = spec.rpartition("=")
        if not sep or not path or not language:
            parser.error(f"--assume-language expects RUN=LANGUAGE, got {spec!r}")
        assumed[path] = language
    result = validate(args.runs, args.task, args.gfx, args.operator_family, args.language,
                      assumed_languages=assumed)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True, default=list) + "\n",
                                   encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(to_markdown(result) + "\n", encoding="utf-8")
    if not args.json and not args.markdown:
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
