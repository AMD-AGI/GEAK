#!/usr/bin/env python3
"""Versioned, hashed dispatch evidence passed from profile to UT generation.

This module inventories existing measurements. It does not capture, infer tensor
values, generate tests, or certify UT coverage. Large traces/tensors stay at their
recorded locations; the handoff indexes the structured evidence consumed by UTs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

CONTRACT = "workload/kernel_profile_handoff/v1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REQUIRED = ("kernel_ranking.json", "selection_targets.json", "sampling_points_final.json",
            "evidence/source_map.json", "evidence/adjacency.json")
OPTIONAL = ("kernel_workload.json", "coverage.json", "multi_config.json", "serving.json",
            "environment.json", "stack_identity.json", "sampling_plan.json", "sampling_projection.json",
            "evidence/compile_variants.json", "evidence/observed_workload.json")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def path_in(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def build(profile_dir, params, verified_refs=None):
    from finalize import configured_scope, scope_findings, denominator, INVENTORY
    root = Path(profile_dir).resolve()
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise ValueError("missing profile evidence: " + ", ".join(missing))
    ranking, selection = read(root / REQUIRED[0]), read(root / REQUIRED[1])
    scope = configured_scope(params)
    issues = scope_findings(selection, ranking, scope)
    if issues:
        raise ValueError("; ".join(note for _, note in issues))
    if denominator(str(root), ranking)[1] is None:
        raise ValueError("profile ranking has no valid measured denominator")
    if any(not ranking.get(k) for k in INVENTORY):
        raise ValueError("profile ranking must retain selection_scope, candidate_inventory and coverage_summary")
    targets = selection.get("targets") or []
    ids = [t.get("id") or t.get("name") for t in targets]
    if not ids or any(not isinstance(k, str) or not k for k in ids) or len(ids) != len(set(ids)):
        raise ValueError("selected kernel identities must be nonempty and unique")
    points_doc = read(root / "sampling_points_final.json")
    points = points_doc.get("points") or []
    pids = [p.get("point_id") for p in points]
    if any(not isinstance(p, str) or not p for p in pids) or len(pids) != len(set(pids)):
        raise ValueError("sampling point IDs must be nonempty and unique")
    files, external, dispatch = {}, {}, {k: [] for k in ids}

    def record(path):
        path = Path(os.path.abspath(path))
        if not path.is_file():
            raise ValueError("missing evidence file: " + str(path))
        try:
            key = str(path.relative_to(root))
            refs = files
        except ValueError:
            key, refs = str(path), external
        if key not in refs:
            cached = (verified_refs[1 if refs is external else 0].get(key)
                      if verified_refs else None)
            refs[key] = cached or {"sha256": sha(path), "bytes": path.stat().st_size}
        return {"path": key, "external": refs is external, **refs[key]}

    for name in REQUIRED + OPTIONAL:
        if (root / name).is_file():
            record(root / name)
    # Preserve all measured shape manifests, including eager source-only windows.
    shape_docs = {}
    for path in sorted((root / "evidence/shapes").glob("*.json")):
        record(path)
        shape_docs[path.stem] = read(path)
    counted = []
    for p in points:
        if not p.get("counted", True):
            continue
        pid = p["point_id"]
        shape_path = root / "evidence/shapes" / (pid + ".json")
        if pid not in shape_docs:
            raise ValueError("counted point has no shape/dispatch manifest: " + pid)
        raw_census = p.get("census") or str(root / "evidence/census" / (pid + ".json"))
        census = record(path_in(root, raw_census))
        # The merged census is also an input to drafting and must stay immutable.
        merged_census = root / "evidence/census" / (pid + ".json")
        if merged_census.is_file():
            record(merged_census)
        c = read(path_in(root, raw_census))
        if c.get("phase") not in (None, p.get("phase")) or c.get("achieved_shape") != p.get("achieved_shape"):
            raise ValueError("point and census disagree about achieved phase/shape: " + pid)
        shape_doc = shape_docs[pid]
        if shape_doc.get("phase") != p.get("phase") or shape_doc.get("achieved_shape") != p.get("achieved_shape"):
            raise ValueError("point and shape manifest disagree: " + pid)
        n = (p.get("represented") or {}).get("instances")
        if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0:
            raise ValueError("invalid represented instance count: " + pid)
        if p.get("probe_result"):
            record(path_in(root, p["probe_result"]))
        point = {"point_id": pid, "config_id": p.get("config_id"), "phase": p.get("phase"),
                 "shape": p.get("achieved_shape"), "context": p.get("achieved_context"),
                 "represented_instances": n, "shapes": record(shape_path), "census": census}
        counted.append(point)
        for k in dispatch:
            if k in (shape_docs[pid].get("kernels") or {}):
                dispatch[k].append(pid)
    if not counted:
        raise ValueError("no measured sampling points")
    mc = read(root / "multi_config.json") if (root / "multi_config.json").is_file() else None
    observation_refs = {}
    if mc:
        for cfg in mc.get("configs") or []:
            directory = path_in(root, cfg["eval_dir"])
            observation_refs[cfg["config_id"]] = record(directory / "evidence/observed_workload.json")
            for name in ("serving.json", "kernel_ranking.json", "sampling_points_final.json"):
                record(directory / name)
    else:
        observation_refs["single"] = record(root / "evidence/observed_workload.json")
    source_map = read(root / "evidence/source_map.json")
    kernels = []
    for target in targets:
        k = target.get("id") or target["name"]
        source = (source_map.get("kernels") or {}).get(k) or {}
        op = source.get("op") or {}
        entry = op.get("entry") or {}
        callable_spec = entry.get("target_callable") if isinstance(entry, dict) else entry
        gaps = []
        if not dispatch[k]:
            gaps.append("no selected identity launch in counted point shape manifests")
        if not callable_spec:
            gaps.append("source map has no resolved operator callable; UT boundary needs source review")
        kernels.append({"id": k, "operator": target.get("op"), "entry": callable_spec,
                        "dispatch_points": dispatch[k], "source_map_key": k,
                        "source_status": "located" if source else "unresolved",
                        "status": "ready_for_ut_review" if not gaps else "needs_review",
                        "gaps": gaps})
    return {"contract": CONTRACT, "profile_root": str(root),
            "scope": "dispatch/shape/source-location evidence; no UT or source-delivery verdict",
            "selection_params": {"selection_top_k": scope[0], "selection_overall_pct_gt": scope[1],
                                 "selection_phases": scope[2]},
            "files": files, "external_files": external, "observations": observation_refs,
            "points": counted, "kernels": kernels,
            "coverage": {"basis": "profile measured-point model", "denominators": ranking["denominators"],
                         "summary": ranking.get("coverage_summary"),
                         "sampling_gaps": points_doc.get("configs") or points_doc.get("phase_summary"),
                         "ut_coverage": None},
            "compile_evidence": "recorded" if "evidence/compile_variants.json" in files else "not_recorded",
            "ut_contracts": "kernel-extract/ut_contracts: select a family during UT generation",
            "read_only_inputs": True}


def verify(path):
    doc = read(path)
    if doc.get("contract") != CONTRACT or doc.get("read_only_inputs") is not True:
        raise ValueError("unsupported profile handoff contract")
    root = Path(doc["profile_root"])
    if not root.is_absolute() or not isinstance(doc.get("files"), dict):
        raise ValueError("handoff needs an absolute profile_root and a file manifest")
    if not set(REQUIRED).issubset(doc["files"]):
        raise ValueError("handoff is missing required evidence references")
    for name, ref in doc["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid relative evidence path: " + name)
        target = root / name
        if not target.is_file() or sha(target) != ref.get("sha256") or target.stat().st_size != ref.get("bytes"):
            raise ValueError("profile evidence changed or disappeared: " + str(target))
    for name, ref in doc.get("external_files", {}).items():
        target = Path(name)
        if (not target.is_absolute() or not target.is_file() or sha(target) != ref.get("sha256")
                or target.stat().st_size != ref.get("bytes")):
            raise ValueError("external profile evidence changed or disappeared: " + name)
    # Re-derive indices and reference closure from the verified files; edited summary
    # fields or omitted shape/observation refs cannot narrow the handoff silently.
    expected = build(root, doc["selection_params"], (doc["files"], doc.get("external_files", {})))
    for key in ("files", "external_files", "observations", "points", "kernels", "coverage", "compile_evidence"):
        if doc.get(key) != expected[key]:
            raise ValueError("handoff %s differs from the verified profile evidence" % key)
    return doc


def finalize_evidence(out, params):
    from emit_result import emit
    try:
        doc = build(out, params)
        write(Path(out) / "profile_handoff.json", doc)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return emit("blocked", artifacts={}, notes="Profile evidence is incomplete: " + str(exc))
    unresolved = sum(bool(k["gaps"]) for k in doc["kernels"])
    return emit("ok", artifacts={"profile_handoff": "profile_handoff.json",
                "kernel_ranking": "kernel_ranking.json", "selection_targets": "selection_targets.json",
                "sampling_points": "sampling_points_final.json"},
                metrics={"points_measured": len(doc["points"]), "kernels_selected": len(doc["kernels"]),
                         "kernels_needing_ut_review": unresolved, "ut_generation_started": False,
                         "handoff_sha256": sha(Path(out) / "profile_handoff.json")},
                notes="Dispatch evidence handed off with hashed inputs and explicit gaps. "
                      "UT generation/validation is a separate stage; no UT coverage is claimed.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--profile-dir", required=True)
    export.add_argument("--params", required=True)
    export.add_argument("--out", help="default: <profile-dir>/profile_handoff.json")
    check = sub.add_parser("verify")
    check.add_argument("--handoff", required=True)
    args = p.parse_args()
    try:
        if args.command == "verify":
            doc = verify(args.handoff)
        else:
            doc = build(args.profile_dir, read(args.params))
            write(args.out or Path(args.profile_dir) / "profile_handoff.json", doc)
        print("%s: %d kernels, %d measured points" % (doc["contract"], len(doc["kernels"]), len(doc["points"])))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.exit(1, "Profile handoff: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
