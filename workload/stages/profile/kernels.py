#!/usr/bin/env python3
"""The profile stage's delivery: one config's selected kernels, where they come from, at
which shapes they ran and how much each shape weighs -- `kernels.json`.

    python3 kernels.py export --profile-dir <EVAL_DIR> --params <params.json> [--out FILE]
    python3 kernels.py verify --kernels <kernels.json>

It indexes evidence the profile already measured; it captures nothing, decides no task
boundary and writes no UT. Every file it summarizes is listed with its SHA-256, so a
consumer can prove the summary still describes the evidence (`verify` re-derives it).

  operators[]   each selected operator: its member kernels, entry callable and share
  kernels[]     each selected kernel:
                  path     entry callable, launching ops and the resolved serving-stack
                           frames (file, line, callable, sha256) from evidence/source_map.json
                  points   every counted sampling point it launched at: phase, achieved
                           shape/context, represented instances, launches per instance,
                           device time per instance, grid/block and host-op operand dims
                  weight   launches over the measured workload (instances x launches per
                           instance), share of whole-workload and per-phase GPU time
                  gaps     what is missing for a UT boundary (no launch at a counted point,
                           no resolved entry)
  denominator   the GPU-time total every share is a fraction of, and coverage gaps
  files         every evidence file the above was read from, with SHA-256
Stdlib only.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

CONTRACT = "workload/kernel_profile/v1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REQUIRED = ("kernel_ranking.json", "selection_targets.json", "sampling_points_final.json",
            "evidence/source_map.json", "evidence/adjacency.json")
OPTIONAL = ("kernel_workload.json", "coverage.json", "serving.json", "environment.json",
            "runtime_identity.json", "sampling_plan.json", "evidence/compile_variants.json",
            "evidence/observed_workload.json")
MAX_FRAMES = 12          # serving-stack frames kept per kernel (innermost first)
MAX_OPERAND_SAMPLES = 4  # distinct host-op operand signatures kept per point


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
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) else None


def path_of(source):
    """The kernel's serving-stack location: entry, launching ops and resolved frames."""
    op = source.get("op") or {}
    entry = op.get("entry") or {}
    entry = entry if isinstance(entry, dict) else {"target_callable": entry}
    frames = [{k: f.get(k) for k in ("target_callable", "path", "line", "sha256")}
              for f in source.get("frames") or [] if f.get("resolved")][:MAX_FRAMES]
    return {"entry": entry.get("target_callable"), "entry_file": entry.get("path"),
            "entry_sha256": entry.get("sha256"), "module_class": op.get("module_class"),
            "operator": op.get("key"),
            "launching_ops": source.get("launching_ops") or [],
            "graph_replayed_only": bool(source.get("graph_replayed_only")),
            "frames": frames}


def operands(variant):
    seen, out = set(), []
    for s in variant.get("samples") or []:
        h = s.get("host_op") or {}
        if not h:
            continue
        sig = {k: h.get(k) for k in ("op", "input_dims", "input_types", "input_strides")}
        key = json.dumps(sig, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(sig)
        if len(out) == MAX_OPERAND_SAMPLES:
            break
    return out


def build(profile_dir, params, verified=None):
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
    total_ms, total_source = denominator(str(root), ranking)
    if total_ms is None:
        raise ValueError("profile ranking has no valid measured denominator")
    if any(not ranking.get(k) for k in INVENTORY):
        raise ValueError("profile ranking must retain selection_scope, candidate_inventory "
                         "and coverage_summary")
    targets = selection.get("targets") or []
    ids = [t.get("id") or t.get("name") for t in targets]
    if not ids or any(not isinstance(k, str) or not k for k in ids) or len(ids) != len(set(ids)):
        raise ValueError("selected kernel identities must be nonempty and unique")
    points_doc = read(root / "sampling_points_final.json")
    files, external = {}, {}

    def record(path):
        path = Path(os.path.abspath(path))
        if not path.is_file():
            raise ValueError("missing evidence file: " + str(path))
        try:
            key, refs = str(path.relative_to(root)), files
        except ValueError:
            key, refs = str(path), external
        if key not in refs:
            cached = (verified[1 if refs is external else 0].get(key) if verified else None)
            refs[key] = cached or {"sha256": sha(path), "bytes": path.stat().st_size}
        return key

    for name in REQUIRED + OPTIONAL:
        if (root / name).is_file():
            record(root / name)
    rank = {r.get("id"): r for r in ranking.get("ranking") or []}
    source_map = (read(root / "evidence/source_map.json").get("kernels") or {})
    per_kernel = {k: [] for k in ids}
    counted = []
    for p in points_doc.get("points") or []:
        if not p.get("counted", True):
            continue
        pid = p["point_id"]
        shape_path = root / "evidence/shapes" / (pid + ".json")
        if not shape_path.is_file():
            raise ValueError("counted point has no shape/dispatch manifest: " + pid)
        shapes = read(shape_path)
        census = Path(p.get("census") or root / "evidence/census" / (pid + ".json"))
        census = census if census.is_absolute() else root / census
        point = {"point_id": pid, "phase": p.get("phase"), "shape": p.get("achieved_shape"),
                 "context": p.get("achieved_context"),
                 "instances": number((p.get("represented") or {}).get("instances")),
                 "shapes": record(shape_path), "census": record(census)}
        if point["instances"] is None or point["instances"] < 0:
            raise ValueError("invalid represented instance count: " + pid)
        if shapes.get("phase") != p.get("phase") or shapes.get("achieved_shape") != p.get("achieved_shape"):
            raise ValueError("point and shape manifest disagree: " + pid)
        counted.append(point)
        for k in ids:
            ks = (shapes.get("kernels") or {}).get(k)
            if not ks:
                continue
            variants = ks.get("variants") or []
            lpi = sum(number(v.get("launches_per_instance")) or 0 for v in variants)
            per_kernel[k].append({
                "point_id": pid, "phase": point["phase"], "shape": point["shape"],
                "context": point["context"], "instances": point["instances"],
                "launches_per_instance": round(lpi, 6),
                "us_per_instance": round(sum(number(v.get("us_per_instance")) or 0 for v in variants), 6),
                "variants": [{"grid": v.get("grid"), "block": v.get("block"),
                              "launches_per_instance": v.get("launches_per_instance"),
                              "operands": operands(v)} for v in variants]})
    if not counted:
        raise ValueError("no measured sampling points")
    kernels = []
    for t in targets:
        k = t.get("id") or t["name"]
        source = source_map.get(k) or {}
        path = path_of(source)
        pts = per_kernel[k]
        gaps = []
        if not pts:
            gaps.append("launched at no counted point's shape manifest")
        if not path["entry"]:
            gaps.append("no resolved operator entry in evidence/source_map.json")
        kernels.append({
            "id": k, "operator": t.get("op"), "path": path, "points": pts,
            "weight": {"launches": round(sum(p["instances"] * p["launches_per_instance"] for p in pts), 3),
                       "overall_pct": number(t.get("overall_pct")),
                       "share_of_phase_pct": (rank.get(k) or {}).get("share_of_phase_pct"),
                       "selected_phases": t.get("selected_phases")},
            "gaps": gaps})
    operators = {}
    for kern in kernels:
        op = kern["operator"] or kern["id"]
        row = operators.setdefault(op, {"operator": op, "kernels": [], "entry": kern["path"]["entry"],
                                        "overall_pct": 0.0})
        row["kernels"].append(kern["id"])
        row["overall_pct"] = round(row["overall_pct"] + (kern["weight"]["overall_pct"] or 0), 6)
    return {"contract": CONTRACT, "profile_root": str(root),
            "scope": "one config's kernel analysis: path, shapes, weights; no task boundary or UT verdict",
            "selection_params": {"selection_top_k": scope[0], "selection_overall_pct_gt": scope[1],
                                 "selection_phases": scope[2]},
            "denominator": {"total_ms": total_ms, "source": total_source,
                            "phase_share": ranking.get("phase_share"),
                            "coverage_summary": ranking.get("coverage_summary"),
                            "sampling_gaps": points_doc.get("phase_summary")},
            "points": counted, "operators": sorted(operators.values(), key=lambda r: -r["overall_pct"]),
            "kernels": kernels,
            "compile_evidence": ("evidence/compile_variants.json"
                                 if "evidence/compile_variants.json" in files else None),
            "files": files, "external_files": external}


def verify(path):
    """The kernels.json at `path`, after proving every listed file is unchanged and that the
    summary is what those files still say."""
    doc = read(path)
    if doc.get("contract") != CONTRACT:
        raise ValueError("unsupported kernel profile contract")
    root = Path(doc["profile_root"])
    if not root.is_absolute() or not set(REQUIRED).issubset(doc.get("files") or {}):
        raise ValueError("kernels.json needs an absolute profile_root and its required evidence")
    for name, ref in list(doc["files"].items()) + list((doc.get("external_files") or {}).items()):
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha(target) != ref.get("sha256") or target.stat().st_size != ref.get("bytes"):
            raise ValueError("profile evidence changed or disappeared: " + str(target))
    expected = build(root, doc["selection_params"], (doc["files"], doc.get("external_files") or {}))
    for key in ("points", "operators", "kernels", "denominator", "files", "external_files"):
        if doc.get(key) != expected[key]:
            raise ValueError("kernels.json %s differs from the verified profile evidence" % key)
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--profile-dir", required=True)
    export.add_argument("--params", required=True)
    export.add_argument("--out", help="default: <profile-dir>/kernels.json")
    check = sub.add_parser("verify")
    check.add_argument("--kernels", required=True)
    a = p.parse_args()
    try:
        if a.command == "verify":
            doc = verify(a.kernels)
        else:
            doc = build(a.profile_dir, read(a.params))
            write(a.out or Path(a.profile_dir) / "kernels.json", doc)
        print("%s: %d kernels in %d operators, %d measured points"
              % (doc["contract"], len(doc["kernels"]), len(doc["operators"]), len(doc["points"])))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.exit(1, "kernels.json: %s\n" % exc)


if __name__ == "__main__":
    raise SystemExit(main())
