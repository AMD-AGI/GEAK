#!/usr/bin/env python3
"""Plan one config's sampling points from its observed workload, optionally reusing the
captures of another profile of the same deployment.

    python3 plan_points.py --eval-dir "$EVAL_DIR" --num-points 24 \
        [--context-tolerance 1.5] [--target-pct 90] [--runtime-identity identity.json] \
        [--reuse-capture <another profile's EVAL_DIR>]

reads   $EVAL_DIR/serving.json  $EVAL_DIR/evidence/observed_workload.json
        [$EVAL_DIR/evidence/server_info.json]  [$EVAL_DIR/runtime_identity.json]
writes  $EVAL_DIR/sampling_points.json  (what run_points.py captures)
        $EVAL_DIR/sampling_plan.json    (planned proxy coverage, reused and unselected cells)

Observed rows are grouped into cells by phase, exact shape, sequence count, observed graph
mode and a context band (`--context-tolerance`; 1 = exact context). The representative of a
cell is an actual observed row, never an invented midpoint. The budget goes greedily to the
cell covering the most uncovered proxy mass of its phase (prefill new tokens, decode
iterations), each phase normalized to 1, until every phase meets `--target-pct` or the budget
is spent. Proxy coverage is a planning figure, not measured GPU time.

Reuse. A cell another profile already captured costs no budget when both deployments are
identical: serving fields and arguments except transport/allocation, and the runtime identity
(image, model/source revision, GPU arch). That profile's point is planned verbatim (same id,
same measurement signature) and its signed census and probe result are linked into this
eval dir, so run_points.py resumes them instead of capturing them again. Everything else is
planned and captured here. Stdlib only.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

TRANSPORT_ARGS = {"--port", "--host", "--nccl-port", "--dist-init-addr"}
IGNORED_SERVING = {"gpus", "host", "port", "concurrency", "conc"}
IDENTITY_KEYS = ("image", "model_revision", "source_revision", "gpu_arch")


def read(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        if default is not None:
            return default
        raise


def write(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return value


def deployment(serving, identity):
    """What decides whether two captures measure the same thing: every serving field and
    argument except transport and allocation, plus the runtime identity. Scheduler, cache and
    graph settings stay significant; `--max-running-requests` is not a client label."""
    doc = {k: v for k, v in serving.items() if k not in IGNORED_SERVING and not k.startswith("_")}
    if not doc.get("model") or not doc.get("tp"):
        raise ValueError("serving.json needs model and tp")
    args = doc.get("args", [])
    if not isinstance(args, list) or any(not isinstance(x, str) for x in args):
        raise ValueError("serving.args must be an argv list")
    kept, i = [], 0
    while i < len(args):
        if args[i].split("=", 1)[0] in TRANSPORT_ARGS:
            if "=" not in args[i]:
                if i + 1 == len(args) or args[i + 1].startswith("--"):
                    raise ValueError("transport argument has no value: " + args[i])
                i += 1
        else:
            kept.append(args[i])
        i += 1
    doc["args"] = kept
    return {"serving": doc, "runtime_identity": identity or {}}


def require_identity(identity, where):
    missing = [k for k in IDENTITY_KEYS if not isinstance((identity or {}).get(k), str)
               or not identity[k].strip()]
    if missing:
        raise ValueError("reuse needs a runtime identity with %s (%s)" % (", ".join(missing), where))


def observations(obs):
    rows = []
    for phase, key in (("PREFILL", "raw_prefill_rows"), ("DECODE", "raw_decode_rows")):
        for index, r in enumerate(obs.get(key, [])):
            shape = r.get("new_token" if phase == "PREFILL" else "running")
            if not shape:
                continue
            positive(shape, "observed shape")
            if int(shape) != shape:
                raise ValueError("observed shapes must be integers")
            seqs = max(1, r.get("new_seq") or 1) if phase == "PREFILL" else int(shape)
            total = (r.get("cached_token", 0) + shape) if phase == "PREFILL" else r.get("token")
            if not total:
                raise ValueError("context is missing from an observed " + phase + " row")
            inst = 1 if phase == "PREFILL" else (r.get("iterations") or obs.get("decode_log_interval") or 1)
            positive(inst, "observed instances")
            rows.append({"phase": phase, "shape": int(shape), "sequences": seqs,
                         "context": positive(total / seqs, "observed context"),
                         "cuda_graph": r.get("cuda_graph"), "instances": inst,
                         "mass": shape if phase == "PREFILL" else inst, "source_row": index})
    if not rows:
        raise ValueError("observed workload has no usable raw rows")
    return rows


def partition(phase, shape, sequences, cuda_graph):
    # Do not interpolate across unobserved M-dependent kernel switches.
    return [phase, shape, sequences, cuda_graph]


def band(context, tolerance):
    return context if tolerance == 1 else math.floor(math.log(context) / math.log(tolerance))


def boundaries(serving, eval_dir):
    """Dispatch boundaries the server states (prefill chunk, graph capture sizes)."""
    info = read(Path(eval_dir) / "evidence/server_info.json", {})
    info = dict(info.get("server_args", info))
    args = serving.get("args", [])
    for i, arg in enumerate(args):
        key, sep, value = arg.partition("=")
        if key in ("--chunked-prefill-size", "--max-prefill-tokens", "--cuda-graph-max-bs",
                   "--cuda-graph-max-bs-decode"):
            info[key[2:].replace("-", "_")] = int(value if sep else args[i + 1])
    b = {"prefill_chunk": None, "graph_bs": [], "notes": []}
    for k in ("chunked_prefill_size", "max_prefill_tokens", "chunk_size"):
        if info.get(k):
            b["prefill_chunk"] = int(info[k])
            break
    for k in ("cuda_graph_bs", "capture_bs", "cuda_graph_batch_sizes"):
        if isinstance(info.get(k), list) and info[k]:
            b["graph_bs"] = sorted(int(x) for x in info[k])
            break
    if not b["graph_bs"] and (info.get("cuda_graph_max_bs") or info.get("cuda_graph_max_bs_decode")):
        b["notes"].append("cuda_graph_max_bs is set but the capture-size list is not exposed: a "
                          "decode point at bs=B may measure the padded size, not B")
    if not info:
        b["notes"].append("no server info: dispatch boundaries are unknown; retain "
                          "evidence/server_info.json from the deployed server")
    return b


def reusable(other_dir, dep, tolerance):
    """{cell key: point} for every point of another profile of this deployment whose
    signed census exists there."""
    other = Path(other_dir).resolve()
    other_serving = read(other / "serving.json")
    other_identity = read(other / "runtime_identity.json", {})
    require_identity(other_identity, str(other / "runtime_identity.json"))
    if deployment(other_serving, other_identity) != dep:
        raise ValueError("--reuse-capture %s is a different deployment (serving fields/args or "
                         "runtime identity differ); its captures do not measure this config" % other)
    plan = read(other / "sampling_points.json")
    run = read(other / "profile_run.json", {}).get("points", {})
    out = {}
    for p in plan.get("points", []):
        pid, sig = p["point_id"], p.get("measurement_signature")
        census = other / "evidence/census" / (pid + ".json")
        if not sig or not census.is_file() or run.get(pid, {}).get("measurement_signature") != sig:
            continue
        part = p.get("sampling_partition")
        if not part:
            shape = p["new_tokens"] if p["phase"] == "PREFILL" else p["batch_size"]
            seqs = p.get("running", 1) if p["phase"] == "PREFILL" else p["batch_size"]
            part = partition(p["phase"], shape, seqs, p.get("cuda_graph"))
        key = digest([part, band(p["context_tokens"], tolerance)])
        out[key] = {"point": p, "dir": str(other), "record": run[pid]}
    return out


def plan(serving, identity, observed, budget, tolerance=1.5, target_pct=90.0, reuse=None):
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("num-points must be a nonnegative integer")
    positive(tolerance, "context-tolerance")
    if tolerance < 1:
        raise ValueError("context-tolerance must be >= 1")
    if not 0 < target_pct <= 100:
        raise ValueError("target-pct must be in (0, 100]")
    dep = deployment(serving, identity)
    gid = digest(dep)[:16]
    cells, totals = {}, {}
    for row in observations(observed):
        part = partition(row["phase"], row["shape"], row["sequences"], row["cuda_graph"])
        key = digest([part, band(row["context"], tolerance)])
        c = cells.setdefault(key, {"key": key, "partition": part,
                                   "context_band": band(row["context"], tolerance), "rows": [],
                                   "instances": 0, "proxy_mass": 0})
        c["rows"].append(row)
        c["instances"] += row["instances"]
        c["proxy_mass"] += row["mass"]
        totals[row["phase"]] = totals.get(row["phase"], 0) + row["mass"]
    for c in cells.values():
        ph, shape, seqs, graph = c["partition"]
        row = min(c.pop("rows"), key=lambda r: (-r["mass"], r["context"], r["source_row"]))
        p = {"phase": ph, "context_tokens": row["context"], "sampling_partition": c["partition"],
             "cuda_graph": graph, "grounds": "observed_row", "representative_row": row["source_row"]}
        if ph == "PREFILL":
            p.update(new_tokens=shape, running=seqs, cached_tokens=max(0, round(row["context"] * seqs - shape)))
        else:
            p.update(batch_size=shape)
        p["point_id"] = "pt_" + digest([gid, c["partition"], row["context"]])[:20]
        p["measurement_signature"] = digest([dep, p])
        c["point"] = p
        c["gain"] = c["proxy_mass"] / totals[ph]
    covered = {ph: 0.0 for ph in totals}
    selected, reused = [], []
    remaining = sorted(cells.values(), key=lambda c: c["point"]["point_id"])
    for c in list(remaining):
        hit = (reuse or {}).get(c["key"])
        if hit:
            c.update(point=hit["point"], reused_from=hit["dir"])
            remaining.remove(c)
            reused.append(c)
            covered[c["partition"][0]] += c["gain"]

    def score(c):
        return min(c["gain"], max(0.0, target_pct / 100 - covered[c["partition"][0]]))
    while remaining and len(selected) < budget:
        best = max(remaining, key=score)
        if score(best) <= 1e-12:
            break
        remaining.remove(best)
        selected.append(best)
        covered[best["partition"][0]] += best["gain"]
    phases = {ph: {"total_proxy_mass": totals[ph],
                   "planned_proxy_coverage_pct": round(100 * covered[ph], 6),
                   "target_met": covered[ph] + 1e-12 >= target_pct / 100} for ph in totals}
    strip = lambda c: {k: c[k] for k in ("point", "partition", "context_band", "instances",
                                         "proxy_mass") if k in c}
    return {"contract": "kernel_extract/sampling_plan/v3", "deployment_id": gid,
            "budget_points": budget, "context_tolerance": tolerance, "target_pct": target_pct,
            "coverage_basis": "planning proxy: prefill new tokens / decode iterations; not measured GPU time",
            "phases": phases, "capture_count": len(selected),
            "captures": [strip(c) for c in selected],
            "reused": [dict(strip(c), reused_from=c["reused_from"]) for c in reused],
            "unselected": [strip(c) for c in remaining],
            "status": "planned" if all(v["target_met"] for v in phases.values()) else "budget_limited"}


def link_reused(eval_dir, reused_cells, reuse):
    """Link each reused point's signed census and probe result, and seed profile_run.json so
    run_points.py resumes them (it checks the measurement signature)."""
    ev = Path(eval_dir)
    run_path = ev / "profile_run.json"
    run = read(run_path, {"points": {}})
    by_id = {h["point"]["point_id"]: h for h in reuse.values()}
    for cell in reused_cells:
        pid = cell["point"]["point_id"]
        src = Path(by_id[pid]["dir"])
        for sub, pattern in (("census", pid + ".json"), ("probe", pid + "*.json")):
            for f in sorted((src / "evidence" / sub).glob(pattern)):
                dest = ev / "evidence" / sub / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.is_symlink() and dest.resolve() == f.resolve():
                    continue
                if dest.exists() or dest.is_symlink():
                    raise ValueError("refusing to overwrite evidence: %s" % dest)
                dest.symlink_to(f.resolve())
        run["points"][pid] = dict(by_id[pid]["record"], reused_from=str(src))
    write(run_path, run)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--num-points", type=int, required=True,
                    help="capture budget for this config (reused points cost none)")
    ap.add_argument("--context-tolerance", type=float, default=1.5)
    ap.add_argument("--target-pct", type=float, default=90.0)
    ap.add_argument("--runtime-identity", help="default: $EVAL_DIR/runtime_identity.json")
    ap.add_argument("--reuse-capture", help="another profile EVAL_DIR of the same deployment")
    a = ap.parse_args(argv)
    ev = Path(a.eval_dir).resolve()
    try:
        serving = read(ev / "serving.json")
        identity = read(a.runtime_identity or ev / "runtime_identity.json", {})
        dep = deployment(serving, identity)
        reuse = None
        if a.reuse_capture:
            require_identity(identity, a.runtime_identity or str(ev / "runtime_identity.json"))
            reuse = reusable(a.reuse_capture, dep, a.context_tolerance)
        doc = plan(serving, identity, read(ev / "evidence/observed_workload.json"),
                   a.num_points, a.context_tolerance, a.target_pct, reuse)
    except (ValueError, OSError, KeyError) as exc:
        print("plan_points: %s" % exc, file=sys.stderr)
        return 1
    doc["boundaries"] = boundaries(serving, ev)
    points = [c["point"] for c in doc["reused"]] + [c["point"] for c in doc["captures"]]
    write(ev / "sampling_points.json", {"contract": "kernel_extract/sampling_points/v2",
                                        "context_tolerance": a.context_tolerance, "points": points})
    write(ev / "sampling_plan.json", doc)
    if reuse:
        link_reused(ev, doc["reused"], reuse)
    print("%d points: %d reused, %d to capture (%s); proxy coverage %s" % (
        len(points), len(doc["reused"]), doc["capture_count"], doc["status"],
        ", ".join("%s %.1f%%" % (ph, v["planned_proxy_coverage_pct"]) for ph, v in sorted(doc["phases"].items()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
