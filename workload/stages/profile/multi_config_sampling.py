#!/usr/bin/env python3
"""Plan shared captures before profiling; project measured costs onto each workload.

Used by ``multi_config.py plan/project``. Planning is CPU-only. Equal deployments
share a capture pool, while observations, weights and coverage remain per config.
No measurement is fabricated by the planner. See kernel-extract/references/multi-config-sampling.md.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from collections import defaultdict

SLOT = Path(__file__).resolve().parent
SCRIPTS = SLOT / "kernel-extract/scripts"
sys.path.insert(0, str(SCRIPTS))
from relabel_points import relabel  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return value


def deployment(serving, identity):
    """Conservative equality: ignore transport/allocation, never scheduler or cache args.

    The supplied runtime identity pins image/packages, model revision and hardware.
    Unknown fields participate in equality; a model name alone is not a source identity.
    """
    for key in ("image", "model_revision", "source_revision", "gpu_arch"):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            raise ValueError("runtime identity needs " + key)
    ignored = {"gpus", "host", "port", "concurrency", "conc"}
    doc = {k: v for k, v in serving.items() if k not in ignored and not k.startswith("_")}
    if not doc.get("model") or not doc.get("tp"):
        raise ValueError("serving.json needs model and tp")
    args = doc.get("args", [])
    if not isinstance(args, list) or any(not isinstance(x, str) for x in args):
        raise ValueError("serving.args must be an argv list")
    # --flag=value and --flag value both occur in CI launch receipts.
    transport = {"--port", "--host", "--nccl-port", "--dist-init-addr"}
    kept, i = [], 0
    while i < len(args):
        arg = args[i]
        if arg.split("=", 1)[0] in transport:
            if "=" not in arg:
                if i + 1 == len(args) or args[i + 1].startswith("--"):
                    raise ValueError("transport argument has no value: " + arg)
                i += 1
        else:
            kept.append(arg)
        i += 1
    doc["args"] = kept
    return {"serving": doc, "runtime_identity": identity}


def boundaries_from_server_info(info):
    b = {"prefill_chunk": None, "graph_bs": [], "notes": []}
    if not info:
        b["notes"].append("no server info supplied: dispatch boundaries are unknown; "
                          "retain evidence/server_info.json from the deployed server.")
        return b
    for k in ("chunked_prefill_size", "max_prefill_tokens", "chunk_size"):
        if info.get(k):
            b["prefill_chunk"] = int(info[k])
            b["notes"].append("prefill chunk ceiling %d from server arg %r" % (b["prefill_chunk"], k))
            break
    for k in ("cuda_graph_bs", "capture_bs", "cuda_graph_batch_sizes"):
        v = info.get(k)
        if isinstance(v, list) and v:
            b["graph_bs"] = sorted(int(x) for x in v)
            b["notes"].append("CUDA-graph capture sizes from server arg %r" % k)
            break
    if not b["graph_bs"]:
        mx = info.get("cuda_graph_max_bs") or info.get("cuda_graph_max_bs_decode")
        if mx:
            b["notes"].append(
                "cuda_graph_max_bs=%s but the capture-size LIST is not exposed; decode "
                "batches are padded to captured sizes, so without the list a point at "
                "bs=B may be measuring the pad, not B." % mx)
    return b


def server_info(serving, directory):
    path = Path(directory) / "evidence/server_info.json"
    info = read(path) if path.exists() else {}
    info = dict(info.get("server_args", info))
    args = serving.get("args", [])
    for i, arg in enumerate(args):
        key, sep, value = arg.partition("=")
        if key in ("--chunked-prefill-size", "--max-prefill-tokens", "--cuda-graph-max-bs",
                   "--cuda-graph-max-bs-decode"):
            value = value if sep else args[i + 1]
            info[key[2:].replace("-", "_")] = int(value)
    return info


def observations(obs):
    """Shape and sequence count are discrete. Context clustering is explicitly a proxy."""
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
            ctx = positive(total / seqs, "observed context")
            inst = 1 if phase == "PREFILL" else (r.get("iterations") or obs.get("decode_log_interval") or 1)
            positive(inst, "observed instances")
            rows.append({"phase": phase, "shape": int(shape), "context": ctx,
                         "sequences": seqs, "cuda_graph": r.get("cuda_graph"),
                         "instances": inst, "mass": shape if phase == "PREFILL" else inst,
                         "source_row": index})
    if not rows:
        raise ValueError("observed workload has no usable raw rows")
    return rows


def partition(row):
    # Do not interpolate across unobserved M-dependent kernel switches. Dense-only
    # UT deduplication can later use the operator's smaller semantic signature.
    return [row["phase"], row["shape"], row["sequences"], row["cuda_graph"]]


def plan(configs, budget, context_tolerance=1.5, target_pct=90.0):
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("num-points must be a positive integer (one GLOBAL budget)")
    positive(context_tolerance, "context-tolerance")
    if context_tolerance < 1:
        raise ValueError("context-tolerance must be >= 1")
    if not 0 < target_pct <= 100 or not math.isfinite(target_pct):
        raise ValueError("target-pct must be in (0, 100]")
    ids = [c["config_id"] for c in configs]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("give each config exactly once")
    groups, cells, totals, cfgs = {}, {}, defaultdict(float), {}
    for cfg in sorted(configs, key=lambda c: c["config_id"]):
        cid = cfg["config_id"]
        dep = deployment(cfg["serving"], cfg["runtime_identity"])
        gid = "deployment_" + digest(dep)[:16]
        groups.setdefault(gid, {"deployment": dep, "config_ids": [], "point_ids": []})["config_ids"].append(cid)
        rows = observations(cfg["observed"])
        cfgs[cid] = {"group_id": gid, "weight": positive(cfg.get("weight", 1.0), "config weight"),
                     "inputs": cfg.get("inputs", {}), "source_eval_dir": cfg.get("eval_dir"),
                     "boundaries": boundaries_from_server_info(cfg.get("server_info", {}))}
        for row in rows:
            ph = row["phase"]
            totals[(cid, ph)] += row["mass"]
            band = (row["context"] if context_tolerance == 1 else
                    math.floor(math.log(row["context"]) / math.log(context_tolerance)))
            key = digest([gid, partition(row), band])
            c = cells.setdefault(key, {"group_id": gid, "partition": partition(row), "rows": [],
                                      "per_config": {}, "context_band": band})
            c["rows"].append((cid, row))
            m = c["per_config"].setdefault(cid, {"instances": 0, "proxy_mass": 0, "source_rows": []})
            m["instances"] += row["instances"]
            m["proxy_mass"] += row["mass"]
            m["source_rows"].append(row["source_row"])
    candidates = []
    for cell in cells.values():
        # Deterministic representative chosen from an actual observation, not an invented midpoint.
        cid, row = min(cell.pop("rows"), key=lambda cr: (-cr[1]["mass"], cr[0], cr[1]["context"]))
        ph, shape, seqs, graph = cell["partition"]
        ctx = row["context"]
        p = {"phase": ph, "context_tokens": ctx, "sampling_partition": cell["partition"],
             "grounds": "multi_config_observed_union", "representative_config": cid,
             "representative_row": row["source_row"]}
        if ph == "PREFILL":
            p.update(new_tokens=shape, running=seqs, cached_tokens=max(0, round(ctx * seqs - shape)))
        else:
            p.update(batch_size=shape)
        p["point_id"] = "mc_" + digest([cell["group_id"], cell["partition"], ctx])[:20]
        p["measurement_signature"] = digest([groups[cell["group_id"]]["deployment"], p])
        cell.update(point=p, point_id=p["point_id"])
        cell["gain"] = {cid: v["proxy_mass"] / totals[(cid, ph)] for cid, v in cell["per_config"].items()}
        candidates.append(cell)
    weight_sum = sum(c["weight"] for c in cfgs.values())
    phase_count = {cid: sum(c == cid for c, _ in totals) for cid in cfgs}
    covered = {key: 0.0 for key in totals}
    selected = []
    remaining = sorted(candidates, key=lambda c: c["point_id"])
    while remaining and len(selected) < budget:
        def score(c):
            ph = c["point"]["phase"]
            return sum(cfgs[cid]["weight"] / weight_sum / phase_count[cid] *
                       min(gain, max(0, target_pct / 100 - covered[(cid, ph)]))
                       for cid, gain in c["gain"].items())
        best = max(remaining, key=score)
        if score(best) <= 1e-12:
            break
        remaining.remove(best)
        selected.append(best)
        groups[best["group_id"]]["point_ids"].append(best["point_id"])
        for cid, gain in best["gain"].items():
            covered[(cid, best["point"]["phase"])] += gain
    for cid, cfg in cfgs.items():
        cfg["phases"] = {ph: {"total_proxy_mass": total,
                               "planned_proxy_coverage_pct": round(100 * covered[(cid, ph)], 6),
                               "target_met": covered[(cid, ph)] + 1e-12 >= target_pct / 100}
                         for (c, ph), total in totals.items() if c == cid}
    return {"contract": "workload/profile/shared_sampling/v1", "budget_points": budget,
            "context_tolerance": context_tolerance, "target_pct": target_pct,
            "coverage_basis": "planning proxy: prefill new tokens / decode iterations; not measured GPU time",
            "shape_policy": "exact shape, sequence count and observed graph mode; context cells within tolerance",
            "candidate_points_before_union": sum(len(c["per_config"]) for c in candidates),
            "candidate_points_after_union": len(candidates), "capture_count": len(selected),
            "config_count": len(cfgs), "groups": groups, "configs": cfgs,
            "captures": [{k: v for k, v in c.items() if k != "gain"} for c in selected],
            "unselected": [{"group_id": c["group_id"], "point_id": c["point_id"],
                            "partition": c["partition"], "context_band": c["context_band"],
                            "per_config": c["per_config"]} for c in remaining],
            "status": "planned" if all(v["target_met"] for c in cfgs.values() for v in c["phases"].values())
                      else "budget_limited"}


def load_configs(items, identity):
    configs = []
    for cid, directory in items:
        directory = Path(directory).resolve()
        serving = directory / "serving.json"
        observed = directory / "evidence/observed_workload.json"
        local_identity = directory / "runtime_identity.json"
        runtime = read(local_identity) if local_identity.exists() else identity
        if runtime is None:
            raise ValueError("provide --runtime-identity or " + str(local_identity))
        configs.append({"config_id": cid, "eval_dir": str(directory), "serving": read(serving),
                        "observed": read(observed), "runtime_identity": runtime,
                        "server_info": server_info(read(serving), directory),
                        "inputs": {"serving": file_ref(serving), "observed": file_ref(observed)}})
        if local_identity.exists():
            configs[-1]["inputs"]["runtime_identity"] = file_ref(local_identity)
    return configs


def write_plan(group_dir, document, configs):
    root = Path(group_dir).resolve()
    by_id = {c["config_id"]: c for c in configs}
    for gid, group in document["groups"].items():
        pool = root / "sampling" / gid
        owner = by_id[group["config_ids"][0]]
        points = [c["point"] for c in document["captures"] if c["group_id"] == gid]
        write(pool / "sampling_points.json", {"contract": "kernel_extract/sampling_points/v2",
              "context_tolerance": document["context_tolerance"], "points": points})
        write(pool / "serving.json", owner["serving"])
        # This pooled observation is for discovery only; project() rebuilds all cost
        # denominators using each original log. Preserve each row's decode interval.
        obs = {"raw_prefill_rows": [], "raw_decode_rows": [], "decode_log_interval": 1}
        for cid in group["config_ids"]:
            src = by_id[cid]["observed"]
            obs["raw_prefill_rows"].extend(src.get("raw_prefill_rows", []))
            obs["raw_decode_rows"].extend(dict(row, iterations=row.get("iterations") or src.get("decode_log_interval") or 1)
                                           for row in src.get("raw_decode_rows", []))
        write(pool / "evidence/observed_workload.json", obs)
        group["eval_dir"] = str(pool)
    write(root / "sampling_plan.json", document)


def checked_inputs(document):
    for cfg in document["configs"].values():
        for ref in cfg["inputs"].values():
            if file_ref(ref["path"])["sha256"] != ref["sha256"]:
                raise ValueError("sampling input changed; replan before projecting: " + ref["path"])


def capture_commands(group_dir):
    """One probe server per compatible deployment, with existing GPU lease handling."""
    root = Path(group_dir).resolve()
    doc = read(root / "sampling_plan.json")
    checked_inputs(doc)
    commands = []
    for gid, group in doc["groups"].items():
        if not group["point_ids"]:
            continue
        pool = Path(group["eval_dir"])
        if deployment(read(pool / "serving.json"), group["deployment"]["runtime_identity"]) != group["deployment"]:
            raise ValueError("capture deployment changed: " + str(pool))
        gpus = str(read(pool / "serving.json").get("gpus", ""))
        if not gpus or any(not g.strip().isdigit() for g in gpus.split(",")):
            raise ValueError("capture pool needs allocated GPU IDs in serving.gpus: " + str(pool))
        commands.append((gid, [sys.executable, str(SLOT / "kernel-extract/shared/gpu_lock_set.py"), gpus, "--",
            sys.executable, str(SCRIPTS / "probe_server.py"), "run",
            "--serving", str(pool / "serving.json"), "--run-dir", str(pool / "probe"), "--",
            sys.executable, str(SCRIPTS / "run_points.py"),
            "--server-state", str(pool / "probe/probe_server.json"),
            "--points", str(pool / "sampling_points.json"),
            "--observed", str(pool / "evidence/observed_workload.json"),
            "--all-evidence", "--eval-dir", str(pool)]))
    return commands


def link(source, dest):
    source, dest = Path(source).resolve(), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() and dest.resolve() == source:
        return
    if dest.exists() or dest.is_symlink():
        raise ValueError("refusing to overwrite evidence: " + str(dest))
    dest.symlink_to(source)


def project(group_dir):
    """CPU only. Share immutable raw evidence; reweight from original logs, never a mean.

    A separate projection tree prevents overwriting independent/previously frozen runs.
    Missing captures stay gaps; an entirely unmeasured phase cannot produce a ranking.
    """
    root = Path(group_dir).resolve()
    doc = read(root / "sampling_plan.json")
    checked_inputs(doc)
    projected, receipts = [], {}
    for cid, cfg in doc["configs"].items():
        pool = Path(doc["groups"][cfg["group_id"]]["eval_dir"])
        dest = root / "projected" / cid
        plan_doc = read(pool / "sampling_points.json")
        expected = {c["point_id"]: c["point"] for c in doc["captures"] if c["group_id"] == cfg["group_id"]}
        if {p["point_id"]: p for p in plan_doc["points"]} != expected:
            raise ValueError("capture pool's plan changed: " + str(pool))
        census_dir, probe_dir = pool / "evidence/census", pool / "evidence/probe"
        run_doc = read(pool / "profile_run.json") if (pool / "profile_run.json").exists() else {}
        for p in plan_doc["points"]:
            path = census_dir / (p["point_id"] + ".json")
            if path.exists():
                contract = run_doc.get("points", {}).get(p["point_id"], {}).get("measurement_signature")
                if contract != p["measurement_signature"]:
                    raise ValueError("capture has no matching measurement signature: " + str(path))
        obs = read(cfg["inputs"]["observed"]["path"])
        final = relabel(obs, dict(plan_doc, _path=str(pool / "sampling_points.json")),
                        str(census_dir), str(probe_dir), 1.25, max(1 + 1e-12, doc["context_tolerance"]))
        final["measurement_reuse"] = {"pool": str(pool), "config_id": cid,
            "basis": "shared capture under identical deployment; this config's own observed frequencies",
            "raw_capture_is_not_an_independent_run_of_this_config": True}
        for p in final["points"]:
            p["measurement_ref"] = {"pool": str(pool), "point_id": p["point_id"],
                                    "signature": p["measurement_signature"]}
            if p.get("counted"):
                p["measurement_ref"]["census"] = file_ref(p["census"])
        write(dest / "sampling_points_final.json", final)
        write(dest / "sampling_points.json", plan_doc)
        write(dest / "serving.json", read(cfg["inputs"]["serving"]["path"]))
        link(cfg["inputs"]["observed"]["path"], dest / "evidence/observed_workload.json")
        for name in ("census", "probe", "shapes", "source_map.json", "adjacency.json", "compile_variants.json"):
            source = pool / "evidence" / name
            if source.exists():
                link(source, dest / "evidence" / name)
        phases_without_measurement = [ph for ph in cfg["phases"]
                                      if not any(p.get("counted") and p["phase"] == ph for p in final["points"])]
        receipt = {"status": "projected", "eval_dir": str(dest), "measurement_pool": str(pool),
                   "missing_phases": phases_without_measurement, "phase_summary": final["phase_summary"]}
        if phases_without_measurement:
            receipt["status"] = "missing_measurements"
            stale = dest / "kernel_ranking.json"
            if stale.exists():
                stale.replace(dest / "kernel_ranking.previous.json")
        else:
            cov = dest / "coverage.json"
            cmd = [sys.executable, str(SCRIPTS / "coverage_ledger.py"), "--points",
                   str(dest / "sampling_points_final.json"), "--out", str(cov)]
            for p in final["points"]:
                if p.get("counted"):
                    cmd += ["--census", p["point_id"] + "=" + p["census"]]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            cmd = [sys.executable, str(SCRIPTS / "build_ranking.py"), "--coverage", str(cov),
                   "--out", str(dest / "kernel_ranking.json")]
            for pid, rec in run_doc.get("points", {}).items():
                if rec.get("capture_mode") == "eager":
                    cmd += ["--eager-point", pid]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            ranking = read(dest / "kernel_ranking.json")
            ranking["measurement_reuse"] = final["measurement_reuse"]
            ranking["warnings"].append("Shared sampling costs reweighted to this config; coverage remains a measured-point model, not exact serving coverage.")
            write(dest / "kernel_ranking.json", ranking)
            projected.append((cid, str(dest)))
        receipts[cid] = receipt
    out = {"contract": "workload/profile/shared_sampling_projection/v1", "configs": receipts,
           "status": "projected" if len(projected) == len(doc["configs"]) else "missing_measurements",
           "merge_configs": [cid + "=" + directory for cid, directory in projected]}
    write(root / "sampling_projection.json", out)
    return out
