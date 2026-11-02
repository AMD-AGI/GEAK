#!/usr/bin/env python3
"""Merge several configs' kernel-extract evidence into ONE eval dir, so a task gets ONE UT.

An InferenceX workload is a set of configs (search-space points: TP/EP/spec/KV-offload x
concurrency), not one point. Each config is extracted on its own first -- its own CI
serving log -> observed_workload.json -> sampling points -> run_points.py, in
`<group>/configs/<cid>/` -- because its shape distribution and its call mass are its own.
This slot script then writes the group dir that the rest of the skill
(draft_task_decls.py, build_task_plan.py, ut_wave.py) reads as if it were one config:

  sampling_points_final.json   every config's points, point_id "<cid>__<pid>", config_id set,
                               and `configs.<cid>.phase_summary` so plan_ut_cases.py plans
                               each config on its own bands
  evidence/shapes, census      the per-point files under the prefixed ids
  evidence/source_map.json     union (a kernel graph-replayed in one config but launched
                               eagerly in another keeps the eager frames)
  evidence/adjacency.json      union of the consecutive-launch pairs
  evidence/compile_variants    union, variant ids prefixed, tensor paths made absolute
  kernel_ranking.json          each row's share = the config-weighted MEAN of its share in
                               each config (absent = a measured 0), configs equal. This is
                               the denominator the slot's selection threshold is applied to.
  multi_config.json            what was merged, with which weights

Correctness of the pooled metric does not depend on this file's arithmetic: every UT case
carries its config_id, and harness_lib normalizes each config's baseline-time mass to its
share before pooling. The ranking mean here only decides WHICH operators get a task.

    python3 multi_config.py merge --group-dir G --config c4=G/configs/c4 --config c8=G/configs/c8

Stdlib only.
"""
import argparse
import json
import os
import shutil
import sys

SEP = "__"


def load(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def dump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def pid(cid, p):
    return "%s%s%s" % (cid, SEP, p)


# --------------------------------------------------------------------------- points
def merge_points(configs):
    """configs: [(cid, eval_dir)] -> merged sampling_points_final.json."""
    points, summaries, tol = [], {}, None
    for cid, d in configs:
        doc = load(os.path.join(d, "sampling_points_final.json"))
        if not doc:
            raise SystemExit("multi_config: %s has no sampling_points_final.json (run run_points.py "
                             "for that config first)" % d)
        tol = tol if tol is not None else doc.get("shape_tolerance")
        for p in doc.get("points", []):
            q = dict(p, point_id=pid(cid, p["point_id"]), config_id=cid,
                     source_point_id=p["point_id"])
            if q.get("census") and not os.path.isabs(q["census"]):
                q["census"] = os.path.abspath(os.path.join(d, q["census"]))
            points.append(q)
        summary = {}
        for ph, s in (doc.get("phase_summary") or {}).items():
            s = dict(s)
            bands = []
            for b in s.get("unrepresented_bands") or []:
                b = dict(b)
                if b.get("nearest_point"):
                    b["nearest_point"] = pid(cid, b["nearest_point"])
                for k in ("bracketing_points", "bracketing_points_2d"):
                    if b.get(k):
                        b[k] = [pid(cid, x) for x in b[k]]
                bands.append(b)
            s["unrepresented_bands"] = bands
            summary[ph] = s
        summaries[cid] = {"phase_summary": summary, "eval_dir": os.path.abspath(d)}
    return {"contract": "kernel_extract/sampling_points_final/v1+multi_config",
            "shape_tolerance": tol, "configs": summaries, "points": points}


def copy_point_files(configs, group, sub):
    """evidence/<sub>/<pid>.json -> group/evidence/<sub>/<cid>__<pid>.json"""
    out = os.path.join(group, "evidence", sub)
    os.makedirs(out, exist_ok=True)
    n = 0
    for cid, d in configs:
        src = os.path.join(d, "evidence", sub)
        if not os.path.isdir(src):
            continue
        for f in sorted(os.listdir(src)):
            if not f.endswith(".json"):
                continue
            doc = load(os.path.join(src, f))
            if doc is None:
                continue
            if isinstance(doc, dict):
                doc["point_id"] = pid(cid, doc.get("point_id") or f[:-5])
                doc["config_id"] = cid
            dump(doc, os.path.join(out, pid(cid, f[:-5]) + ".json"))
            n += 1
    return n


# --------------------------------------------------------------------------- ranking
def merge_ranking(configs, weights):
    docs = []
    for cid, d in configs:
        doc = load(os.path.join(d, "kernel_ranking.json"))
        if not doc or not doc.get("ranking"):
            raise SystemExit("multi_config: %s has no kernel_ranking.json" % d)
        docs.append((cid, doc))
    total_w = sum(weights[c] for c, _ in docs)
    phases = sorted({ph for _, doc in docs for r in doc["ranking"]
                     for ph in (r.get("share_of_phase_pct") or {})})

    def fold(key):
        rows = {}
        for cid, doc in docs:
            w = weights[cid] / total_w
            for r in doc.get(key) or []:
                m = rows.setdefault(r["id"], {"id": r["id"], "name": r.get("name", r["id"]),
                                              "kind": r.get("kind"), "selection_pct_gpu_time": 0.0,
                                              "share_of_phase_pct": {ph: 0.0 for ph in phases},
                                              "per_config_pct": {}})
                m["selection_pct_gpu_time"] += w * r.get("selection_pct_gpu_time", 0.0)
                for ph, v in (r.get("share_of_phase_pct") or {}).items():
                    m["share_of_phase_pct"][ph] += w * v
                m["per_config_pct"][cid] = r.get("selection_pct_gpu_time", 0.0)
        for m in rows.values():
            for cid, _ in docs:
                m["per_config_pct"].setdefault(cid, 0.0)
            m["selection_pct_gpu_time"] = round(m["selection_pct_gpu_time"], 4)
            m["share_of_phase_pct"] = {k: round(v, 4) for k, v in m["share_of_phase_pct"].items()}
            # relative mass on the merged basis: build_task_plan picks a task's device_kernel by it
            m["mass_us"] = m["selection_pct_gpu_time"]
        return sorted(rows.values(), key=lambda r: (-r["selection_pct_gpu_time"], r["id"]))

    return {
        "contract": "kernel_extract/kernel_ranking/v1",
        "denominators": {
            "basis": "config-weighted mean of each config's whole-workload share; mass_us is that "
                     "share in percent, not device time",
            "config_weights": {c: weights[c] / total_w for c, _ in docs},
            "per_config": {c: doc.get("denominators") for c, doc in docs},
        },
        "phase_share": {"basis": "per config, see per_config", "per_config":
                        {c: doc.get("phase_share") for c, doc in docs}},
        "warnings": [("%s: %s" % (c, w)) for c, doc in docs for w in doc.get("warnings") or []],
        "ranking": fold("ranking"),
        "non_selectable": fold("non_selectable"),
    }


# --------------------------------------------------------------------------- evidence
def merge_source_map(configs):
    out = {"contract": "kernel_extract/source_map/v1", "packages": [], "graph_replayed_only": [],
           "kernels": {}, "files": {}, "unresolved_frames": [], "configs": {}}
    replayed = None
    for cid, d in configs:
        sm = load(os.path.join(d, "evidence", "source_map.json"))
        if not sm:
            continue
        out["configs"][cid] = os.path.abspath(os.path.join(d, "evidence", "source_map.json"))
        out["packages"] = sorted(set(out["packages"]) | set(sm.get("packages") or []))
        # replayed-only only if it never launched eagerly anywhere
        g = set(sm.get("graph_replayed_only") or [])
        replayed = g if replayed is None else (replayed & g) | (g - set(out["kernels"]))
        for name, k in (sm.get("kernels") or {}).items():
            have = out["kernels"].get(name)
            if have is None or (not have.get("frames") and k.get("frames")):
                out["kernels"][name] = dict(k, from_config=cid)
        out["files"].update(sm.get("files") or {})
        out["unresolved_frames"] = sorted(set(out["unresolved_frames"])
                                          | set(sm.get("unresolved_frames") or []))
    out["graph_replayed_only"] = sorted(n for n in (replayed or set())
                                        if not out["kernels"].get(n, {}).get("frames"))
    return out


def merge_adjacency(configs):
    out = {"contract": "kernel_extract/adjacency/v1", "traces": [], "launches": {}, "pairs": []}
    pairs = {}
    for cid, d in configs:
        adj = load(os.path.join(d, "evidence", "adjacency.json"))
        if not adj:
            continue
        for k in ("criterion", "note"):
            if k in adj and k not in out:
                out[k] = adj[k]
        out["traces"] += list(adj.get("traces") or [])
        for k, v in (adj.get("launches") or {}).items():
            out["launches"][k] = out["launches"].get(k, 0) + v
        for r in adj.get("pairs") or []:
            key = (r.get("from"), r.get("to"), r.get("kind"))
            pairs.setdefault(key, dict(r, configs=[]))["configs"].append(cid)
    out["pairs"] = list(pairs.values())
    return out


def merge_compile_variants(configs):
    out, n = None, 0
    for cid, d in configs:
        cv = load(os.path.join(d, "evidence", "compile_variants.json"))
        if not cv:
            continue
        if out is None:
            out = {k: v for k, v in cv.items() if k != "variants"}
            out["variants"], out["configs"] = [], {}
        out["configs"][cid] = {"server_mode": cv.get("server_mode"), "watch_dir": cv.get("watch_dir")}
        for r in cv.get("variants") or []:
            r = dict(r, id=pid(cid, r.get("id")), config_id=cid)
            if r.get("tensors") and cv.get("watch_dir") and not os.path.isabs(r["tensors"]):
                r["tensors"] = os.path.join(cv["watch_dir"], r["tensors"])
            out["variants"].append(r)
            n += 1
    return out


# --------------------------------------------------------------------------- cli
def parse_kv(items, what, cast=str):
    out = []
    for it in items or []:
        if "=" not in it:
            raise SystemExit("multi_config: --%s expects cid=value, got %r" % (what, it))
        k, v = it.split("=", 1)
        if not k or SEP in k or "/" in k:
            raise SystemExit("multi_config: config id %r must be nonempty without %r or '/'" % (k, SEP))
        out.append((k, cast(v)))
    return out


def cmd_merge(a):
    configs = parse_kv(a.config, "config")
    if len({c for c, _ in configs}) != len(configs) or not configs:
        raise SystemExit("multi_config: give each config once")
    # Equal by policy. The UT side (harness_lib._config_weights) also equal-weights unless
    # meta.workload.serving_weight_model.config_weights says otherwise; keep the two in step.
    weights = {c: 1.0 for c, _ in configs}
    g = os.path.abspath(a.group_dir)
    os.makedirs(g, exist_ok=True)
    dump(merge_points(configs), os.path.join(g, "sampling_points_final.json"))
    n_shapes = copy_point_files(configs, g, "shapes")
    n_census = copy_point_files(configs, g, "census")
    dump(merge_ranking(configs, weights), os.path.join(g, "kernel_ranking.json"))
    dump(merge_source_map(configs), os.path.join(g, "evidence", "source_map.json"))
    dump(merge_adjacency(configs), os.path.join(g, "evidence", "adjacency.json"))
    cv = merge_compile_variants(configs)
    if cv:
        dump(cv, os.path.join(g, "evidence", "compile_variants.json"))
    first = configs[0][1]
    if os.path.exists(os.path.join(first, "serving.json")) and not os.path.exists(os.path.join(g, "serving.json")):
        shutil.copy2(os.path.join(first, "serving.json"), os.path.join(g, "serving.json"))
    dump({"contract": "workload/profile/multi_config/v1",
          "configs": [{"config_id": c, "eval_dir": os.path.abspath(d), "weight": weights[c]}
                      for c, d in configs],
          "weighting": "per-config normalized in the UT (harness_lib._config_weights); "
                       "kernel_ranking.json is the config-weighted mean share",
          "shapes": n_shapes, "census": n_census},
         os.path.join(g, "multi_config.json"))
    print("merged %d configs into %s (%d shape manifests, %d censuses)"
          % (len(configs), g, n_shapes, n_census))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("merge")
    p.add_argument("--group-dir", required=True)
    p.add_argument("--config", action="append", required=True, help="cid=<config eval dir>")
    p.set_defaults(fn=cmd_merge)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
