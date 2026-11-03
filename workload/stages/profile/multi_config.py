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

Selection (`select`) is per config, then a union: an operator is a candidate as soon as
its share of ANY one config's own workload is above the threshold. A config-averaged share
would bury an operator that dominates some concurrencies and is absent at others -- which
is exactly the M-dependent path switch this merge exists for (Muse Spark: the large-M dense
GEMM path is 5.7% of c64 and 0.2% of c1, mean 1.8%). Each config is selected by the
UNCHANGED select_targets.py on its own kernel_ranking.json, against the merged source map so
operator keys agree across configs, and the per-config target lists are unioned into the
group's selection_targets.json. The UT then pools only the configs the operator runs in.

Correctness of the pooled metric does not depend on this file's arithmetic: every UT case
carries its config_id, and harness_lib normalizes each config's baseline-time mass to its
share before pooling. The ranking mean here only decides WHICH operators get a task.

    python3 multi_config.py merge --group-dir G --config c4=G/configs/c4 --config c8=G/configs/c8
    python3 multi_config.py select --group-dir G [--output G/selection_targets.json]
        (inside the stage: PARAMS supplies selection_top_k / _overall_pct_gt / _phases)

Stdlib only.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
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


def cmd_select(a):
    """Per-config selection (select_targets.py, unchanged) -> union -> selection_targets.json."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from select_candidates import selector_args
    g = os.path.abspath(a.group_dir)
    mc = load(os.path.join(g, "multi_config.json"))
    if not mc:
        raise SystemExit("multi_config: %s has no multi_config.json; run merge first" % g)
    params = load(a.params or os.environ.get("PARAMS") or "", {}) or {}
    arguments = selector_args(params)
    skill = a.skill_root or os.environ.get("SKILL_ROOT") or os.path.join(here, "kernel-extract")
    selector = os.path.join(skill, "scripts", "select_targets.py")
    smap = os.path.join(g, "evidence", "source_map.json")
    merged = {r["id"]: r for r in (load(os.path.join(g, "kernel_ranking.json"), {}) or {}).get("ranking", [])}
    out = os.path.abspath(a.output or os.path.join(g, "selection_targets.json"))
    per_config, union, policy, denoms = {}, {}, None, {}
    for c in mc["configs"]:
        cid, d = c["config_id"], c["eval_dir"]
        ranking = os.path.join(d, "kernel_ranking.json")
        dest = os.path.join(g, "selection", "%s.selection_targets.json" % cid)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        cmd = [sys.executable, selector, ranking, "--output", dest, *arguments]
        if os.path.exists(smap):
            cmd += ["--source-map", smap]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("multi_config: selection failed for %s: %s" % (cid, (r.stderr or r.stdout)[-800:]))
        doc = load(dest)
        policy = policy or doc.get("policy")
        denoms[cid] = doc.get("denominators")
        per_config[cid] = {"targets": len(doc.get("targets") or []), "file": os.path.relpath(dest, g)}
        for t in doc.get("targets") or []:
            u = union.setdefault(t["id"], dict(t, selected_in_configs=[], config_pct={}))
            u["selected_in_configs"].append(cid)
            u["config_pct"][cid] = t.get("overall_pct")
    targets = []
    for tid, t in union.items():
        t["max_config_pct"] = max(v for v in t["config_pct"].values() if v is not None)
        # the share reported downstream is on the merged ranking's basis (the group mean),
        # so it adds up against the same denominator as kernel_ranking.json
        t["overall_pct"] = (merged.get(tid) or {}).get("selection_pct_gpu_time", t["max_config_pct"])
        targets.append(t)
    targets.sort(key=lambda t: (-t["max_config_pct"], t["id"]))
    doc = {"scope": "extraction_target_selection_only",
           "policy": dict(policy or {}, combination="per_config_then_union",
                          rule="an operator is selected when its share of any single config's "
                               "own workload exceeds overall_pct_gt"),
           "denominators": {"basis": "per config (each config's own workload)", "per_config": denoms},
           "per_config": per_config, "targets": targets, "target_count": len(targets),
           "selection_invocation": {
               "arguments": arguments, "per_config": True,
               "selector_sha256": hashlib.sha256(open(selector, "rb").read()).hexdigest(),
               "source_map_sha256": hashlib.sha256(open(smap, "rb").read()).hexdigest()
               if os.path.exists(smap) else None}}
    dump(doc, out)
    print("selected %d targets (union over %d configs) -> %s" % (len(targets), len(per_config), out))
    for t in targets:
        print("  %6.2f%% max  %-40s in %s" % (t["max_config_pct"], t.get("op") or t["name"][:40],
                                             ",".join(t["selected_in_configs"])))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("merge")
    p.add_argument("--group-dir", required=True)
    p.add_argument("--config", action="append", required=True, help="cid=<config eval dir>")
    p.set_defaults(fn=cmd_merge)
    p = sub.add_parser("select")
    p.add_argument("--group-dir", required=True)
    p.add_argument("--output", default=None, help="default <group>/selection_targets.json")
    p.add_argument("--params", default=None, help="default $PARAMS")
    p.add_argument("--skill-root", default=None, help="default $SKILL_ROOT, else ./kernel-extract")
    p.set_defaults(fn=cmd_select)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
