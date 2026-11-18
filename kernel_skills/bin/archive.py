#!/usr/bin/env python3
"""Search archive for a kernel run: every measured node with per-case results, and the views a search
policy decides on.

Usage:
  archive.py build  --run RUN                    # v4 layout: RUN/cands/*/{meta,result}.json -> RUN/archive.json
  archive.py build  --ut-reports 'GLOB' --out F  # any set of UT reports (grouped by snapshot_sha256)
  archive.py views  ARCHIVE [--noise 0.01] [--cap RUN/roofline/cap.json] [--md OUT.md]

A node = one code version: id, parents, design-family tag, cut, status, cost, per-case {speedup, weight, n}.
Repeated measurements of the same node are pooled per case (median, spread). Views:
  best        the node with the best weighted score (score = sum(w) / sum(w / s), the GEAK metric)
  front       noise-aware Pareto front over regimes: a node is dominated only if another is better by
              more than NOISE on every regime and better overall
  per_case    the fastest node per case and, beyond noise, the "dispatch composite": the score if each
              regime ran its best node (an upper bound for crossover by per-regime dispatch)
  families    best score and node count per design family (diversity / exhausted families)
  gap         per regime: best speedup vs roofline cap, and the weighted gap left (needs --cap)
"""
import argparse, collections, glob, json, os, statistics, sys


def wscore(cases):
    w = sum(c["weight"] for c in cases.values())
    return w / sum(c["weight"] / c["speedup"] for c in cases.values()) if w else None


def regime_of(sig, case):
    return case.get("regime") or sig.split("_")[0]


def build_run(run):
    nodes = {}
    for meta_f in glob.glob(os.path.join(run, "cands", "*", "meta.json")):
        d = os.path.dirname(meta_f)
        meta = json.load(open(meta_f))
        nid = meta.get("id") or os.path.basename(d)
        res = {}
        for rf in sorted(glob.glob(os.path.join(d, "result*.json"))):
            try:
                res = json.load(open(rf))
            except Exception:  # noqa: BLE001
                continue
        per = collections.defaultdict(list)
        weights = {}
        rows = res.get("per_case") or []
        if isinstance(rows, dict):   # {sig: row} or {sig: speedup}
            rows = [dict(v, sig=k) if isinstance(v, dict) else {"sig": k, "speedup": v} for k, v in rows.items()]
        for c in rows:
            if not isinstance(c, dict):
                continue
            sig = c.get("sig") or c.get("name")
            if sig and c.get("speedup"):
                per[sig].append(c["speedup"])
                weights[sig] = (c.get("weight", 1.0), c.get("regime", ""))
        sp = res.get("speedup")
        rs = (res.get("status") or "").lower()
        if rs in ("incorrect", "apply_failed", "frozen_touched", "broken", "invalid"):
            evo_status = "incorrect"
        elif not isinstance(sp, (int, float)):
            evo_status = "unverified"
        else:
            evo_status = "slower" if sp <= 1.0 else "correct"
        patch = os.path.join(d, "patch.diff")
        sha = None
        if os.path.exists(patch):
            import hashlib
            sha = hashlib.sha256(open(patch, "rb").read()).hexdigest()
        parents = meta.get("parents", [])
        nodes[nid] = {"id": nid, "parent_id": parents[0] if parents else "ORIGINAL", "donor_ids": meta.get("donors", parents[1:]),
                      "approach": meta.get("approach") or meta.get("family", ""), "evo_status": evo_status,
                      "patch_ref": {"path": patch if sha else None, "sha256": sha},
                      "parents": parents, "family": meta.get("approach") or meta.get("family", ""),
                      "cut": meta.get("cut", ""), "role": meta.get("role", ""), "status": res.get("status", meta.get("status", "")),
                      "official": res.get("speedup"), "cost_usd": meta.get("cost_usd"),
                      "cases": {s: {"speedup": statistics.median(v), "n": len(v), "weight": weights[s][0],
                                    "regime": weights[s][1]} for s, v in per.items()}}
    best = None
    sp_f = os.path.join(run, "shipped", "ID")
    if os.path.exists(sp_f):
        best = open(sp_f).read().strip()
    if not best:
        ok = [n for n in nodes.values() if n["evo_status"] == "correct" and isinstance(n.get("official"), (int, float))]
        best = max(ok, key=lambda n: n["official"])["id"] if ok else "ORIGINAL"
    # geak.evolution/1-compatible view (PR #580): entries with parent/donors/approach/status/patch_ref
    entries = [{"id": n["id"], "parent_id": n["parent_id"], "donor_ids": n["donor_ids"], "approach": n["approach"],
                "specialty": n.get("cut", ""), "patch_ref": n["patch_ref"], "verified": n["evo_status"] in ("correct", "slower"),
                "speedup": n.get("official"), "status": n["evo_status"], "case_rows": len(n["cases"])}
               for n in nodes.values()]
    return {"schema": "geak.evolution/1", "base": os.path.join(run, "orig"), "best_id": best,
            "entries": entries, "source": run, "nodes": nodes}


def build_reports(pattern):
    groups = collections.defaultdict(list)
    for f in glob.glob(pattern):
        try:
            r = json.load(open(f))
            agg = r["gates"]["timing"]["aggregate"]
        except Exception:  # noqa: BLE001
            continue
        groups[r.get("snapshot_sha256", f)].append((f, agg))
    nodes = {}
    for sha, runs in groups.items():
        per = collections.defaultdict(list)
        meta = {}
        for _, agg in runs:
            for c in agg.get("per_case", []):
                if c.get("included", True) and c.get("speedup"):
                    per[c["sig"]].append(c["speedup"])
                    meta[c["sig"]] = (c.get("weight", 1.0), c.get("regime", ""))
        nid = sha[:10]
        nodes[nid] = {"id": nid, "parents": [], "family": "", "cut": "", "role": "", "status": "measured",
                      "official": statistics.median(a.get("primary") or a.get("weighted") or 0 for _, a in runs),
                      "runs": len(runs),
                      "cases": {s: {"speedup": statistics.median(v), "n": len(v), "spread": (max(v) - min(v)) if len(v) > 1 else None,
                                    "weight": meta[s][0], "regime": meta[s][1]} for s, v in per.items()}}
    return {"source": pattern, "nodes": nodes}


def views(arch, noise, cap_path):
    nodes = {k: n for k, n in arch["nodes"].items() if n["cases"]}
    if not nodes:
        sys.exit("archive has no node with per-case results")
    # common case set: cases measured on the baseline-weighted majority of nodes
    counts = collections.Counter(s for n in nodes.values() for s in n["cases"])
    common = {s for s, k in counts.items() if k >= max(1, len(nodes) // 2)}
    full = {k: n for k, n in nodes.items() if common <= set(n["cases"])}
    if not full:
        sys.exit("no node covers the common case set")
    ref = next(iter(full.values()))["cases"]
    W = {s: ref[s]["weight"] for s in common}
    R = {s: regime_of(s, ref[s]) for s in common}
    score = {k: wscore({s: n["cases"][s] for s in common}) for k, n in full.items()}
    best = max(score, key=score.get)
    regimes = sorted(set(R.values()))
    def reg_score(n, r):
        return wscore({s: n["cases"][s] for s in common if R[s] == r})
    regs = {k: {r: reg_score(n, r) for r in regimes} for k, n in full.items()}
    front = []
    for a in full:
        dominated = any(score[b] > score[a] + noise and all(regs[b][r] > regs[a][r] + noise for r in regimes)
                        for b in full if b != a)
        if not dominated:
            front.append(a)
    # per-regime dispatch composite: each regime runs the node best on that regime (beyond noise)
    pick = {}
    for r in regimes:
        cand = max(full, key=lambda k: regs[k][r])
        pick[r] = cand if regs[cand][r] > regs[best][r] + noise else best
    comp_cases = {s: full[pick[R[s]]]["cases"][s] for s in common}
    composite = wscore(comp_cases)
    # finer: regime x size band (m from the case row or the sig), still a legal dispatch key
    def band(s):
        c = ref[s]; m = c.get("m")
        if m is None:
            import re
            mm = re.search(r"(?:_m|_b)(\d+)", s); m = int(mm.group(1)) if mm else 0
        return f"{R[s]}:{'S' if m <= 16 else 'M' if m <= 64 else 'L'}"
    B = {s: band(s) for s in common}
    bpick = {}
    for g in sorted(set(B.values())):
        ss = [s for s in common if B[s] == g]
        bs = lambda k: wscore({s: full[k]["cases"][s] for s in ss})
        cand = max(full, key=bs)
        bpick[g] = cand if bs(cand) > bs(best) + noise else best
    band_comp = wscore({s: full[bpick[B[s]]]["cases"][s] for s in common})
    fam = collections.defaultdict(list)
    for k, n in full.items():
        fam[n.get("family") or "-"].append(score[k])
    out = {"n_nodes": len(full), "regimes": regimes, "best": {"id": best, "score": round(score[best], 4)},
           "front": sorted(({"id": k, "score": round(score[k], 4), "regimes": {r: round(v, 4) for r, v in regs[k].items()}}
                            for k in front), key=lambda x: -x["score"]),
           "dispatch_composite": {"score": round(composite, 4), "gain_over_best": round(composite / score[best] - 1, 4),
                                  "pick": pick},
           "band_composite": {"score": round(band_comp, 4), "gain_over_best": round(band_comp / score[best] - 1, 4),
                              "pick": bpick},
           "families": {f: {"n": len(v), "best": round(max(v), 4)} for f, v in fam.items()}}
    if cap_path and os.path.exists(cap_path):
        cap = json.load(open(cap_path))
        caps = {r["name"]: r["cap"] for r in cap.get("rows", [])}
        out["gap"] = {r: {"best": round(max(regs[k][r] for k in full), 4)} for r in regimes}
        out["cap"] = cap.get("cap")
        if caps:
            out["gap_note"] = "per-case caps available in cap.json rows; compare regime bests with them"
    return out


def to_md(v):
    L = [f"Nodes with full per-case results: {v['n_nodes']}; regimes: {', '.join(v['regimes'])}",
         f"Best: {v['best']['id']} {v['best']['score']}",
         f"Dispatch composite (each regime runs its best node beyond noise): {v['dispatch_composite']['score']} "
         f"(+{v['dispatch_composite']['gain_over_best']:.2%} over best) pick={v['dispatch_composite']['pick']}",
         f"Band composite (regime x size band): {v['band_composite']['score']} (+{v['band_composite']['gain_over_best']:.2%})",
         "", "| front node | score | " + " | ".join(v["regimes"]) + " |", "|---|---|" + "---|" * len(v["regimes"])]
    for f in v["front"]:
        L.append(f"| {f['id']} | {f['score']} | " + " | ".join(str(f["regimes"][r]) for r in v["regimes"]) + " |")
    L.append("\nFamilies: " + ", ".join(f"{k}: n={x['n']} best={x['best']}" for k, x in v["families"].items()))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--run"); b.add_argument("--ut-reports"); b.add_argument("--out")
    w = sub.add_parser("views"); w.add_argument("archive"); w.add_argument("--noise", type=float, default=0.01)
    w.add_argument("--cap"); w.add_argument("--md"); w.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "build":
        arch = build_run(a.run) if a.run else build_reports(a.ut_reports)
        out = a.out or os.path.join(a.run, "archive.json")
        json.dump(arch, open(out, "w"), indent=1)
        print(f"{len(arch['nodes'])} nodes -> {out}")
    else:
        v = views(json.load(open(a.archive)), a.noise, a.cap)
        md = to_md(v)
        print(md)
        if a.md:
            open(a.md, "w").write(md + "\n")
        if a.json:
            json.dump(v, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
