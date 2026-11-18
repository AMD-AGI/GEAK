#!/usr/bin/env python3
"""Checkpoint migration between independently launched lineages (GEAK-evo `population_lineage`, PR #580).

Usage:
  population.py publish --run RUN --root POP --lineage A --id <cand> [--round N]
  population.py import  --run RUN --root POP --lineage A --peers B,C

publish: copy a committed (shipped, verified) candidate's cumulative patch, meta and result to
  POP/A/r<N>.{diff,json}, bound to this run's base (sha256 of RUN/orig's editable files) and metric.
import:  for each named peer, take checkpoints with the same base hash and metric that this run has not
  imported yet, and create local nodes `peer_<B>_r<N>` under RUN/cands with status `imported`. They become
  selectable parents/donors. Nothing replaces the local incumbent: an imported node must pass the local
  geak-benchmark measure + geak-ship audit before it can ship.
"""
import argparse, glob, hashlib, json, os, shutil, time


def base_hash(run):
    h = hashlib.sha256()
    cmd_env = os.path.join(run, "cmd.env")
    editable = "kernel_src"
    if os.path.exists(cmd_env):
        for line in open(cmd_env):
            if line.startswith("EDITABLE="):
                editable = line.split("=", 1)[1].strip().strip('"').split()[0]
    root = os.path.join(run, "orig", editable)
    files = sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)) if os.path.isdir(root) else [root]
    for f in files:
        if os.path.isfile(f):
            h.update(os.path.relpath(f, run).encode()); h.update(open(f, "rb").read())
    return h.hexdigest()


def metric(run):
    c = os.path.join(run, "COMMANDMENT.md")
    for line in open(c) if os.path.exists(c) else []:
        if "METRIC" in line:
            return line.strip()[:200]
    return "unknown"


def publish(a):
    src = os.path.join(a.run, "cands", a.id)
    res = json.load(open(os.path.join(src, "result.json")))
    if res.get("status") not in ("ok", "shipped", "verified", "correct"):
        raise SystemExit(f"{a.id}: status {res.get('status')} is not a committed, verified candidate")
    d = os.path.join(a.root, a.lineage); os.makedirs(d, exist_ok=True)
    n = a.round if a.round is not None else len(glob.glob(os.path.join(d, "r*.json")))
    shutil.copy(os.path.join(src, "patch.diff"), os.path.join(d, f"r{n}.diff"))
    meta = json.load(open(os.path.join(src, "meta.json")))
    ck = {"schema": "geak.population.checkpoint/1", "lineage_id": a.lineage, "round": n, "source_id": a.id,
          "base_hash": base_hash(a.run), "metric": metric(a.run), "speedup": res.get("speedup"),
          "approach": meta.get("approach") or meta.get("family"), "per_case": res.get("per_case"),
          "patch_sha256": hashlib.sha256(open(os.path.join(d, f"r{n}.diff"), "rb").read()).hexdigest(),
          "published": time.strftime("%FT%TZ", time.gmtime())}
    json.dump(ck, open(os.path.join(d, f"r{n}.json"), "w"), indent=1)
    print(f"published {a.lineage}/r{n} ({a.id}, {res.get('speedup')})")


def do_import(a):
    mine, met = base_hash(a.run), metric(a.run)
    got = 0
    for peer in [p for p in a.peers.split(",") if p]:
        for f in sorted(glob.glob(os.path.join(a.root, peer, "r*.json"))):
            ck = json.load(open(f))
            nid = f"peer_{peer}_r{ck['round']}"
            dst = os.path.join(a.run, "cands", nid)
            if os.path.exists(dst):
                continue
            if ck["base_hash"] != mine or ck["metric"] != met:
                print(f"skip {nid}: incompatible base or metric"); continue
            diff = f[:-5] + ".diff"
            if hashlib.sha256(open(diff, "rb").read()).hexdigest() != ck["patch_sha256"]:
                print(f"skip {nid}: patch hash mismatch"); continue
            os.makedirs(dst)
            shutil.copy(diff, os.path.join(dst, "patch.diff"))
            json.dump({"id": nid, "parents": ["ORIGINAL"], "role": "import", "approach": ck.get("approach"),
                       "peer_checkpoint": {"lineage": peer, "round": ck["round"], "source_id": ck["source_id"]},
                       "status": "imported"}, open(os.path.join(dst, "meta.json"), "w"), indent=1)
            json.dump({"id": nid, "status": "imported", "speedup": None, "peer_speedup": ck.get("speedup"),
                       "per_case": ck.get("per_case"), "notes": "peer-reported; measure locally before any decision"},
                      open(os.path.join(dst, "result.peer.json"), "w"), indent=1)
            got += 1
            print(f"imported {nid} (peer reported {ck.get('speedup')})")
    print(f"{got} new checkpoints")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish")
    for k in ("--run", "--root", "--lineage", "--id"):
        p.add_argument(k, required=True)
    p.add_argument("--round", type=int)
    i = sub.add_parser("import")
    for k in ("--run", "--root", "--lineage", "--peers"):
        i.add_argument(k, required=True)
    a = ap.parse_args()
    publish(a) if a.cmd == "publish" else do_import(a)


if __name__ == "__main__":
    main()
