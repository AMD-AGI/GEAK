#!/usr/bin/env python3
"""Check one round's search decision before dispatch (the search shell's "check" step).

Usage: check_decision.py --run RUN --decision RUN/decisions/round_<n>.json --strategy SK/v4/search/<name>
                         [--gpus N] [--max-heavy K]
Checks:
  1. schema (v4/search/decision.schema.json; uses `jsonschema` when installed, else built-in checks);
  2. ops are within the strategy's `## Actions used` section (STRATEGY.md);
  3. referenced nodes exist (`ORIGINAL` / `c000` or `RUN/cands/<id>`), and parents of mutate / combine have a patch;
  4. custom roles exist under the strategy's `roles/`;
  5. GPU-heavy actions (mutate, fresh, combine, measure) ≤ max-heavy (default: 2 per GPU, at least 2);
  6. at most one `stop`, and `stop` is alone.
Exit 0 = OK (warnings may be printed), 1 = errors (printed one per line).
"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, "..", "v4", "search", "decision.schema.json")
VERBS = {"mutate", "fresh", "combine", "measure", "investigate", "ship", "custom", "stop"}
NEED = {"mutate": ["parent", "cut", "direction"], "fresh": ["cut", "direction"], "combine": ["parents", "kind"],
        "measure": ["nodes"], "investigate": ["mode", "question"], "ship": ["node"], "custom": ["role"]}
HEAVY = {"mutate", "fresh", "combine", "measure"}


def declared_ops(strategy_dir):
    s = open(os.path.join(strategy_dir, "STRATEGY.md")).read()
    m = re.search(r"^## Actions used\s*\n(.*?)(?=^## |\Z)", s, re.S | re.M)
    if not m:
        return None
    return {v for v in VERBS if re.search(rf"\b{v}\b", m.group(1))}


def node_ok(run, nid, need_patch):
    if nid in ("ORIGINAL", "c000"):
        return True, ""
    d = os.path.join(run, "cands", nid)
    if not os.path.isdir(d):
        return False, f"node {nid} does not exist"
    if need_patch and not os.path.exists(os.path.join(d, "patch.diff")):
        return False, f"node {nid} has no patch.diff (cannot be a parent)"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--decision", required=True)
    ap.add_argument("--strategy", required=True); ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--max-heavy", type=int)
    a = ap.parse_args()
    err, warn = [], []
    d = json.load(open(a.decision))
    try:
        import jsonschema
        jsonschema.validate(d, json.load(open(SCHEMA)))
    except ImportError:
        if not isinstance(d.get("actions"), list):
            err.append("decision.actions must be a list")
        for k in ("round", "strategy"):
            if k not in d:
                err.append(f"decision.{k} missing")
    except Exception as e:  # noqa: BLE001  (jsonschema.ValidationError)
        err.append(f"schema: {getattr(e, 'message', e)}")
    acts = d.get("actions") or []
    ops = declared_ops(a.strategy)
    if ops is None:
        warn.append("strategy has no '## Actions used' section; op set not checked")
    for i, act in enumerate(acts):
        op = act.get("op")
        tag = f"action {i} ({op})"
        if op not in VERBS:
            err.append(f"{tag}: unknown op"); continue
        if ops is not None and op not in ops:
            err.append(f"{tag}: op not declared by the strategy (declared: {sorted(ops)})")
        for k in NEED.get(op, []):
            if k not in act:
                err.append(f"{tag}: missing '{k}'")
        if len(act.get("why", "")) < 10:
            err.append(f"{tag}: 'why' must cite the state it rests on")
        for nid in ([act["parent"]] if "parent" in act else []) + act.get("parents", []):
            ok, msg = node_ok(a.run, nid, True)
            if not ok:
                err.append(f"{tag}: {msg}")
        for nid in act.get("nodes", []) + ([act["node"]] if "node" in act else []) + act.get("refs", []):
            ok, msg = node_ok(a.run, nid, False)
            if not ok:
                err.append(f"{tag}: {msg}")
        if op == "custom" and not os.path.exists(os.path.join(a.strategy, "roles", f"{act.get('role')}.md")):
            err.append(f"{tag}: custom role {act.get('role')} not found in {a.strategy}/roles/")
    heavy = sum(1 for x in acts if x.get("op") in HEAVY)
    cap = a.max_heavy or max(2, 2 * a.gpus)
    if heavy > cap:
        err.append(f"{heavy} GPU-heavy actions > {cap} for {a.gpus} GPU(s)")
    stops = [x for x in acts if x.get("op") == "stop"]
    if stops and len(acts) > 1:
        err.append("stop must be the only action of its round")
    for w in warn:
        print("WARN", w)
    for e in err:
        print("ERROR", e)
    print("OK" if not err else f"{len(err)} error(s)")
    sys.exit(1 if err else 0)


if __name__ == "__main__":
    main()
