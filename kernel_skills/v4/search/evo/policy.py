#!/usr/bin/env python3
"""evo policy hook (reference example of the search shell's optional policy.py).

python3 policy.py --run RUN --archive RUN/archive.json --state RUN/search/evo/state.json [--noise 0.01] [--k 2]
Prints {"state": ..., "proposals": [...], "notes": ...}; writes the state file. Never edits the archive.
Islands = values of the `family` tag. Proposals: for the regimes with the largest gap (cap if available,
else 1/score), a tournament over front nodes on that regime picks the parent of a `mutate`.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "bin"))
import archive  # noqa: E402  (kernel_skills/bin/archive.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--archive", required=True)
    ap.add_argument("--state", required=True); ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--k", type=int, default=2, help="mutation proposals per round")
    a = ap.parse_args()
    arch = json.load(open(a.archive))
    prev = json.load(open(a.state)) if os.path.exists(a.state) else {"round": 0, "islands": {}}
    cap = os.path.join(a.run, "roofline", "cap.json")
    v = archive.views(arch, a.noise, cap if os.path.exists(cap) else None, "family")
    nodes = arch["nodes"]
    front = {f["id"] for f in v["front"]}
    children = {}
    for n in nodes.values():
        for p in n.get("parents", []):
            children[p] = children.get(p, 0) + 1
    # islands: best score, front members, rounds since last gain
    isl = {}
    for nid, sc in v["scores"].items():
        fam = (nodes[nid].get("tags") or {}).get("family") or nodes[nid].get("family") or "-"
        i = isl.setdefault(fam, {"best": 0, "nodes": 0, "front": 0})
        i["best"] = max(i["best"], sc); i["nodes"] += 1; i["front"] += nid in front
    for fam, i in isl.items():
        old = prev["islands"].get(fam, {})
        gained = i["best"] > old.get("best", 0) + a.noise
        i["stale_rounds"] = 0 if gained else old.get("stale_rounds", 0) + 1
        i["retire_candidate"] = i["stale_rounds"] >= 2 and i["front"] == 0
    # regime gaps: distance of the regime's best to 1/score-proxy (cap per regime is not in core views yet)
    regs = v["regimes"]
    best_reg = {r: max(s[r] for s in v["regime_scores"].values()) for r in regs}
    order = sorted(regs, key=lambda r: best_reg[r])        # weakest regime first
    proposals = []
    for r in (order * a.k)[:a.k]:
        cand = sorted(front, key=lambda nid: (-v["regime_scores"][nid][r], children.get(nid, 0)))
        if cand:
            proposals.append({"op": "mutate", "parent": cand[0], "cut": f"regime:{r}",
                              "direction": f"<lead: name the lever for regime {r}>",
                              "why": f"front tournament on regime {r} (best {best_reg[r]}); {cand[0]} leads it with "
                                     f"{children.get(cand[0], 0)} children"})
    if any(i["front"] for i in isl.values()) and len([i for i in isl.values() if i["front"]]) == 1 \
            and max(i["stale_rounds"] for i in isl.values()) >= 2:
        proposals.append({"op": "fresh", "cut": f"regime:{order[0]}", "direction": "<lead: a new organization>",
                          "why": "one island holds the whole front and has not gained for 2 rounds"})
    state = {"round": prev.get("round", 0) + 1, "islands": isl}
    os.makedirs(os.path.dirname(a.state) or ".", exist_ok=True)
    json.dump(state, open(a.state, "w"), indent=1)
    print(json.dumps({"state": state, "proposals": proposals,
                      "notes": f"{len(isl)} islands; front {sorted(front)}; dispatch composite "
                               f"+{v['band_composite']['gain_over_best']:.2%} over best"}, indent=1))


if __name__ == "__main__":
    main()
