#!/usr/bin/env python3
"""Observed distribution + a budget of N  ->  N profile sampling points.

Step two of turning a completed AgentX baseline into a profile. `observed_workload.py`
says what the measured leg executed; this says which few shapes to actually profile so
that the resulting kernel census stands for that leg rather than for the probe.

The problem it solves
---------------------
A profile window is expensive -- on a stack with a shape-discovery hook, arming one
costs minutes, and the capture bytes are budgeted. So a leg of tens of thousands of
forwards has to be represented by a handful of measurements. Which handful is not a
matter of taste: pick by instance count and the big prefill chunks vanish (they are
4% of batches and 29% of tokens); pick by token count and the modal short extend
vanishes; pick round numbers and you land between dispatch boundaries and measure a
path production never takes.

So points are chosen on three grounds, and each point records which one put it there:

  modal      the shapes most of the mass actually sits at
  boundary   where dispatch changes -- the chunked-prefill ceiling, CUDA-graph capture
             sizes, the dense/sparse attention switch. A kernel can differ on both
             sides, and interpolating across a boundary is not measurement
  tail        material mass far from the mode. Included when it carries enough mass to
             change a conclusion, excluded loudly when it does not

Every observed instance is then assigned to its nearest point, so the output states,
for each point, exactly which mass it stands for and how far it sits from the
instances it represents. Mass that no point represents well is listed as uncovered
rather than quietly folded into the nearest sample.

A caution about the phase split
-------------------------------
Dividing the budget between PREFILL and DECODE needs device-time mass, which is not
known until something has been profiled. This script splits on a PROXY (prefill
tokens, decode iterations) and says so in its output. The proxy is good enough to
start; after the first census, re-run with --phase-budget to spend the remaining
points where the time actually is. Re-planning is cheap, a wrong split is not.

Stdlib only.
"""
import argparse
import json
import math
import sys

# --------------------------------------------------------------------------- #
# Dispatch boundaries. These are properties of the deployment, read from the
# server's effective args -- not guessed, and not defaulted to round numbers.
# --------------------------------------------------------------------------- #
def boundaries_from_server_info(info):
    b = {"prefill_chunk": None, "graph_bs": [], "notes": []}
    if not info:
        b["notes"].append("no server info supplied: dispatch boundaries are UNKNOWN and "
                          "no boundary points were planned. Pass --server-info.")
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


def snap_up(x, sizes):
    """What production actually runs: a batch is padded up to the next captured size."""
    for s in sizes:
        if s >= x:
            return s
    return sizes[-1] if sizes else x


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #
def prefill_candidates(ow, bnd):
    """One candidate per observed token-magnitude bucket, plus the chunk ceiling."""
    cands = []
    for b in ow["prefill"]["buckets"]:
        cands.append({
            "phase": "PREFILL",
            "new_tokens": b["new_token_median"],
            "running": b["running_mode"],
            "cached_tokens": b["cached_token_median"],
            "bucket": b["bucket"],
            "proxy_mass": b["new_token_sum"],
            "instances": b["instances"],
            "grounds": "modal",
        })
    chunk = bnd.get("prefill_chunk")
    if chunk:
        at_ceiling = [b for b in ow["prefill"]["buckets"] if b["new_token_max"] >= chunk]
        if at_ceiling:
            # A batch that hit the ceiling was split by the scheduler; it is a different
            # dispatch from one that fit, even at a similar token count.
            for c in cands:
                if c["bucket"] == at_ceiling[-1]["bucket"]:
                    c["new_tokens"] = chunk
                    c["grounds"] = "modal+boundary"
                    c["boundary"] = "chunked_prefill_size=%d (batches at the ceiling are " \
                                    "scheduler-split; a full chunk is its own dispatch)" % chunk
    return cands


def decode_candidates(ow, bnd):
    cands = []
    graph_bs = bnd.get("graph_bs") or []
    for b in ow["decode"]["buckets"]:
        bs = b["batch_size"]
        padded = snap_up(bs, graph_bs) if graph_bs else None
        cands.append({
            "phase": "DECODE",
            "batch_size": bs,
            "padded_to": padded,
            "context_tokens": b["per_req_context_median"],
            "accept_len": b["accept_len_mean"],
            "proxy_mass": b["iterations"],
            "instances": b["iterations"],
            "grounds": "modal",
        })
    # Under CUDA graph, batches sharing a capture size run the SAME kernel shapes, so
    # measuring both is a wasted window. Collapse them and say so.
    if graph_bs:
        by_pad = {}
        for c in cands:
            by_pad.setdefault(c["padded_to"], []).append(c)
        merged = []
        for pad, group in sorted(by_pad.items()):
            lead = max(group, key=lambda c: c["proxy_mass"])
            if len(group) > 1:
                lead = dict(lead)
                lead["proxy_mass"] = sum(c["proxy_mass"] for c in group)
                lead["instances"] = sum(c["instances"] for c in group)
                lead["represents_batch_sizes"] = [c["batch_size"] for c in group]
                lead["grounds"] = "modal+boundary"
                lead["boundary"] = ("batch sizes %s all pad to captured size %s and run "
                                    "identical kernel shapes" %
                                    ([c["batch_size"] for c in group], pad))
            merged.append(lead)
        cands = merged
    return cands


# --------------------------------------------------------------------------- #
# Budget allocation + selection
# --------------------------------------------------------------------------- #
def allocate(n, phases, explicit=None):
    """Split N across phases. Proportional to proxy mass, floor of 1 per material phase."""
    if explicit:
        return explicit
    total = sum(p["proxy_mass"] for p in phases.values()) or 1
    raw = {k: n * v["proxy_mass"] / total for k, v in phases.items()}
    alloc = {k: max(1, int(math.floor(v))) for k, v in raw.items()}
    # hand out the remainder by largest fractional part
    while sum(alloc.values()) > n and len(alloc) > 1:
        k = min(alloc, key=lambda k: raw[k])
        if alloc[k] > 1:
            alloc[k] -= 1
        else:
            break
    for _ in range(n - sum(alloc.values())):
        k = max(raw, key=lambda k: raw[k] - alloc[k])
        alloc[k] += 1
    return alloc


def select(cands, k):
    """Greedy by proxy mass, but never spend the last point on a candidate that adds
    nothing: boundary candidates are taken before equal-mass modal ones, because a
    boundary cannot be interpolated across and a modal neighbour can."""
    if k >= len(cands):
        return list(cands), []
    ranked = sorted(cands, key=lambda c: (-c["proxy_mass"], "boundary" not in c["grounds"]))
    chosen = ranked[:k]
    # Force in any boundary candidate that greedy missed, displacing the lowest-mass pick.
    bnds = [c for c in ranked[k:] if "boundary" in c["grounds"]]
    for b in bnds:
        if len(chosen) < k:
            chosen.append(b)
            continue
        weakest = min(chosen, key=lambda c: c["proxy_mass"])
        if "boundary" not in weakest["grounds"] and b["proxy_mass"] > 0:
            chosen.remove(weakest)
            chosen.append(b)
    rejected = [c for c in cands if c not in chosen]
    return chosen, rejected


def assign(cands, chosen, key):
    """Map every candidate's mass to its nearest chosen point, log-distance in `key`.

    This is the ledger: it is what lets a reader see that a point stands for 29% of
    the tokens and that the furthest instance it speaks for is 1.7x away in shape.
    """
    out = {id(c): {"mass": 0, "instances": 0, "represents": [], "max_ratio": 1.0}
           for c in chosen}
    per_cand = {}
    unassigned = []
    for c in cands:
        best, bestd = None, None
        for p in chosen:
            if p["phase"] != c["phase"]:
                continue
            a, b = float(c.get(key) or 0), float(p.get(key) or 0)
            if a <= 0 or b <= 0:
                d = 0.0 if a == b else float("inf")
            else:
                d = abs(math.log(a / b))
            if bestd is None or d < bestd:
                best, bestd = p, d
        if best is None:
            unassigned.append(c)
            continue
        e = out[id(best)]
        e["mass"] += c["proxy_mass"]
        e["instances"] += c["instances"]
        ratio = round(math.exp(bestd), 3) if bestd != float("inf") else None
        e["represents"].append({key: c.get(key), "proxy_mass": c["proxy_mass"],
                                "shape_ratio": ratio})
        if ratio is not None:
            e["max_ratio"] = max(e["max_ratio"], ratio)
        per_cand[id(c)] = {"point": best, "ratio": ratio}
    return out, unassigned, per_cand


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--observed", required=True, help="observed_workload.json")
    ap.add_argument("-n", "--num-points", type=int, required=True,
                    help="how many profile windows the budget affords")
    ap.add_argument("--server-info", default=None,
                    help="JSON from /get_server_info; supplies dispatch boundaries")
    ap.add_argument("--phase-budget", default=None,
                    help="override the proxy split, e.g. PREFILL=3,DECODE=2")
    ap.add_argument("--far-ratio", type=float, default=2.0,
                    help="an instance further than this factor from its point is "
                         "reported as poorly represented")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ow = json.load(open(args.observed))
    info = json.load(open(args.server_info)) if args.server_info else None
    bnd = boundaries_from_server_info(info)

    pf = prefill_candidates(ow, bnd)
    dc = decode_candidates(ow, bnd)

    explicit = None
    if args.phase_budget:
        explicit = {}
        for part in args.phase_budget.split(","):
            k, v = part.split("=")
            explicit[k.strip().upper()] = int(v)

    phases = {"PREFILL": {"proxy_mass": sum(c["proxy_mass"] for c in pf)},
              "DECODE": {"proxy_mass": sum(c["proxy_mass"] for c in dc)}}
    # The two proxies are in different units (tokens vs iterations) and are NOT
    # comparable as device time. Normalize each to a share of its own phase, then
    # split the budget evenly unless the caller knows better. Pretending tokens and
    # iterations are the same currency would silently decide the split.
    if not explicit:
        phases = {k: {"proxy_mass": 1.0} for k in phases}
    alloc = allocate(args.num_points, phases, explicit)

    pf_sel, pf_rej = select(pf, alloc.get("PREFILL", 0))
    dc_sel, dc_rej = select(dc, alloc.get("DECODE", 0))

    pf_led, _, pf_near = assign(pf, pf_sel, "new_tokens")
    dc_led, _, dc_near = assign(dc, dc_sel, "batch_size")

    points, uncovered = [], []
    for i, (sel, led, key, rej, near) in enumerate(
            ((pf_sel, pf_led, "new_tokens", pf_rej, pf_near),
             (dc_sel, dc_led, "batch_size", dc_rej, dc_near))):
        phase_total = sum(c["proxy_mass"] for c in (pf if i == 0 else dc)) or 1
        for p in sel:
            e = led[id(p)]
            far = [r for r in e["represents"] if r["shape_ratio"] > args.far_ratio]
            pt = dict(p)
            pt.pop("proxy_mass", None)
            pt["point_id"] = "%s_%s" % (p["phase"][:2].lower(), p.get(key))
            pt["represented"] = {
                "instances": e["instances"],
                "proxy_mass": e["mass"],
                "phase_proxy_mass_share_pct": round(100.0 * e["mass"] / phase_total, 3),
                "max_shape_ratio": e["max_ratio"],
                "poorly_represented": far,
            }
            points.append(pt)
        for r in rej:
            n = near.get(id(r)) or {}
            npt, ratio = n.get("point"), n.get("ratio")
            uncovered.append({
                "phase": r["phase"], key: r.get(key),
                "proxy_mass": r["proxy_mass"],
                "phase_proxy_mass_share_pct": round(100.0 * r["proxy_mass"] / phase_total, 3),
                "nearest_point": ("%s_%s" % (r["phase"][:2].lower(), npt.get(key))
                                  if npt else None),
                "shape_ratio_to_nearest": ratio,
                "reason": "outside the budget of %d points for this phase; its mass is "
                          "attributed to the nearest point, %sx away in shape"
                          % (alloc.get(r["phase"], 0), ratio),
            })

    out = {
        "observed_workload": args.observed,
        "budget_points": args.num_points,
        "phase_budget": alloc,
        "phase_budget_basis": ("explicit --phase-budget" if explicit else
                               "EVEN split. The prefill proxy (tokens) and the decode "
                               "proxy (iterations) are different units and cannot be "
                               "compared as device time. Re-run with --phase-budget "
                               "after the first census to spend points where the time is."),
        "dispatch_boundaries": bnd,
        "points": points,
        "uncovered_candidates": uncovered,
        "how_to_read": {
            "proxy_mass": "PREFILL: new tokens. DECODE: iterations. A within-phase "
                          "ranking proxy, NOT device time.",
            "max_shape_ratio": "how far, multiplicatively, the furthest instance this "
                               "point stands for sits from the point itself. 1.0 is exact; "
                               "large values mean the point is extrapolating.",
            "next_step": "run each point with serving_probe.py, then coverage_ledger.py "
                         "to convert measured per-point kernel time into whole-leg mass.",
        },
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    print("budget %d -> %s" % (args.num_points, alloc))
    for p in points:
        r = p["represented"]
        print("  %-14s %-28s %6.2f%% of phase proxy mass, %d instances, max shape ratio %.2f%s"
              % (p["point_id"],
                 ("new_tokens=%s running=%s" % (p.get("new_tokens"), p.get("running")))
                 if p["phase"] == "PREFILL"
                 else ("bs=%s pad=%s ctx=%s" % (p.get("batch_size"), p.get("padded_to"),
                                                p.get("context_tokens"))),
                 r["phase_proxy_mass_share_pct"], r["instances"], r["max_shape_ratio"],
                 "  [%s]" % p["grounds"]))
    if uncovered:
        print("uncovered candidates: %d (see uncovered_candidates in %s)"
              % (len(uncovered), args.out))
    for n in bnd["notes"]:
        print("  boundary: %s" % n)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    sys.exit(main())
