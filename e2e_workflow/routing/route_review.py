#!/usr/bin/env python3
"""route_review.py — the cost ladder's after-run review. Reports; never changes the router.

Reads one finished run and writes two files:

  route_review.md        where every dollar went, by lane: retries, escalations and their reasons,
                         cache reuse, savings, and a closing "WHERE DID I OVERSPEND?" section.
  proposed_floors.json   per-scope starting floors for the NEXT run. A proposal only: nothing reads
                         this file. An operator who agrees passes it as the `route_floors` workflow arg.

Inputs:
  --calls    the ledger's reports/geak_calls.jsonl for the routed run (llm_ledger.py writes it)
  --routing  any JSON holding the kernel lane's `routing` block (the lane result, or a workflow return
             that nests it); optional, but without it there are no decisions to review
  --control  geak_calls.jsonl of a MATCHED run with routing OFF; optional. The only source of a real
             savings number. Without it, savings are a repricing counterfactual and are labelled so.

    python3 e2e_workflow/routing/route_review.py --calls <eval>/reports/geak_calls.jsonl \\
        --routing <eval>/workflow_return.json [--control <ctl>/reports/geak_calls.jsonl] --out <dir>

Policy: e2e_workflow/routing/SKILL.md.
"""
import argparse
import json
import os
import sys
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import llm_ledger as L  # noqa: E402  (rates and cost functions: one source of truth)

BRAIN_MODEL = "claude-opus-5-5"
# Models on the tokenizer from before Claude Opus 4.7: the same text counts about 30% fewer tokens on them,
# which repricing their tokens at BRAIN_MODEL rates cannot see.
OLD_TOKENIZER = ("claude-opus-4-6", "claude-opus-4-5", "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-haiku-4-5")


def scope_of(label):
    """Python twin of kernel_lane.js __routeScope: 'eng d3:memory' -> 'eng:memory'."""
    parts = str(label or "").split(" ")
    rest = parts[1] if len(parts) > 1 else ""
    return parts[0] + ":" + rest[rest.index(":") + 1:] if ":" in rest else parts[0]


def load_calls(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_routing(obj):
    """Depth-first search for every dict that looks like a lane's routing block."""
    found = []
    if isinstance(obj, dict):
        if "lanes" in obj and "audit" in obj:
            found.append(obj)
        for v in obj.values():
            found += find_routing(v)
    elif isinstance(obj, list):
        for v in obj:
            found += find_routing(v)
    return found


def cost(row, rates):
    return row["cost_usd"] if "cost_usd" in row else L.cost_of(row, rates)


def _tok(row, k):
    return int(row.get(k) or 0)


def outcomes(routing):
    """Per scope, each dispatch in order with its result: 'fail' if a fail event for the scope came
    before the scope's next dispatch, else 'ok'. Also the decisions and reasons the ladder recorded."""
    lanes = routing.get("lanes") or []
    per = OrderedDict()
    for ev in routing.get("audit") or []:
        sc = ev.get("scope")
        if sc is None:
            continue
        s = per.setdefault(sc, {"dispatches": [], "decisions": [], "skips": 0})
        kind = ev.get("event")
        if kind == "dispatch":
            s["dispatches"].append({"model": ev.get("model"), "lane": lanes.index(ev["model"]) if ev.get("model") in lanes else None,
                                    "capped": bool(ev.get("capped")), "result": "ok", "evidence": ""})
        elif kind == "fail" and s["dispatches"]:
            s["dispatches"][-1]["result"] = "fail"
            s["dispatches"][-1]["evidence"] = ev.get("evidence", "")
        elif kind == "decide":
            s["decisions"].append({"lane": ev.get("lane"), "why": ev.get("why", "")})
        elif kind == "skip":
            s["skips"] += 1
    return per


def review(calls, routing, control, rates):
    lanes = (routing or {}).get("lanes") or list(L.DEFAULT_RATES.keys())
    total = sum(cost(r, rates) for r in calls)
    router = sum(r.get("cost_breakdown", {}).get("router", 0.0) for r in calls)
    by_model = defaultdict(lambda: {"calls": 0, "usd": 0.0})
    by_scope_model = defaultdict(float)
    for r in calls:
        m = r.get("model") or "?"
        by_model[m]["calls"] += 1
        by_model[m]["usd"] += cost(r, rates)
        by_scope_model[(scope_of(r.get("agent_label")), m)] += cost(r, rates)
    inp = sum(L.total_input(r) for r in calls if "input_tokens" in r)
    reads = sum(_tok(r, "cache_read_input_tokens") for r in calls)
    # Counterfactual: the same token buckets, all billed at the brain model's card.
    brain_rates = dict(rates, **{"_default": rates.get(BRAIN_MODEL, rates["_default"])})
    repriced = sum(L.cost_of(dict(r, model="_default"), brain_rates) for r in calls if "input_tokens" in r)
    ctl = sum(cost(r, rates) for r in control) if control is not None else None

    per = outcomes(routing) if routing else OrderedDict()
    # Cost of failed dispatches, per (scope, model): the scope-model cost split evenly over that pair's
    # dispatches. An allocation, not a measurement — the ledger does not tag calls with dispatch ids.
    wasted, wasted_rows = 0.0, []
    for sc, s in per.items():
        by_m = defaultdict(lambda: [0, 0])
        for d in s["dispatches"]:
            by_m[d["model"]][0] += 1
            by_m[d["model"]][1] += d["result"] == "fail"
        for m, (n, f) in by_m.items():
            if f:
                usd = by_scope_model.get((sc, m), 0.0) * f / n
                wasted += usd
                wasted_rows.append((sc, m, f, n, usd))

    proposals, notes = {}, []
    for sc, s in per.items():
        ds = [d for d in s["dispatches"] if d["lane"] is not None]
        if not ds:
            continue
        first = ds[0]["lane"]
        ok = [d["lane"] for d in ds if d["result"] == "ok"]
        if ok and min(ok) > first:
            proposals[sc] = min(ok)
            notes.append(f"`{sc}`: started on {lanes[first]}, first success on {lanes[min(ok)]} — "
                         f"propose floor {min(ok)} to skip the failed lanes next time.")
        elif ok and all(d["result"] == "ok" for d in ds) and first > 0 and len(ds) >= 3:
            notes.append(f"`{sc}`: {len(ds)}/{len(ds)} first-try successes on {lanes[first]} — a lower lane is "
                         f"UNTESTED, not proven; worth one trial run at lane {first - 1}, not a floor change.")
    return dict(total=total, router=router, by_model=by_model, inp=inp, reads=reads, repriced=repriced,
                control=ctl, per=per, wasted=wasted, wasted_rows=wasted_rows, proposals=proposals,
                notes=notes, lanes=lanes, n_calls=len(calls))


def _usd(x):
    return "$%.2f" % x


def render(R, routing):
    # Name the decider the run actually used (it is in the lane's routing block), so a review of an older
    # run is not relabelled with today's model.
    decider = "`%s` decider" % routing["decider"] if routing and routing.get("decider") else "decider"
    L_ = ["# Routing review", ""]
    L_ += ["## Spend by lane", "", "| model | calls | spend | share |", "|---|---:|---:|---:|"]
    for m, v in sorted(R["by_model"].items(), key=lambda kv: -kv[1]["usd"]):
        L_.append("| `%s` | %d | %s | %.1f%% |" % (m, v["calls"], _usd(v["usd"]),
                                                  100 * v["usd"] / R["total"] if R["total"] else 0))
    L_ += ["", "- total: **%s** over %d calls" % (_usd(R["total"]), R["n_calls"]),
           "- of which the router (%s): %s (%.1f%%)" % (
               decider, _usd(R["router"]), 100 * R["router"] / R["total"] if R["total"] else 0),
           "- cache reuse: %.1f%% of input tokens were cache reads" % (100 * R["reads"] / R["inp"] if R["inp"] else 0), ""]

    L_ += ["## Savings", ""]
    if R["control"] is not None:
        d = R["control"] - R["total"]
        L_.append("**Measured** against a matched routing-OFF run: control %s, routed %s, saved %s (%.1f%%)."
                  % (_usd(R["control"]), _usd(R["total"]), _usd(d), 100 * d / R["control"] if R["control"] else 0))
        L_.append("Check the two runs' speedups before quoting this: a saving that bought a worse kernel is not a saving.")
    else:
        d = R["repriced"] - R["total"]
        L_.append("No control run given, so there is **no measured saving**. Counterfactual only: the same tokens "
                  "billed at `%s` rates would cost %s, so routing's price difference is %s (%.1f%%)."
                  % (BRAIN_MODEL, _usd(R["repriced"]), _usd(d), 100 * d / R["repriced"] if R["repriced"] else 0))
        old = sorted(m for m in R["by_model"] if L.rate_key(m, dict.fromkeys(OLD_TOKENIZER)))
        tok = (", and %s use%s an older tokenizer that counts about 30%% fewer tokens for the same text"
               % (" and ".join("`%s`" % m for m in old), "s" if len(old) == 1 else "")) if old else ""
        L_.append("This is an estimate, not a fact: a different model writes different outputs%s. Pass "
                  "`--control` with a matched routing-OFF run to get the real number." % tok)
    L_.append("")

    if routing:
        th = routing.get("thresholds", {})
        L_ += ["## Ladder", "",
               "Thresholds: escalate below confidence %s; max %s failures per lane; max %s worker dispatch(es) on "
               "the top lane; output-token cap %s; floors %s."
               % (th.get("conf_escalate"), th.get("max_retries_per_lane"), th.get("max_top_escalations"),
                  th.get("max_output_tokens"), json.dumps(th.get("floors") or {})), ""]
        if routing.get("killed"):
            L_ += ["**The output-token kill switch tripped** at %s output tokens. Work after that point was skipped."
                   % routing.get("spent_output_tokens"), ""]
        L_ += ["| scope | dispatches | lanes used | failures | why (last decision) |", "|---|---:|---|---:|---|"]
        for sc, s in R["per"].items():
            ds = s["dispatches"]
            used = " → ".join(OrderedDict.fromkeys(d["model"].replace("claude-", "") for d in ds if d["model"]))
            why = s["decisions"][-1]["why"] if s["decisions"] else ""
            L_.append("| `%s` | %d | %s | %d | %s |" % (sc, len(ds), used, sum(d["result"] == "fail" for d in ds), why))
        L_.append("")
        esc = [(sc, d["why"]) for sc, s in R["per"].items() for d in s["decisions"] if d["why"].startswith("escalate")]
        L_ += ["### Escalations", ""] + (["- `%s`: %s" % e for e in esc] or ["- none"]) + [""]

    L_ += ["## WHERE DID I OVERSPEND?", ""]
    items = []
    if R["wasted_rows"]:
        items.append("**Failed attempts: about %s.** Dispatches whose deterministic check failed, cost split "
                     "evenly across each scope's dispatches on that model (an allocation, not a measurement):" % _usd(R["wasted"]))
        items += ["  - `%s` on `%s`: %d of %d failed, ~%s" % (sc, m, f, n, _usd(u)) for sc, m, f, n, u in R["wasted_rows"]]
    if R["router"]:
        items.append("**Choosing cost %s.** Every decider classification is overhead; it pays only when it "
                     "moves work to a cheaper lane than the default would have." % _usd(R["router"]))
    top = [(sc, d) for sc, s in R["per"].items() for d in s["dispatches"] if d["lane"] == len(R["lanes"]) - 1]
    if top:
        items.append("**Top-lane worker dispatches: %d**, %d failed anyway." % (len(top), sum(d["result"] == "fail" for _, d in top)))
    capped = sum(d["capped"] for s in R["per"].values() for d in s["dispatches"])
    if capped:
        items.append("**%d dispatch(es) wanted Opus 5.5 but were capped to Opus 4.6.** If they then failed, the cap cost quality." % capped)
    if R["control"] is not None and R["total"] > R["control"]:
        items.append("**Routing cost more than it saved:** routed %s vs control %s." % (_usd(R["total"]), _usd(R["control"])))
    L_ += (["- " + i if not i.startswith("  ") else i for i in items] or ["- Nothing found in this run's evidence."]) + [""]

    L_ += ["## Proposed changes (for your approval — nothing is applied)", ""]
    L_ += ["- " + n for n in R["notes"]] or ["- none"]
    if R["proposals"]:
        L_ += ["", "To apply, pass this as the `route_floors` workflow arg of the next run:", "",
               "```json", json.dumps(R["proposals"], sort_keys=True), "```"]
    return "\n".join(L_) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calls", required=True)
    ap.add_argument("--routing", default=None)
    ap.add_argument("--control", default=None)
    ap.add_argument("--out", default=".")
    a = ap.parse_args(argv)
    calls = load_calls(a.calls)
    routing = None
    if a.routing:
        with open(a.routing, encoding="utf-8") as fh:
            blocks = find_routing(json.load(fh))
        if blocks:
            routing = blocks[0]
            for b in blocks[1:]:          # several lanes: one review over all of them
                routing = dict(routing, audit=routing["audit"] + b["audit"],
                               killed=routing.get("killed") or b.get("killed"))
        else:
            print("route_review: no routing block in %s (was routing ON?)" % a.routing, file=sys.stderr)
    control = load_calls(a.control) if a.control else None
    R = review(calls, routing, control, L.DEFAULT_RATES)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "route_review.md"), "w", encoding="utf-8") as fh:
        fh.write(render(R, routing))
    with open(os.path.join(a.out, "proposed_floors.json"), "w", encoding="utf-8") as fh:
        json.dump(R["proposals"], fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("route_review: wrote %s" % os.path.join(a.out, "route_review.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
