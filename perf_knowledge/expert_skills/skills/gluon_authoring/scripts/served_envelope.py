#!/usr/bin/env python3
"""Weigh a win by the SERVED mix, not by microseconds on one shape.

A per-kernel roofline, however well calibrated, cannot answer "should this round go to decode or to
prefill". Ranking gaps by their microseconds silently answers it with "whichever shape is slower",
which is almost never the served answer: a shape that is individually fast but served constantly
can dominate the served time, so one microsecond removed there is worth tens of microseconds
removed on the slow shape. A gap-sorted queue spends the whole budget on the wrong side of that
exchange rate, and nothing in the per-shape numbers reveals the mistake.

Two numbers, both cheap:

    served speedup   S = T_baseline / T_current = W / sum_k(W_k / speedup_k)
                         (a weighted HARMONIC mean of per-point speedups -- an arithmetic mean of
                         speedups overstates, because time adds and rates do not)
    marginal value   MV_k = -dS/dt_k  = f_k * T_baseline / T_current^2
                         "%served per unit of time removed at point k". The RATIO between two
                         points is the exchange rate that decides where the next round goes.

THE WEIGHT TRAP. Weights come in two kinds and they are not interchangeable:

    calls  W_k = how often point k is served (a call/traffic share)
    time   W_k = how much BASELINE TIME point k contributes (calls x baseline_ms)

If a `time` weight is fed into a formula that multiplies by baseline_ms again, the heavy-baseline
point gets counted twice and the ranking can INVERT -- the majority shareholder of served time can
come out looking like a minority one, reversing the strategic priority while every individual
number still looks reasonable. So `weight_kind` is REQUIRED here, and every report prints what the
shares would be under the other interpretation, flagging it when the ranking flips. A number you
cannot misread by accident is worth more than a number with a comment.

Usage:
  served_envelope.py --champion <bundle>/plain_champion.json   # reads its served_range table
  served_envelope.py --points served.json [--json out.json]
  served_envelope.py --weight-kind time \\
      --point "decode,weight=<w>,baseline_ms=<b>,current_ms=<c>" \\
      --point "prefill,weight=<w>,baseline_ms=<b>,current_ms=<c>"

served.json:
  {"weight_kind": "time",
   "points": [{"name": "<point>", "weight": <w>,
               "baseline_ms": <b>, "current_ms": <c>}, ...]}
"""
from __future__ import annotations

import argparse
import json
import sys

WEIGHT_KINDS = ("calls", "time")


def _call_weights(points, weight_kind):
    """Normalize whatever the caller gave into CALL weights f_k, the only kind the envelope math
    can consume. A `time` weight already contains baseline_ms, so it is divided back out here --
    exactly once, in one place, instead of at each use site."""
    out = []
    for p in points:
        w, b = float(p["weight"]), float(p["baseline_ms"])
        if w < 0:
            raise ValueError(f"negative weight on {p.get('name')!r}")
        if b <= 0:
            raise ValueError(f"baseline_ms must be > 0 on {p.get('name')!r}")
        out.append(w if weight_kind == "calls" else w / b)
    if not any(out):
        raise ValueError("all weights are zero")
    return out


def envelope(points, weight_kind):
    """Served speedup, per-point time shares, and the marginal value of a unit of time at each
    point. Pure arithmetic; no I/O."""
    if weight_kind not in WEIGHT_KINDS:
        raise ValueError(f"weight_kind must be one of {WEIGHT_KINDS}, got {weight_kind!r}")
    if not points:
        raise ValueError("no operating points")
    f = _call_weights(points, weight_kind)
    b = [float(p["baseline_ms"]) for p in points]
    c = [float(p.get("current_ms", p["baseline_ms"])) for p in points]
    for p, ci in zip(points, c):
        if ci <= 0:
            raise ValueError(f"current_ms must be > 0 on {p.get('name')!r}")
    t_base = sum(fi * bi for fi, bi in zip(f, b))
    t_cur = sum(fi * ci for fi, ci in zip(f, c))
    served = t_base / t_cur
    rows = []
    for p, fi, bi, ci in zip(points, f, b, c):
        rows.append({
            "name": p.get("name"),
            "call_weight": fi,
            "baseline_share": fi * bi / t_base,        # share of SERVED BASELINE TIME
            "current_share": fi * ci / t_cur,
            "speedup": bi / ci,
            # %served gained per ms removed at this point, at the CURRENT operating point
            "marginal_value_pct_per_ms": 100.0 * fi * t_base / (t_cur ** 2),
        })
    mvs = [r["marginal_value_pct_per_ms"] for r in rows]
    best = max(range(len(rows)), key=lambda i: mvs[i])
    for i, r in enumerate(rows):
        r["exchange_rate_vs_best"] = (mvs[best] / mvs[i]) if mvs[i] else float("inf")
    return {"weight_kind": weight_kind, "served_speedup": served,
            "baseline_total_ms": t_base, "current_total_ms": t_cur,
            "points": rows, "highest_marginal_value": rows[best]["name"]}


def inversion_check(points, weight_kind):
    """What the shares would be if the weights were the OTHER kind, and whether that flips the
    ranking. This is the guard for the double-counted-time-weight bug: the failure mode is not a
    slightly-off number, it is a reversed priority, and it is invisible unless you compute both."""
    other = "time" if weight_kind == "calls" else "calls"
    try:
        alt = envelope(points, other)
    except ValueError as e:
        return {"other_kind": other, "computable": False, "why": str(e)}
    this = envelope(points, weight_kind)
    # ordered by index, not by name: duplicate or missing names must not hide a flip
    order_a = sorted(range(len(points)), key=lambda i: -this["points"][i]["baseline_share"])
    order_b = sorted(range(len(points)), key=lambda i: -alt["points"][i]["baseline_share"])
    return {"other_kind": other, "computable": True,
            "shares_this": [(r["name"], r["baseline_share"]) for r in this["points"]],
            "shares_other": [(r["name"], r["baseline_share"]) for r in alt["points"]],
            "ranking_flips": order_a != order_b}


def _print(env, inv):
    print(f"\n=== served envelope ({len(env['points'])} operating points, "
          f"weight_kind={env['weight_kind']}) ===")
    print(f"  served speedup   {env['served_speedup']:.4f}x   "
          f"(baseline {env['baseline_total_ms']:.4f} -> current {env['current_total_ms']:.4f} "
          f"weighted ms)")
    print(f"  {'point':<24s} {'share%':>8s} {'speedup':>9s} {'%served/ms':>12s} {'1 ms here =':>14s}")
    for r in env["points"]:
        rate = r["exchange_rate_vs_best"]
        eq = "the best point" if rate == 1 else f"{rate:.1f} ms there"
        print(f"  {str(r['name'])[:24]:<24s} {100*r['baseline_share']:>7.2f}% "
              f"{r['speedup']:>8.4f}x {r['marginal_value_pct_per_ms']:>12.4f} {eq:>14s}")
    print(f"  spend the next round on: {env['highest_marginal_value']}  "
          f"(highest marginal value, not the largest gap)")
    if inv.get("computable"):
        if inv["ranking_flips"]:
            print(f"  *** WEIGHT-KIND WARNING *** read as `{inv['other_kind']}` weights these shares "
                  f"become "
                  + ", ".join(f"{k}={100*v:.2f}%" for k, v in inv["shares_other"])
                  + ",\n    which REVERSES the ranking. Confirm the weight kind against the "
                    "harness's own per-case baseline before\n    spending a round on this: a time "
                    "weight fed in as a call weight double-counts baseline_ms.")
        else:
            print(f"  (weight-kind check: reading them as `{inv['other_kind']}` does not flip the "
                  f"ranking)")
    print()


_NUMERIC_KEYS = ("weight", "baseline_ms", "current_ms")


def from_champion(bundle):
    """Read a `plain_champion` bundle's served table as operating points.

    Each row carries the champion's latency (`ms`) and its ratio to the shipped default
    (`vs_default`), so the baseline is `ms * vs_default` -- which is how the served speedup lands on
    the same comparator the rest of the handoff uses. A row without `weight` cannot be ranked, and is
    refused rather than silently equal-weighted: assuming a uniform served mix is a claim about the
    deployment that nobody measured."""
    rows = bundle.get("served_range") or []
    if not rows:
        raise ValueError("bundle has an empty served_range (Phase 4 unfinished)")
    kind = bundle.get("served_weight_kind")
    if kind not in WEIGHT_KINDS:
        raise ValueError(
            "bundle's served_range has no `served_weight_kind` -- an unweighted table can say "
            "'no shape regressed' but cannot be ranked. Add weights + their kind at emit time.")
    points = []
    for r in rows:
        if "weight" not in r:
            raise ValueError(f"served_range row {r.get('shape')!r} has no `weight`")
        ms, vs = r.get("ms"), r.get("vs_default")
        if not ms:
            raise ValueError(f"served_range row {r.get('shape')!r} has no `ms`")
        if not vs:
            raise ValueError(f"served_range row {r.get('shape')!r} has no `vs_default`, so it has no "
                             f"baseline to speed up from")
        points.append({"name": r.get("shape", "?"), "weight": r["weight"],
                       "baseline_ms": ms * vs, "current_ms": ms})
    return points, kind


def _parse_point(s):
    """`name=..,weight=..,baseline_ms=..,current_ms=..`. Any token that is not one of the known keys
    is taken whole as the name, so shape labels that contain `=` ("decode M=32") survive."""
    d = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, _, v = tok.partition("=")
        k = k.strip()
        if k in _NUMERIC_KEYS:
            try:
                d[k] = float(v)
            except ValueError:
                sys.exit(f"[served_envelope] {k}= must be a number, got {v!r}")
        elif k == "name":
            d["name"] = v.strip()
        else:
            d["name"] = tok
    for need in ("weight", "baseline_ms"):
        if need not in d:
            sys.exit(f"[served_envelope] --point needs {need}=: {s!r}")
    d.setdefault("name", "point")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", help="JSON file: {weight_kind, points:[{name,weight,baseline_ms,current_ms}]}")
    ap.add_argument("--champion", help="plain_champion.json: read its served_range table directly")
    ap.add_argument("--point", action="append", default=[],
                    help="inline point: name=..,weight=..,baseline_ms=..,current_ms=..")
    ap.add_argument("--weight-kind", choices=WEIGHT_KINDS,
                    help="calls = traffic share; time = calls x baseline_ms (REQUIRED: the two are "
                         "not interchangeable and mixing them can invert the ranking)")
    ap.add_argument("--json", help="write the record here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    kind, points = a.weight_kind, [_parse_point(s) for s in a.point]
    if a.points:
        doc = json.load(open(a.points))
        points = doc.get("points", []) + points
        kind = kind or doc.get("weight_kind")
    if a.champion:
        try:
            cpts, ckind = from_champion(json.load(open(a.champion)))
        except (ValueError, KeyError) as e:
            sys.exit(f"[served_envelope] --champion {a.champion}: {e}")
        points = cpts + points
        kind = kind or ckind
    if not points:
        ap.error("give --champion BUNDLE, --points FILE, and/or one or more --point")
    if not kind:
        ap.error("--weight-kind is required (or `weight_kind` in the JSON): a `time` weight already "
                 "contains baseline_ms, and using one as the other inverts the shares")
    try:
        env = envelope(points, kind)
    except ValueError as e:
        sys.exit(f"[served_envelope] {e}")
    inv = inversion_check(points, kind)
    _print(env, inv)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({**env, "weight_kind_check": inv}, f, indent=2)
        print(f"  wrote {a.json}")
    return 0


def _selftest():
    # A measured fused-MoE deployment: decode dominates served time even though prefill is the
    # slower kernel by three orders of magnitude per call.
    pts = [{"name": "decode M=32", "weight": 138.41, "baseline_ms": 0.05644, "current_ms": 0.05352},
           {"name": "prefill M=7937", "weight": 41.71, "baseline_ms": 0.58705, "current_ms": 0.58705},
           {"name": "decode M=1", "weight": 0.048, "baseline_ms": 0.0301, "current_ms": 0.0301}]
    env = envelope(pts, "time")
    shares = {r["name"]: r["baseline_share"] for r in env["points"]}
    assert abs(shares["decode M=32"] + shares["decode M=1"] - 0.7685) < 0.002, shares
    assert abs(shares["prefill M=7937"] - 0.2315) < 0.002, shares
    assert abs(sum(shares.values()) - 1.0) < 1e-9
    # the served speedup is the weighted HARMONIC mean: a 5.2% win on 77% of served time is ~4%,
    # NOT the 5.2% the single-shape number would suggest
    assert 1.03 < env["served_speedup"] < 1.045, env["served_speedup"]
    assert env["highest_marginal_value"].startswith("decode"), env
    mv = {r["name"]: r["marginal_value_pct_per_ms"] for r in env["points"]}
    rate = mv["decode M=32"] / mv["prefill M=7937"]
    # The exchange rate is the ratio of CALL weights, nothing else -- so it is checked against that
    # identity rather than against a remembered constant. The load-bearing fact is the order of
    # magnitude: one microsecond of the served-heavy point is worth tens of microseconds of the
    # slow one, so a gap-sorted queue points the wrong way.
    f = {r["name"]: r["call_weight"] for r in env["points"]}
    assert abs(rate - f["decode M=32"] / f["prefill M=7937"]) < 1e-9, (rate, f)
    assert 30 < rate < 45, rate
    # ...and it is reported per point, so nobody has to derive it
    dec = next(r for r in env["points"] if r["name"] == "decode M=32")
    pre = next(r for r in env["points"] if r["name"] == "prefill M=7937")
    assert dec["exchange_rate_vs_best"] == 1.0 and abs(pre["exchange_rate_vs_best"] - rate) < 1e-9

    # THE TRAP: the same numbers read as `calls` weights double-count baseline_ms and reverse the
    # ranking. The tool must detect that and say so rather than print the reversed number.
    inv = inversion_check(pts, "time")
    assert inv["computable"] and inv["ranking_flips"], inv
    wrong = envelope(pts, "calls")
    w_shares = {r["name"]: r["baseline_share"] for r in wrong["points"]}
    assert w_shares["prefill M=7937"] > 0.7, w_shares      # 23% -> >70%: the inversion
    w_mv = {r["name"]: r["marginal_value_pct_per_ms"] for r in wrong["points"]}
    # the exchange rate collapses by an order of magnitude too, which is how a lopsided asymmetry
    # gets mistaken for a near-tie
    assert (w_mv["decode M=32"] / w_mv["prefill M=7937"]) < 0.15 * rate
    # a mix where the interpretation does NOT matter must not cry wolf
    flat = [{"name": "a", "weight": 1.0, "baseline_ms": 1.0, "current_ms": 0.5},
            {"name": "b", "weight": 1.0, "baseline_ms": 1.0, "current_ms": 1.0}]
    assert not inversion_check(flat, "calls")["ranking_flips"]
    assert abs(envelope(flat, "calls")["served_speedup"] - (2.0 / 1.5)) < 1e-12

    # reading a champion bundle's served table: baseline = ms * vs_default, so the served speedup is
    # quoted against the same shipped default the rest of the handoff uses
    bundle = {"served_weight_kind": "calls",
              "served_range": [{"shape": "M=1", "ms": 9.31, "vs_default": 1.20, "weight": 4.0},
                               {"shape": "M=8192", "ms": 40.0, "vs_default": 1.00, "weight": 1.0}]}
    cpts, ckind = from_champion(bundle)
    assert ckind == "calls" and abs(cpts[0]["baseline_ms"] - 9.31 * 1.20) < 1e-12
    cenv = envelope(cpts, ckind)
    assert abs(cenv["points"][0]["speedup"] - 1.20) < 1e-12
    assert 1.0 < cenv["served_speedup"] < 1.20      # the small shape's win is diluted by the big one
    # a served table that cannot be ranked must SAY so rather than assume a uniform mix
    for bad in ({"served_range": []},
                {"served_range": [{"shape": "M=1", "ms": 9.31, "vs_default": 1.2, "weight": 1}]},
                {"served_weight_kind": "calls", "served_range": [{"shape": "M=1", "ms": 9.31}]},
                {"served_weight_kind": "calls",
                 "served_range": [{"shape": "M=1", "ms": 9.31, "weight": 1}]},
                {"served_weight_kind": "calls",
                 "served_range": [{"shape": "M=1", "vs_default": 1.2, "weight": 1}]}):
        try:
            from_champion(bad)
            raise AssertionError(f"from_champion accepted {bad!r}")
        except ValueError:
            pass

    # the CLI surface: a shape label carrying its own `=` must not be eaten by the key=value split
    p = _parse_point("decode M=32,weight=138.41,baseline_ms=0.05644,current_ms=0.05352")
    assert p == {"name": "decode M=32", "weight": 138.41,
                 "baseline_ms": 0.05644, "current_ms": 0.05352}, p
    assert _parse_point("name=prefill,weight=1,baseline_ms=2")["name"] == "prefill"
    # and distinct names must survive into the flip check (duplicates would mask an inversion)
    assert inversion_check([_parse_point("decode M=32,weight=138.41,baseline_ms=0.05644"),
                            _parse_point("prefill M=7937,weight=41.71,baseline_ms=0.58705")],
                           "time")["ranking_flips"]

    # no-op case: current == baseline everywhere -> exactly 1.0, and shares still sum to 1
    noop = envelope([{"name": "x", "weight": 3.0, "baseline_ms": 2.0}], "calls")
    assert noop["served_speedup"] == 1.0 and noop["points"][0]["baseline_share"] == 1.0
    # input hygiene: the arithmetic must refuse rather than silently produce a number
    for bad, kind in (([], "calls"),
                      ([{"name": "z", "weight": 0.0, "baseline_ms": 1.0}], "calls"),
                      ([{"name": "z", "weight": 1.0, "baseline_ms": 0.0}], "calls"),
                      ([{"name": "z", "weight": -1.0, "baseline_ms": 1.0}], "calls"),
                      ([{"name": "z", "weight": 1.0, "baseline_ms": 1.0, "current_ms": 0}], "calls"),
                      ([{"name": "z", "weight": 1.0, "baseline_ms": 1.0}], "shares")):
        try:
            envelope(bad, kind)
            raise AssertionError(f"envelope accepted {bad!r} / {kind!r}")
        except ValueError:
            pass
    print("[served_envelope] SELFTEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
