#!/usr/bin/env python3
"""Re-time the a16w16 tuner's winners against what currently ships, in a graph-captured
interleaved harness.

Why this exists: aiter's tuner times eagerly, one candidate at a time, with a sync per
iteration. On gfx950 that undercounts the clock ramp and overcounts launch overhead --
its numbers for the shapes here came out ~25% above what the same kernels do inside a
graph. That is fine for *ranking within one tuner run*, but it is not a number you can
compare against a differently-measured baseline, and tuning-kb/qwen3-8b records a case
where a claimed 19% tuner win re-timed as a dead tie.

So: take the tuner's per-shape top candidates out of profile_m64.csv, plus whatever the
live config resolves to today, and race them all through analysis/gbench.py.

The second thing this has to get right is the cache. MI355X carries a 256 MB LLC and every
weight here fits in it (o_proj 34 MB, qkv 50 MB, down_proj 117 MB, gate_up 235 MB), so a
graph that replays the *same* B tensor `reps` times measures an LLC-resident GEMM -- which
is how you get gate_up reading 101% of the HBM roofline. Production is not like that: a
decode step streams ~2.28 GB of KV per layer, which evicts the LLC between every pair of
GEMMs, so the weights are read from HBM every time. `--footprint-mb` allocates that many
megabytes of distinct B buffers and the graph rotates through them, so the working set
overflows the LLC the way the real model does. `--footprint-mb 0` restores the (wrong,
but informative) resident behaviour for comparison.

Usage:
    python3 analysis/race_gemm_candidates.py --profile analysis/gemm_tune/profile_m64.csv \
        --m 64 --json analysis/gemm_candidates_m64.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbench import race  # noqa: E402

import aiter  # noqa: E402
from aiter.tuned_gemm import get_GEMM_A16W16_config, solMap  # noqa: E402
from aiter.jit.utils.chip_info import get_cu_num, get_gfx  # noqa: E402

NAMES = {
    (6144, 4096): "qkv_proj",
    (4096, 4096): "o_proj",
    (28672, 4096): "gate_up_proj",
    (4096, 14336): "down_proj",
}
HBM_PEAK_TBS = 6.87


def make_step(cfg, a, bs, sink, key):
    """`bs` is a list of distinct B buffers; successive calls rotate through them so the
    graph's weight working set is `len(bs)` x one weight, not one weight replayed."""
    libtype = cfg["libtype"]
    ctr = {"i": 0}
    if libtype == "torch":
        def _s():
            b = bs[ctr["i"] % len(bs)]
            ctr["i"] += 1
            sink[key] = torch.nn.functional.linear(a, b)
        return _s
    fn = solMap[libtype]
    solidx = int(cfg.get("solidx", 0) or 0)

    def _s():
        b = bs[ctr["i"] % len(bs)]
        ctr["i"] += 1
        sink[key] = fn(a, b, solidx, None, a.dtype, None, None, None, False, config=cfg)
    return _s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="analysis/gemm_tune/profile_m64.csv")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--topk", type=int, default=3, help="top candidates per libtype family")
    ap.add_argument("--err-max", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=64)
    ap.add_argument("--footprint-mb", type=int, default=1024,
                    help="MB of distinct B buffers to rotate through; must exceed the "
                         "256 MB LLC or the GEMM is measured cache-resident. 0 = one buffer.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    prof = pd.read_csv(args.profile)
    prof = prof[(prof.M == args.m) & (prof.err_ratio <= args.err_max) & (prof.us > 0)]
    print(f"gfx={get_gfx()} cu_num={get_cu_num()} profile={args.profile} rows={len(prof)}")

    out = {"m": args.m, "gfx": get_gfx(), "cu_num": get_cu_num(),
           "footprint_mb": args.footprint_mb, "shapes": {}}
    for (n, k), label in NAMES.items():
        g = prof[(prof.N == n) & (prof.K == k)].sort_values("us")
        if g.empty:
            continue
        print(f"\n=== {label}  M={args.m} N={n} K={k} ===")

        a = torch.randn(args.m, k, device="cuda", dtype=torch.bfloat16) / (k ** 0.5)
        w_mb = n * k * 2 / 2**20
        nbuf = max(1, -(-args.footprint_mb // int(w_mb))) if args.footprint_mb else 1
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / (k ** 0.5)
        # The rotating copies only have to differ in *address*, not in content -- the
        # reference and err_ratio stay valid, and cloning is cheaper than re-randomising.
        bs = [b] + [b.clone() for _ in range(nbuf - 1)]
        print(f"  weight={w_mb:.0f} MB x {nbuf} buffers = {w_mb * nbuf:.0f} MB working set "
              f"(LLC is 256 MB)")
        ref = torch.nn.functional.linear(a.float(), b.float())
        refmax = ref.abs().max().item()

        # candidate set: what the live config resolves to now, plus the tuner's best
        # `topk` per libtype family (families differ in how they fail, so do not just
        # take the global top N -- that would be all hipblaslt on one shape).
        live = get_GEMM_A16W16_config(args.m, n, k, False, str(torch.bfloat16),
                                      str(torch.bfloat16), False, False)
        cands = {f"LIVE:{live['libtype']}": dict(live)}
        for lib, sub in g.groupby("libtype"):
            for _, r in sub.sort_values("us").head(args.topk).iterrows():
                cfg = {"libtype": lib, "solidx": int(r.solidx),
                       "splitK": int(r.splitK) if pd.notna(r.splitK) else None,
                       "kernelName": r.kernelName}
                cands[f"{lib}:{int(r.solidx)}"] = cfg
        if "torch:0" not in cands:
            cands["torch:0"] = {"libtype": "torch", "solidx": 0, "splitK": None,
                                "kernelName": "native"}

        steps, sink, errs, tuner_us = {}, {}, {}, {}
        for name, cfg in cands.items():
            try:
                st = make_step(cfg, a, bs, sink, name)
                st()
                torch.cuda.synchronize()
                errs[name] = (sink[name].float() - ref).abs().max().item() / (refmax + 1e-9)
                if errs[name] > args.err_max:
                    print(f"  {name:<22} REJECT err={errs[name]:.4f}")
                    continue
                steps[name] = st
                m = g[(g.libtype == cfg["libtype"]) & (g.solidx == (cfg.get("solidx") or 0))]
                tuner_us[name] = float(m.us.iloc[0]) if len(m) else None
            except Exception as e:  # noqa: BLE001
                print(f"  {name:<22} SETUP ERROR {type(e).__name__}: {str(e)[:110]}")

        res = race(steps, rounds=args.rounds, reps=args.reps)
        byt = (args.m * k + n * k + args.m * n) * 2
        rows = []
        for name in steps:
            r = res[name]
            if "us" not in r:
                print(f"  {name:<22} {r['mode']}")
                continue
            tbs = byt / (r["us"] * 1e-6) / 1e12
            rows.append({"cand": name, "libtype": cands[name]["libtype"],
                         "solidx": cands[name].get("solidx"),
                         "kernelName": cands[name].get("kernelName"),
                         "us": r["us"], "tbs": tbs, "roofline_pct": 100 * tbs / HBM_PEAK_TBS,
                         "err_ratio": errs[name], "round_spread_pct": r["spread_pct"],
                         "tuner_us": tuner_us.get(name)})
        rows.sort(key=lambda x: x["us"])
        livekey = f"LIVE:{live['libtype']}"
        liveus = next((r["us"] for r in rows if r["cand"] == livekey), None)
        for r in rows:
            d = "" if liveus is None else f" {100*(liveus-r['us'])/liveus:+6.2f}% vs LIVE"
            t = "" if r["tuner_us"] is None else f" tuner={r['tuner_us']:.2f}"
            print(f"  {r['cand']:<22} {r['us']:8.3f} us ({r['roofline_pct']:5.1f}% roof)"
                  f"  err={r['err_ratio']:.4f} rspread={r['round_spread_pct']:.1f}%{t}{d}")
        out["shapes"][label] = {"n": n, "k": k, "live": livekey, "live_us": liveus,
                                "live_kernelName": live.get("kernelName"),
                                "weight_mb": w_mb, "nbuf": nbuf, "cands": rows}

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
