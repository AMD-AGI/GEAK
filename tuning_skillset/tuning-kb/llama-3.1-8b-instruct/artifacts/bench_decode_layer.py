#!/usr/bin/env python3
"""Race decode GEMM configs *in situ* -- inside a whole decoder layer -- not in isolation.

Why this exists. `race_gemm_candidates.py` times one GEMM shape at a time in its own graph
and says the hipblaslt rows beat the shipping ones by 7.70% (o_proj), 4.55% (down_proj) and
2.15% (qkv). Deployed, the same rows cost 0.394% of end-to-end throughput and 0.52% of TPOT.
Both numbers were measured carefully, so one of the two is measuring something production
does not do.

The isolated harness differs from production in exactly one structural way: production runs
each GEMM sandwiched between a 2.28 GB paged-attention read and the next GEMM, on a device
whose LLC is 256 MB. `--footprint-mb` already forces the *weights* out of cache, but it
cannot reproduce what the attention kernel does to the rest of the machine -- its own cache
state, its occupancy tail, the L2 it leaves behind.

So this builds the real thing: qkv -> PA -> o_proj -> gate_up -> silu_and_mul -> down_proj,
one Llama-3.1-8B decoder layer at the trace's decode shape, captured in a graph and
interleaved across arms exactly like every other harness here. Same q, same KV pool, same
weights across arms; the only variable is which GEMM config each of the three deployed
shapes resolves to.

Arms are the full candidate set plus one-at-a-time, so if the regression reproduces it also
says which row causes it.

    PYTHONPATH=/sgl-workspace/aiter python3 analysis/bench_decode_layer.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbench import race  # noqa: E402

import aiter  # noqa: F401,E402
from aiter.tuned_gemm import get_GEMM_A16W16_config, solMap  # noqa: E402
from aiter.jit.utils.chip_info import get_cu_num, get_gfx  # noqa: E402

NUM_SEQS = 64
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = 4096
INTER = 14336
CTX = 8704            # mean of the 8192..9216 the workload sweeps
MAX_CONTEXT = 11264   # --context-length of the frozen config
PARTITION = 256
POOL_PAGES = 1_476_702

# The three rows patches/rejected/0001 deploys. gate_up is not in the patch (torch wins it),
# but the layer runs it anyway because production does and it evicts cache like production.
CAND = {
    "qkv_proj":  dict(n=6144, k=4096, libtype="hipblaslt", solidx=438484, splitK=None),
    "o_proj":    dict(n=4096, k=4096, libtype="hipblaslt", solidx=438482, splitK=None),
    "down_proj": dict(n=4096, k=14336, libtype="hipblaslt", solidx=440151, splitK=None),
}


def live_cfg(n, k):
    return dict(get_GEMM_A16W16_config(
        64, n, k, False, str(torch.bfloat16), str(torch.bfloat16), False, False))


def gemm(cfg, a, b):
    if cfg["libtype"] == "torch":
        return torch.nn.functional.linear(a, b)
    return solMap[cfg["libtype"]](a, b, int(cfg.get("solidx") or 0), None, a.dtype,
                                  None, None, None, False, config=cfg)


def build_kv():
    """Production page layout: ISL 8192 with chunked-prefill 16384 means each request's
    prefill KV is one allocation -- one contiguous 8192-slot run -- and only the decode
    tail is handed out one slot per sequence per step. bench_pa_scatter.py measured this
    layout at 389.24 us against 389.73 us for fully contiguous, i.e. no locality penalty."""
    dev = "cuda"
    k_cache = torch.empty(POOL_PAGES, 1, NUM_KV_HEADS, HEAD_DIM,
                          device=dev, dtype=torch.bfloat16).normal_(0, 0.5)
    v_cache = torch.empty(POOL_PAGES, 1, NUM_KV_HEADS, HEAD_DIM,
                          device=dev, dtype=torch.bfloat16).normal_(0, 0.5)
    kv_indptr = torch.arange(0, NUM_SEQS + 1, dtype=torch.int32).mul_(CTX)
    PREFILL, tail = 8192, CTX - 8192
    idx = torch.empty(NUM_SEQS * CTX, dtype=torch.int32)
    tail_base = NUM_SEQS * PREFILL
    for s in range(NUM_SEQS):
        run = torch.arange(s * PREFILL, (s + 1) * PREFILL, dtype=torch.int32)
        tl = tail_base + torch.arange(tail, dtype=torch.int32) * NUM_SEQS + s
        idx[s * CTX:(s + 1) * CTX] = torch.cat([run, tl])
    return (k_cache, v_cache, kv_indptr.to(dev), idx.to(dev),
            torch.ones(NUM_SEQS, dtype=torch.int32, device=dev))


SHAPES = {"qkv_proj": (6144, 4096), "o_proj": (4096, 4096),
          "gate_up": (2 * INTER, 4096), "down_proj": (4096, INTER)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--search", default=None, choices=sorted(SHAPES),
                    help="race the tuner's candidates for ONE shape in situ, holding the "
                         "other three at their live config")
    ap.add_argument("--arm", action="append", default=[],
                    help="explicit combo arm, 'label|shape=lib:solidx|...'; splitK and "
                         "kernelName are looked up in --profile. Repeatable.")
    ap.add_argument("--profile", default="analysis/gemm_tune/profile_m64.csv")
    ap.add_argument("--topk", type=int, default=3, help="per libtype family")
    ap.add_argument("--err-max", type=float, default=0.05)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.json is None:
        args.json = (f"analysis/insitu_{args.search}.json" if args.search
                     else "analysis/insitu_combo.json" if args.arm
                     else "analysis/decode_layer_insitu.json")

    torch.manual_seed(0)
    dev = "cuda"
    print(f"gfx={get_gfx()} cu_num={get_cu_num()}  layer shape: seqs={NUM_SEQS} ctx={CTX}")

    w = {
        "qkv_proj":  torch.randn(6144, HIDDEN, device=dev, dtype=torch.bfloat16) / 64,
        "o_proj":    torch.randn(HIDDEN, HIDDEN, device=dev, dtype=torch.bfloat16) / 64,
        "gate_up":   torch.randn(2 * INTER, HIDDEN, device=dev, dtype=torch.bfloat16) / 64,
        "down_proj": torch.randn(HIDDEN, INTER, device=dev, dtype=torch.bfloat16) / 120,
    }
    live = {name: live_cfg(*nk) for name, nk in SHAPES.items()}

    k_cache, v_cache, kv_indptr, kv_indices, kv_lpl = build_kv()
    hidden = torch.randn(NUM_SEQS, HIDDEN, device=dev, dtype=torch.bfloat16) / 8
    attn_out = torch.empty(NUM_SEQS, NUM_Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    mnp = (MAX_CONTEXT + PARTITION - 1) // PARTITION
    ws = torch.empty(NUM_SEQS * NUM_Q_HEADS * mnp * HEAD_DIM * 4
                     + 2 * NUM_SEQS * NUM_Q_HEADS * mnp * 4, device=dev, dtype=torch.uint8)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    ones = torch.ones(1, dtype=torch.float32, device=dev)

    from csrc.cpp_itfs.pa.pa_ragged import paged_attention_ragged as pa_core

    kv_gb = NUM_SEQS * CTX * NUM_KV_HEADS * HEAD_DIM * 2 * 2 / 2**30
    wt_mb = sum(t.numel() * 2 for t in w.values()) / 2**20
    print(f"  KV read per layer-step {kv_gb:.2f} GB, layer weights {wt_mb:.0f} MB "
          f"(LLC 256 MB -- attention flushes it between GEMMs, as in production)")

    def make_layer(cfgs, sink, key):
        qkv_c, o_c, gu_c, down_c = (cfgs["qkv_proj"], cfgs["o_proj"],
                                    cfgs["gate_up"], cfgs["down_proj"])

        def step():
            qkv = gemm(qkv_c, hidden, w["qkv_proj"])
            q = qkv[:, :HIDDEN].contiguous().view(NUM_SEQS, NUM_Q_HEADS, HEAD_DIM)
            pa_core(attn_out, ws, q,
                    k_cache.view(-1, 1, NUM_KV_HEADS, HEAD_DIM),
                    v_cache.view(-1, 1, NUM_KV_HEADS, HEAD_DIM),
                    scale, kv_indptr, kv_indices, kv_lpl,
                    1, mnp, None, "auto", "NHD", 0.0, ones, ones, None, PARTITION)
            o = gemm(o_c, attn_out.view(NUM_SEQS, HIDDEN), w["o_proj"])
            gu = gemm(gu_c, o, w["gate_up"])
            act = torch.nn.functional.silu(gu[:, :INTER]) * gu[:, INTER:]
            sink[key] = gemm(down_c, act, w["down_proj"])
        return step

    for n, c in live.items():
        print(f"  live {n:<10} {c['libtype']}:{c.get('solidx')}")

    if args.arm:
        import pandas as pd
        prof = pd.read_csv(args.profile)
        arms = {"live": dict(live)}
        for spec in args.arm:
            parts = spec.split("|")
            a = dict(live)
            for p in parts[1:]:
                shape, rhs = p.split("=")
                lib, sol = rhs.split(":")
                n, k = SHAPES[shape]
                r = prof[(prof.M == 64) & (prof.N == n) & (prof.K == k)
                         & (prof.libtype == lib) & (prof.solidx == int(sol))]
                if lib == "torch":
                    a[shape] = {"libtype": "torch", "solidx": 0, "splitK": None,
                                "kernelName": "native"}
                    continue
                if r.empty:
                    raise SystemExit(f"no profile row for {shape} {lib}:{sol}")
                r = r.iloc[0]
                a[shape] = {"libtype": lib, "solidx": int(sol),
                            "splitK": int(r.splitK) if pd.notna(r.splitK) else None,
                            "kernelName": r.kernelName, "tuner_us": float(r.us)}
            arms[parts[0]] = a
        print(f"\n=== in-situ combo race: {len(arms) - 1} arms vs live ===")
    elif args.search:
        import pandas as pd
        n, k = SHAPES[args.search]
        prof = pd.read_csv(args.profile)
        prof = prof[(prof.M == 64) & (prof.N == n) & (prof.K == k)
                    & (prof.err_ratio <= args.err_max) & (prof.us > 0)]
        arms = {"live": dict(live)}
        seen = {(live[args.search]["libtype"], int(live[args.search].get("solidx") or 0))}
        for lib, sub in prof.groupby("libtype"):
            for _, r in sub.sort_values("us").head(args.topk).iterrows():
                keyt = (lib, int(r.solidx))
                if keyt in seen:
                    continue
                seen.add(keyt)
                a = dict(live)
                a[args.search] = {"libtype": lib, "solidx": int(r.solidx),
                                  "splitK": int(r.splitK) if pd.notna(r.splitK) else None,
                                  "kernelName": r.kernelName, "tuner_us": float(r.us)}
                arms[f"{lib}:{int(r.solidx)}"] = a
        if ("torch", 0) not in seen:
            a = dict(live)
            a[args.search] = {"libtype": "torch", "solidx": 0, "splitK": None,
                              "kernelName": "native"}
            arms["torch:0"] = a
        print(f"\n=== in-situ search: {args.search} M=64 N={n} K={k}, "
              f"{len(arms) - 1} candidates vs live ===")
    else:
        arms = {"live": dict(live),
                "cand_all": {**live, **{n: dict(CAND[n]) for n in CAND}}}
        for n in CAND:
            arms[f"cand_{n}"] = {**live, n: dict(CAND[n])}

    sink, steps, errs = {}, {}, {}
    ref = None
    for name, cfgs in arms.items():
        try:
            st = make_layer(cfgs, sink, name)
            st()
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  arm {name:<18} SETUP ERROR {type(e).__name__}: {str(e)[:90]}")
            continue
        if ref is None:
            ref = sink[name].float()
        # Relative to the *live* layer output, not fp32: this gates "does swapping this
        # kernel move the layer's answer", which is the question the gsm8k gate then
        # settles end to end. bf16 accumulation noise alone lands around 5e-3 here.
        errs[name] = ((sink[name].float() - ref).abs().max()
                      / (ref.abs().max() + 1e-9)).item()
        steps[name] = st

    res = race(steps, rounds=args.rounds, reps=args.reps, iters=args.iters, warmup=6)

    base = res["live"]["us"]
    print(f"\n  {'arm':<18} {'us/layer-step':>14} {'vs live':>10} {'rspread':>8} "
          f"{'err':>7} {'tuner_us':>9}")
    rows = []
    for name in steps:
        r = res[name]
        if "us" not in r:
            print(f"  {name:<18} {r['mode']}")
            continue
        d = 100 * (base - r["us"]) / base
        swapped = arms[name][args.search] if args.search else None
        tu = swapped.get("tuner_us") if swapped else None
        print(f"  {name:<18} {r['us']:14.3f} {d:+9.2f}% {r['spread_pct']:7.1f}% "
              f"{errs[name]:7.4f} {('%9.2f' % tu) if tu else '        -'}")
        rows.append({"arm": name, "us": r["us"], "delta_pct_vs_live": d,
                     "round_spread_pct": r["spread_pct"], "err_vs_live": errs[name],
                     "tuner_us": tu,
                     "kernelName": swapped.get("kernelName") if swapped else None,
                     "rounds": r["rounds"]})

    out = {"ctx": CTX, "seqs": NUM_SEQS, "reps": args.reps, "rounds": args.rounds,
           "search": args.search, "gfx": get_gfx(),
           "live": {n: {"libtype": c["libtype"], "solidx": c.get("solidx")}
                    for n, c in live.items()},
           "arms": rows}
    with open(args.json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
