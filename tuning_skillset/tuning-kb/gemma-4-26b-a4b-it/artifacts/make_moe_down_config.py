#!/usr/bin/env python3
"""Build the SPLIT fused-MoE config tables for Gemma-4-26B-A4B-it on MI355X (patch 006).

Patch 002 gave this model one tuned tile table.  SGLang runs `fused_moe_kernel` TWICE per
layer, on two structurally different GEMMs:

    gate/up   A=[M*topk, 2816]  B=w1[128, 704, 2816]   -> N_out=704,  K=2816
    down      A=[M*topk,  352]  B=w2[128, 2816, 352]   -> N_out=2816, K=352

and it supports a separate table for the second one: `get_config_file_name(..., down_moe=True)`
appends `_down` to the filename and `try_get_optimal_moe_config(..., return_down_config=True)`
looks it up.  With no `_down` file present, `get_moe_configs(down_moe=True)` falls back to
returning the UP table (logging "Down MoE config file not found ... reusing the tuned
up-projection config"), so today both GEMMs run one tile shape tuned for the pair jointly.

This script emits BOTH tables: the `_down` file, and the one-key edit to patch 002's up table.

WHAT WAS MEASURED, AND WHERE
----------------------------
`analysis/moe_down_bench.py --verify` times four corners INTERLEAVED IN ONE PROCESS through the
real file-lookup path (not `override_config`, which cannot express a separate down table at all).
Run 3, MI355X, GPU 4 idle, `analysis/moe_down_verify_r3*.json`:

    M=16384   up002 both (today)   1493.90 us      M=64   up002 both (today)    151.48 us
              up002 + downWin      1465.14 +1.96%         up002 + downWin       146.98 +3.07%
              upWin + downWin      1454.14 +2.73%         upWin + downWin       143.84 +5.31%
              upWin both (no down) 1481.02 +0.87%         upWin both (no down)  170.12 -10.96%

    M=8192    up002 both (today)    862.97 us      M=8    up002 both (today)     69.44 us
              up002 + downWin       846.61 +1.93%         up002 + downWin        69.97 -0.76%
              upWin + downWin       847.17 +1.87%         upWin + downWin        68.56 +1.28%
              upWin both (no down)  856.89 +0.71%         upWin both (no down)   80.81 -14.07%

Every arm is bit-exact against the first (relerr 0.00e+00 at all four M).

TWO THINGS THAT TABLE SETTLES, AND THEY DECIDE THE DESIGN
---------------------------------------------------------
1. **The fourth corner is a free repeatability control at M=16384 and M=8192**, because at those
   sizes the up re-sweep found nothing better and `upWin == up002` -- so arms 1 and 4 are the
   SAME configuration timed twice in the same rotation.  They differ by 0.87% and 0.71%.  That
   is this microbenchmark's own noise, and it is the number the gains have to be read against:
   +2.73% is ~3x it, +1.93% is ~2.7x it, and M=8's +1.28% is INSIDE it.

2. **At M=64 the up re-tune is not separable from the down table.**  `upWin` alone, serving both
   GEMMs, is -10.96% -- a large regression.  It only becomes a +5.31% win once the down GEMM has
   its own table.  So this patch is one change, not two that happen to be bundled: n32k256 is a
   good gate/up tile and a bad down tile, and the shared table could not express that.

WHY THE UNTUNED KEYS ARE NO-OPS AND NOT EXTRAPOLATION
-----------------------------------------------------
The down table must carry the same dense key grid as the up table, for exactly the reason
documented in analysis/make_moe_config.py: `try_get_optimal_moe_config` picks the numerically
nearest key and does not interpolate, so an absent key is a routing decision.  But a key that
was never swept does not have to be a guess.  Setting `down[k] = up[k]` reproduces today's
fallback behaviour BIT-FOR-BIT at that key -- `down_config or config` receives an identical
dict -- so every untuned key is a provable no-op rather than an untested tile.  Only the keys
below are anything else:

    key 64            -> down winner   (measured at M=64,    +3.07% over shared)
    keys 256..16384   -> down winner   (measured at M=16384 and M=8192; 256..4096 extrapolated,
                                        and this workload never produces those M)
    key 64 (up table) -> up winner     (measured at M=64, valid ONLY with the down table above)

M=8 IS DELIBERATELY LEFT ALONE.  It is a real shape for this workload (the decode tail), it was
verified, and the answer was "inside the control": +1.28% against a 0.7-0.9% repeatability floor,
with the down winner alone actually -0.76%.  Changing it would be tuning to noise, so key 8 keeps
today's behaviour and this comment is the record of why.

Regenerate with:  python3 analysis/make_moe_down_config.py [--write]
"""

import argparse
import json
import os

CFG_DIR = (
    "/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/"
    "configs/triton_3_7_0"
)
UP_DEST = os.path.join(CFG_DIR, "E=128,N=352,device_name=AMD_Instinct_MI355X.json")
DOWN_DEST = os.path.join(CFG_DIR, "E=128,N=352,device_name=AMD_Instinct_MI355X_down.json")

# --- measured winners --------------------------------------------------------------------
# BLOCK_SIZE_M is NOT free: try_get_optimal_moe_config forces the down table to adopt the up
# table's BLOCK_SIZE_M (both GEMMs consume one moe_align_block_size sort) and warns if they
# differ.  Both winners below already match the up table's value at the keys they are used at
# -- 16 in the decode range, 128 in the prefill range -- so no override ever fires.  That is a
# property to preserve when re-tuning, not a coincidence.

DOWN_MEDIUM = {  # M=64: 146.98 us vs 151.48 shared (+3.07%); with UP_MEDIUM, 143.84 (+5.31%)
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 1,
}

DOWN_LARGE = {  # M=16384: 1465.14 us vs 1493.90 shared (+1.96%); M=8192: 846.61 vs 862.97
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 1,
}

UP_MEDIUM = {  # M=64 re-tune, valid ONLY alongside DOWN_MEDIUM (see docstring point 2)
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 256,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 3,
}

# Keys that get a measured winner.  Everything else mirrors the up table and is a no-op.
DOWN_OVERRIDES = {64: DOWN_MEDIUM}
DOWN_OVERRIDES.update({k: DOWN_LARGE for k in (256, 512, 1024, 1536, 2048, 3072, 4096, 8192, 16384)})
UP_OVERRIDES = {64: UP_MEDIUM}


def build(up_table):
    """Return (new_up_table, down_table).  Key grid is inherited from the up table."""
    up = {k: dict(v) for k, v in up_table.items()}
    for k, cfg in UP_OVERRIDES.items():
        assert str(k) in up, f"up table has no key {k}"
        up[str(k)] = dict(cfg)
    down = {k: dict(DOWN_OVERRIDES[int(k)]) if int(k) in DOWN_OVERRIDES else dict(v)
            for k, v in up.items()}
    return up, down


def selftest(up_old, up_new, down):
    """Assert the tables do what the patch claims, at the M values this workload runs."""
    assert set(up_new) == set(down), "up and down tables must share a key grid"
    keys = [int(k) for k in down]

    def res(t, m):
        return t[str(min(keys, key=lambda x: abs(x - m)))]

    # 1. The four shapes the server actually dispatches resolve to the intended pair.
    want = {
        8:     ("decode M=8 (tail)",   up_old["8"],     up_old["8"]),
        64:    ("decode M=64",         UP_MEDIUM,       DOWN_MEDIUM),
        8192:  ("prefill M=8192",      up_old["8192"],  DOWN_LARGE),
        16384: ("prefill M=16384",     up_old["16384"], DOWN_LARGE),
    }
    for m, (label, wu, wd) in want.items():
        gu, gd = res(up_new, m), res(down, m)
        assert gu == wu, f"{label}: up resolved to {gu}, wanted {wu}"
        assert gd == wd, f"{label}: down resolved to {gd}, wanted {wd}"
        tag = "no-op" if gd == gu else "SPLIT"
        print(f"  ok  {label:20s} up n{gu['BLOCK_SIZE_N']}k{gu['BLOCK_SIZE_K']}"
              f"g{gu['GROUP_SIZE_M']}s{gu['num_stages']:d}   "
              f"down n{gd['BLOCK_SIZE_N']}k{gd['BLOCK_SIZE_K']}"
              f"g{gd['GROUP_SIZE_M']}s{gd['num_stages']:d}   [{tag}]")

    # 2. BLOCK_SIZE_M agrees at EVERY key, so SGLang never has to override it.  A mismatch is
    #    silently repaired at runtime with a warning_once, which would mean the tile actually
    #    run is not the tile that was measured.
    for k in down:
        assert down[k]["BLOCK_SIZE_M"] == up_new[k]["BLOCK_SIZE_M"], (
            f"key {k}: BLOCK_SIZE_M {down[k]['BLOCK_SIZE_M']} != up {up_new[k]['BLOCK_SIZE_M']} "
            "-- SGLang would override it and the measured tile would not be the one running")
    print(f"  ok  BLOCK_SIZE_M matches at all {len(down)} keys (no runtime override)")

    # 3. No USE_TMA anywhere.  A truthy USE_TMA switches on a TMA path nothing here measured.
    for k, v in down.items():
        assert "USE_TMA" not in v, f"key {k}: USE_TMA must not be set (untested path)"
    print("  ok  no USE_TMA key (TMA path deliberately not enabled)")

    # 4. Every key that is NOT a measured winner must be byte-identical to the up entry, i.e.
    #    a provable no-op rather than an untested tile.
    noop = [k for k in down if int(k) not in DOWN_OVERRIDES]
    for k in noop:
        assert down[k] == up_new[k], f"key {k} claims to be a no-op but differs from up"
    print(f"  ok  {len(noop)} untuned keys are exact no-ops "
          f"({', '.join(sorted(noop, key=int))})")

    # 5. The up table is changed at exactly one key.
    changed = sorted((k for k in up_new if up_new[k] != up_old[k]), key=int)
    assert changed == ["64"], f"up table changed at {changed}, expected only key 64"
    print(f"  ok  up table changed at exactly one key: {changed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="install into the sglang tree")
    ap.add_argument("--up", default=UP_DEST)
    ap.add_argument("--down", default=DOWN_DEST)
    args = ap.parse_args()

    up_old = json.load(open(args.up))
    up_new, down = build(up_old)
    print(f"{len(up_new)} keys in each table")
    selftest(up_old, up_new, down)

    if args.write:
        for path, table in ((args.up, up_new), (args.down, down)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(json.dumps(table, indent=4) + "\n")
            print(f"wrote {path}")
    else:
        print(json.dumps({"up": up_new, "down": down}, indent=2)[:400] + " ...")


if __name__ == "__main__":
    main()
