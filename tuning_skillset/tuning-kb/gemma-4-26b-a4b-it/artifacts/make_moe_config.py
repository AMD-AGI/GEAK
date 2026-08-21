#!/usr/bin/env python3
"""Build the tuned fused-MoE config table for Gemma-4-26B-A4B-it on MI355X.

Emits the JSON that SGLang's `get_moe_configs()` loads for this model's expert
shape.  The server builds the path itself and says so in its own log when the
file is missing:

    Config file not found at /sgl-workspace/sglang/python/sglang/srt/layers/moe/
    moe_runner/triton_utils/configs/triton_3_7_0/
    E=128,N=352,device_name=AMD_Instinct_MI355X.json

E=128 experts, N=352 = moe_intermediate_size 704 sharded over TP=2.

WHY THE KEY GRID LOOKS LIKE THIS
--------------------------------
`try_get_optimal_moe_config()` does not interpolate.  It picks

    configs[min(configs.keys(), key=lambda x: abs(x - M))]

i.e. the numerically nearest key, over whatever keys happen to be in the file.
That makes the *absent* keys as load-bearing as the present ones.  A file
containing only the three keys actually tuned (8, 64, 16384) would mis-serve
the prefill chunk this workload runs most:

    M=8192 -> |8192-64| = 8128  <  |8192-16384| = 8192  -> picks the M=64 tile

so the 16384-token prefill GEMM would silently run a 16x128 decode tile.  The
grid below is therefore dense enough that every key routes to the winner that
was actually measured for its magnitude.

Only three points were tuned, on the three shapes this workload actually
executes (see FINDINGS.md 2.2).  Keys 256..4096 are extrapolation from the
M=16384 winner and are NOT exercised by this workload -- documented as such in
the patch's RESULT.md rather than presented as tuned.

Regenerate with:  python3 analysis/make_moe_config.py [--write]
"""

import argparse
import json
import os

# --- measured winners -------------------------------------------------------
# analysis/moe_tuned_M8_M16384.json, analysis/moe_tuned_M64.json.
# Each was the best of a 289-config coarse sweep followed by an interleaved
# re-time of the finalists (measurement.md Rule 5), all at relerr 0.00e+00
# against the default config's output.

SMALL = {  # M=8   : 68.31 us vs 72.61 us default  (+6.3%)
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 3,
}

MEDIUM = {  # M=64  : 148.40 us vs 166.90 us default (+12.5%, 5.13 vs 4.56 TB/s)
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
}

LARGE = {  # M=16384: 1578.60 us vs 2622.36 us default (+66.1%, 494 vs 297 TFLOP/s)
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 4,
    "num_warps": 4,
    "num_stages": 2,
}

# key -> winner.  Boundaries sit where the nearest-key rule would otherwise
# stray: the last SMALL key is 16 and the first MEDIUM key is 24, so M<=19
# takes SMALL; the last MEDIUM key is 128 and the first LARGE key is 256, so
# M<=192 takes MEDIUM.
GRID = (
    [(k, SMALL) for k in (1, 2, 4, 8, 16)]
    + [(k, MEDIUM) for k in (24, 32, 48, 64, 96, 128)]
    + [
        (k, LARGE)
        for k in (256, 512, 1024, 1536, 2048, 3072, 4096, 8192, 16384)
    ]
)

DEST = (
    "/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/"
    "configs/triton_3_7_0/E=128,N=352,device_name=AMD_Instinct_MI355X.json"
)


def build():
    return {str(k): v for k, v in GRID}


def selftest(table):
    """Assert every M this workload runs resolves to the intended winner."""
    keys = [int(k) for k in table]

    def resolve(m):
        return table[str(min(keys, key=lambda x: abs(x - m)))]

    # (M, expected) for the shapes observed in the server log: decode 8 and 64,
    # prefill 8192 and 16384.
    for m, want, name in (
        (8, SMALL, "decode M=8"),
        (64, MEDIUM, "decode M=64"),
        (8192, LARGE, "prefill M=8192"),
        (16384, LARGE, "prefill M=16384"),
    ):
        got = resolve(m)
        assert got == want, f"{name}: resolved to {got}, wanted {want}"
        print(f"  ok  {name:18s} -> BLOCK_M={got['BLOCK_SIZE_M']:3d} "
              f"BLOCK_N={got['BLOCK_SIZE_N']:3d} BLOCK_K={got['BLOCK_SIZE_K']} "
              f"GROUP_M={got['GROUP_SIZE_M']} w{got['num_warps']} s{got['num_stages']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="install into the sglang tree")
    ap.add_argument("--dest", default=DEST)
    args = ap.parse_args()

    table = build()
    print(f"{len(table)} keys")
    selftest(table)

    blob = json.dumps(table, indent=4) + "\n"
    if args.write:
        os.makedirs(os.path.dirname(args.dest), exist_ok=True)
        with open(args.dest, "w") as f:
            f.write(blob)
        print(f"wrote {args.dest}")
    else:
        print(blob)


if __name__ == "__main__":
    main()
