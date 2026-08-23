#!/usr/bin/env python3
"""Turn the gemm_tune/*.jsonl search output into an aiter bf16_tuned_gemm CSV.

One row per (M,N,K) where the best candidate beats what the stack does today by
more than MIN_GAIN. Rows below that are left out on purpose: a 1% "win" is inside
the run-to-run spread of the search itself and only risks a regression.
"""
import json, glob, os, sys, csv
from collections import defaultdict

MIN_GAIN = 1.03
SRC = "/work/analysis/gemm_tune"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/analysis/qwen3_8_2_4t_a95b_bf16_tuned_gemm.csv"

# asm auto-selects a kernel when kernelName is None; the CSV has to name it
# explicitly, so it is resolved once here (measured in asm_name.py).
ASM_KERNEL = "_ZN5aiter39bf16gemm_fp32bf16_tn_64x64_splitk_cleanE"

HEADER = ["gfx", "cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB",
          "bpreshuffle", "libtype", "solidx", "splitK", "us", "kernelName",
          "err_ratio", "tflops", "bw"]

# shapes that exist in the model but are not part of a fused-GEMM patch
SKIP_STEMS = {"fused1024"}

# (M,N,K) already carried by another model's config. aiter merges every
# model_configs/*bf16_tuned_gemm*.csv into one table and *raises* on a duplicate
# shape key after rewriting the source files, so a collision would break the
# launch and damage llama70B's config. llama70B already tunes (8192|16384, 8192,
# 2048) and picks torch there; our hipblaslt rows are only 1.04-1.06x better, so
# they are not worth the collision.
SKIP_SHAPES = {(8192, 8192, 2048), (16384, 8192, 2048)}

# Rejected by validate_config.py. The search timed each candidate once, in its
# own process; validate_config.py re-times tuned-vs-torch interleaved, medians
# over 5 rounds. These four won the search but do not beat torch by 3% under the
# fairer comparison, so the search was measuring drift, not a kernel.
#   (64, 512, 8192)     search 1.078x -> 0.976x
#   (8192, 32, 8192)    search 1.234x -> 1.017x
#   (8192, 4608, 8192)  search 1.053x -> 0.997x
#   (16384, 32, 8192)   search 1.129x -> 0.965x
SKIP_SHAPES |= {(64, 512, 8192), (8192, 32, 8192), (8192, 4608, 8192), (16384, 32, 8192)}


def load(path):
    cur, oks, done = None, [], False
    M = N = K = None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("kind") == "current":
            cur = r
        elif r.get("kind") == "ok":
            oks.append(r)
        elif r.get("kind") == "done":
            done = True
        if r.get("M"):
            M, N, K = r["M"], r["N"], r["K"]
    return cur, oks, done, M, N, K


rows, report = [], []
for path in sorted(glob.glob(f"{SRC}/*.jsonl")):
    stem = os.path.basename(path)[:-6]
    name = stem.split("_", 1)[1]
    if name in SKIP_STEMS:
        continue
    cur, oks, done, M, N, K = load(path)
    if (M, N, K) in SKIP_SHAPES:
        report.append(f"{stem:24s} M={M:<6d} N={N:<5d} K={K:<5d} SKIPPED (shape owned by another model config)")
        continue
    if not cur or not oks:
        report.append(f"{stem}: no data"); continue
    oks.sort(key=lambda r: r["us"])
    best = oks[0]
    gain = cur["us"] / best["us"]
    if gain < MIN_GAIN:
        report.append(f"{stem:24s} M={M:<6d} N={N:<5d} K={K:<5d} "
                      f"{cur['lib']}/{cur['sol']} {cur['us']:.2f}us -> best {best['lib']} "
                      f"{best['us']:.2f}us  {gain:.3f}x  SKIPPED (< {MIN_GAIN}x){'' if done else '  [partial]'}")
        continue
    lib = best["lib"]
    splitK = best["sol"] if lib == "asm" else 0
    solidx = best["sol"] if lib == "hipblaslt" else 0
    kname = ASM_KERNEL if lib == "asm" else ("triton" if lib == "triton" else "")
    flops = 2 * M * N * K
    bytes_ = 2 * (M * K + N * K + M * N)
    rows.append([
        "gfx950", 256, M, N, K, False, "torch.bfloat16", "torch.bfloat16", False, False,
        lib, solidx, splitK, round(best["us"], 4), kname, 0.0,
        round(flops / best["us"] / 1e6, 2), round(bytes_ / best["us"] / 1e3, 2),
    ])
    report.append(f"{stem:24s} M={M:<6d} N={N:<5d} K={K:<5d} "
                  f"{cur['lib']}/{cur['sol']} {cur['us']:.2f}us -> {lib}/{best['sol']} "
                  f"{best['us']:.2f}us  {gain:.3f}x  KEPT{'' if done else '  [partial]'}")

rows.sort(key=lambda r: (r[2], r[3], r[4]))
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(HEADER)
    w.writerows(rows)

print("\n".join(report))
print(f"\n{len(rows)} rows -> {OUT}")
