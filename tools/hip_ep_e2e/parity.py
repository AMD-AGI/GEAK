#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Output parity of two hip-ep packages on the same OGA model.

Runs parity_dump.py once per package (prefill-path and decode-path teacher-forced
logits over fixed texts), then compares:
  top1   argmax agreement over every prefill position and decode step
  relL2  ||a - b|| / ||a|| over all saved logits, in fp32
The text is cut to each of --lens (the timed TTFT lengths), so the prefill passes
hit the same GEMM M as the A/B.

    python3 parity.py --model M --base A --cand B --out DIR [--lens 1024,2048]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import hipep_env  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = os.path.join(HERE, "data", "parity_text.txt")
SAMPLE = 64        # prefill positions whose full logits are compared, per text
DECODE_STEPS = 32  # one-token decode steps compared, per text


def texts(lens):
    return [{"file": TEXT, "len": L} for L in lens]


def compare(path_a, path_b):
    a, b = np.load(path_a), np.load(path_b)
    n = sum(1 for k in a.files if k.startswith("ids_"))
    agree = {"prefill": [0, 0], "decode": [0, 0]}
    diff2 = ref2 = 0.0
    worst = 0.0
    for i in range(n):
        if not np.array_equal(a[f"ids_{i}"], b[f"ids_{i}"]):
            raise ValueError(f"text {i}: the dumps read different token ids")
        for kind, key in (("prefill", "argmax"), ("decode", "dargmax")):
            x, y = a[f"{key}_{i}"], b[f"{key}_{i}"]
            agree[kind][0] += int((x == y).sum())
            agree[kind][1] += x.size
        for key in ("logits", "dlogits"):
            x = a[f"{key}_{i}"].astype(np.float32)
            y = b[f"{key}_{i}"].astype(np.float32)
            d2 = ((x - y) ** 2).sum(-1)
            r2 = (x ** 2).sum(-1)
            diff2 += float(d2.sum())
            ref2 += float(r2.sum())
            worst = max(worst, float(np.sqrt(d2 / np.maximum(r2, 1e-30)).max()))
    hits = agree["prefill"][0] + agree["decode"][0]
    total = agree["prefill"][1] + agree["decode"][1]
    rel = float(np.sqrt(diff2 / ref2)) if ref2 else 0.0
    return {"top1": hits / total, "prefill_top1": agree["prefill"][0] / agree["prefill"][1],
            "decode_top1": agree["decode"][0] / agree["decode"][1], "relL2": rel,
            "relL2_max_pos": worst, "exact": hits == total and diff2 == 0.0,
            "kernel_lib_a": str(a["kernel_lib"]), "kernel_lib_b": str(b["kernel_lib"])}


# hip-ep's own config switch (lookup vs online autotune) moves the logits by
# relL2 7.99e-3 on Qwen3.8-27B and 1.09e-2 on Qwen3-0.6B, so the bound sits at
# about 3x that: a candidate may change the arithmetic order as much as hip-ep's
# autotune does, not more.
REL_MAX = 3e-2


def verdict(cmp, top1_min=0.99, rel_max=REL_MAX):
    return cmp["top1"] >= top1_min and cmp["relL2"] <= rel_max


def check_kernel_lib(mapped, pkg):
    """The dump must have mapped exactly one kernel library, from `pkg`. /proc/self/maps
    shows resolved paths, so the package path is resolved too."""
    if not mapped or ";" in mapped:
        raise RuntimeError(f"no kernel library (or several) mapped: {mapped!r}")
    if os.path.dirname(os.path.dirname(os.path.realpath(mapped))) != os.path.realpath(pkg):
        raise RuntimeError(f"kernel library {mapped} is not from package {pkg}")


def dump(env, model, texts_json, out_npz, timeout=3600):
    home = hipep_env.pkg_home(os.path.dirname(out_npz), env.pkg)
    args = [os.path.join(HERE, "parity_dump.py"), "--model", model, "--texts", texts_json,
            "--out", out_npz, "--sample", str(SAMPLE), "--decode-steps", str(DECODE_STEPS)]
    log = out_npz[:-len(".npz")] + ".log"
    with hipep_env.serving_lock(env):
        p = subprocess.run(hipep_env.pkg_argv(env, env.python, args), cwd=env.lib,
                           env=hipep_env.pkg_env(env, home), capture_output=True,
                           text=True, timeout=timeout, check=False)
    with open(log, "w") as f:
        f.write(p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"parity_dump exited {p.returncode}; log: {log}")
    check_kernel_lib(str(np.load(out_npz)["kernel_lib"]), env.pkg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lens", type=bench._lens, default="1024,2048",
                    help="the timed prompt lengths; the text is cut to each")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "parity.json")
    if os.path.exists(path):
        os.remove(path)  # a failed run must not leave an earlier verdict behind
    texts_json = os.path.join(a.out, "texts.json")
    with open(texts_json, "w") as f:
        json.dump(texts(a.lens), f)

    npz = {}
    for side, pkg in (("base", a.base), ("cand", a.cand)):
        npz[side] = os.path.join(a.out, f"{side}.npz")
        dump(hipep_env.Env.from_environ(pkg), a.model, texts_json, npz[side])
    cmp = compare(npz["base"], npz["cand"])
    res = {"model": a.model, "base": a.base, "cand": a.cand, **cmp, "pass": verdict(cmp)}
    hipep_env.write_json(path, res)
    for p in npz.values():
        os.remove(p)
    print(f"[parity] top1 {cmp['top1']:.4f} (prefill {cmp['prefill_top1']:.4f}, decode "
          f"{cmp['decode_top1']:.4f}), relL2 {cmp['relL2']:.2e} (worst position "
          f"{cmp['relL2_max_pos']:.2e}), exact={cmp['exact']}: {'PASS' if res['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
