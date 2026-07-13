#!/usr/bin/env python3
"""Probe per-shape output -> meta.json m_bucket lists (path-1 adapter).

Turns the per-shape probe product (real measured shapes from an enforce-eager run) into the
`decode_m_buckets` / `prefill_m_buckets` lists that `kernel_extractor` writes into meta.json and that
`attribute_weights.attribute_gemm` consumes. This REPLACES the inferred "M ≈ WORKLOAD.conc" guess with
measured M values — the rest of the pipeline (attribute_gemm / unittest / attribute_weights) is
unchanged, only the m_bucket VALUES change from guessed to measured.

Extraction: for the target GEMM kernel, take each case's activation M (dims[0][0]) with its call
count. Split into decode (small M, the steady-state running batch ~ conc) vs prefill (large M, prompt
chunk tokens) by a conc-anchored threshold, then keep the buckets that carry real traffic (drop the
long tail below a count-share floor so a couple of stray shapes don't pollute the bucket list).

Usage:
    python3 probe_to_mbuckets.py --probe <per_shape_probe.json> --conc 64 \
        [--kernel-match matmul_ogs] [--decode-max-mult 8] [--min-count-share 0.001]
Prints a JSON: {"decode_m_buckets":[...], "prefill_m_buckets":[...], "notes": "..."}
The extractor merges these two keys into meta.json.
"""
import argparse, json, sys


def extract(probe, conc, kernel_match, decode_max_mult, min_count_share):
    kernels = probe.get("kernels", [])
    # pick the target kernel: name/label contains kernel_match (or the single captured GEMM)
    cand = [k for k in kernels if k.get("probe_status") == "captured"
            and (not kernel_match or kernel_match.lower() in (k.get("label", "") + k.get("target", "")).lower())]
    if not cand:
        return None, f"no captured kernel matching '{kernel_match}'"
    # if several match, take the one with the most calls
    k = max(cand, key=lambda x: x.get("probe_total_calls", 0))

    # (M -> total count) over cases that exposed a real activation shape
    m_count = {}
    for c in k.get("cases", []):
        dims = c.get("dims") or []
        if not dims or not dims[0]:
            continue
        M = dims[0][0]
        if isinstance(M, int):
            m_count[M] = m_count.get(M, 0) + c.get("count", 0)
    if not m_count:
        return None, f"kernel '{k.get('label')}' has no real activation shapes (all dims empty)"

    total = sum(m_count.values()) or 1
    decode_thr = conc * decode_max_mult   # M at/below this = decode regime (running batch scale)
    decode, prefill = [], []
    for M, cnt in m_count.items():
        if cnt / total < min_count_share:
            continue  # drop long-tail shapes with negligible traffic
        (decode if M <= decode_thr else prefill).append(M)

    decode = sorted(set(decode))
    prefill = sorted(set(prefill))
    notes = (f"kernel={k.get('label')} conc={conc} decode_thr={decode_thr} "
             f"total_calls={total} distinct_M={len(m_count)} "
             f"kept_decode={len(decode)} kept_prefill={len(prefill)} "
             f"(dropped M with count-share < {min_count_share})")
    return {"decode_m_buckets": decode, "prefill_m_buckets": prefill, "notes": notes}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="per-shape probe json (postprocess output)")
    ap.add_argument("--conc", type=int, required=True, help="WORKLOAD concurrency (decode batch scale)")
    ap.add_argument("--kernel-match", default="", help="substring to pick the target GEMM kernel")
    ap.add_argument("--decode-max-mult", type=float, default=8.0,
                    help="M <= conc*this is decode; above is prefill (default 8, covers conc*topk)")
    ap.add_argument("--min-count-share", type=float, default=0.001,
                    help="drop M whose call-count share is below this (long-tail noise)")
    ap.add_argument("--out", default="", help="write JSON here; else stdout")
    args = ap.parse_args()

    with open(args.probe) as fh:
        probe = json.load(fh)
    res, err = extract(probe, args.conc, args.kernel_match, args.decode_max_mult, args.min_count_share)
    if err:
        sys.stderr.write(f"[probe_to_mbuckets] {err}\n")
        sys.exit(2)
    out = json.dumps(res, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(out)
    print(out)


if __name__ == "__main__":
    main()
