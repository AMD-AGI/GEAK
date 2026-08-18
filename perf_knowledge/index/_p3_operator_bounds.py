#!/usr/bin/env python3
"""Section-scoped `bound_type` / `levers` labelling for `operators/` and friends (plan P3).

`bound_type` is the roofline routing key: a kernel_class query without it returns navigation docs,
and the lane's `--bound <x>` (fed from PROFILE_SUMMARY) can only reach files that declare it. P1's
rule pass emits none — a path can tell you a file's COST, never its BOTTLENECK. P2 hand-labelled the
82 cross-cutting docs. That leaves the 441-file `operators/` grid, which is what this covers.

Two signals, combined — evidence first, physics as the floor:

  1. EVIDENCE. Scan the operator's `overview.md` for the phrases the KB actually uses to state a
     bottleneck ("bandwidth-bound", "latency-bound", "bank conflict", …). Only unambiguous
     bottleneck phrases count; a bare mention of "MFMA" or "LDS" does not, because measured over the
     whole tree that kind of naive matching tags ~4.5 labels/file of pure noise.
  2. CLASS PRIOR. Every kernel_class has a bottleneck that follows from what the operator DOES — a
     GEMV decode kernel is bandwidth-bound whether or not its overview says so. The prior is the
     recall floor; `bound_type` is the one field where a miss is fatal (unreachable) and a
     false positive is merely a card the agent reads and discards.

Union of the two, evidence-backed bounds first, capped at 2 per file (the plan's cap — a card that
claims every bottleneck routes to none of them). `numerics.md` is skipped: it describes a precision
property, not a bottleneck.

Rows land in kb_labels.yaml as `src: llm` with the citation in `evidence`, so they rank BELOW P2's
hand labels and are spot-checkable. Nothing here touches `cost` — cost is precision-first (a wrong
value both mis-sorts and gets pruned under --max-cost) and stays with P1's rules and P2's humans.

  python3 index/_p3_operator_bounds.py            # dry-run
  python3 index/_p3_operator_bounds.py --write    # merge into kb_labels.yaml
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _kb_vocab as V  # noqa: E402
from _label_kb import load_labels, write_labels, validate_row  # noqa: E402

PK_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(PK_ROOT)
OPS = os.path.join(PK_ROOT, "operators")

# Unambiguous bottleneck phrasing only. Anything looser than this is the 4.5-tags/file noise trap.
EVIDENCE_RE = [
    ("hbm_bw", r"(?:bandwidth|memory|HBM|BW)[- ]bound"),
    ("mfma_compute", r"compute[- ]bound|MFMA[- ]bound|matrix[- ]core[- ]bound"),
    ("launch_overhead", r"launch overhead|launch floor|per-call floor|latency[- ]bound"),
    ("lds_bank", r"bank conflict"),
    ("l2_locality", r"L2 locality|XCD locality|L2 hit rate|cross-XCD"),
    ("sync", r"collective latency|all-reduce latency|RCCL|ring latency|barrier"),
    ("occupancy", r"occupancy[- ](?:bound|limited)|register pressure"),
    ("host_bound", r"host[- ]bound|host overhead|CPU[- ]bound|dispatch overhead"),
]

# What the operator DOES decides its bottleneck. Keyed by kernel_class so a new operator inherits it.
CLASS_PRIOR = {
    "gemm.dense": ["mfma_compute"],
    "gemm.batched": ["mfma_compute"],
    "gemm.grouped_moe": ["mfma_compute", "hbm_bw"],
    "gemm.epilogue_fused": ["mfma_compute", "hbm_bw"],
    "gemm.scaled_quant": ["mfma_compute"],
    "gemm.splitk_streamk": ["mfma_compute", "occupancy"],
    "gemm.skinny_decode": ["hbm_bw"],              # GEMV: one pass over the weights, no reuse
    "attn.prefill": ["mfma_compute", "hbm_bw"],
    "attn.mla": ["mfma_compute", "hbm_bw"],
    "attn.gqa_mqa": ["mfma_compute", "hbm_bw"],
    "attn.decode_paged": ["hbm_bw"],               # streams the KV cache, arithmetic intensity ~1
    "attn.linear": ["mfma_compute", "hbm_bw"],
    "attn.sparse": ["hbm_bw", "l2_locality"],      # gathered blocks: scattered reads, poor locality
    "attn.spec_decode": ["hbm_bw", "launch_overhead"],
    "norm_act": ["hbm_bw", "launch_overhead"],     # one pass over activations, tiny per-call work
    "elementwise_reduction": ["hbm_bw", "launch_overhead"],
    "data_movement": ["hbm_bw", "l2_locality"],
    "collective": ["sync", "hbm_bw"],
    "quant": ["hbm_bw"],
    "positional": ["hbm_bw", "launch_overhead"],
    "embedding_sampling": ["hbm_bw", "launch_overhead"],
    "moe.routing": ["launch_overhead", "hbm_bw"],  # tiny tensors, many small kernels
    "moe.dispatch": ["sync", "hbm_bw"],
    "conv": ["mfma_compute", "hbm_bw"],
}

# `levers` for the two per-operator files whose lever follows from the file's PURPOSE rather than
# from its prose. Everything else keeps whatever P1/P2 gave it (or stays unset — precision first).
FUSION_LEVERS = ["fusion.epilogue", "fusion.prologue"]
FUSION_LEVERS_NORM = ["fusion.norm-quant", "fusion.epilogue"]
OP_LEVERS = {
    "splitk_streamk_gemm": ["tile.splitk", "tile.streamk"],
}

CAP = 2


def evidence_bounds(text):
    """Return [(bound, quoted_span)] for every unambiguous bottleneck phrase in the doc."""
    out = []
    for bound, pat in EVIDENCE_RE:
        m = re.search(pat, text, re.I)
        if not m:
            continue
        line = next((ln.strip() for ln in text.splitlines() if m.group(0).lower() in ln.lower()), "")
        out.append((bound, (line or m.group(0))[:80]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rows, stats = [], {"evidence": 0, "prior": 0, "skipped": 0}
    for op in sorted(os.listdir(OPS)):
        opdir = os.path.join(OPS, op)
        if not os.path.isdir(opdir):
            continue
        kc = V.kernel_class_for_operator(op)
        prior = CLASS_PRIOR.get(kc, [])
        if not prior:
            print(f"  [no prior] {op} (kernel_class={kc}) — add it to CLASS_PRIOR", file=sys.stderr)

        ov = os.path.join(opdir, "overview.md")
        ev = evidence_bounds(open(ov, encoding="utf-8").read()) if os.path.isfile(ov) else []
        ev_map = dict(ev)

        # The prior's HEAD is definitional and always keeps a slot: an all-reduce card whose prose
        # happens to say "bandwidth-bound" is still the thing a `--bound sync` query must find.
        # Evidence fills the remaining slot(s), then the rest of the prior.
        bounds, cites = [], []

        def take(b, why):
            if b and b not in bounds and len(bounds) < CAP:
                bounds.append(b)
                cites.append(why)

        take(prior[0] if prior else "", f"kernel_class {kc} prior (definitional)")
        for b, span in ev:
            take(b, f'overview.md: "{span}"')
        for b in prior[1:]:
            take(b, f"kernel_class {kc} prior")
        if not bounds:
            stats["skipped"] += 1
            continue
        stats["evidence" if ev_map else "prior"] += 1
        evidence = "; ".join(cites)

        for dp, _dn, fn in os.walk(opdir):
            for f in sorted(fn):
                if not f.endswith(".md") or f in ("README.md", "numerics.md"):
                    continue                    # numerics.md states a precision property, not a bound
                rel = os.path.relpath(os.path.join(dp, f), REPO)
                r = {"path": rel, "src": "llm", "bound_type": bounds, "evidence": evidence}
                if f == "fusion.md":
                    r["levers"] = FUSION_LEVERS_NORM if kc == "norm_act" else FUSION_LEVERS
                elif f == "tuning.md" and op in OP_LEVERS:
                    r["levers"] = OP_LEVERS[op]
                errs = validate_row(r)
                if errs:
                    print(f"INVALID {rel}: {', '.join(errs)}", file=sys.stderr)
                    return 1
                rows.append(r)

    keep = [r for r in load_labels() if r.get("src") != "llm"]
    print(f"{len(rows)} llm rows over {stats['evidence'] + stats['prior']} operators "
          f"({stats['evidence']} with in-doc evidence, {stats['prior']} on the class prior alone), "
          f"{stats['skipped']} operators skipped. {len(keep)} rule/human rows preserved.")
    if a.write:
        write_labels(keep + rows)
        print("kb_labels.yaml updated.")
    else:
        print("dry-run — pass --write to merge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
