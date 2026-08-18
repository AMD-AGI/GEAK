#!/usr/bin/env python3
"""Unified KB retrieval layer (plan Part 1.5 / P3) — the ONE query entry point.

Reads kb_manifest.yaml (built by _gen_index.py across all three layers) and returns a
sorted, de-duplicated list of `path + one-line hook`. Ordering (plan Part 1.5):

  1. cost ASCENDING (L0 -> L3)                — "try the cheap lever first"
  2. kernel_class EXACT match  >  class-agnostic (cross-cutting)
  3. platform EXACT match  >  platform-independent
  4. layer/lifecycle: learned(active) > reference(active) > candidate > stale
  5. verified_on freshness (newer first)

ADD-only philosophy (learned/README.md): the resolver only RANKS "read these first"; it never
prunes the agent's own candidates. `stale`/`archived` cards are surfaced too, down-weighted and
marked ⚠/⛔. The footer always reports the full matched total and any display truncation.

An UNSET field on a card means "applies to everything", never "applies to nothing" — that holds
uniformly for platforms, skus and kernel_class, and is what keeps the cross-cutting docs
(hardware/, optimization/, profiling/, quantization/) reachable from an operator-specific query.

Usage:
  kb_resolve.py --gfx gfx942 --kernel-class gemm.dense --bound hbm_bw --max-cost L2
  kb_resolve.py --gfx gfx942 --operator dense_gemm --limit 12      # operator id -> kernel_class
  kb_resolve.py --gfx gfx950 --sku mi355x --layer reference,learned --limit 20 --json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _kb_vocab as V  # noqa: E402

try:
    import yaml
except Exception:
    yaml = None

MANIFEST = os.path.join(HERE, "kb_manifest.yaml")

LAYER_RANK = {"learned": 0, "reference": 1, "artifact": 2}
LIFE_RANK = {"active": 0, "candidate": 1, "stale": 2, "archived": 3}


def load_manifest(path):
    if not os.path.isfile(path):
        return []
    text = open(path, encoding="utf-8").read()
    if yaml is not None:
        try:
            d = yaml.safe_load(text) or {}
            return d.get("records", []) if isinstance(d, dict) else []
        except Exception:
            pass
    return []


def matches(r, a):
    if a.gfx:
        gfx = V.SKU_TO_GEN.get(a.gfx, a.gfx) if a.gfx in V.SKU_IDS else a.gfx
        plats = r.get("platforms") or []
        if plats and gfx not in plats:              # platform-independent (empty) always eligible
            return False
    if a.sku:
        skus = r.get("skus") or []
        if skus and a.sku not in skus:              # sku-agnostic (empty) still eligible
            return False
    kc = r.get("kernel_class")
    # An UNSET kernel_class is a cross-cutting doc (hardware/, optimization/, profiling/, …) and is
    # eligible for every class — same wildcard semantics as platforms/skus above. Only a card that
    # declares a DIFFERENT class is dropped.
    if a.kernel_class and kc and kc != a.kernel_class:
        return False
    if a.bound and a.bound not in (r.get("bound_type") or []):
        return False
    if a.layer:
        want = {x.strip() for x in a.layer.split(",") if x.strip()}
        if r.get("layer") not in want:
            return False
    if a.max_cost:
        cap = V.COST_RANK.get(a.max_cost)
        c = r.get("cost")
        # A card with an explicit cost above the cap is dropped; an UNSET cost is never pruned by cost.
        if c in V.COST_RANK and cap is not None and V.COST_RANK[c] > cap:
            return False
    if not a.include_archived and r.get("lifecycle") == "archived":
        return False
    return True


def sort_key(r, gfx, kernel_class=""):
    cost = V.COST_RANK.get(r.get("cost"), len(V.COSTS))     # unset cost sorts AFTER explicit costs
    # A card that DECLARES the queried class outranks a class-agnostic cross-cutting doc, which in
    # turn outranks anything else. Mirrors plat_exact; only meaningful once --kernel-class is given.
    kc = r.get("kernel_class")
    kc_exact = 0 if (kernel_class and kc == kernel_class) else (1 if not kc else 2)
    plats = r.get("platforms") or []
    plat_exact = 0 if (gfx and gfx in plats) else (1 if not plats else 2)
    layer = LAYER_RANK.get(r.get("layer"), 3)
    life = LIFE_RANK.get(r.get("lifecycle"), 4)
    return (cost, kc_exact, plat_exact, life, layer, _inv_date(r.get("verified_on")))


def _inv_date(s):
    """Invert a YYYY-MM-DD string so ascending sort == newest first; missing sorts last."""
    if not s:
        return (0,)
    try:
        return tuple(-int(x) for x in str(s).replace("-", " ").split()[:3])
    except Exception:
        return (0,)


def mark(r):
    life = r.get("lifecycle")
    m = {"stale": " ⚠stale", "archived": " ⛔archived", "candidate": " ·candidate"}.get(life, "")
    if not r.get("verified_on") and r.get("layer") == "artifact":
        m += " ⚠unverified"
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gfx", default="")
    ap.add_argument("--sku", default="")
    ap.add_argument("--kernel-class", dest="kernel_class", default="")
    ap.add_argument("--operator", default="",
                    help="taxonomy operator id (e.g. dense_gemm); translated to --kernel-class via "
                         "_kb_vocab.OPERATOR_KERNEL_CLASS. Ignored when --kernel-class is given.")
    ap.add_argument("--bound", default="")
    ap.add_argument("--max-cost", dest="max_cost", default="", choices=["", "L0", "L1", "L2", "L3"])
    ap.add_argument("--layer", default="")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--include-archived", dest="include_archived", action="store_true")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    # --operator is the ONE bridge between the taxonomy operator ids the workflows carry around
    # (kk_operator: dense_gemm) and the dotted kernel_class ids the manifest is keyed by (gemm.dense).
    # Keeping it here means no prompt has to hardcode the 50-row map. An unknown operator must NOT
    # silently degrade into "match everything" — it stays unset and says so on stderr.
    if a.operator and not a.kernel_class:
        kc = V.kernel_class_for_operator(a.operator)
        if not kc:
            print(f"kb_resolve: unknown operator '{a.operator}' — not in _kb_vocab.OPERATOR_KERNEL_CLASS; "
                  f"see index/taxonomy.md. Proceeding WITHOUT a class filter.", file=sys.stderr)
        a.kernel_class = kc or ""

    recs = load_manifest(a.manifest)
    if not recs:
        print("kb_resolve: kb_manifest.yaml missing or empty — run index/_gen_index.py first.",
              file=sys.stderr)

    gfx = V.SKU_TO_GEN.get(a.gfx, a.gfx) if a.gfx in V.SKU_IDS else a.gfx
    hits = [r for r in recs if matches(r, a)]
    hits.sort(key=lambda r: sort_key(r, gfx, a.kernel_class))
    shown = hits[: max(0, a.limit)] if a.limit else hits

    if a.json:
        print(json.dumps({"total": len(hits), "shown": len(shown), "truncated": len(hits) - len(shown),
                          "candidates": shown}, ensure_ascii=False))
        return 0

    q = ", ".join(f"{k}={v}" for k, v in [("gfx", a.gfx), ("sku", a.sku), ("operator", a.operator),
                  ("kernel_class", a.kernel_class), ("bound", a.bound), ("max_cost", a.max_cost),
                  ("layer", a.layer)] if v)
    print(f"# kb_resolve — {q or 'no filters'}")
    print(f"# {len(hits)} candidates matched (cost-ascending; ADD-only: nothing pruned).\n")
    for r in shown:
        cost = r.get("cost") or "--"
        lev = ",".join(r.get("levers") or []) or "-"
        print(f"[{cost}] {r['path']}{mark(r)}\n      {r.get('hook','')}  «{lev}»")
    if len(hits) > len(shown):
        print(f"\n… {len(hits) - len(shown)} more matched but not shown (raise --limit). "
              f"Total candidate count = {len(hits)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
