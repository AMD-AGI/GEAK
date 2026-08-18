#!/usr/bin/env python3
"""Validate the KB frontmatter + structured index (plan Part 2.7 / P6 consistency check).

Checks (ERROR fails CI; WARN is advisory):
  * ERROR  unknown value: kernel_class / lever / cost / risk / bound_type / lifecycle / layer / sku
           not defined in _kb_vocab.py.
  * ERROR  taxonomy.md is missing an id that _kb_vocab.py defines (the human mirror drifted).
  * ERROR  orphan: a content file with frontmatter that _gen_index.py would index but that is
           absent from kb_manifest.yaml (stale manifest — re-run _gen_index.py).
  * WARN   levers / cost / bound_type unset on a card (structured axes not yet filled).
  * WARN   lifecycle:active but verified_on:null on an ARTIFACT card (never on-box reproduced).
  * WARN   status:sota on a sota_card whose body carries no measured quantity (an unbacked claim).

Run from perf_knowledge/:  python3 index/_validate_kb.py
Exit code 0 = no errors (warnings allowed); 1 = at least one error.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _kb_vocab as V  # noqa: E402

try:
    import yaml
except Exception:
    yaml = None

PK_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(PK_ROOT)
LEARNED_ROOT = os.path.join(REPO, "e2e_workflow", "knowledge", "learned")
KW_KNOW = os.path.join(REPO, "kernel_workflow", "knowledge")
MANIFEST = os.path.join(HERE, "kb_manifest.yaml")
TAXONOMY = os.path.join(HERE, "taxonomy.md")

FM_RE = re.compile(r"^---\n(.*?)\n---\n?", re.S)
# A measured quantity: a number carrying a performance unit. Kept in sync with the identically
# named constant in _demote_thin.py — both answer "did anyone actually measure this?".
NUM = re.compile(
    r"\d+(?:\.\d+)?\s*(?:[x×]\b|%|TFLOP|GFLOP|GB/s|TB/s|\bms\b|\bus\b|µs|tok/s|tokens/s)",
    re.I)
EXCLUDE = (os.sep + "expert_skills" + os.sep, os.sep + "index" + os.sep, os.sep + "_templates" + os.sep)
SKIP_NAMES = {"INDEX.md", "README.md", "_archive.md"}

errors, warns = [], []


def err(p, m):
    errors.append(f"ERROR {p}: {m}")


def warn(p, m):
    warns.append(f"WARN  {p}: {m}")


def parse_fm(text):
    m = FM_RE.match(text)
    if not m:
        return None
    if yaml is not None:
        try:
            d = yaml.safe_load(m.group(1))
            if isinstance(d, dict):
                return d
        except Exception:
            return {}
    return {}


def as_list(v):
    if v is None:
        return []
    return [str(x).strip() for x in (v if isinstance(v, list) else [v]) if str(x).strip()]


def check_vocab(rel, fm, body=""):
    is_learned = LEARNED_ROOT in os.path.join(REPO, rel)
    layer = str(fm.get("layer", "learned" if is_learned else "reference")).strip()
    if layer and layer not in V.LAYERS:
        err(rel, f"unknown layer '{layer}'")

    kc = str(fm.get("kernel_class", "")).strip()
    if kc and kc not in V.KERNEL_CLASSES:
        err(rel, f"unknown kernel_class '{kc}'")

    for lev in as_list(fm.get("levers")):
        if lev not in V.LEVERS:
            err(rel, f"unknown lever '{lev}'")

    cost = str(fm.get("cost", "")).strip()
    if cost and cost not in V.COST_RANK:
        err(rel, f"unknown cost '{cost}'")

    risk = str(fm.get("risk", "")).strip()
    if risk and risk not in V.RISKS:
        err(rel, f"unknown risk '{risk}'")

    for b in as_list(fm.get("bound_type")):
        if b not in V.BOUND_TYPES:
            err(rel, f"unknown bound_type '{b}'")

    life = str(fm.get("lifecycle", "")).strip()
    if life and life not in V.LIFECYCLES:
        err(rel, f"unknown lifecycle '{life}'")

    for s in as_list(fm.get("skus")):
        if s not in V.SKU_IDS:
            err(rel, f"unknown sku '{s}'")

    # WARN: unfilled structured axes on a real card (has kernel_class or is a sota_card)
    if kc or str(fm.get("kind", "")).strip() == "sota_card":
        if not as_list(fm.get("levers")):
            warn(rel, "levers unset")
        if not cost:
            warn(rel, "cost unset")

    # WARN: a SOTA claim with nothing measured behind it. perf_knowledge/README.md promises that
    # "every performance number is measured", and `status: sota` is the strongest claim a card can
    # make — but most sota cards carry no number at all, so the promise is currently aspirational.
    # This makes that debt countable instead of invisible; it stays a WARN because the fix is on-box
    # measurement, not an edit, and gating CI on it would only invite deleting the claim.
    if str(fm.get("kind", "")).strip() == "sota_card" \
            and str(fm.get("status", "")).strip() == "sota" and not NUM.search(body):
        warn(rel, "status:sota but no measured quantity in the body (unbacked SOTA claim)")

    # WARN: active artifact without on-box verification
    vo = fm.get("verified_on")
    if layer == "artifact" and life == "active" and (vo is None or str(vo).strip() in ("", "null")):
        warn(rel, "lifecycle:active but verified_on:null (never on-box reproduced)")


def check_taxonomy():
    try:
        txt = open(TAXONOMY, encoding="utf-8").read()
    except OSError:
        err("index/taxonomy.md", "missing")
        return
    ids = (V.KERNEL_CLASSES | V.LEVERS | V.BOUND_TYPES | V.LIFECYCLES
           | V.GEN_IDS | set(V.COSTS) | V.RISKS)
    for i in sorted(ids):
        if i not in txt:
            err("index/taxonomy.md", f"vocab id '{i}' defined in _kb_vocab.py but absent from taxonomy.md")


def load_manifest_paths():
    if not os.path.isfile(MANIFEST):
        err("index/kb_manifest.yaml", "missing — run _gen_index.py")
        return set()
    txt = open(MANIFEST, encoding="utf-8").read()
    if yaml is not None:
        try:
            d = yaml.safe_load(txt) or {}
            return {r["path"] for r in d.get("records", []) if isinstance(r, dict) and r.get("path")}
        except Exception:
            pass
    return set(re.findall(r"^\s*- path:\s*(.+?)\s*$", txt, re.M))


def main():
    check_taxonomy()
    manifest_paths = load_manifest_paths()

    files = []
    for root in [PK_ROOT, LEARNED_ROOT, KW_KNOW]:
        for dp, _dn, fn in os.walk(root):
            if any(x in dp + os.sep for x in EXCLUDE):
                continue
            for f in fn:
                if f.endswith(".md") and f not in SKIP_NAMES:
                    files.append(os.path.join(dp, f))

    n_cards = 0
    for path in sorted(set(files)):
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            continue
        fm = parse_fm(text)
        if fm is None:
            continue  # no frontmatter -> not an indexed card (backfill/no-fm handled elsewhere)
        n_cards += 1
        rel = os.path.relpath(path, REPO)
        m = FM_RE.match(text)
        check_vocab(rel, fm, text[m.end():] if m else "")
        if rel not in manifest_paths:
            err(rel, "orphan — has frontmatter but not in kb_manifest.yaml (re-run _gen_index.py)")

    for w in warns:
        print(w)
    for e in errors:
        print(e)
    print(f"\nvalidated {n_cards} carded files: {len(errors)} error(s), {len(warns)} warning(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
