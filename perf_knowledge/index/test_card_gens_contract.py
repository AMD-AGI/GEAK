#!/usr/bin/env python3
"""test_card_gens_contract.py — a card that CLAIMS an arch must also SCOPE its evidence.

The gap this closes. The gfx1151 adaptation added `gfx1151` to `gens:` on 95 of the
226 SOTA cards, and appended a disclaimer to each saying the same thing in prose: the
gen is listed because the backend's SOURCE is portable to RDNA, not because anything
here was measured there, and any fp8/fp4 dtype in the row is CDNA-only because gfx1151
has no fp8 matrix instruction — an fp8 candidate runs EMULATED, passing correctness
and losing performance silently.

That correspondence is 95/95 today, and it held only because one commit wrote both
halves at once. Nothing re-checked it. The two halves are in different places in the
file — frontmatter at the top, prose at the bottom — so the cheap edit (add a gen to
a `gens:` line) is exactly the one that drops the expensive half.

Why that matters more here than it looks. `index/capability_index.yaml` and
`sota_registry.yaml` are GENERATED from this frontmatter, and the documented query
flow is "filter by (operator, gen, dtype, regime) -> candidate backends -> read card
-> MEASURE". A gen with no disclaimer therefore hands an agent a CDNA ranking, a CDNA
tuning recipe and a CDNA roofline under an RDNA filter, with nothing on the page
saying so. The failure is silent in both directions: nobody sees a wrong number, they
see a plausible one.

The arch knowledge files in kernel_workflow/knowledge already have this contract
pinned (tests/test_arch_knowledge_routing.py). The 95 cards are the larger and
newer half of the same idea and had nothing. This file is that test, and like the
routing one it derives the card set FROM THE TREE so a new card cannot opt out by
simply not being listed here.

    python3 -m pytest -q perf_knowledge/index/test_card_gens_contract.py
    python3 perf_knowledge/index/test_card_gens_contract.py
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(HERE)
CARDS = sorted(glob.glob(os.path.join(PK, "operators", "*", "backends", "*.md")))

# The arch this contract is about. Kept as a table rather than a constant: a second
# non-CDNA target (gfx1200, say) joins by adding a row, and inherits the test.
NON_CDNA = {
    "gfx1151": {
        # Prose that must appear on any card claiming the gen. Matched loosely enough
        # to survive rewording, strictly enough that the claim has to actually be made.
        # `[\s*_]+` rather than `\s+` between words: the cards bold the negation
        # ("It is **not** a claim that anything..."), and a plain \s+ silently fails to
        # span the asterisks — which made this test red on 95/95 correct cards the first
        # time it ran. A contract test that fires on the healthy tree teaches people to
        # ignore it, so the emphasis has to be part of the grammar, not an exception.
        "scope": re.compile(r"not[\s*_]+a[\s*_]+claim[\s*_]+that[\s*_]+anything", re.I),
        "heading": re.compile(r"^##\s+On\s+gfx1151\b", re.I | re.M),
        # Extra clause required only when the row advertises a dtype the card cannot
        # deliver in hardware there.
        "emulated": re.compile(r"EMULAT", re.I),
    },
}

# fp8/fp4/mxfp entries are the dangerous ones: they do not fail on RDNA, they emulate.
_SOFT_DTYPE = re.compile(r"\b(fp8|fp4|mxfp|e4m3|e5m2|e2m1)\w*", re.I)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _frontmatter(txt):
    """Return the YAML-ish frontmatter block, or '' when the card has none."""
    if not txt.startswith("---"):
        return ""
    end = txt.find("\n---", 3)
    return txt[3:end] if end != -1 else ""


def _list_field(fm, name):
    m = re.search(r"^%s:\s*\[(.*?)\]\s*$" % re.escape(name), fm, re.M)
    if not m:
        return []
    return [v.strip() for v in m.group(1).split(",") if v.strip()]


def _cards():
    assert CARDS, "no SOTA cards found under %s/operators — wrong tree?" % PK
    return CARDS


def _rel(path):
    return os.path.relpath(path, PK)


def test_every_card_claiming_a_non_cdna_gen_scopes_its_evidence():
    missing = []
    for path in _cards():
        txt = _read(path)
        gens = _list_field(_frontmatter(txt), "gens")
        for gen, rules in NON_CDNA.items():
            if gen not in gens:
                continue
            if not (rules["heading"].search(txt) and rules["scope"].search(txt)):
                missing.append("%s (gens has %s)" % (_rel(path), gen))
    assert not missing, (
        "%d card(s) list a non-CDNA gen with no section scoping the evidence to CDNA.\n"
        "The generated index will offer these as candidates under an RDNA filter, and the\n"
        "page they point at reads as if its numbers apply:\n  %s"
        % (len(missing), "\n  ".join(missing)))


def test_soft_dtypes_on_a_non_cdna_gen_are_marked_emulated():
    missing = []
    for path in _cards():
        txt = _read(path)
        fm = _frontmatter(txt)
        gens = _list_field(fm, "gens")
        soft = [d for d in _list_field(fm, "dtypes") if _SOFT_DTYPE.search(d)]
        for gen, rules in NON_CDNA.items():
            if gen in gens and soft and not rules["emulated"].search(txt):
                missing.append("%s (%s + %s)" % (_rel(path), gen, ",".join(soft)))
    assert not missing, (
        "%d card(s) advertise an fp8/fp4 dtype on a gen with no hardware for it, and never\n"
        "say the path is emulated. That candidate passes correctness and loses performance\n"
        "with nothing to attribute the loss to:\n  %s"
        % (len(missing), "\n  ".join(missing)))


def test_the_disclaimer_is_not_present_without_the_claim():
    """The mirror image: prose about gfx1151 on a card the index will never surface for it.

    Harmless to a reader, but it means someone edited one half and not the other — and
    the direction that loses the disclaimer is the same edit. Catching both directions is
    what makes this a contract rather than a lint.
    """
    orphaned = []
    for path in _cards():
        txt = _read(path)
        gens = _list_field(_frontmatter(txt), "gens")
        for gen, rules in NON_CDNA.items():
            if rules["heading"].search(txt) and gen not in gens:
                orphaned.append("%s (section for %s, not in gens)" % (_rel(path), gen))
    assert not orphaned, (
        "%d card(s) carry an arch section for a gen their frontmatter does not claim:\n  %s"
        % (len(orphaned), "\n  ".join(orphaned)))


def test_generated_index_agrees_with_the_cards():
    """The index is generated, so it can only disagree if it is STALE — and a stale index
    is read as authoritative by anything doing the documented (operator, gen, dtype)
    filter. Compare the gen sets rather than re-running the generator, so this stays a
    check and never becomes a write."""
    idx = os.path.join(HERE, "capability_index.yaml")
    if not os.path.exists(idx):
        return  # nothing generated in this tree; the cards are still the source of truth
    body = _read(idx)
    for gen in NON_CDNA:
        in_cards = sum(1 for p in _cards() if gen in _list_field(_frontmatter(_read(p)), "gens"))
        in_index = len(re.findall(r"\b%s\b" % re.escape(gen), body))
        assert in_index >= in_cards, (
            "capability_index.yaml names %s %d time(s) but %d card(s) claim it — the index "
            "is stale. Regenerate with `python3 index/_gen_registry.py` from perf_knowledge/."
            % (gen, in_index, in_cards))


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   %s" % name)
            except AssertionError as exc:
                fails += 1
                print("FAIL %s\n%s" % (name, exc))
    sys.exit(1 if fails else 0)
