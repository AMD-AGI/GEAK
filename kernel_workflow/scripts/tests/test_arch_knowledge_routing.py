#!/usr/bin/env python3
"""test_arch_knowledge_routing.py — arch knowledge must route, and must not lie.

The failure this prevents: before `amd_rdna.md` existed, the knowledge base was
CDNA throughout. `amd_instinct.md` opens by declaring the workflow runs on
MI-series parts, and a repo-wide count was gfx950 x3322, gfx942 x2653,
gfx1151 x3. An agent on a wave32 part loaded it and produced confidently
CDNA-shaped advice -- MFMA, AGPRs, HBM bandwidth -- none of which exists there.

Routing is currently prompt-driven: the agent reads a guard and picks a file. A
test cannot check what the agent does, so it checks the invariants that make the
guard possible at all, and the ones that make the picked file honest.

Following test_kb_switch.py, the file set is derived FROM THE TREE, not from a
list here, so a third arch file cannot be added without joining the contract.

    python3 -m pytest -q kernel_workflow/scripts/tests/test_arch_knowledge_routing.py
    python3 kernel_workflow/scripts/tests/test_arch_knowledge_routing.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.dirname(os.path.dirname(HERE))
KNOW = os.path.join(WF, "knowledge")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# An arch knowledge file is a top-level knowledge/*.md that names a gfx target
# or a matrix instruction: that is what makes it arch-specific, and therefore
# what obliges it to say which arch it is for.
_ARCH_MARKERS = re.compile(r"\bgfx\d{3,4}\b|\bMFMA\b|\bWMMA\b", re.I)


def arch_files():
    out = []
    for name in sorted(os.listdir(KNOW)):
        path = os.path.join(KNOW, name)
        if name.endswith(".md") and os.path.isfile(path) and _ARCH_MARKERS.search(_read(path)):
            out.append(name)
    return out


def _rdna_file():
    return next((n for n in arch_files() if "rdna" in n.lower()), None)


def _cdna_file():
    return next((n for n in arch_files() if "instinct" in n.lower() or "cdna" in n.lower()), None)


def test_both_families_have_a_file():
    names = arch_files()
    assert len(names) >= 2, (
        "expected at least an RDNA file and a CDNA file; found %r. If the RDNA "
        "file was removed, the CDNA file is unconditionally governing again, "
        "which is the bug this guards." % names)
    assert _rdna_file(), "no RDNA arch knowledge file among %r" % names
    assert _cdna_file(), "no CDNA/Instinct arch knowledge file among %r" % names


# A guard is an explicit statement that this file may be the wrong one, not any
# sentence that happens to contain "wavefront". An earlier, looser pattern
# passed triton_optimization.md on the strength of the line "Ensure BLOCK_SIZE
# is a multiple of 64 (AMD wavefront size)" -- a CDNA number asserted as a
# universal fact, i.e. exactly the defect this test exists to catch.
_GUARD = re.compile(
    r"arch guard|DETECT THE BOX|detect .{0,20}first|"
    r"(read|see|use)\s+`?(knowledge/)?amd_(rdna|instinct)\.md|"
    r"not applicable (on|to)|does not apply (on|to)", re.I)


def test_every_arch_file_carries_a_guard():
    for name in arch_files():
        head = "\n".join(_read(os.path.join(KNOW, name)).split("\n")[:40])
        assert _GUARD.search(head), (
            "%s has no routing guard in its first 40 lines: an agent that opens "
            "it has no way to learn it may be the wrong file" % name)


def test_the_two_families_name_each_other():
    rdna, cdna = _rdna_file(), _cdna_file()
    for a, b in ((rdna, cdna), (cdna, rdna)):
        assert b in _read(os.path.join(KNOW, a)), (
            "%s never names %s, so a reader that landed on the wrong file is "
            "not told where to go" % (a, b))


# Mentioning MFMA/AGPR is not the problem -- the RDNA file exists to say they
# are absent, and it contrasts them against WMMA in tables. The problem is
# RECOMMENDING them, so match advisory phrasing only.
_RECOMMENDS = re.compile(
    r"\b(use|prefer|switch to|budget|target|schedule|pack|tune|allocate|stage|"
    r"issue)\b[^.]{0,60}\b(mfma|agprs?)\b", re.I)


def test_rdna_file_does_not_recommend_cdna_hardware():
    name = _rdna_file()
    for line in _read(os.path.join(KNOW, name)).split("\n"):
        if "mfma" not in line.lower() and "agpr" not in line.lower():
            continue
        if line.lstrip().startswith("|"):
            continue                      # comparison table row: CDNA vs RDNA
        assert not _RECOMMENDS.search(line), (
            "%s line %r RECOMMENDS MFMA/AGPR -- that is the CDNA-shaped advice "
            "this file exists to displace" % (name, line.strip()[:90]))


def test_cited_repro_scripts_ship():
    name = _rdna_file()
    cited = set(re.findall(r"`([A-Za-z0-9_\-]+\.(?:py|sh))`", _read(os.path.join(KNOW, name))))
    present = set()
    for _root, _dirs, files in os.walk(WF):
        present.update(files)
    missing = sorted(s for s in cited if s not in present)
    assert not missing, (
        "%s cites %r as evidence but they do not ship anywhere under "
        "kernel_workflow/ -- [measured] claims nobody can re-take" % (name, missing))


def test_profiling_cost_figures_state_their_conditions():
    name = _rdna_file()
    body = _read(os.path.join(KNOW, name))
    if not (re.search(r"85\s*s", body) and re.search(r"27\s*s", body)):
        return
    i = body.find("85 s")
    window = body[max(0, i - 1500):i + 1500]
    assert re.search(r"warm|fresh|process start|per pass", window, re.I), (
        "%s gives both a ~85 s and a 27 s profiling figure without stating the "
        "conditions that separate them; they differ 16x and read as a "
        "contradiction" % name)


def test_no_file_claims_l2_is_unobtainable():
    """gfx1151 hides the raw TCC_/TCP_ names, but GL2C_HIT / GL2C_MISS are
    collectable and were collected: a real run measured GL2 hit rate 50.37%.
    A guard that says L2 hit rate is unobtainable tells the agent to stop
    trying, which is worse than saying nothing."""
    bad = re.compile(r"(l1/l2|l2)[^.\n]{0,60}(unobtainable|not available|cannot be measured)", re.I)
    for name in arch_files():
        for n, line in enumerate(_read(os.path.join(KNOW, name)).split("\n"), 1):
            assert not bad.search(line), (
                "%s:%d claims L2 hit rate is unobtainable: %r. GL2C_HIT/GL2C_MISS "
                "are available on this part (measured GL2 hit 50.37%%)."
                % (name, n, line.strip()[:90]))


def _main():
    fails = []
    for fn in sorted(k for k in globals() if k.startswith("test_")):
        try:
            globals()[fn]()
        except AssertionError as exc:
            fails.append("%s: %s" % (fn, exc))
    if fails:
        print("FAIL: %d problem(s)" % len(fails))
        for f in fails:
            print("  - %s" % f)
        return 1
    print("PASS: %d arch knowledge file(s) route and cite honestly: %s"
          % (len(arch_files()), ", ".join(arch_files())))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
