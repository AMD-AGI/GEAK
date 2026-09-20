#!/usr/bin/env python3
"""test_arch_knowledge_routing.py — arch knowledge must route, and must not lie.

The failure this prevents: before `amd_rdna.md` existed, the knowledge base was
CDNA throughout. `amd_instinct.md` opens by declaring the workflow runs on
MI-series parts, and a repo-wide count was gfx950 x3322, gfx942 x2653,
gfx1151 x3. An agent on a wave32 part loaded it and produced confidently
CDNA-shaped advice -- MFMA, AGPRs, HBM bandwidth -- none of which exists there.

Routing is currently prompt-driven: the agent reads a guard and picks a file. A
test cannot check what the agent does, so it checks the invariants that make the
guard possible at all, and the ones that make the picked file honest:

  1. every arch knowledge file carries a routing guard near the top, so neither
     can be read as unconditionally governing;
  2. the RDNA file does not carry the CDNA-only advice it exists to displace;
  3. each file points at the other, so a reader that lands on the wrong one is
     told where to go;
  4. every script the RDNA file cites as its evidence actually ships, so a
     `[measured]` claim can be re-taken rather than taken on faith;
  5. the two profiling-cost figures both state their conditions -- they differ
     16x and read as a contradiction without them.

Following test_kb_switch.py: the file set is derived FROM THE TREE, not from a
list here, so a third arch file cannot be added without joining the contract.

    python3 kernel_workflow/scripts/tests/test_arch_knowledge_routing.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.dirname(os.path.dirname(HERE))
KNOW = os.path.join(WF, "knowledge")

failures = []


def fail(msg):
    failures.append(msg)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# --- derive the arch knowledge files from the tree --------------------------
# An arch knowledge file is a top-level knowledge/*.md that names a gfx target
# or a matrix instruction: that is what makes it arch-specific, and therefore
# what obliges it to say which arch it is for.
ARCH_MARKERS = re.compile(r"\bgfx\d{3,4}\b|\bMFMA\b|\bWMMA\b", re.I)
arch_files = []
for name in sorted(os.listdir(KNOW)):
    path = os.path.join(KNOW, name)
    if not name.endswith(".md") or not os.path.isfile(path):
        continue
    if ARCH_MARKERS.search(read(path)):
        arch_files.append(name)

if len(arch_files) < 2:
    fail("expected at least two arch knowledge files (an RDNA one and a CDNA "
         "one); found %r -- if the RDNA file was removed, the CDNA file is "
         "unconditionally governing again, which is the bug this guards"
         % arch_files)

# --- 1. every arch file carries a routing guard near the top ----------------
# A guard is an explicit statement that this file may be the wrong one, not any
# sentence that happens to contain "wavefront". An earlier, looser pattern
# passed triton_optimization.md on the strength of the line "Ensure BLOCK_SIZE
# is a multiple of 64 (AMD wavefront size)" -- a CDNA number asserted as a
# universal fact, i.e. exactly the defect this test exists to catch.
GUARD_HINT = re.compile(
    r"arch guard|DETECT THE BOX|detect .{0,20}first|"
    r"(read|see|use)\s+`?(knowledge/)?amd_(rdna|instinct)\.md|"
    r"not applicable (on|to)|does not apply (on|to)", re.I)
for name in arch_files:
    head = "\n".join(read(os.path.join(KNOW, name)).split("\n")[:40])
    if not GUARD_HINT.search(head):
        fail("%s has no routing guard in its first 40 lines: an agent that "
             "opens it has no way to learn it may be the wrong file" % name)

# --- 2/3. the RDNA and CDNA files must name each other ----------------------
rdna = [n for n in arch_files if "rdna" in n.lower()]
cdna = [n for n in arch_files if "instinct" in n.lower() or "cdna" in n.lower()]
if not rdna:
    fail("no RDNA arch knowledge file found among %r" % arch_files)
if not cdna:
    fail("no CDNA/Instinct arch knowledge file found among %r" % arch_files)

for a, b in ((rdna, cdna), (cdna, rdna)):
    if a and b:
        body = read(os.path.join(KNOW, a[0]))
        if b[0] not in body:
            fail("%s never names %s, so a reader that landed on the wrong file "
                 "is not told where to go" % (a[0], b[0]))

# --- 2b. the RDNA file must not recommend CDNA-only hardware ----------------
if rdna:
    body = read(os.path.join(KNOW, rdna[0]))
    # Mentioning MFMA/AGPR is not the problem -- the file exists to say they are
    # absent, and it contrasts them against WMMA in tables. The problem is
    # RECOMMENDING them, so match advisory phrasing only.
    RECOMMENDS = re.compile(
        r"\b(use|prefer|switch to|budget|target|schedule|pack|tune|allocate|"
        r"stage|issue)\b[^.]{0,60}\b(mfma|agprs?)\b", re.I)
    for line in body.split("\n"):
        if "mfma" not in line.lower() and "agpr" not in line.lower():
            continue
        if line.lstrip().startswith("|"):
            continue                      # comparison table row: CDNA vs RDNA
        if not RECOMMENDS.search(line):
            continue
        fail("%s line %r RECOMMENDS MFMA/AGPR -- that is the CDNA-shaped advice "
             "this file exists to displace" % (rdna[0], line.strip()[:90]))

# --- 4. cited evidence must ship -------------------------------------------
if rdna:
    body = read(os.path.join(KNOW, rdna[0]))
    cited = set(re.findall(r"`([A-Za-z0-9_\-]+\.(?:py|sh))`", body))
    # Build the set of script basenames anywhere in the workflow tree, so a
    # script may live in repro/ or scripts/ without this test caring.
    present = set()
    for root, _dirs, files in os.walk(WF):
        present.update(files)
    for script in sorted(cited):
        if script not in present:
            fail("%s cites `%s` as evidence but it does not ship anywhere under "
                 "kernel_workflow/ -- a [measured] claim nobody can re-take"
                 % (rdna[0], script))

# --- 5. the two profiling-cost figures must state their conditions ----------
if rdna:
    body = read(os.path.join(KNOW, rdna[0]))
    if re.search(r"85\s*s", body) and re.search(r"27\s*s", body):
        window = body[max(0, body.find("85 s") - 1500): body.find("85 s") + 1500]
        if not re.search(r"warm|fresh|process start|per pass", window, re.I):
            fail("%s gives both a ~85 s and a 27 s profiling figure without "
                 "stating the conditions that separate them; they differ 16x "
                 "and read as a contradiction" % rdna[0])

# --- report ----------------------------------------------------------------
if failures:
    print("FAIL: %d problem(s)" % len(failures))
    for f in failures:
        print("  - %s" % f)
    sys.exit(1)

print("PASS: %d arch knowledge file(s) route and cite honestly: %s"
      % (len(arch_files), ", ".join(arch_files)))
sys.exit(0)
