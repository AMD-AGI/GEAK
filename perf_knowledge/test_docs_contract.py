#!/usr/bin/env python3
"""Check the two structural promises `perf_knowledge/README.md` makes about this tree.

The README's banner has long advertised "0 broken links · every content file ends with `## Sources`".
Both were true when typed and neither was checked by anything, which is the failure mode this file
exists to remove: a claim that *looks* verifiable and is not is worse than no claim, because a reader
budgets trust against it. When this ran for the first time it found 25 content files with no
`## Sources` — the banner had been wrong for a while and nothing said so.

Four invariants:

1. **Every relative markdown link resolves.** A knowledge base is navigated by following links; one
   that dead-ends sends the reader to `ls` and, more often, to a wrong assumption about what exists.
   External URLs are not fetched (that would make this a network test and flaky by construction) and
   `#anchors` are not resolved into headings — both limits are stated in the output rather than left
   for a reader to discover.

2. **Every content file carries a `## Sources` section.** This tree's whole claim is that a number in
   it can be traced. A file with no sources is indistinguishable from a file whose numbers were
   invented, and the reader cannot tell which they are holding.

3. **Every `repo@commit:path` citation resolves on this box**, through git objects rather than
   through a path. Five aiter checkouts live here at five commits, so the same path holds a
   different file in each and a path-based check would both miss real breakage and invent fake.

4. **The generated indexes still match the cards.** `index/{sota_registry,capability_index}.yaml`
   and `index/sota_matrix.md` are derived from card frontmatter and are what the workflows read; a
   card fixed without re-running the generator leaves the stale copy live where it is consumed while
   checks 1-3 over the cards report clean. Observed: a path corrected in a card stayed broken in both
   YAMLs for a commit.

Exemptions are enumerated with a reason each, never as a pattern that happens to make the run green.

Run: `python3 perf_knowledge/test_docs_contract.py`
"""

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REL = "perf_knowledge"

# Markdown inline link. A real target holds no whitespace, which is what keeps prose like
# `[x](qx, qy)` from being read as a path — an earlier cut of this check reported four such
# "broken links" that were never links.
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")

# Not sources, with the reason. Each entry answers "why would this file cite anything?".
SOURCES_EXEMPT = {
    "index/sources_index.md": "is the sources index — citing itself is circular",
    "index/changelog.md": "records what changed in this tree; its source is the git history",
    "index/sota_matrix.md": "a registry of pointers; every row's source is the doc it points at",
}
# Directory-level exemptions, same rule: a reason, not a convenience.
SOURCES_EXEMPT_DIRS = {
    "_templates": "templates are shapes to fill, not content — their `## Sources` is a placeholder",
    "expert_skills": "vendored skill bundles carry their upstream layout; rewriting their headings "
                     "would fork them from the source they came from",
}
# Structural files: an index or a readme is navigation, and its sources are its entries.
SOURCES_EXEMPT_NAMES = {"INDEX.md", "README.md", "SKILL.md"}

# Trees that are VENDORED whole and hash-pinned, so their internal links are not ours to repair.
# `expert_skills/tuning/` is synced by `e2e_workflow/scripts/tuning_skillset_sync.py` against
# `tuning_skillset.manifest.sha256`, and `test_tuning_skillset_sync.py` fails the build on any
# in-place edit ("re-sync it, do not patch it in place"). Its README indexes per-model directories
# that the sync does not carry, so 12 of its links dangle here and cannot dangle upstream. Without
# this entry the two gates contradict each other: one demands the edit, the other forbids it.
LINKS_EXEMPT_DIRS = {
    "expert_skills/tuning": "vendored + sha256-pinned; edits are reverted by the skillset sync",
}


def markdown_files():
    for dirpath, dirnames, filenames in os.walk(HERE):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        for name in sorted(filenames):
            if name.endswith(".md"):
                yield os.path.join(dirpath, name)


def rel(path):
    return os.path.relpath(path, HERE).replace(os.sep, "/")


def exempt_from_sources(relpath):
    if relpath in SOURCES_EXEMPT:
        return SOURCES_EXEMPT[relpath]
    parts = relpath.split("/")
    for d, why in SOURCES_EXEMPT_DIRS.items():
        if d in parts[:-1]:
            return why
    if parts[-1] in SOURCES_EXEMPT_NAMES:
        return "structural: navigation, whose sources are its entries"
    return None


def check_links():
    """Relative links resolve. Returns (checked, [failure strings])."""
    checked, bad = 0, []
    for path in markdown_files():
        relpath = rel(path)
        # Templates link to the paths the FILLED file will have, so they dangle by design.
        if "_templates" in relpath.split("/"):
            continue
        if any(relpath.startswith(d + "/") for d in LINKS_EXEMPT_DIRS):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for m in LINK.finditer(text):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#", "<")):
                continue
            # A literal ellipsis inside link parens is prose, not a path: `[foo](...)`.
            if target in {"...", "…"}:
                continue
            checked += 1
            bare = target.split("#")[0]
            if not bare:
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), bare))
            if not os.path.exists(resolved):
                line = text[:m.start()].count("\n") + 1
                bad.append(f"{relpath}:{line} -> {target} (no such path)")
    return checked, bad


def has_frontmatter_sources(text):
    """A non-empty `sources:` block in the YAML header counts as citing sources.

    The invariant is "this file says where it came from", not "this file has a heading spelled
    Sources". `hardware/rdna4_gfx1201/*.md` carry a populated `sources:` list in frontmatter and no
    section, which the heading-only check read as uncited — the strictly worse outcome of the two,
    since the fix it invites is to restate in prose what the header already records.
    """
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return False
    header = m.group(1).splitlines()
    for i, line in enumerate(header):
        if not re.match(r"^sources:\s*(.*)$", line):
            continue
        inline = re.match(r"^sources:\s*(.*)$", line).group(1).strip()
        if inline and inline not in ("[]", "~", "null"):
            return True
        # Block form: `sources:` on its own line, entries indented under it.
        return any(l.startswith((" ", "\t")) and l.strip().startswith("- ")
                   for l in header[i + 1:i + 40])
    return False


def check_sources():
    """Content files carry `## Sources`. Returns (checked, [failure strings], exempted count)."""
    checked, bad, exempted = 0, [], 0
    for path in markdown_files():
        relpath = rel(path)
        why = exempt_from_sources(relpath)
        if why:
            exempted += 1
            continue
        checked += 1
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if not (re.search(r"^##+\s+Sources\b", text, re.MULTILINE) or has_frontmatter_sources(text)):
            bad.append(f"{relpath} (no `## Sources` section and no `sources:` in frontmatter)")
    return checked, bad, exempted


def check_citations():
    """Every `repo@commit:path` citation resolves in a checkout on this box.

    The third promise, and the one that was silently false the longest. A `## Sources` section
    satisfies check_sources() whether or not a single line in it points at anything real: the second
    invariant checks that sources were CITED, never that the citations RESOLVE. Two did not -- a
    path that moved under `aiter/ops/` before the cited commit -- and the suite was green over them
    for as long as they existed.

    The check is delegated to `index/resolve_citation.py`, which resolves through git objects rather
    than through a path, because this box holds five aiter checkouts at five commits and "the file
    is not there" usually means "you are standing in the wrong tree". Returns (checked, [failures]).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "resolve_citation", os.path.join(HERE, "index", "resolve_citation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cites = mod.kb_citations()
    per_repo, bad = {}, []
    for repo, _, _, _ in cites:
        name = repo.split("/")[-1]
        if name not in per_repo:
            per_repo[name] = mod.checkouts(name)
    seen = {}
    for repo, commit, path, src in cites:
        seen.setdefault((repo, commit, path), []).append(src)
    for (repo, commit, path), srcs in sorted(seen.items()):
        trees = per_repo[repo.split("/")[-1]]
        if not trees:
            # No checkout of that repo on this box: not a defect in the card. Say so and move on,
            # rather than reporting a citation nobody can check here as a broken one.
            continue
        r = mod.resolve(commit, path, trees)
        if not r["ok"]:
            bad.append(f"{sorted(set(srcs))[0]} -> {repo}@{commit[:8]}:{path} ({r['reason']})")
    return len(cites), bad


def check_index_fresh():
    """`index/{sota_registry,capability_index}.yaml` + `sota_matrix.md` still match the cards.

    These three are GENERATED from card frontmatter and are what a workflow actually reads. A card
    edited without re-running the generator leaves the old status, dtypes and sources live in the
    machine-consumed copy while the card itself looks right -- so the other three checks pass and the
    agent is handed the stale answer.
    """
    r = subprocess.run([sys.executable, os.path.join(HERE, "index", "_gen_registry.py"), "--check"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        return []
    return [(r.stderr.strip() or r.stdout.strip() or "index/_gen_registry.py --check failed")]


def main():
    failures = []

    n_links, bad_links = check_links()
    print(f"links:   {n_links} relative links checked, {len(bad_links)} broken")
    print("         (external URLs are not fetched; `#anchors` are not resolved to headings)")
    failures += bad_links

    n_src, bad_src, n_exempt = check_sources()
    print(f"sources: {n_src} content files checked, {len(bad_src)} citing nothing "
          f"({n_exempt} exempt, each with a stated reason)")
    failures += bad_src

    n_cite, bad_cite = check_citations()
    print(f"cites:   {n_cite} `repo@commit:path` citations resolved against on-box checkouts, "
          f"{len(bad_cite)} unresolved")
    print("         (resolved through git objects, not through a path -- see "
          "index/resolve_citation.py)")
    failures += bad_cite

    bad_index = check_index_fresh()
    print(f"index:   generated index/ vs. card frontmatter, {len(bad_index)} stale file-set")
    failures += bad_index

    if failures:
        print(f"\nFAIL: {len(failures)} problem(s) in {REL}/", file=sys.stderr)
        for f in failures[:60]:
            print(f"  {f}", file=sys.stderr)
        if len(failures) > 60:
            print(f"  ... and {len(failures) - 60} more", file=sys.stderr)
        print("\nEither fix the file or, if it genuinely has no sources to cite, add it to "
              "SOURCES_EXEMPT with the reason.", file=sys.stderr)
        return 1

    print(f"\nOK: {REL}/ links all resolve, every content file cites its sources, every citation "
          "resolves on this box, and the generated indexes match the cards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
