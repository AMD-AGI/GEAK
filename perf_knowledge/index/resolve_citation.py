#!/usr/bin/env python3
"""Resolve a `repo@commit:path:line` citation against the checkouts actually on this box.

    python3 perf_knowledge/index/resolve_citation.py --list
    python3 perf_knowledge/index/resolve_citation.py 'aiter@4faf8713:aiter/fused_moe.py:4530'
    python3 perf_knowledge/index/resolve_citation.py --audit          # every aiter@ citation in the KB
    python3 perf_knowledge/index/resolve_citation.py --audit --json

WHY THIS EXISTS
This box carries NINE aiter checkouts at nine different commits. The knowledge base cites two of
them -- 585 citations at `a6bb4993` (which is `/sgl-workspace/aiter`) and 11 at `4faf8713` (which is
`/sgl-workspace/geak_flydsl/aiter_main/aiter`) -- and almost every card writes the path of the FIRST
one into its prose, because when those cards were written there was only one tree. A reader who
takes a `@4faf8713` citation to `/sgl-workspace/aiter` finds that the file does not exist, or worse,
finds a same-named file with different contents at the cited line.

That is not hypothetical. A fresh agent given full instructions fell in within minutes and reported
back that every grouped-MoE citation was broken, that `fused_moe.py` was 1967 lines (it is 5309 at
the cited commit), and that four named kernel files did not exist. All four resolve exactly. The
verdict was confidently wrong and nothing in the repo could contradict it, because a citation is
just text and nobody had written the mapping down.

So: a commit is the identity, a path is not. Given a commit this resolves it through git objects,
which means ANY clone holding that commit can answer -- the file comes out of `git show`, pinned,
regardless of which tree's HEAD happens to be sitting where. A path-matched working-tree read is
only used when the tree is at that commit AND the file is clean, and the dirty case is reported
rather than silently served.

Exit status: 0 if everything asked for resolved, 1 otherwise. `--audit` is meant for CI.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(HERE)
ROOT = os.path.dirname(PK)

# Where to look for checkouts. Depth-limited: a clone's own `aiter/aiter/` package directory is not
# a second checkout, and a venv's site-packages copy is not a git tree at all.
SEARCH_ROOTS = [r for r in (os.environ.get("GEAK_CITATION_ROOTS") or "/sgl-workspace").split(":") if r]
SEARCH_DEPTH = 5

# `ROCm/aiter@a6bb4993:aiter/tuned_gemm.py:120-140`, with the repo prefix and the lines optional.
CITATION = re.compile(
    r"(?:(?P<repo>[\w.-]+/[\w.-]+)@)?(?P<commit>[0-9a-f]{7,40}):(?P<path>[\w./+-]+?)"
    r"(?::(?P<lo>\d+)(?:-(?P<hi>\d+))?)?$"
)


def _git(tree, *args):
    r = subprocess.run(["git", "-C", tree, *args], capture_output=True, text=True, check=False)
    return r.returncode, r.stdout, r.stderr


def checkouts(name="aiter"):
    """Every git checkout named `name` under the search roots, with HEAD and remote.

    Returned in a stable order so two runs on an unchanged box agree.
    """
    found = []
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        r = subprocess.run(
            ["find", root, "-maxdepth", str(SEARCH_DEPTH), "-type", "d", "-name", name,
             "-not", "-path", f"*/{name}/{name}*"],
            capture_output=True, text=True, check=False)
        for d in r.stdout.split():
            rc, head, _ = _git(d, "rev-parse", "HEAD")
            if rc != 0:
                continue  # not a checkout -- a vendored package directory, or a plain copy
            # `git -C` answers for the ENCLOSING repo, so a plain directory that happens to be
            # named `aiter` inside some other repo reports that repo's HEAD. The knowledge base
            # contains exactly such a directory (`perf_knowledge/backends/aiter/`), and listing it
            # as a checkout at the GEAK commit is the same confusion this file exists to prevent.
            _, top, _ = _git(d, "rev-parse", "--show-toplevel")
            if os.path.realpath(top.strip() or "/") != os.path.realpath(d):
                continue
            _, origin, _ = _git(d, "remote", "get-url", "origin")
            _, dirty, _ = _git(d, "status", "--porcelain")
            found.append({
                "path": os.path.abspath(d),
                "head": head.strip(),
                "origin": origin.strip() or None,
                "dirty_files": len([l for l in dirty.splitlines() if len(l) > 3]),
            })
    return sorted(found, key=lambda c: c["path"])


def _has_commit(tree, commit):
    rc, _, _ = _git(tree, "cat-file", "-e", commit + "^{commit}")
    return rc == 0


def resolve(commit, path, trees=None):
    """Find a tree that can serve `path` AT `commit`, and say how.

    Resolution order, best first:
      1. a tree holding the commit as an object -> `git show commit:path`, exact and pinned;
      2. nothing. A tree merely sitting at the right HEAD adds nothing over (1), and a tree at the
         WRONG head is the trap this module exists to close, so it is never used as a fallback.
    """
    trees = trees if trees is not None else checkouts()
    holders = [t for t in trees if _has_commit(t["path"], commit)]
    for t in holders:
        rc, blob, err = _git(t["path"], "show", f"{commit}:{path}")
        if rc == 0:
            at_head = t["head"].startswith(commit)
            return {
                "ok": True,
                "tree": t["path"],
                "via": "git show",
                "tree_at_cited_commit": at_head,
                # Only meaningful when the tree is at that commit: it says whether reading the
                # working file by line number would have agreed with the pinned blob.
                "working_tree_trustworthy": at_head and t["dirty_files"] == 0,
                "blob": blob,
            }
    if holders:
        return {"ok": False, "reason": "path-not-at-commit",
                "detail": f"{len(holders)} checkout(s) hold {commit[:8]} but none contains {path} there",
                "holders": [t["path"] for t in holders]}
    return {"ok": False, "reason": "commit-not-on-box",
            "detail": f"no checkout under {':'.join(SEARCH_ROOTS)} holds commit {commit[:8]}; "
                      f"`git -C <a-checkout> fetch origin {commit}` first",
            "holders": []}


def kb_citations():
    """`(repo, commit, path, source_file)` for every `<repo>@<sha>:<path>` in the knowledge markdown.

    Prose citations only -- the YAML frontmatter `sources:` lists use the same syntax and are caught
    by the same regex, which is intended: a frontmatter source that does not resolve is the same
    defect as a body one.

    `repo` is carried because the knowledge base cites more than aiter (ROCm/mori, vLLM), and an
    audit that resolved every citation against aiter checkouts would report a perfectly good mori
    citation as missing -- manufacturing exactly the kind of false "this is broken" verdict the rest
    of this file exists to prevent.
    """
    pat = re.compile(r"(?P<repo>[\w.-]+/[\w.-]+)@(?P<commit>[0-9a-f]{7,40}):(?P<path>[\w./+-]+\.\w+)")
    out = []
    for base in (os.path.join(PK, "operators"), os.path.join(PK, "backends"),
                 os.path.join(PK, "corpus"), os.path.join(PK, "index"),
                 os.path.join(PK, "landscape")):
        for dirpath, _, files in os.walk(base):
            for fn in files:
                # `.yaml` too: sota_registry / capability_index are the machine-consumed copies of
                # the card frontmatter, and they are GENERATED -- so a card fixed without re-running
                # `index/_gen_registry.py` leaves the broken citation live in exactly the file the
                # workflows read, while the audit over the cards reports clean. (Observed.)
                if not (fn.endswith(".md") or fn.endswith(".yaml")):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    with open(p, encoding="utf-8") as fh:
                        text = fh.read()
                except OSError:
                    continue
                for m in pat.finditer(text):
                    out.append((m.group("repo"), m.group("commit"), m.group("path"),
                                os.path.relpath(p, ROOT)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("citation", nargs="*", help="repo@commit:path[:line[-line]]")
    ap.add_argument("--list", action="store_true", help="every on-box checkout, with HEAD")
    ap.add_argument("--audit", action="store_true", help="resolve every aiter@ citation in the KB")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--name", default=None,
                    help="checkout directory name to search for (default: the citation's own repo, "
                         "else aiter)")
    ap.add_argument("--context", type=int, default=0, help="lines of context around a cited line")
    a = ap.parse_args()

    trees = checkouts(a.name or "aiter")

    if a.list:
        if a.json:
            print(json.dumps(trees, indent=2))
        else:
            print(f"{len(trees)} `{a.name}` checkout(s) under {':'.join(SEARCH_ROOTS)}:")
            for t in trees:
                flag = f"  [{t['dirty_files']} dirty]" if t["dirty_files"] else ""
                print(f"  {t['head'][:8]}  {t['path']}{flag}")
        return 0

    if a.audit:
        cites = kb_citations()
        by_key = {}
        for repo, commit, path, src in cites:
            by_key.setdefault((repo, commit, path), []).append(src)
        # One checkout scan per cited repository, keyed on the repo's own name.
        per_repo = {}
        for repo, _, _, _ in cites:
            name = repo.split("/")[-1]
            if name not in per_repo:
                per_repo[name] = checkouts(name)
        bad = []
        for (repo, commit, path), srcs in sorted(by_key.items()):
            r = resolve(commit, path, per_repo[repo.split("/")[-1]])
            if not r["ok"]:
                bad.append({"repo": repo, "commit": commit, "path": path, "reason": r["reason"],
                            "detail": r["detail"], "cited_by": sorted(set(srcs))})
        summary = {"citations": len(cites), "distinct": len(by_key), "unresolved": len(bad),
                   "checkouts": {k: len(v) for k, v in sorted(per_repo.items())}, "failures": bad}
        if a.json:
            print(json.dumps(summary, indent=2))
        else:
            counts = ", ".join(f"{k}:{len(v)}" for k, v in sorted(per_repo.items()))
            print(f"{len(cites)} citations / {len(by_key)} distinct (repo, commit, path) "
                  f"/ checkouts {{{counts}}} -> {len(bad)} unresolved")
            for b in bad:
                print(f"  MISS {b['repo']}@{b['commit'][:8]}:{b['path']}  [{b['reason']}]")
                for s in b["cited_by"][:3]:
                    print(f"       cited by {s}")
        return 1 if bad else 0

    if not a.citation:
        ap.print_help()
        return 1

    rc = 0
    for raw in a.citation:
        m = CITATION.search(raw.strip().strip("`"))
        if not m:
            print(f"UNPARSED  {raw}", file=sys.stderr)
            rc = 1
            continue
        commit, path = m.group("commit"), m.group("path")
        # Search the checkouts named after the citation's OWN repo, the way `--audit` already does.
        # Hardcoding `aiter` here made every `ROCm/mori@...` citation report
        # "no checkout under /sgl-workspace holds commit 96ffa169; git fetch origin ... first" while
        # `/sgl-workspace/mori` sat at exactly that commit -- a false MISS whose advice was to fetch a
        # commit already present. The role guardrails tell every role to resolve citations with this
        # command, so the single-citation path has to key on the repo as well.
        repo = m.group("repo")
        scope = trees if a.name or not repo else checkouts(repo.split("/")[-1])
        r = resolve(commit, path, scope)
        if not r["ok"]:
            print(f"MISS  {raw}\n      {r['detail']}", file=sys.stderr)
            rc = 1
            continue
        print(f"OK    {raw}\n      {r['tree']}  (tree at cited commit: {r['tree_at_cited_commit']})")
        lo, hi = m.group("lo"), m.group("hi")
        if lo:
            lines = r["blob"].splitlines()
            lo_i = max(1, int(lo) - a.context)
            hi_i = min(len(lines), int(hi or lo) + a.context)
            for i in range(lo_i, hi_i + 1):
                print(f"      {i:6d}| {lines[i - 1]}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
