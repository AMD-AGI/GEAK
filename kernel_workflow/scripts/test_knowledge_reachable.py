#!/usr/bin/env python3
"""test_knowledge_reachable.py — the knowledge loop is closed in BOTH directions.

The return path (run -> card -> digest) has tests and CI gates. The read path had none, and it broke
silently: `run_recurrence.md`, the GEMM corpus and `languages/flydsl/version_map.md` were all
generated, tested, gated and linked from three READMEs — and named by no role prompt at all. Every
`--check` passed. A human browsing the base would find them; the agent that plans directions never
would. A doc no role reads is a comment with a CI gate on it.

Two directions, and they fail for different reasons:

1. **Every `KERNEL_KNOWLEDGE_DIR/<path>` a role prompt names must resolve.** A role prompt is not
   code, so a renamed file leaves the prompt pointing at nothing and the agent silently reads less
   than it was told to — no error, just a worse plan. Paths with `<placeholder>` segments are
   resolved as far as their fixed prefix and counted, since `operators/<op>/overview.md` cannot be
   checked without knowing the op.

2. **Every artifact in `MUST_BE_READ` must be named by at least one role prompt.** Deliberately an
   explicit list rather than "everything marked GENERATED": most generated files here are machine
   inputs (`evidence/*.yaml`), human navigation (`sota_matrix.md`) or skill internals, so a blanket rule
   would need an exemption list longer than the rule itself. Being explicit also puts the burden in
   the right place — adding a knowledge artifact should require saying who reads it, which is the
   question that went unasked.

    python3 kernel_workflow/scripts/test_knowledge_reachable.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.dirname(HERE)
ROOT = os.path.dirname(WF)
PK = os.path.join(ROOT, "perf_knowledge")

# The corpus (`corpus/`) lives in the knowledge repository, not here, and the role prompts name it
# as `KERNEL_KNOWLEDGE_DIR/corpus/...` for exactly that reason. So an artifact can be reachable
# while being absent from this checkout, and resolving every reference against the repo-local tree
# would report the loop broken on a box that has simply not checked the corpus out.
#
# Resolution order matches what the lane does: the configured knowledge dir first, this repo's own
# perf_knowledge/ second. A reference that resolves in NEITHER is still a failure -- that is the
# dangling-reference case this file exists to catch, and the split must not be allowed to turn it
# into a blanket excuse.
CONFIGURED = os.environ.get("KERNEL_KNOWLEDGE_DIR", "")

# Resolution order matches what the lane does: the configured knowledge dir first, this repo's own
# perf_knowledge/ second. `corpus/` straddles the two -- the data, the catalog and the generated
# pages ship in the knowledge repository, the consumer modules a role invokes (`_select_candidates`,
# `benchmarks/merge_results`, ...) stay here -- so the split is per-file and cannot be decided from
# the prefix. Hence: resolve against both, and only then ask who should have had it.
KNOWLEDGE_DIRS = [d for d in (CONFIGURED, PK) if d]

# A path that resolves in NEITHER is normally a dangling reference, which is the failure this file
# exists to catch. The one exception is a `corpus/` path on a box with no KERNEL_KNOWLEDGE_DIR set:
# there is nothing to resolve it against, and failing would make the gate permanently red for a
# reason no edit to this repo can fix -- which is how a gate gets deleted. So it SKIPS, counted and
# named, on the same terms as the corpus tests: a skip says the seam went unverified on this run,
# not that it passed. Set the dir -- as CI does -- and a missing path is a hard failure again.
REMOTE_PREFIXES = ("corpus/",)


def pk_exists(relative):
    return any(os.path.exists(os.path.join(d, relative)) for d in KNOWLEDGE_DIRS)


def unverifiable(relative):
    return not CONFIGURED and relative.startswith(REMOTE_PREFIXES) and not pk_exists(relative)

# BOTH role trees. The e2e side is not decoration here: `_gen_run_recurrence.py` aggregates the e2e
# learned cards along with the kernel ones, so an e2e run that never reads the digest is contributing
# to a prior it cannot spend. A first version of this file scanned only kernel_workflow/roles and
# reported the loop closed while that asymmetry was live — the guard had a blind spot in exactly the
# place the bug was.
ROLE_DIRS = [os.path.join(ROOT, "kernel_workflow", "roles"),
             os.path.join(ROOT, "e2e_workflow", "roles")]

# path in perf_knowledge -> why a role has to name it. The reason is not decoration: it is what makes
# a future removal a decision rather than an accident.
MUST_BE_READ = {
    "index/run_recurrence.md":
        "per-axis base rates from this workflow's own cards. If no planning role reads it, the "
        "return path computes priors that nothing ever prices a round against, and the whole "
        "feedback loop terminates in a file.",
    "corpus/gemm_decisions.md":
        "conditioned, actionable GEMM candidate cards grounded in source observations or shipped "
        "configuration seeds. The source index alone leaves the author to infer recommendations from "
        "regex-hit counts; this is the semantic layer that turns evidence into something usable "
        "without bypassing the separate switches for measured learned cards and expert skills.",
    "corpus/catalog.yaml":
        "operator/pattern discovery for conditioned corpus cards. Without it every new family needs "
        "another GEMM-specific prompt and validation-code edit.",
    "languages/flydsl/version_map.md":
        "which FlyDSL symbol moved where across versions. Unread, an agent porting a recipe guesses "
        "at an API instead of looking it up, which is the specific failure it was built to remove.",
}

# Segments a prompt legitimately cannot fill in: the op and language are per-run, and `optimization/*`
# names a directory's worth of files on purpose.
PLACEHOLDER = re.compile(r"<[^>]+>|\{[^}]*\}|\*")


def role_files():
    out = []
    for roles in ROLE_DIRS:
        if not os.path.isdir(roles):
            continue
        for base, _, names in os.walk(roles):
            out += [os.path.join(base, n) for n in sorted(names) if n.endswith(".md")]
    return sorted(out)


def expand_braces(path):
    """`index/{decision_trees,recipes}.md` -> both. Prompts use this to name a pair compactly."""
    m = re.search(r"\{([^}]*)\}", path)
    if not m:
        return [path]
    out = []
    for alt in m.group(1).split(","):
        out += expand_braces(path[:m.start()] + alt.strip() + path[m.end():])
    return out


SEG = r"[A-Za-z0-9_/{},.<>*-]+"

# The spellings the two role trees actually use for the same base. kernel_workflow interpolates the
# variable it is handed; e2e prompts write the path out, sometimes repo-relative. Matching only the
# first form is why the e2e references were invisible to an earlier version of this check.
REF_PATTERNS = (
    re.compile(rf"KERNEL_KNOWLEDGE_DIR/({SEG})"),
    re.compile(rf"(?:GEAK/)?perf_knowledge/({SEG})"),
)


def base_relative_pattern():
    """Bare `index/foo.md` inside a role file, anchored on the real top-level dirs of the base.

    Needed because a nested bullet list states the base once in the parent and leaves the children
    relative — `system_architect.md` names `index/capability_index.yaml` and `index/run_recurrence.md`
    that way, and the prefixed patterns above see neither. Anchoring on `os.listdir(PK)` rather than a
    hardcoded list keeps it from drifting as the base grows, and keeps it tight enough that an English
    sentence containing the word "index" is not mistaken for a path.

    Plus `REMOTE_PREFIXES`: `corpus/` is a top-level dir of the knowledge repository and of nothing
    here, so anchoring on the local tree alone would stop matching bare `corpus/gemm_decisions.md` —
    which quietly empties the set direction 2 checks against and turns a real "no role names it"
    into a pass.

    Deliberately NOT `os.listdir()` over the configured knowledge dir. That repository's other top
    level dirs — `knowledge/`, `kernels/`, `tests/` — are also directory names in THIS tree, under
    `kernel_workflow/` and `e2e_workflow/`, and a role naming `knowledge/learned/` means its own
    workflow's copy. Taking the union made those prose strings parse as knowledge-base paths and
    then demanded they resolve under a base they were never relative to: nine such references, all
    fine, reported as dangling the moment KERNEL_KNOWLEDGE_DIR was set — which is to say, only in
    CI. The set of prefixes that genuinely straddle the split is the one already written down.
    """
    tops = sorted({d for d in os.listdir(PK) if os.path.isdir(os.path.join(PK, d))}
                  | {p.rstrip("/") for p in REMOTE_PREFIXES})
    # Must END like a path — an extension, a trailing slash, a glob, or a placeholder. Without that,
    # prose collides with the vocabulary: "a stale index/mask from a prior call" and "the
    # reference/synth that raised" both parse as paths under `index/` and `reference/`, and both were
    # reported as broken references on the first run of this pattern. `/` here means "or".
    return re.compile(rf"(?<![\w/])((?:{'|'.join(map(re.escape, tops))})/{SEG}"
                      r"(?:\.[A-Za-z0-9]{1,5}|/|\*|>))")


def referenced_paths():
    """Every path a role prompt names under the knowledge base, with the role that names it.

    "Names" is as far as a path check can go: it cannot tell an instruction to read a file from a
    mention of one, so `update_experience.md` counts as naming `run_recurrence.md` even though what it
    says is "do not regenerate this by hand". That is a known limit, not a hole — the failure this
    guards is an artifact NO prompt mentions at all, and telling instruction from mention would need a
    reader of English rather than of paths.
    """
    patterns = (*REF_PATTERNS, base_relative_pattern())
    hits = {}
    for f in role_files():
        with open(f, encoding="utf-8") as fh:
            body = fh.read()
        for pat in patterns:
            for raw in pat.findall(body):
                for p in expand_braces(raw.rstrip(".,;:)`")):
                    hits.setdefault(p, set()).add(os.path.basename(f))
    return hits


def main():
    if not any(os.path.isdir(d) for d in KNOWLEDGE_DIRS):
        print("  skip  no perf_knowledge/ in this checkout")
        return 0

    refs = referenced_paths()
    fails, skipped, unchecked = [], [], []

    # Direction 1: what the prompts name must exist.
    for p, roles in sorted(refs.items()):
        if unverifiable(p):
            unchecked.append(p)
            continue
        if PLACEHOLDER.search(p):
            # Check the fixed prefix only — `operators/<op>/overview.md` at least proves
            # `operators/` is there, which is the part a rename would break.
            prefix = p.split("<")[0].split("{")[0].rstrip("/")
            if prefix and not pk_exists(prefix):
                fails.append(f"{p}: fixed prefix `{prefix}` does not exist "
                             f"(named by {', '.join(sorted(roles))})")
            else:
                skipped.append(p)
            continue
        if not pk_exists(p):
            fails.append(f"{p}: named by {', '.join(sorted(roles))} but not present in "
                         f"perf_knowledge/ — the role reads nothing and says nothing")

    # Direction 2: what exists and matters must be named.
    named = set(refs)
    for p, why in sorted(MUST_BE_READ.items()):
        if unverifiable(p):
            unchecked.append(p)
            continue
        if not pk_exists(p):
            skipped.append(f"{p} (not generated in this checkout)")
            continue
        if p not in named:
            fails.append(f"{p}: exists but NO role prompt names it. {why}")

    for f in fails:
        print(f"  FAIL  {f}")
    if fails:
        print(f"\n{len(fails)} unreachable or dangling reference(s). A knowledge artifact is only "
              f"in the loop when a role is told to read it.")
        return 1

    reachable = sum(1 for p in MUST_BE_READ if pk_exists(p))
    print(f"  ok    all {len(refs) - len(set(unchecked) & set(refs))} knowledge paths named by "
          f"roles resolve ({len(skipped)} per-run placeholder(s) checked to their fixed prefix)")
    if unchecked:
        print(f"  skip  {len(set(unchecked))} reference(s) live in the knowledge repository and "
              f"went UNVERIFIED: no KERNEL_KNOWLEDGE_DIR is set")
        for p in sorted(set(unchecked)):
            print(f"          {p}")
    print(f"  ok    all {reachable} loop-critical artifact(s) present here are named by at least "
          f"one role")
    for p in sorted(MUST_BE_READ):
        if p in refs and pk_exists(p):
            print(f"          {p} <- {', '.join(sorted(refs[p]))}")
    print("\nall green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
