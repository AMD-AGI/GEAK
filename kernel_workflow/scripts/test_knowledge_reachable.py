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
    """
    tops = sorted(d for d in os.listdir(PK) if os.path.isdir(os.path.join(PK, d)))
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
    if not os.path.isdir(PK):
        print("  skip  no perf_knowledge/ in this checkout")
        return 0

    refs = referenced_paths()
    fails, skipped = [], []

    # Direction 1: what the prompts name must exist.
    for p, roles in sorted(refs.items()):
        if PLACEHOLDER.search(p):
            # Check the fixed prefix only — `operators/<op>/overview.md` at least proves
            # `operators/` is there, which is the part a rename would break.
            prefix = p.split("<")[0].split("{")[0].rstrip("/")
            if prefix and not os.path.exists(os.path.join(PK, prefix)):
                fails.append(f"{p}: fixed prefix `{prefix}` does not exist "
                             f"(named by {', '.join(sorted(roles))})")
            else:
                skipped.append(p)
            continue
        if not os.path.exists(os.path.join(PK, p)):
            fails.append(f"{p}: named by {', '.join(sorted(roles))} but not present in "
                         f"perf_knowledge/ — the role reads nothing and says nothing")

    # Direction 2: what exists and matters must be named.
    named = set(refs)
    for p, why in sorted(MUST_BE_READ.items()):
        if not os.path.exists(os.path.join(PK, p)):
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

    reachable = sum(1 for p in MUST_BE_READ if os.path.exists(os.path.join(PK, p)))
    print(f"  ok    all {len(refs)} knowledge paths named by roles resolve "
          f"({len(skipped)} per-run placeholder(s) checked to their fixed prefix)")
    print(f"  ok    all {reachable} loop-critical artifact(s) are named by at least one role")
    for p in sorted(MUST_BE_READ):
        if p in refs:
            print(f"          {p} <- {', '.join(sorted(refs[p]))}")
    print("\nall green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
