#!/usr/bin/env python3
"""Rule-derived retrieval labels for the KB (plan P1) — `cost` / `levers` / `risk`, no LLM.

The structured index only earns its keep once `cost`, `levers` and `bound_type` are actually set;
`_backfill_kb.py` deliberately left those four fields to a human. This script fills the mechanically
derivable ones from the KB's own regular skeleton, and leaves the judgement calls to P2 (hand
labelling of the cross-cutting dirs) and P3 (section-scoped LLM pass).

Why rules work here: `operators/` is a Cartesian grid — 54 operators ×
{overview, tuning, numerics, fusion} + `backends/<backend>.md`, where the backend filename IS the
taxonomy id. A ~30-line table therefore covers ~380 files with zero guessing.

  path selector                                   -> cost   levers                    risk
  operators/*/backends/<library backend>.md          L2     [backend.swap]
  operators/*/backends/<DSL / hand-written>.md       L3     -
  operators/*/tuning.md                              L1     [config.per-shape-tune]
  operators/*/numerics.md                            -      -                         numerics-affecting
  operators/*/{overview,fusion}.md                   -      -
  languages/**                                       L3     -

`overview.md` / `numerics.md` get NO cost on purpose: kb_resolve sorts an unset cost AFTER every
explicit one, so navigation docs stop out-ranking actionable cards without being pruned.

Precision/recall is chosen per field by how kb_resolve PUNISHES a mistake:
  * `cost`    — a wrong value mis-sorts AND gets pruned under `--max-cost`; a missing one only sorts
                last and is never pruned.  => PRECISION first. Rules only, no guessing.
  * `levers`  — display-only today.         => PRECISION first, ≤3.
  * `bound_type` — a missing value is UNREACHABLE by `--bound` (today's failure mode). => RECALL
                first — but it is NOT rule-derivable, so this script never emits it. P2/P3 own it.

Deliberately NOT done: naive keyword matching for `levers`. Measured over all 616 content files it
tags `backend.swap` on 472, `dtype.downcast` on 463, `env.flag` on 391 — 4.49 tags/file of pure noise.

`kb_labels.yaml` is the auditable proposal + provenance file (checked in, diffable). Every row carries
`src: rule|llm|human` and an `evidence` string. `--rules` only ever recomputes `src: rule` rows, so it
is idempotent and can never clobber a human decision. Precedence on apply:

    file frontmatter  >  src: human  >  src: llm  >  src: rule

The frontmatter is always the single source of truth for `_gen_index.py`; `kb_labels.yaml` is a
proposal ledger, never a second index.

Usage:
  python3 index/_label_kb.py --rules              # (re)derive rule rows -> kb_labels.yaml
  python3 index/_label_kb.py --apply              # dry-run: what WOULD land in frontmatter
  python3 index/_label_kb.py --apply --write      # apply (ADD-only, never overwrites)
  python3 index/_label_kb.py --emit-batch         # files still missing levers/bound_type, for P3
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _kb_vocab as V  # noqa: E402
from _backfill_kb import split_fm, fm_keys, yl  # noqa: E402  (ADD-only semantics, shared)

try:
    import yaml
except Exception:
    yaml = None

PK_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(PK_ROOT)
LABELS = os.path.join(HERE, "kb_labels.yaml")

FIELDS = ("cost", "levers", "risk", "bound_type")
LIST_FIELDS = ("levers", "bound_type")
SRC_RANK = {"human": 0, "llm": 1, "rule": 2}

# Backend filenames are the canonical taxonomy ids (perf_knowledge/README.md "SOTA registry"), so the
# two groups below are exhaustive over `ls operators/*/backends/`. Keep them in sync if a backend is
# added — _selftest() fails loudly when a backend card matches neither group.
LIB_BACKENDS = {"aiter", "vllm_kernels", "ck", "fa_rocm", "hipblaslt", "rccl",
                "sglang_kernels", "miopen", "mori", "pytorch_inductor"}
DSL_BACKENDS = {"triton", "hip", "asm", "flydsl", "hipkittens", "tilelang", "gluon", "rocwmma"}


def _rule_for(rel):
    """Return (rule_id, labels, evidence) for a repo-relative path, or None when no rule applies."""
    parts = rel.split("/")
    name = os.path.splitext(parts[-1])[0]

    if len(parts) >= 5 and parts[1] == "operators" and parts[3] == "backends":
        if name in LIB_BACKENDS:
            # Swapping in an existing library is an integration seam, not a rewrite.
            return ("backend.library", {"cost": "L2", "levers": ["backend.swap"]},
                    "operators/*/backends/<library backend>.md")
        if name in DSL_BACKENDS:
            # Authoring in a DSL / by hand is a rewrite. No lever: WHICH lever the card teaches is a
            # per-card judgement (P3), not something the filename can tell us.
            return ("backend.dsl", {"cost": "L3"},
                    "operators/*/backends/<DSL or hand-written backend>.md")
        return None

    if len(parts) >= 4 and parts[1] == "operators":
        if name == "tuning":
            return ("operator.tuning", {"cost": "L1", "levers": ["config.per-shape-tune"]},
                    "operators/*/tuning.md — per-shape config search, no source change")
        if name == "numerics":
            # No cost: a numerics doc describes a property, it is not a lever you can pull.
            return ("operator.numerics", {"risk": "numerics-affecting"},
                    "operators/*/numerics.md")
        if name in ("overview", "fusion"):
            # Navigation / concept docs: intentionally cost-less so they sort behind actionable cards.
            return ("operator.nav", {}, f"operators/*/{name}.md — navigational, cost left unset")
        return None

    if len(parts) >= 2 and parts[1] == "languages":
        return ("language.guide", {"cost": "L3"},
                "languages/** — writing the kernel in this language is a rewrite")

    return None


def content_files():
    """The same content set _backfill_kb.py walks: no expert_skills/, no index/, no _templates/."""
    EXCLUDE = (os.sep + "expert_skills" + os.sep, os.sep + "index" + os.sep,
               os.sep + "_templates" + os.sep)
    out = []
    for dp, _dn, fn in os.walk(PK_ROOT):
        if any(x in dp + os.sep for x in EXCLUDE):
            continue
        for f in fn:
            if f.endswith(".md"):
                out.append(os.path.join(dp, f))
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# kb_labels.yaml I/O — hand-rolled writer so the file is byte-stable under CI's
# `git diff --exit-code` regardless of the installed PyYAML version.
# --------------------------------------------------------------------------- #
def load_labels():
    if not os.path.isfile(LABELS) or yaml is None:
        return []
    try:
        d = yaml.safe_load(open(LABELS, encoding="utf-8").read()) or {}
    except Exception:
        return []
    rows = d.get("labels", []) if isinstance(d, dict) else []
    return [r for r in rows if isinstance(r, dict) and r.get("path")]


def _q(s):
    s = str(s)
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def write_labels(rows):
    rows = sorted(rows, key=lambda r: (r["path"], SRC_RANK.get(r.get("src"), 9)))
    out = ["# kb_labels.yaml — retrieval-label PROPOSALS + provenance (plan P1).",
           "#",
           "# NOT an index. File frontmatter is the only source of truth _gen_index.py reads; this file",
           "# records where a label came from so a rule pass can be re-run without erasing human work.",
           "#   src: rule   — derived by index/_label_kb.py --rules; recomputed on every run.",
           "#   src: llm    — proposed by the P3 section-scoped pass; must carry a quoted evidence span.",
           "#   src: human  — hand-labelled (P2). NEVER touched by --rules.",
           "# Precedence when applying: frontmatter > human > llm > rule.",
           "",
           f"# rows: {len(rows)}",
           "labels:"]
    for r in rows:
        out.append(f"  - path: {_q(r['path'])}")
        out.append(f"    src: {r.get('src', 'rule')}")
        if r.get("rule"):
            out.append(f"    rule: {r['rule']}")
        for k in FIELDS:
            if k not in r:
                continue
            v = r[k]
            if k in LIST_FIELDS:
                out.append(f"    {k}: [{', '.join(str(x) for x in (v or []))}]")
            else:
                out.append(f"    {k}: {v}")
        if r.get("evidence"):
            out.append(f"    evidence: {_q(r['evidence'])}")
    open(LABELS, "w", encoding="utf-8").write("\n".join(out) + "\n")


def validate_row(r):
    """Reject a row that would put a value outside the controlled vocabulary into frontmatter."""
    errs = []
    if r.get("cost") and r["cost"] not in V.COSTS:
        errs.append(f"cost '{r['cost']}'")
    if r.get("risk") and r["risk"] not in V.RISKS:
        errs.append(f"risk '{r['risk']}'")
    for x in r.get("levers") or []:
        if x not in V.LEVERS:
            errs.append(f"lever '{x}'")
    for x in r.get("bound_type") or []:
        if x not in V.BOUND_TYPES:
            errs.append(f"bound_type '{x}'")
    if (r.get("src") == "llm") and not r.get("evidence"):
        errs.append("src:llm without evidence")
    return errs


# --------------------------------------------------------------------------- #
# Modes.
# --------------------------------------------------------------------------- #
def cmd_rules():
    kept = [r for r in load_labels() if r.get("src") in ("human", "llm")]
    rows, by_rule, unmatched = list(kept), {}, []
    for path in content_files():
        rel = os.path.relpath(path, REPO)
        got = _rule_for(rel)
        if not got:
            unmatched.append(rel)
            continue
        rule, labels, evidence = got
        by_rule[rule] = by_rule.get(rule, 0) + 1
        if not labels:                       # matched a deliberate no-op rule (operator.nav)
            continue
        rows.append({"path": rel, "src": "rule", "rule": rule, "evidence": evidence, **labels})
    write_labels(rows)
    print(f"kb_labels.yaml: {len(rows)} rows ({len(kept)} human/llm preserved).")
    for k in sorted(by_rule):
        print(f"  {k:22s} {by_rule[k]:4d} files matched")
    print(f"  (no rule)              {len(unmatched):4d} files — P2/P3 territory")
    return 0


def cmd_apply(write):
    rows = load_labels()
    best = {}
    for r in rows:
        errs = validate_row(r)
        if errs:
            print(f"  [reject] {r['path']}: {', '.join(errs)}")
            continue
        p = r["path"]
        if p not in best or SRC_RANK.get(r.get("src"), 9) < SRC_RANK.get(best[p].get("src"), 9):
            best[p] = r

    changed = missing = untouched = skipped = 0
    for rel, r in sorted(best.items()):
        path = os.path.join(REPO, rel)
        if not os.path.isfile(path):
            missing += 1
            print(f"  [gone] {rel}")
            continue
        text = open(path, encoding="utf-8").read()
        fm_body, rest = split_fm(text)
        if fm_body is None:
            skipped += 1
            continue
        present = fm_keys(fm_body)
        # ADD-only: a value already in the frontmatter always wins over any proposal.
        adds = []
        for k in FIELDS:
            if k in present or k not in r:
                continue
            v = r[k]
            if k in LIST_FIELDS:
                if v:
                    adds.append((k, yl([str(x) for x in v])))
            elif v:
                adds.append((k, str(v)))
        if not adds:
            untouched += 1
            continue
        if write:
            if not os.access(path, os.W_OK):
                skipped += 1
                print(f"  [skip:ro] {rel}")
                continue
            new_fm = fm_body.rstrip("\n") + "\n" + "\n".join(f"{k}: {v}" for k, v in adds)
            open(path, "w", encoding="utf-8").write("---\n" + new_fm + "\n---\n" + rest)
        changed += 1
        print(f"  [+] {rel}: " + "  ".join(f"{k}={v}" for k, v in adds))

    print(f"\n{'APPLIED' if write else 'DRY-RUN'}: {changed} files changed, {untouched} already "
          f"complete, {missing} label rows point at a missing file, {skipped} skipped.")
    return 0


SECTIONS = ("## TL;DR", "## Config space", "## The levers", "## Pitfalls & anti-patterns")


def cmd_emit_batch():
    """List the files P3 still has to look at, with only the decision-bearing sections.

    P3 must NOT read whole bodies: the judgement is 'does this section give an EXECUTABLE
    instruction', and feeding it prose it cannot cite is how a labeller starts inventing levers.
    """
    n = 0
    for path in content_files():
        rel = os.path.relpath(path, REPO)
        text = open(path, encoding="utf-8").read()
        fm_body, _ = split_fm(text)
        if fm_body is None:
            continue
        present = fm_keys(fm_body)
        if "levers" in present and "bound_type" in present:
            continue
        want = [k for k in ("levers", "bound_type") if k not in present]
        heads = [h for h in SECTIONS
                 if re.search(r"^" + re.escape(h), text, re.M)]
        n += 1
        print(f"{rel}\tneeds={','.join(want)}\tsections={'|'.join(heads) or '-'}")
    print(f"# {n} files still missing levers and/or bound_type", file=sys.stderr)
    return 0


def _selftest():
    """Cheap invariant: every backend card must fall in exactly one of the two backend groups."""
    unknown = set()
    for path in content_files():
        parts = os.path.relpath(path, REPO).split("/")
        if len(parts) >= 5 and parts[1] == "operators" and parts[3] == "backends":
            nm = os.path.splitext(parts[-1])[0]
            if nm not in LIB_BACKENDS and nm not in DSL_BACKENDS:
                unknown.add(nm)
    if unknown:
        print(f"_label_kb: backend(s) in no cost group: {sorted(unknown)} — add them to "
              f"LIB_BACKENDS or DSL_BACKENDS.", file=sys.stderr)
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", action="store_true", help="(re)derive src:rule rows into kb_labels.yaml")
    ap.add_argument("--apply", action="store_true", help="merge kb_labels.yaml into file frontmatter")
    ap.add_argument("--emit-batch", dest="emit_batch", action="store_true",
                    help="list files still missing levers/bound_type (P3 worklist)")
    ap.add_argument("--write", action="store_true", help="with --apply: actually write (default dry-run)")
    a = ap.parse_args()
    if _selftest():
        return 1
    if a.rules:
        return cmd_rules()
    if a.apply:
        return cmd_apply(a.write)
    if a.emit_batch:
        return cmd_emit_batch()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
