#!/usr/bin/env python3
"""Demote evidence-free thin cards to `lifecycle: candidate` (plan P7 — no prose is rewritten).

A card claiming `lifecycle: active` asserts that this is settled, reproduced knowledge. Roughly a
tenth of `operators/` does not clear that bar: a short stub with no measured number anywhere in the
body. Rewriting those into real cards is weeks of on-box work; relabelling them is honest today and
costs zero prose. `kb_resolve.py` then does the rest by itself — `LIFE_RANK` sorts `candidate` below
`active` at equal cost and the hit is printed with a `·candidate` marker, so a thin card stops
outranking a substantive one without ever being hidden (ADD-only: nothing is pruned).

The criterion is deliberately mechanical and conservative, so this is re-runnable and auditable
rather than a one-off judgement call:

  * under `operators/` (the only tree with a regular enough skeleton for a structural rule);
  * `lifecycle: active` today;
  * fewer than MIN_PROSE non-blank, non-heading body lines;
  * AND no measured quantity anywhere in the body (see NUM).

Both halves are required. A short card that cites a number is evidence, and a long card without one
is a different debt — that one is surfaced as a `_validate_kb.py` WARN, not demoted here.

  python3 index/_demote_thin.py            # dry-run: list what would change
  python3 index/_demote_thin.py --write    # rewrite the `lifecycle:` line in place
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _backfill_kb import split_fm  # noqa: E402

PK_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(PK_ROOT)
OPERATORS = os.path.join(PK_ROOT, "operators")

MIN_PROSE = 22
SKIP_NAMES = {"INDEX.md", "README.md", "_archive.md"}

# A measured quantity: a number carrying a performance unit. Kept in sync with _validate_kb.py's
# copy (same constant, same name) — both answer the question "did anyone actually measure this?".
NUM = re.compile(
    r"\d+(?:\.\d+)?\s*(?:[x×]\b|%|TFLOP|GFLOP|GB/s|TB/s|\bms\b|\bus\b|µs|tok/s|tokens/s)",
    re.I)


def prose_lines(body):
    return [ln for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def is_thin(body):
    return len(prose_lines(body)) < MIN_PROSE and not NUM.search(body)


def scan():
    """Yield (path, fm_body, body) for every active operators/ card that is thin and unmeasured."""
    hits = []
    for dp, _dn, fn in os.walk(OPERATORS):
        for f in sorted(fn):
            if not f.endswith(".md") or f in SKIP_NAMES:
                continue
            path = os.path.join(dp, f)
            text = open(path, encoding="utf-8").read()
            fm, body = split_fm(text)
            if fm is None:
                continue
            if not re.search(r"^lifecycle:\s*active\s*$", fm, re.M):
                continue
            if is_thin(body):
                hits.append((path, fm, body))
    return sorted(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="apply; default is a dry-run listing")
    a = ap.parse_args()

    hits = scan()
    for path, _fm, body in hits:
        print(f"  {len(prose_lines(body)):>3} prose lines  {os.path.relpath(path, REPO)}")

    if not a.write:
        print(f"\n_demote_thin: {len(hits)} card(s) would become lifecycle: candidate (dry-run).")
        return 0

    for path, _fm, _body in hits:
        text = open(path, encoding="utf-8").read()
        fm, rest = split_fm(text)
        new_fm = re.sub(r"^lifecycle:\s*active\s*$", "lifecycle: candidate", fm, count=1, flags=re.M)
        open(path, "w", encoding="utf-8").write(f"---\n{new_fm}\n---\n{rest}")
    print(f"\n_demote_thin: {len(hits)} card(s) -> lifecycle: candidate. "
          "Re-run index/_gen_index.py + index/_validate_kb.py and commit together.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
