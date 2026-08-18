#!/usr/bin/env python3
"""One-shot, idempotent backfill of the structured-index frontmatter (plan Part 1.3 / P1).

ADDS only the machine-derivable keys, NEVER overwrites an existing value:
  * layer         reference (perf_knowledge/) | learned (learned/ cards)
  * platforms     derived from `gens:` (perf_knowledge) or a gfxNNNN token (learned filename/key)
  * kernel_class  from the operator map (sota_card / operator_overview only)
  * lifecycle     `active` default when the file is a real card (has status:/kind: or is a learned card)

`levers` / `cost` / `bound_type` / `verified_on` / `upstream_rev` are deliberately LEFT for a human (or a
later governance pass); the validator flags their absence as a warning, not an error.

Usage:
  python3 index/_backfill_kb.py            # dry-run: report what WOULD change
  python3 index/_backfill_kb.py --write    # apply
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _kb_vocab as V  # noqa: E402

PK_ROOT = os.path.dirname(HERE)                                  # perf_knowledge/
REPO = os.path.dirname(PK_ROOT)
LEARNED_ROOT = os.path.join(REPO, "e2e_workflow", "knowledge", "learned")

FM_RE = re.compile(r"^---\n(.*?)\n---\n?", re.S)


def split_fm(text):
    """Return (fm_body, rest) or (None, text) when there is no frontmatter block."""
    m = FM_RE.match(text)
    if not m:
        return None, text
    return m.group(1), text[m.end():]


def fm_keys(fm_body):
    keys = set()
    for line in fm_body.splitlines():
        km = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):", line)
        if km:
            keys.add(km.group(1))
    return keys


def fm_get_list(fm_body, key):
    """Read a `key: [a, b]` inline list from a frontmatter body (best-effort)."""
    m = re.search(r"^%s:\s*\[(.*?)\]\s*$" % re.escape(key), fm_body, re.M)
    if not m:
        return []
    return [x.strip() for x in m.group(1).split(",") if x.strip()]


def fm_get_scalar(fm_body, key):
    m = re.search(r"^%s:\s*(.+?)\s*$" % re.escape(key), fm_body, re.M)
    return m.group(1).strip() if m else None


def yl(xs):
    return "[" + ", ".join(xs) + "]"


def derive(path, fm_body):
    """Return an ordered list of (key, value_str) to append; empty if nothing to add."""
    present = fm_keys(fm_body)
    additions = []
    rel = os.path.relpath(path, REPO)
    is_learned = os.path.abspath(path).startswith(os.path.abspath(LEARNED_ROOT) + os.sep)

    # layer -------------------------------------------------------------------
    if "layer" not in present:
        additions.append(("layer", "learned" if is_learned else "reference"))

    # platforms ---------------------------------------------------------------
    if "platforms" not in present:
        plats = []
        if is_learned:
            # learned cards carry the gfx in the filename (…-gfx942.md) and the `key:` line.
            toks = re.findall(r"gfx\d+", os.path.basename(path) + " " + (fm_get_scalar(fm_body, "key") or ""))
            for t in toks:
                t = t.lower()
                if t in V.GEN_IDS and t not in plats:
                    plats.append(t)
        else:
            plats = V.gens_to_platforms(fm_get_list(fm_body, "gens"))
        if plats:
            additions.append(("platforms", yl(plats)))

    # kernel_class ------------------------------------------------------------
    if "kernel_class" not in present and not is_learned:
        op = fm_get_scalar(fm_body, "operator")
        if not op:
            # operator_overview cards live at operators/<op>/overview.md
            parts = os.path.abspath(path).split(os.sep)
            if "operators" in parts:
                i = parts.index("operators")
                if i + 1 < len(parts):
                    op = parts[i + 1]
        kc = V.kernel_class_for_operator(op) if op else None
        if kc:
            additions.append(("kernel_class", kc))

    # lifecycle ---------------------------------------------------------------
    if "lifecycle" not in present:
        looks_like_card = is_learned or ("status" in present) or ("kind" in present)
        if looks_like_card:
            additions.append(("lifecycle", "active"))

    return additions, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="apply changes (default: dry-run)")
    a = ap.parse_args()

    # Content layers only. expert_skills/ (self-contained skill packages), the index/ tooling dir, and
    # _templates/ are NOT KB content and are excluded.
    EXCLUDE = (os.sep + "expert_skills" + os.sep, os.sep + "index" + os.sep, os.sep + "_templates" + os.sep)
    targets = []
    for root in [PK_ROOT, LEARNED_ROOT, os.path.join(REPO, "kernel_workflow", "knowledge")]:
        for dp, _dn, fn in os.walk(root):
            if any(x in dp + os.sep for x in EXCLUDE):
                continue
            for f in fn:
                if f.endswith(".md"):
                    targets.append(os.path.join(dp, f))

    changed = no_fm = untouched = skipped = 0
    for path in sorted(set(targets)):
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            continue
        fm_body, rest = split_fm(text)
        if fm_body is None:
            no_fm += 1
            print(f"  [no-fm] {os.path.relpath(path, REPO)}")
            continue
        additions, rel = derive(path, fm_body)
        if not additions:
            untouched += 1
            continue
        add_str = "  ".join(f"{k}={v}" for k, v in additions)
        if a.write:
            # Some learned cards are root-owned (created by a container run); skip cleanly rather than abort.
            if not os.access(path, os.W_OK):
                skipped += 1
                print(f"  [skip:ro] {rel}")
                continue
            new_fm = fm_body.rstrip("\n") + "\n" + "\n".join(f"{k}: {v}" for k, v in additions)
            try:
                open(path, "w", encoding="utf-8").write("---\n" + new_fm + "\n---\n" + rest)
            except OSError as e:
                skipped += 1
                print(f"  [skip:err] {rel}: {e}")
                continue
        changed += 1
        print(f"  [+] {rel}: {add_str}")

    print(f"\n{'APPLIED' if a.write else 'DRY-RUN'}: {changed} files changed, "
          f"{untouched} already complete, {no_fm} without frontmatter, {skipped} skipped (read-only).")


if __name__ == "__main__":
    main()
