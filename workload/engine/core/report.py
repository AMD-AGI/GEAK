#!/usr/bin/env python3
"""Render one run-level REPORT.md from what the stages already wrote.

Nothing here is retyped. Every figure is read back from `STATE.json` and the stages'
own artifacts, so the report cannot disagree with the run -- and a stage that did not
run simply has no section rather than a stale one.

This file knows no individual stage and no individual skill. It renders two things: the
target the run declared, and the stage contract -- status, skill, metrics, notes, which
every stage must produce whatever implements it. Everything below that is the stage's
own `$OUT/REPORT.md`, concatenated verbatim.

That division is PRINCIPLES R3 and R6 together. The connector passes paths and reads
`result.json`; the moment it starts knowing that `optimize` writes `optimized.json` with
a `kernels` list, or that only the cold-verified column counts, that knowledge belongs to
the slot that enforces it -- and it used to live in both places, which is two copies of a
rule and therefore a rule that drifts. A slot owns what it accepts, so it owns what it
reports. A replacement skill inherits the section by writing the file; one that writes
nothing still appears in the table.

It is also the renderer `engine/core/watch.py` calls on every tick, so the report an
unattended run leaves behind mid-flight is the same document as the one it finishes
with -- and a campaign that never reaches `applyback` still has a readable one. That is
why `stage_rows` exists as a function rather than as a loop inside `main`: one mapping
from artifacts to a row, used by both.

Usage:
    python3 report.py --run-root <dir> [--out REPORT.md]
"""
import argparse, json, os


def rj(*parts):
    p = os.path.join(*parts)
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def summary_rejection(summary):
    """Consume a native producer's rejection, including the window before result.json."""
    if not isinstance(summary, dict):
        return None
    if summary.get('error'):
        return str(summary['error'])
    if summary.get('overlay_loaded_by_every_leg') is False:
        return 'native summary did not verify every serving arm'
    if summary.get('independent_arms') is False:
        return 'native summary rejected non-independent arms'
    return None


def stage_rows(root, state=None):
    """One row per stage the run knows about: skill, status, key metrics, notes.

    The contract view -- the part that holds for every stage whatever skill implements
    it -- and the only place that maps a stage's artifacts onto a row. A stage may report
    before the driver records it, but a driver rejection remains authoritative until
    validation succeeds. Retained result.json must not turn that rejection into `ok`.
    """
    state = (rj(root, "STATE.json") or {}) if state is None else state
    stages = state.get("stages") or state
    order = ["baseline", "profile", "roofline", "optimize", "applyback"]
    rows = []
    for name in order + [k for k in stages if k not in order]:
        st = stages.get(name)
        res = rj(root, name, "out", "result.json")
        if not isinstance(st, dict):
            if not isinstance(res, dict):
                continue
            st = {}
        res = res if isinstance(res, dict) else st
        status = res.get('status') or st.get('status') or '—'
        notes = res.get('notes') or ''
        if st.get('status') in ('blocked', 'fail'):
            status = st['status']
            notes = st.get('notes') or '; '.join(st.get('errors') or []) or notes
        if res.get('status') == 'fail':
            status = 'fail'
        if name == 'applyback':
            rejected = summary_rejection(rj(root, name, 'out', 'e2e', 'bench_summary.json'))
            if rejected:
                status, notes = 'fail', 'Latest native summary rejected: ' + rejected
        m = (res.get("metrics") or {}) if status == 'ok' else {}
        rows.append({"stage": name,
                     "skill": res.get("skill") or st.get("skill") or "—",
                     "status": status,
                     "metrics": {k: v for k, v in m.items() if not k.startswith("_")},
                     "notes": notes})
    return rows


def stage_section(root, stage):
    """A stage's own `$OUT/REPORT.md`, if it wrote one.

    Written either by the skill (profile's does) or by the slot's `finalize.py`, which
    already has every number in hand at the moment it decides whether to accept them. It
    is pasted in unread: this renderer does not get to summarise, reorder or second-guess
    a stage's account of itself.
    """
    try:
        with open(os.path.join(root, stage, "out", "REPORT.md")) as f:
            text = f.read().strip()
    except OSError:
        return ""
    return text + "\n" if text else ""


def stage_table(rows, limit=4):
    L = ["| Stage | Skill | Status | Key metrics |", "| --- | --- | --- | --- |"]
    for r in rows:
        keys = list(r["metrics"])[:limit]
        L.append("| `%s` | `%s` | %s | %s |"
                 % (r["stage"], r["skill"], r["status"],
                    ", ".join("%s=%s" % (k, r["metrics"][k]) for k in keys) or "—"))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = os.path.abspath(a.run_root)
    out = a.out or os.path.join(root, "REPORT.md")

    state = rj(root, "STATE.json") or {}
    L = ["# Workload run — %s\n" % os.path.basename(root)]

    # --- What was optimized, stated next to the run's name. A report that names only the
    # run invites the reader to infer the stack from the directory, which is exactly how
    # a stale label becomes a false claim.
    tgt = state.get("target") or {}
    if tgt.get("model"):
        L.append("**Target stack** — model `%s`%s%s. Declared in the config and "
                 "cross-checked against what the stages observed: **%s**.\n"
                 % (tgt["model"],
                    (", image `%s`" % tgt["image"]) if tgt.get("image") else "",
                    (", " + ", ".join("%s `%s`" % (k, v)
                                      for k, v in sorted((tgt.get("commits") or {}).items())))
                    if tgt.get("commits") else "",
                    tgt.get("verdict", "unchecked")))
    else:
        L.append("**Target stack** — not declared by this run's config, so the run name is "
                 "the only thing naming the stack and nothing checked it.\n")
    if os.path.exists(os.path.join(root, "RENAME.md")):
        L.append("This run was renamed after it finished; see `RENAME.md`.\n")

    # --- The contract view: every stage, whatever its skill.
    rows = stage_rows(root, state)
    admitted = {row['stage'] for row in rows if row['status'] == 'ok'}
    L.append("## Stages\n")
    L.extend(stage_table(rows))

    # --- Each stage's own section, exactly as the stage wrote it.
    # Only from a stage the driver admitted: an unadmitted stage's REPORT.md describes
    # work whose verdict did not hold, and pasting it in makes a rejected run look
    # finished.
    for row in rows:
        if row["stage"] not in admitted:
            continue
        section = stage_section(root, row["stage"])
        if section:
            L.append("\n" + section)

    # --- Whatever each stage flagged about itself.
    said = [(r["stage"], r["notes"]) for r in rows if r["notes"]]
    if said:
        L.append("\n## What the stages recorded\n")
        L.extend("- **%s** — %s" % (n, t.replace("\n", " ")) for n, t in said)

    L.append("")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
