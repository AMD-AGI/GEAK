#!/usr/bin/env python3
"""Render one run-level REPORT.md from what the stages already wrote.

Nothing here is retyped. Every figure is read back from `STATE.json` and the stages'
own artifacts, so the report cannot disagree with the run -- and a stage that did not
run simply has no section rather than a stale one.

The file knows no individual skill. It renders the stage contract (status + metrics +
notes, which every stage must produce) for all five stages, and adds a detail section
only when a well-known artifact is present. A replacement skill that writes the same
artifact gets the same section; one that does not still appears in the summary.

It is also the renderer `scripts/watch.py` calls on every tick, so the report an
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


def pct(x):
    return "%+.4f%%" % x if isinstance(x, (int, float)) else "—"


def x_(v, nd=4):
    return ("%%.%df×" % nd) % v if isinstance(v, (int, float)) else "—"


def stage_rows(root, state=None):
    """One row per stage the run knows about: skill, status, key metrics, notes.

    The contract view -- the part that holds for every stage whatever skill implements
    it -- and the only place that maps a stage's artifacts onto a row. `result.json` wins
    over STATE.json where both exist, because the stage wrote one and the driver wrote
    the other, and a stage that has written its result but is not recorded in STATE.json
    yet still gets a row: mid-run that is the ordinary state of the stage that has just
    finished, and a reporter that skipped it would describe the run as emptier than it is.
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
        m = res.get("metrics") or {}
        rows.append({"stage": name,
                     "skill": res.get("skill") or st.get("skill") or "—",
                     "status": res.get("status") or st.get("status") or "—",
                     "metrics": {k: v for k, v in m.items() if not k.startswith("_")},
                     "notes": res.get("notes") or ""})
    return rows


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
    L.append("## Stages\n")
    L.extend(stage_table(rows))

    # --- Baseline.
    b = rj(root, "baseline", "out", "baseline.json")
    pb = rj(root, "profile", "out", "pristine_baseline.json")
    if b or pb:
        L.append("\n## Baseline\n")
        L.append("| Reference | tok/s | Provenance |\n| --- | --- | --- |")
        if b:
            L.append("| adopted receipt | %s | %s |"
                     % (b.get("throughput"), (b.get("provenance") or {}).get("method")))
        if pb:
            L.append("| node-local pristine leg | %s | measured on this node by the same "
                     "harness as the candidate legs |" % pb.get("output_tok_per_s"))
        if pb and b:
            L.append("\nThe A/B below is against the **node-local** leg. A receipt from "
                     "another machine is a cross-check, not a reference: only the local "
                     "leg shares the harness, the client flags and the corpus with the "
                     "candidate.\n")

    # --- Optimize: claimed vs cold-verified, per kernel.
    o = rj(root, "optimize", "out", "optimized.json")
    if o:
        L.append("\n## Kernel optimization\n")
        L.append("Credited %s GPU-h across %s tasks; %s passed cold re-verification.\n"
                 % (o.get("gpu_hours"), o.get("tasks_total"), len(o.get("kept") or [])))
        L.append("| Kernel | Attempts | Credited | Claimed | **Cold-verified** | Verdict |"
                 "\n| --- | --- | --- | --- | --- | --- |")
        for k in o.get("kernels") or []:
            L.append("| `%s` | %s | %.1f h | %s | **%s** | %s |"
                     % (k.get("kernel"), k.get("attempts"),
                        (k.get("credited_seconds") or 0) / 3600.0,
                        x_(k.get("claimed_speedup")), x_(k.get("verified_speedup")),
                        k.get("verdict")))
        notes = [(k["kernel"], n) for k in o.get("kernels") or [] for n in k.get("notes") or []]
        if notes:
            L.append("")
            L.extend("- `%s`: %s" % (k, n) for k, n in notes)
        L.append("\nThe claimed column is what the worker measured while optimizing. Only "
                 "the cold-verified column counts: a fresh process per repeat, the "
                 "minimum across repeats, and a code-identity check that calls an "
                 "unchanged source what it is.\n")

    # --- Applyback: the only end-to-end claim in the run.
    ab = rj(root, "applyback", "out", "ab.json")
    summ = rj(root, "applyback", "out", "e2e", "bench_summary.json")
    if ab:
        L.append("\n## End to end\n")
        base, cand = ab.get("baseline") or {}, ab.get("candidate") or {}
        L.append("| | tok/s |\n| --- | --- |")
        L.append("| reference | %s |" % base.get("throughput"))
        for leg in (summ or {}).get("legs") or []:
            L.append("| candidate leg `%s` | %s |" % (leg.get("tag"), leg.get("throughput")))
        L.append("| **candidate median (%s legs)** | **%s** |"
                 % (cand.get("repeats"), cand.get("throughput")))
        L.append("\n| | |\n| --- | --- |")
        L.append("| operator speedup, peak | %s |" % x_(ab.get("op_speedup_max")))
        L.append("| **whole machine** | **%s** |" % pct(ab.get("e2e_delta_pct")))
        L.append("| noise band | ±%.2f%% |" % (ab.get("noise_band_pct") or 0))
        L.append("| verdict | **%s** |" % ab.get("verdict"))
        L.append("| overlay shadowed every target | %s |" % ab.get("overlay_all_shadowed"))
        L.append("| overlay loaded by every leg | %s |"
                 % (summ or {}).get("overlay_loaded_by_every_leg"))
        if (summ or {}).get("spread_pct") is not None:
            L.append("| spread between legs | %.4f%% |" % summ["spread_pct"])
        L.append("\n| Kernel installed | Operator speedup | Target |\n| --- | --- | --- |")
        for op in ab.get("op_level") or []:
            L.append("| `%s` | %s | `%s` |"
                     % (op.get("kernel"), x_(op.get("op_speedup")), op.get("target")))
        L.append("\nThe two numbers are stated separately because in this scenario they "
                 "come apart: an operator win with no system movement is the normal "
                 "outcome, not the exception. A delta inside the band means the machine "
                 "did not move, and is reported as such rather than rounded upward.\n")

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
