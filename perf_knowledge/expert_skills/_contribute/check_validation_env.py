#!/usr/bin/env python3
"""Report whether each validated skill records the environment its numbers were measured in.

A skill's speedup is evidence about one environment: a FlyDSL version, a ROCm build, an aiter
revision, a gfx target. Reused on a different one, the LOGIC still transfers -- that is the whole
point of the portability rule -- but the NUMBER does not, and the reader can only know which parts
to re-measure if the record says what the number came from.

Today it mostly does not. Of six skills declaring `to_backend: flydsl` and `status: validated`, two
point at no artifact at all and one points at a directory of scripts; of the three real measurement
records, two pin `flydsl_version` and one does not, and none records a ROCm version or an aiter
revision. Silence there reads as "the environment does not matter", which is the one thing it never
means.

So the rule is about SILENCE, not about completeness:

    an environment field must be present and either name a value or say `not_captured`

`not_captured` is a legitimate, permanent answer for a run already in the past -- inventing a
plausible commit hash would be far worse. What this refuses is the field being absent, because then
a reader cannot tell "not recorded" from "same as mine".

THIS IS NOT A FILTER. It reports at contribution time; it never edits a skill, never writes a
frontmatter gate, and nothing here may be used to skip a skill. A version difference is not a reason
to skip a skill (see _contribute/test_flydsl_skill_portability.py) -- it is a reason to re-measure.

Run from the repo root or anywhere:
    python3 perf_knowledge/expert_skills/_contribute/check_validation_env.py [--json] [SKILL_ID ...]
"""
import argparse
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERT_SKILLS = os.path.dirname(HERE)
SKILLS = os.path.join(EXPERT_SKILLS, "skills")
REPO = os.path.dirname(os.path.dirname(EXPERT_SKILLS))

NOT_CAPTURED = "not_captured"
# Always required, whatever the language.
BASE_FIELDS = ("rocm_version", "aiter_commit", "gpu")


def load_frontmatter(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    return yaml.safe_load(parts[1])


def resolve_artifact(artifact):
    """Two path conventions coexist in the tree: expert_skills-relative and repo-relative."""
    for base in (EXPERT_SKILLS, REPO):
        p = os.path.join(base, artifact)
        if os.path.exists(p):
            return p
    return None


def required_fields(backend):
    """`flydsl_version` for a FlyDSL skill, `triton_version` for a Triton one, and so on."""
    fields = list(BASE_FIELDS)
    if backend:
        fields.insert(0, f"{backend}_version")
    return fields


def check_skill(skill_id):
    """One report row. `problems` is non-empty only for silence, never for `not_captured`."""
    path = os.path.join(SKILLS, skill_id, "skill.md")
    row = {"skill": skill_id, "problems": [], "recorded": {}, "not_captured": []}
    fm = load_frontmatter(path)
    if fm is None:
        row["problems"].append("skill.md has no YAML frontmatter")
        return row

    val = fm.get("validation") or {}
    row["status"] = str(val.get("status", "") or "")
    backend = (fm.get("match") or {}).get("to_backend") or ""
    row["backend"] = backend
    if row["status"] != "validated":
        return row                      # only a validated claim owes an environment

    artifact = str(val.get("artifact", "") or "").strip()
    if not artifact:
        row["problems"].append(
            "status is 'validated' but validation.artifact is empty: the claim has no record, so "
            "there is no environment to read and nothing to re-measure against")
        return row

    resolved = resolve_artifact(artifact)
    row["artifact"] = artifact
    if resolved is None:
        row["problems"].append(f"validation.artifact {artifact!r} does not exist")
        return row
    if os.path.isdir(resolved):
        row["problems"].append(
            f"validation.artifact {artifact!r} is a directory, not a measurement record: a folder of "
            f"scripts cannot say which FlyDSL, ROCm or aiter revision produced the numbers")
        return row

    try:
        with open(resolved, encoding="utf-8") as f:
            evidence = yaml.safe_load(f.read()) or {}
    except Exception as e:  # noqa: BLE001 - any unreadable record is reported, not raised
        row["problems"].append(f"validation.artifact {artifact!r} is unreadable: {e}")
        return row

    for field in required_fields(backend):
        if field not in evidence:
            row["problems"].append(
                f"{field} is absent from {os.path.basename(resolved)}. Absent is not the same as "
                f"unknown: write `{field}: {NOT_CAPTURED}` when it was not recorded, so a reader "
                f"cannot mistake silence for a matching environment")
        elif str(evidence[field]).strip() == NOT_CAPTURED:
            row["not_captured"].append(field)
        else:
            row["recorded"][field] = evidence[field]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("skills", nargs="*", help="skill ids to check (default: all)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    ids = a.skills or sorted(d for d in os.listdir(SKILLS)
                             if os.path.isfile(os.path.join(SKILLS, d, "skill.md")))
    rows = [check_skill(s) for s in ids]
    failing = [r for r in rows if r["problems"]]

    if a.json:
        print(json.dumps({"skills_checked": len(rows), "skills_with_silent_fields": len(failing),
                          "rows": rows}, ensure_ascii=False, indent=2))
    else:
        for r in rows:
            if r.get("status") != "validated":
                print(f"  --   {r['skill']}: status={r.get('status') or 'none'} — no claim to support")
                continue
            mark = "FAIL" if r["problems"] else "ok  "
            print(f"  {mark} {r['skill']}")
            for k, v in r["recorded"].items():
                print(f"         recorded      {k}: {v}")
            if r["not_captured"]:
                print(f"         re-measure    {', '.join(r['not_captured'])} "
                      f"(not captured at the time — reuse the logic, re-measure the numbers)")
            for p in r["problems"]:
                print(f"         SILENT        {p}")
        print(f"\n{len(rows)} skill(s) checked; {len(failing)} with a silently absent field.")

    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
