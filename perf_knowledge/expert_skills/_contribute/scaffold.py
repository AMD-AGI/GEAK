#!/usr/bin/env python3
"""scaffold.py — create a new expert skill skeleton and (re)generate the selector index.

Usage:
  # create a new skill skeleton from the template + register it (status: draft)
  python _contribute/scaffold.py --id my_skill --operator dense_gemm --scope e2e \
      --title "..." --author you --gens gfx942 --dtypes fp8_e4m3_fnuz --regimes prefill,decode

  # regenerate index.yaml from every skills/*.md frontmatter (no new skill)
  python _contribute/scaffold.py --reindex

The index is AUTO-GENERATED from each skill file's frontmatter, so contributors only ever edit one
markdown file; the selector stays consistent and merge-conflict-free.
"""
import argparse, os, re, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # .../expert_skills
SKILLS_DIR = os.path.join(ROOT, "skills")
TEMPLATE = os.path.join(ROOT, "_template", "SKILL_TEMPLATE.md")
INDEX = os.path.join(ROOT, "index.yaml")
CAP_INDEX = os.path.normpath(os.path.join(ROOT, "..", "index", "capability_index.yaml"))

FM_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)


def read_frontmatter(path):
    with open(path) as f:
        txt = f.read()
    m = FM_RE.match(txt)
    if not m:
        raise ValueError(f"{path}: no YAML frontmatter")
    return yaml.safe_load(m.group(1)), m.group(2), txt


def known_operators():
    """The set of operator names declared in capability_index.yaml (skills must align)."""
    if not os.path.exists(CAP_INDEX):
        return None  # cannot validate -> caller warns, does not block
    data = yaml.safe_load(open(CAP_INDEX))
    return {c["operator"] for c in (data.get("candidates") or []) if "operator" in c}


def reindex():
    ops = known_operators()
    entries = []
    # Each skill is a SUBDIRECTORY skills/<id>/ with a main skill.md (plus any extra files it needs).
    for sub in (sorted(os.listdir(SKILLS_DIR)) if os.path.isdir(SKILLS_DIR) else []):
        sub_path = os.path.join(SKILLS_DIR, sub)
        skill_md = os.path.join(sub_path, "skill.md")
        if not os.path.isdir(sub_path) or sub.startswith("_") or not os.path.exists(skill_md):
            continue
        fm, _, _ = read_frontmatter(skill_md)
        op = (fm.get("match") or {}).get("operator")
        if ops is not None and op not in ops:
            print(f"  WARN: {sub}/skill.md: operator '{op}' not in capability_index.yaml", file=sys.stderr)
        entries.append({
            "id": fm["id"],
            "file": f"skills/{sub}/skill.md",
            "scope": fm.get("scope", "kernel"),
            "match": fm.get("match", {}),
            "expects": fm.get("expects", {}),
            "validation_status": (fm.get("validation") or {}).get("status", "draft"),
        })
    header = (
        "# index.yaml — expert_skills selector (AUTO-MAINTAINED by _contribute/scaffold.py + "
        "validate_skill.py).\n"
        "# Regenerate with:  python _contribute/scaffold.py --reindex\n"
        "# NOT a ranking. Filter by (operator, gen, arch_class, [from->to], status==validated) -> MEASURE.\n"
        "# Only 'validated' skills are auto-applied by the workflows (advisory priors, never override A/B).\n\n"
        "schema: {id, file, scope, match, expects, validation_status}\n\n"
    )
    with open(INDEX, "w") as f:
        f.write(header)
        f.write(yaml.safe_dump({"skills": entries}, sort_keys=False, allow_unicode=True, width=100))
    print(f"reindexed {len(entries)} skill(s) -> {os.path.relpath(INDEX, ROOT)}")


def create(args):
    ops = known_operators()
    if ops is not None and args.operator not in ops:
        sys.exit(f"ERROR: operator '{args.operator}' is not in capability_index.yaml. "
                 f"Use an existing operator name so the selector can match. Known e.g.: "
                 f"{', '.join(sorted(list(ops))[:8])} ...")
    skill_dir = os.path.join(SKILLS_DIR, args.id)
    dest = os.path.join(skill_dir, "skill.md")
    if os.path.exists(dest):
        sys.exit(f"ERROR: {dest} already exists.")
    os.makedirs(skill_dir, exist_ok=True)
    fm, body, _ = read_frontmatter(TEMPLATE)
    fm["id"] = args.id
    fm["title"] = args.title or fm["title"]
    fm["authors"] = [a.strip() for a in args.author.split(",")] if args.author else fm["authors"]
    fm["scope"] = args.scope
    fm["match"]["operator"] = args.operator
    if args.gens:    fm["match"]["gens"] = args.gens.split(",")
    if args.dtypes:  fm["match"]["dtypes"] = args.dtypes.split(",")
    if args.regimes: fm["match"]["regimes"] = args.regimes.split(",")
    if args.arch:    fm["match"]["arch_class"] = args.arch.split(",")
    if args.from_backend: fm["match"]["from_backend"] = args.from_backend
    if args.to_backend:   fm["match"]["to_backend"] = args.to_backend
    fm.setdefault("validation", {})["status"] = "draft"
    with open(dest, "w") as f:
        f.write("---\n")
        f.write(yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, width=100))
        f.write("---\n")
        f.write(body)
    print(f"created {os.path.relpath(dest, ROOT)} (status: draft)")
    print("Next: fill in When-to-use / Mechanism / Procedure / Do-no-harm, then run "
          "_contribute/make_pr.sh " + args.id)
    reindex()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reindex", action="store_true", help="regenerate index.yaml from skills/*.md")
    p.add_argument("--id"); p.add_argument("--operator"); p.add_argument("--scope",
        choices=["kernel", "e2e"], default="kernel")
    p.add_argument("--title", default=""); p.add_argument("--author", default="")
    p.add_argument("--gens", default=""); p.add_argument("--dtypes", default="")
    p.add_argument("--regimes", default=""); p.add_argument("--arch", default="")
    p.add_argument("--from-backend", dest="from_backend", default="")
    p.add_argument("--to-backend", dest="to_backend", default="")
    a = p.parse_args()
    if a.reindex and not a.id:
        return reindex()
    if not (a.id and a.operator):
        p.error("provide --id and --operator to create a skill, or --reindex to rebuild the index")
    create(a)


if __name__ == "__main__":
    main()
