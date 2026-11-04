#!/usr/bin/env python3
"""Integrity gate for third-party skills vendored in as-is.

`stages/roofline/kernel-roofline/` is vendored verbatim from an external archive and must
not change by a single byte. This script writes and checks a `MANIFEST.json` for them,
kept in the **slot directory** rather than inside the skill, so the manifest does not
itself alter what it is verifying.

    skill_manifest.py write  <skill_dir> --origin <source archive> [--note ...]
    skill_manifest.py verify <skill_dir>
    skill_manifest.py verify-all

A failing verify means the vendored skill has been touched — either an accidental edit or
an upstream version change. Both must be handled explicitly; the pipeline must not spend
48 hours running a skill that does not match its MANIFEST.
"""

import argparse
import hashlib
import json
import os
import sys

WORKLOAD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = os.path.join(WORKLOAD_ROOT, "stages")
SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache"}


def manifest_path(skill_dir):
    skill_dir = os.path.abspath(skill_dir.rstrip("/"))
    return os.path.join(os.path.dirname(skill_dir),
                        "MANIFEST.%s.json" % os.path.basename(skill_dir))


def digest_tree(skill_dir):
    out = {}
    for root, dirs, files in os.walk(skill_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            if name.endswith(".pyc"):
                continue
            full = os.path.join(root, name)
            if os.path.islink(full):
                out[os.path.relpath(full, skill_dir)] = "symlink:" + os.readlink(full)
                continue
            h = hashlib.sha256()
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out[os.path.relpath(full, skill_dir)] = h.hexdigest()
    return out


def cmd_write(args):
    skill_dir = os.path.abspath(args.skill_dir.rstrip("/"))
    files = digest_tree(skill_dir)
    tree = hashlib.sha256(
        "".join("%s %s\n" % (k, files[k]) for k in sorted(files)).encode()
    ).hexdigest()
    man = {
        "skill": os.path.basename(skill_dir),
        "stage": os.path.basename(os.path.dirname(skill_dir)),
        "origin": args.origin,
        "note": args.note or "",
        "modified_from_origin": False,
        "file_count": len(files),
        "tree_sha256": tree,
        "files": files,
    }
    p = manifest_path(skill_dir)
    with open(p, "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
        f.write("\n")
    print("%s  %d files  tree=%s" % (p, len(files), tree[:16]))
    return 0


def verify_one(skill_dir, quiet=False):
    skill_dir = os.path.abspath(skill_dir.rstrip("/"))
    p = manifest_path(skill_dir)
    if not os.path.exists(p):
        print("missing MANIFEST: %s" % p, file=sys.stderr)
        return 1
    with open(p) as f:
        man = json.load(f)
    now = digest_tree(skill_dir)
    old = man["files"]

    missing = sorted(set(old) - set(now))
    added = sorted(set(now) - set(old))
    changed = sorted(k for k in set(old) & set(now) if old[k] != now[k])

    if missing or added or changed:
        print("%s does not match its MANIFEST:" % man["skill"], file=sys.stderr)
        for k in missing:
            print("  - missing %s" % k, file=sys.stderr)
        for k in added:
            print("  + added   %s" % k, file=sys.stderr)
        for k in changed:
            print("  ! changed %s" % k, file=sys.stderr)
        print("  If this update is intentional, rerun `skill_manifest.py write` and state "
              "the reason in the note.", file=sys.stderr)
        return 1
    if not quiet:
        print("OK  %s/%s  %d files" % (man["stage"], man["skill"], man["file_count"]))
    return 0


def cmd_verify(args):
    return verify_one(args.skill_dir)


def cmd_verify_all(args):
    rc = 0
    found = 0
    for stage in sorted(os.listdir(STAGES)):
        sd = os.path.join(STAGES, stage)
        if not os.path.isdir(sd):
            continue
        for name in sorted(os.listdir(sd)):
            if name.startswith("MANIFEST.") and name.endswith(".json"):
                skill = name[len("MANIFEST."):-len(".json")]
                found += 1
                rc |= verify_one(os.path.join(sd, skill))
    if not found:
        print("no MANIFEST was found")
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("write")
    p.add_argument("skill_dir")
    p.add_argument("--origin", required=True)
    p.add_argument("--note")
    p.set_defaults(fn=cmd_write)
    p = sub.add_parser("verify")
    p.add_argument("skill_dir")
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("verify-all")
    p.set_defaults(fn=cmd_verify_all)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
