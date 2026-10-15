#!/usr/bin/env python3
"""Register the overlay as a case of the reference harness, without patching the harness.

The launcher this targets already has a hook for exactly this -- a case entry carrying
`python_overlay` -- and it asserts the overlay path lies inside the harness root, so the
overlay is copied there rather than referenced from the stage's own output directory.

Two deliberate choices:

* The new case copies `base_case` verbatim and adds nothing but `python_overlay`. The
  candidate leg must differ from the reference leg in the kernels and in nothing else; a
  stray server argument would make the A/B measure two changes at once.
* The original case table is backed up and the registration recorded, so removing the
  entry and the copied directory returns the harness to its exact prior state.
"""
import argparse, hashlib, json, os, shutil


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--case-file", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--base-case", required=True)
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--overlay-subdir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    harness = os.path.abspath(a.harness)
    cf = os.path.join(harness, a.case_file)
    cases = json.load(open(cf))
    if a.base_case not in cases:
        raise SystemExit("the harness has no case %r to copy the reference leg's server "
                         "arguments from" % a.base_case)

    dest = os.path.join(harness, a.overlay_subdir.strip("/"))
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(a.overlay, dest)
    files = {os.path.relpath(os.path.join(r, n), dest): sha256(os.path.join(r, n))
             for r, _, ns in os.walk(dest) for n in ns}

    base = cases[a.base_case]
    entry = {"server_args": dict(base.get("server_args") or {}),
             "environment": dict(base.get("environment") or {}),
             "python_overlay": os.path.relpath(dest, harness),
             "hypothesis": "GEAK workload applyback: the reference leg's server, with the "
                           "cold-re-verified kernels layered in through a reversible "
                           "sitecustomize overlay and nothing else changed"}
    backup = cf + ".before_" + a.case
    if not os.path.exists(backup):
        shutil.copyfile(cf, backup)
    cases[a.case] = entry
    with open(cf, "w") as f:
        json.dump(cases, f, indent=2)

    rec = {"harness": harness, "case_file": cf, "case": a.case, "base_case": a.base_case,
           "entry": entry, "overlay_copied_to": dest, "overlay_files": files,
           "case_file_backup": backup,
           "how_to_revert": "delete the %r entry from %s (or restore %s) and remove %s"
                            % (a.case, cf, backup, dest)}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("registered case %s with %d overlay files under %s" % (a.case, len(files), dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
