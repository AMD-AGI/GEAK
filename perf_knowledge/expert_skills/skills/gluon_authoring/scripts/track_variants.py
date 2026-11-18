#!/usr/bin/env python3
"""Materialize one source file at several git revisions of the track, for a side-by-side A/B.

An A/B between the kept line and a candidate (or between two candidates of a round, or between a
prerequisite and its pair) needs each arm's source as its own file. Copying them by hand per round
drifts: the copy is edited, the tag moves, nobody can say which commit an arm was. This writes each
revision's file from git, with its commit and sha256, into one directory and a manifest the adapter
reads, so every arm is named by the revision it came from.

Usage:
  track_variants.py --track <git checkout> --file <path in the repo> --out <dir> <rev> [<rev> ...]
  track_variants.py --selftest

<rev> is anything git resolves to a commit (a tag such as r3_d1, a branch, a sha). Writes
<out>/<rev>.<ext> per revision (characters outside [A-Za-z0-9_.-] become '_') and
<out>/variants.json:
  {"track": ..., "file": ..., "variants": {"<rev>": {"path": ..., "commit": ..., "sha256": ...}}}
An adapter for ab_bench.py --module or edge_shapes.py --module loads each path from variants.json.
Existing files are moved aside (<name>.old_<epoch>), never deleted.
Exit: 0 written | 1 a revision or the file does not resolve | 2 usage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _git(track, *args):
    return subprocess.run(["git", "-C", str(track), *args], capture_output=True, text=False)


def _move_aside(p: Path):
    if p.exists():
        p.rename(p.with_name(f"{p.name}.old_{int(time.time())}_{os.getpid()}"))


def materialize(track, file, out, revs):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ext = Path(file).suffix
    manifest = {"track": str(Path(track).resolve()), "file": file, "variants": {}}
    errors = []
    for rev in revs:
        c = _git(track, "rev-parse", "--verify", "-q", f"{rev}^{{commit}}")
        if c.returncode != 0:
            errors.append(f"{rev}: not a revision of {track}")
            continue
        commit = c.stdout.decode().strip()
        blob = _git(track, "show", f"{commit}:{file}")
        if blob.returncode != 0:
            errors.append(f"{rev}: {file} does not exist at {commit[:12]}")
            continue
        dst = out / (re.sub(r"[^A-Za-z0-9_.-]", "_", rev) + ext)
        _move_aside(dst)
        dst.write_bytes(blob.stdout)
        manifest["variants"][rev] = {"path": str(dst.resolve()), "commit": commit,
                                     "sha256": hashlib.sha256(blob.stdout).hexdigest()}
    mpath = out / "variants.json"
    _move_aside(mpath)
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest, errors


def _selftest():
    fail = 0

    def chk(cond, msg):
        nonlocal fail
        if not cond:
            print("FAIL:", msg)
            fail = 1

    with tempfile.TemporaryDirectory() as t:
        repo = Path(t) / "track"
        repo.mkdir()
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
               "GIT_COMMITTER_EMAIL": "t@t"}
        run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, env=env)
        run("init", "-q")
        (repo / "src").mkdir()
        (repo / "src" / "k.py").write_text("v = 1\n")
        run("add", "-A"); run("commit", "-qm", "base"); run("tag", "base")
        (repo / "src" / "k.py").write_text("v = 2\n")
        run("commit", "-qam", "r1"); run("tag", "r1_d0")
        m, errs = materialize(repo, "src/k.py", Path(t) / "v", ["base", "r1_d0", "nope"])
        chk(Path(m["variants"]["base"]["path"]).read_text() == "v = 1\n", "base arm is the base revision")
        chk(Path(m["variants"]["r1_d0"]["path"]).read_text() == "v = 2\n", "candidate arm is its tag")
        chk(len(errs) == 1 and "nope" in errs[0], "an unknown revision is reported, not skipped silently")
        chk(json.loads((Path(t) / "v" / "variants.json").read_text())["variants"]["r1_d0"]["commit"]
            == m["variants"]["r1_d0"]["commit"], "manifest records the commit")
        materialize(repo, "src/k.py", Path(t) / "v", ["base"])
        chk(any(p.name.startswith("base.py.old_") for p in (Path(t) / "v").iterdir()), "a rewrite moves the old file aside")
    print("track_variants selftest:", "FAIL" if fail else "OK")
    return fail


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--selftest"]:
        return _selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, help="git checkout holding the revisions (the workflow's TRACK)")
    ap.add_argument("--file", required=True, help="path of the source file inside the repo")
    ap.add_argument("--out", required=True, help="directory for the per-revision files and variants.json")
    ap.add_argument("revs", nargs="+", help="revisions (tags, branches, shas)")
    a = ap.parse_args(argv)
    m, errors = materialize(a.track, a.file, a.out, a.revs)
    for rev, v in m["variants"].items():
        print(f"  {rev:20s} {v['commit'][:12]}  {v['path']}")
    for e in errors:
        print(f"  ERROR {e}", file=sys.stderr)
    print(f"wrote {Path(a.out) / 'variants.json'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
