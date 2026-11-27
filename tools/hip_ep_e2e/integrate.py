#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Turn a kernel_workflow result into a hip-ep overlay.

The kernel task mirrors the hip-ep tree under src/ (make_task.py), so the
lane's final patch (a git diff from the task root) maps one to one onto hip-ep
paths. This applies the patch to a copy of the task, refuses it if it changed
anything in the task but the editable files, and writes an overlay: the base
overlay plus the edited files, at their hip-ep paths. A new file outside src/ is left
out: kernel_lane commits a round with `git add -A`, so whatever an agent left in its
workspace (a scratch harness, notes) rides along, and none of it reaches the library.

    python3 integrate.py --task T --patch final_patch.diff --out OVERLAY [--base-overlay O]
"""
import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile

import yaml


def _files(root):
    """{relative path: path} of every file under root, without build/ and .git/ dirs."""
    out = {}
    for dirpath, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("build", ".git")]
        for n in names:
            p = os.path.join(dirpath, n)
            out[os.path.relpath(p, root)] = p
    return out


def changed_files(task, patched):
    """Files that differ between the task and its patched copy, anywhere in the task:
    edited, added or deleted."""
    a, b = _files(task), _files(patched)
    return [rel for rel in sorted(set(a) | set(b))
            if rel not in a or rel not in b or not filecmp.cmp(a[rel], b[rel], shallow=False)]


def integrate(task, patch, out, base_overlay=None):
    with open(os.path.join(task, "config.yaml")) as f:
        editable = set(yaml.safe_load(f)["source_file_path"])
    shutil.rmtree(out, ignore_errors=True)  # a rerun replaces what an earlier attempt left
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "task")
        shutil.copytree(task, work, ignore=shutil.ignore_patterns("build"))
        # Inside a git work tree `git apply` patches paths relative to that tree and skips the
        # rest, so git must not find one above the copy.
        p = subprocess.run(["git", "apply", "--whitespace=nowarn", os.path.abspath(patch)],
                           cwd=work, capture_output=True, text=True, check=False,
                           env={**os.environ, "GIT_CEILING_DIRECTORIES": tmp})
        if p.returncode != 0:
            raise RuntimeError(f"the patch does not apply to {task}: {p.stderr.strip()[:500]}")
        changed = changed_files(task, work)
        ignored = [c for c in changed
                   if not c.startswith("src/") and not os.path.exists(os.path.join(task, c))]
        changed = [c for c in changed if c not in ignored]
        frozen = [c for c in changed if c not in editable]
        if frozen:
            raise RuntimeError(f"the patch edits files outside the task's editable set: {frozen}")
        if not changed:
            raise RuntimeError("the patch changes no editable file")
        gone = [c for c in changed if not os.path.exists(os.path.join(work, c))]
        if gone:
            raise RuntimeError(f"the patch deletes editable files: {gone}")
        if base_overlay:
            shutil.copytree(base_overlay, out)
        else:
            os.makedirs(out)
        for rel in changed:
            dst = os.path.join(out, os.path.relpath(rel, "src"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(work, rel), dst)
    return {"files": [os.path.relpath(c, "src") for c in changed], "ignored": ignored}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-overlay", default=None)
    a = ap.parse_args()
    try:
        info = integrate(a.task, a.patch, a.out, a.base_overlay)
    except RuntimeError as e:
        sys.exit(f"[integrate] {e}")
    print(f"[integrate] overlay {a.out}: {', '.join(info['files'])}" +
          (f"; left out (new, outside src/): {', '.join(info['ignored'])}" if info["ignored"] else ""))


if __name__ == "__main__":
    main()
