#!/usr/bin/env python3
"""Shape manifests' Python frames -> each kernel's live callable and the files behind it.

`with_stack` recorded, around every sampled launch, the frames of the serving process:
`<file>(<line>): <func>`. They name the file that was actually imported on this
deployment, so a task's `target_callable` comes from them rather than from a package
name. For every kernel this keeps the frames that belong to the serving stack, innermost
first, resolves each to an installed file, hashes it and recovers the qualified name
(`module:Class.method`) from the source, and proposes seam candidates:

  launcher   the innermost serving-stack frame (what calls the kernel or its op)
  op_seam    for a group of kernels: the innermost frame they ALL pass through, which
             is the smallest boundary that contains the whole chain

    python3 source_map.py --shapes-dir evidence/shapes --out evidence/source_map.json \
        [--stack-package sglang --stack-package aiter ...] [--root /sgl-workspace/sglang/python]

Run it where the serving stack is installed (inside the serving container): frame paths
are resolved through importlib first, then against --root. A frame that cannot be
resolved is kept with `resolved: false`; nothing is guessed.

This is evidence, not a verdict: a frame proves the call reached that file; a task still
has to prove its candidate overlay executes.
Stdlib only.
"""
import argparse
import ast
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict

FRAME_RE = re.compile(r"^(?P<path>.+?\.py)\((?P<line>\d+)\): (?P<func>.+)$")
DEFAULT_PACKAGES = ("sglang", "sgl_kernel", "aiter", "vllm", "flashinfer", "flash_mla")
# Plumbing between a callable and its launch: never a useful seam.
PLUMBING = re.compile(r"(^|/)(torch/_ops\.py|torch_guard\.py|triton/runtime/|"
                      r"torch/nn/modules/module\.py|torch/utils/_contextlib\.py|"
                      r"torch/_dynamo/|torch/_library/)")


class Resolver:
    def __init__(self, roots, packages):
        self.roots, self.packages, self.cache, self.qual = roots, packages, {}, {}

    def path(self, rel):
        if rel in self.cache:
            return self.cache[rel]
        found = rel if os.path.isabs(rel) and os.path.exists(rel) else None
        top = rel.lstrip("/").split("/", 1)[0]
        if found is None and top in self.packages:
            try:
                spec = importlib.util.find_spec(top)
            except (ImportError, ValueError):
                spec = None
            for loc in (spec.submodule_search_locations or []) if spec else []:
                cand = os.path.join(os.path.dirname(loc), rel)
                if os.path.exists(cand):
                    found = cand
                    break
        if found is None:
            for r in self.roots:
                cand = os.path.join(r, rel.lstrip("/"))
                if os.path.exists(cand):
                    found = cand
                    break
        self.cache[rel] = found
        return found

    def qualname(self, path, line, func):
        """Class.method for the function enclosing `line`, from the source."""
        key = (path, line)
        if key in self.qual:
            return self.qual[key]
        best = func
        try:
            tree = ast.parse(open(path).read())

            def walk(node, prefix):
                nonlocal best
                for ch in ast.iter_child_nodes(node):
                    if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        end = getattr(ch, "end_lineno", None) or ch.lineno
                        name = prefix + ch.name
                        if ch.lineno <= line <= end:
                            if not isinstance(ch, ast.ClassDef) and ch.name == func:
                                best = name
                            walk(ch, name + ".")
            walk(tree, "")
        except (OSError, SyntaxError, ValueError):
            pass
        self.qual[key] = best
        return best


def module_of(rel):
    m = rel.lstrip("/")
    for marker in ("site-packages/", "dist-packages/"):
        if marker in m:
            m = m.split(marker, 1)[1]
    return m[:-3].replace("/", ".").removesuffix(".__init__")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frames_of(kernel_entry, packages):
    """All distinct serving-stack frame stacks seen for one kernel, innermost first, as
    (stack, graph_replayed). A replayed launch's stack names only the graph runner."""
    stacks = []
    for v in kernel_entry["variants"]:
        for s in v["samples"]:
            replayed = bool(s.get("graph_replayed"))
            st = []
            for f in s.get("py_frames") or []:
                m = FRAME_RE.match(f)
                if not m or PLUMBING.search(m.group("path")):
                    continue
                if m.group("path").lstrip("/").split("/", 1)[0] not in packages:
                    continue
                st.append((m.group("path"), int(m.group("line")), m.group("func")))
            if st and (st, replayed) not in stacks:
                stacks.append((st, replayed))
    return stacks


def build(shapes_files, roots, packages):
    res = Resolver(roots, packages)
    files, kernels, ops = {}, {}, defaultdict(set)
    stacks_by_kernel = defaultdict(list)
    replayed_only = set()
    for p in shapes_files:
        d = json.load(open(p))
        for name, entry in d["kernels"].items():
            for st in frames_of(entry, packages):
                if st not in stacks_by_kernel[name]:
                    stacks_by_kernel[name].append(st)
            for v in entry["variants"]:
                for s in v["samples"]:
                    if s.get("host_op"):
                        ops[name].add(s["host_op"]["op"])

    def frame_rec(path, line, func):
        full = res.path(path)
        rec = {"frame": "%s(%d): %s" % (path, line, func), "module": module_of(path),
               "line": line, "resolved": bool(full)}
        if full:
            q = res.qualname(full, line, func)
            rec.update(path=full, target_callable="%s:%s" % (module_of(path), q))
            if full not in files:
                files[full] = sha256(full)
            rec["sha256"] = files[full]
        else:
            rec["target_callable"] = "%s:%s" % (module_of(path), func)
        return rec

    for name, stacks in stacks_by_kernel.items():
        # An eager launch names the op and the source; a graph replay names only the
        # runner. Use the first eager stack when any manifest has one.
        eager = [st for st, rep in stacks if not rep]
        inner = (eager or [st for st, _ in stacks] or [[]])[0]
        if not eager:
            replayed_only.add(name)
        kernels[name] = {
            "launching_ops": sorted(ops.get(name, ())),
            "stacks_seen": len(stacks), "graph_replayed_only": not eager,
            "frames": [frame_rec(*f) for f in inner],
            "launcher": frame_rec(*inner[0]) if inner else None,
        }
    for name in ops:
        kernels.setdefault(name, {"launching_ops": sorted(ops[name]), "stacks_seen": 0,
                                  "frames": [], "launcher": None})
    return {"contract": "kernel_extract/source_map/v1", "packages": list(packages),
            "graph_replayed_only": sorted(replayed_only),
            "kernels": kernels, "files": files,
            "unresolved_frames": sorted({f["frame"] for k in kernels.values()
                                         for f in k["frames"] if not f["resolved"]})}


def common_seam(source_map, names):
    """Innermost frame every kernel in `names` passes through, or None."""
    if any((source_map["kernels"].get(n) or {}).get("graph_replayed_only") for n in names):
        return None
    stacks = [[(f["module"], f["target_callable"]) for f in
               source_map["kernels"].get(n, {}).get("frames", [])] for n in names]
    if not stacks or not all(stacks):
        return None
    others = [set(s) for s in stacks[1:]]
    for key in stacks[0]:
        if all(key in o for o in others):
            return next(f for f in source_map["kernels"][names[0]]["frames"]
                        if (f["module"], f["target_callable"]) == key)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shapes-dir", required=True, help="kernel_shapes.py outputs")
    ap.add_argument("--stack-package", action="append", default=None,
                    help="top-level package counted as the serving stack (repeatable)")
    ap.add_argument("--root", action="append", default=[],
                    help="directory a relative frame path is resolved against (repeatable)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.shapes_dir, "*.json")))
    if not files:
        raise SystemExit("source_map: no shape manifests in %s" % a.shapes_dir)
    out = build(files, a.root, tuple(a.stack_package or DEFAULT_PACKAGES))
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for name, k in sorted(out["kernels"].items()):
        l = k["launcher"]
        print("  %-50s %s" % (name[:50], l["target_callable"] if l else "(no serving-stack frame)"))
    if out["unresolved_frames"]:
        print("  %d frame(s) not resolved to an installed file (run inside the serving "
              "container, or pass --root)" % len(out["unresolved_frames"]))
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
