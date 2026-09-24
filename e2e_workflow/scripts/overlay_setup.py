#!/usr/bin/env python3
"""Build a reversible, COMPOUNDING overlay for an installed package — without editing site-packages.

Why not "copy a subtree + __init__.py onto PYTHONPATH": a regular package (one with __init__.py) on an
earlier path entry FULLY shadows the install — Python does not merge regular packages across path
entries, so every sibling submodule disappears and `import sglang` breaks. The correct, reversible
mechanism is a `sitecustomize.py` (auto-run by Python at interpreter startup, before anything imports
the target) that either (a) loads a PATCHED submodule file when its dotted name is first imported,
or (b) imports the real module and REBINDS one attribute (monkeypatch), or (c) installs a capture
hook. All three are driven by a manifest so multiple overlays COMPOUND (each accepted kernel appends).

Layout produced:
    <overlay>/sitecustomize.py          # generic, manifest-driven (idempotent)
    <overlay>/_overlay_manifest.json    # {"modules":[...], "rebinds":[...], "captures":[...]}
    <overlay>/_patched/<dotted>.py      # patched submodule sources (for module-inject entries)
    <overlay>/<impl files>              # copied impl modules (for rebind/capture entries)
Launch with:  PYTHONPATH=<overlay>:$PYTHONPATH

Commands:
  add-module    inject a patched submodule file in place of the installed one (whole-file source swap)
                --overlay O --module sglang.srt.layers.activation
                (--patched-file F  |  --patch D  [--src-file S])   # S defaults to the install's file
  add-rebind    rebind module:attr -> impl_module.impl_attr (single function/kernel swap; the default)
                --overlay O --target sglang.srt.layers.activation:silu_and_mul
                --impl-module fast_act --impl-attr fast_silu_and_mul [--impl-file fast_act.py]
  add-capture   install a shape/IO capture hook on module:attr (uses capture_shapes.py)
                --overlay O --target sglang...:fn --out <task_dir> [--max 5] [--capture-file capture_shapes.py]
  add-marker    install a marker-only hook on one candidate seam (uses seam_trace.py)
                --overlay O --target sglang...:fn [--marker-file seam_trace.py]
  check         print where a module resolves from (run with the overlay on PYTHONPATH)
                --module sglang.srt.layers.activation [--path-only]
  Every add-* takes --from BASE to SEED a new overlay from an existing one, so a candidate overlay is
  "the live stack + ONE entry". Stacking by PYTHONPATH does NOT work: only the first sitecustomize on
  sys.path is imported, so a second overlay dir is silently dead. Seeding by copy is the only way.

Back-compat aliases: `monkeypatch` == add-rebind, `copy-subtree` == add-module (file granularity).
Stdlib only.
"""
import argparse
import importlib
import json
import os
import shutil
import subprocess

SITECUSTOMIZE = r'''# Auto-generated reversible overlay (e2e_workflow). Drop this dir from PYTHONPATH to revert.
import json, os, sys, importlib, importlib.machinery, importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAN = os.path.join(_HERE, "_overlay_manifest.json")
try:
    with open(_MAN) as _fh:
        _m = json.load(_fh)
except Exception as _e:
    _m = {"modules": [], "rebinds": [], "markers": [], "captures": []}

# (a) register every patched submodule before rebind/capture/marker imports run.
# Python also starts sitecustomize in multiprocessing helpers and compiler probes.
# Those processes must not import an inference framework just because an overlay
# is on PYTHONPATH. Normal import machinery loads the requested replacement,
# binds its parent attribute, and removes a partial module if execution fails.
class _OverlayModuleLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        try:
            super().exec_module(module)
        except Exception as error:
            sys.stderr.write("[overlay] module inject FAILED %s <- %s: %r\n" % (self.name, self.path, error))
            raise
        sys.stderr.write("[overlay] injected module %s <- %s\n" % (self.name, self.path))


class _OverlayModuleFinder:
    def __init__(self, files):
        self.files = files
        self.overlay_root = _HERE

    def find_spec(self, fullname, path=None, target=None):
        filename = self.files.get(fullname)
        if filename is None:
            return None
        loader = _OverlayModuleLoader(fullname, filename)
        return importlib.util.spec_from_file_location(fullname, filename, loader=loader)


_module_files = {}
for _e in _m.get("modules", []):
    try:
        _name = _e["module"]
        _file = os.path.join(_HERE, _e["file"])
        if _name.split(".")[-1] == "__init__" or os.path.basename(_file) == "__init__.py":
            raise ValueError("package __init__.py replacement is unsupported")
        if _name in sys.modules and getattr(sys.modules[_name], "__file__", None) != _file:
            sys.stderr.write("[overlay] module registration FAILED %s: target already imported before overlay registration\n" % _name)
            continue
        _module_files[_name] = _file
    except Exception as _ex:
        sys.stderr.write("[overlay] module registration FAILED %r: %r\n" % (_e, _ex))
if _module_files:
    for _finder in sys.meta_path:
        if getattr(_finder, "overlay_root", None) == _HERE:
            _finder.files.update(_module_files)
            break
    else:
        sys.meta_path.insert(0, _OverlayModuleFinder(_module_files))

# (b) rebind single attributes (monkeypatch).
for _e in _m.get("rebinds", []):
    try:
        _modname, _attr = _e["target"].split(":")
        _t = importlib.import_module(_modname)
        _impl = importlib.import_module(_e["impl_module"])
        setattr(_t, _attr, getattr(_impl, _e["impl_attr"]))
        sys.stderr.write("[overlay] rebound %s -> %s.%s\n" % (_e["target"], _e["impl_module"], _e["impl_attr"]))
    except Exception as _ex:
        sys.stderr.write("[overlay] rebind FAILED %r: %r\n" % (_e, _ex))

# (c) capture hooks (shape/IO oracle recording) go on FIRST, so the capture wrapper is the innermost
# stand-in and is already bound before any marker install imports a module that does
# `from <capture target module> import <attr>` (which would otherwise alias the un-captured function).
for _e in _m.get("captures", []):
    try:
        import capture_shapes
        capture_shapes.install(_e["target"], _e["out"], int(_e.get("max", 5)))
    except Exception as _ex:
        sys.stderr.write("[overlay] capture install FAILED %r: %r\n" % (_e, _ex))

# (d) marker-only hooks used to compare every candidate seam in one trace.
for _e in _m.get("markers", []):
    try:
        import seam_trace
        seam_trace.install(_e["target"])
    except Exception as _ex:
        sys.stderr.write("[overlay] seam marker install FAILED %r: %r\n" % (_e, _ex))
'''


def pkg_root(package):
    mod = importlib.import_module(package)
    f = getattr(mod, "__file__", None)
    if f:
        return os.path.dirname(f)
    p = list(getattr(mod, "__path__", []))
    if not p:
        raise SystemExit(f"cannot locate package root for {package}")
    return p[0]


def module_file(dotted):
    """Absolute path of the installed file backing a dotted module name."""
    spec = importlib.util.find_spec(dotted)
    if not spec or not spec.origin or spec.origin == "namespace":
        raise SystemExit(f"cannot find a file for module {dotted}")
    return spec.origin


def _ensure_overlay(overlay, base=""):
    # Two overlay dirs on PYTHONPATH do NOT compound (only the first sitecustomize is imported), so
    # --from BASE stacks by seeding a copy.
    if base and not os.path.exists(os.path.join(overlay, "_overlay_manifest.json")):
        if os.path.isdir(base):
            shutil.copytree(base, overlay, dirs_exist_ok=True)
    os.makedirs(overlay, exist_ok=True)
    sc = os.path.join(overlay, "sitecustomize.py")
    if not os.path.exists(sc):
        with open(sc, "w") as fh:
            fh.write(SITECUSTOMIZE)
    man = os.path.join(overlay, "_overlay_manifest.json")
    if not os.path.exists(man):
        with open(man, "w") as fh:
            json.dump({"modules": [], "rebinds": [], "markers": [], "captures": []}, fh, indent=2)
    return man


def _load_man(man):
    with open(man) as fh:
        return json.load(fh)


def _save_man(man, m):
    with open(man, "w") as fh:
        json.dump(m, fh, indent=2)


def _try_apply(patch, target_file=None, cwd=None):
    """Apply a unified diff. If target_file given, try patching that exact file directly first."""
    attempts = []
    if target_file:
        attempts += [["patch", target_file, "-i", patch],
                     ["git", "apply", "--unsafe-paths", f"--directory={os.path.dirname(target_file)}", patch]]
    if cwd:
        attempts += [["git", "apply", patch], ["patch", "-p1", "-i", patch]]
    for args in attempts:
        try:
            r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
            if r.returncode == 0:
                return True
        except FileNotFoundError:
            continue
    return False


def cmd_add_module(a):
    for filename in (a.patched_file, getattr(a, "src_file", "")):
        if filename and os.path.basename(filename) == "__init__.py":
            raise SystemExit("add-module does not support package __init__.py replacement")
    search_path = None
    spec = None
    for index, part in enumerate(a.module.split("."), 1):
        spec = importlib.machinery.PathFinder.find_spec(".".join(a.module.split(".")[:index]), search_path)
        if spec is None:
            break
        search_path = spec.submodule_search_locations
        if search_path is None:
            break
    if a.module.split(".")[-1] == "__init__" or (spec and spec.submodule_search_locations is not None):
        raise SystemExit("add-module does not support package __init__.py replacement")
    man = _ensure_overlay(a.overlay, getattr(a, "base", ""))
    patched_dir = os.path.join(a.overlay, "_patched")
    os.makedirs(patched_dir, exist_ok=True)
    dst = os.path.join(patched_dir, a.module + ".py")
    if a.patched_file:
        shutil.copy2(a.patched_file, dst)
    else:
        src = a.src_file or module_file(a.module)
        shutil.copy2(src, dst)
        if a.patch and not _try_apply(a.patch, target_file=dst):
            raise SystemExit(f"failed to apply patch {a.patch} to {dst}")
    m = _load_man(man)
    m["modules"] = [e for e in m.get("modules", []) if e["module"] != a.module]
    m["modules"].append({"module": a.module, "file": os.path.join("_patched", a.module + ".py")})
    _save_man(man, m)
    print(f"OVERLAY_DIR={a.overlay}")
    print(f"add-module {a.module} -> {dst}")
    print(f"launch with: PYTHONPATH={a.overlay}:$PYTHONPATH")


def cmd_add_rebind(a):
    man = _ensure_overlay(a.overlay, getattr(a, "base", ""))
    if a.impl_file:
        shutil.copy2(a.impl_file, os.path.join(a.overlay, os.path.basename(a.impl_file)))
    m = _load_man(man)
    m["rebinds"] = [e for e in m.get("rebinds", []) if e["target"] != a.target]
    m["rebinds"].append({"target": a.target, "impl_module": a.impl_module, "impl_attr": a.impl_attr})
    _save_man(man, m)
    print(f"OVERLAY_DIR={a.overlay}")
    print(f"add-rebind {a.target} -> {a.impl_module}.{a.impl_attr}")
    print(f"launch with: PYTHONPATH={a.overlay}:$PYTHONPATH")


def cmd_add_capture(a):
    man = _ensure_overlay(a.overlay, getattr(a, "base", ""))
    cap = a.capture_file or os.path.join(os.path.dirname(os.path.abspath(__file__)), "capture_shapes.py")
    shutil.copy2(cap, os.path.join(a.overlay, "capture_shapes.py"))
    m = _load_man(man)
    m["captures"] = [e for e in m.get("captures", []) if e["target"] != a.target]
    m["captures"].append({"target": a.target, "out": a.out, "max": a.max})
    _save_man(man, m)
    print(f"OVERLAY_DIR={a.overlay}")
    print(f"add-capture {a.target} -> {a.out}")
    print(f"launch with: PYTHONPATH={a.overlay}:$PYTHONPATH")


def cmd_add_marker(a):
    man = _ensure_overlay(a.overlay, getattr(a, "base", ""))
    marker = a.marker_file or os.path.join(os.path.dirname(os.path.abspath(__file__)), "seam_trace.py")
    shutil.copy2(marker, os.path.join(a.overlay, "seam_trace.py"))
    m = _load_man(man)
    m["markers"] = [e for e in m.get("markers", []) if e["target"] != a.target]
    m["markers"].append({"target": a.target})
    _save_man(man, m)
    print(f"OVERLAY_DIR={a.overlay}")
    print(f"add-marker {a.target}")
    print(f"launch with: PYTHONPATH={a.overlay}:$PYTHONPATH")


def cmd_check(a):
    f = module_file(a.module)
    if getattr(a, "path_only", False):
        print(f)
        return
    print(f"{a.module} -> {f}")
    print("OVERLAY_ACTIVE" if os.sep + "_patched" + os.sep in f else
          ("INJECTED" if f.endswith(a.module + ".py") else "INSTALL (overlay not shadowing this module)"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("add-module", "copy-subtree"):
        p = sub.add_parser(name)
        p.add_argument("--overlay", required=True)
        p.add_argument("--module", help="dotted module name to replace, e.g. sglang.srt.layers.activation")
        # back-compat: copy-subtree used --package/--subpath; accept and convert.
        p.add_argument("--package", default="")
        p.add_argument("--subpath", default="")
        p.add_argument("--patched-file", default="")
        p.add_argument("--src-file", default="")
        p.add_argument("--patch", default="")
        p.add_argument("--from", dest="base", default="", help="seed the overlay from this existing overlay dir")
        p.set_defaults(func=_dispatch_add_module)

    for name in ("add-rebind", "monkeypatch"):
        p = sub.add_parser(name)
        p.add_argument("--overlay", required=True)
        p.add_argument("--target", required=True, help="module:attr to rebind")
        p.add_argument("--impl-module", required=True, dest="impl_module")
        p.add_argument("--impl-attr", required=True, dest="impl_attr")
        p.add_argument("--impl-file", default="", dest="impl_file")
        p.add_argument("--from", dest="base", default="", help="seed the overlay from this existing overlay dir")
        p.set_defaults(func=cmd_add_rebind)

    p = sub.add_parser("add-capture")
    p.add_argument("--overlay", required=True)
    p.add_argument("--target", required=True, help="module:attr to hook")
    p.add_argument("--out", required=True, help="task dir to flush reference_io.pt + meta.json into")
    p.add_argument("--max", type=int, default=5)
    p.add_argument("--capture-file", default="", dest="capture_file")
    p.add_argument("--from", dest="base", default="", help="seed the overlay from this existing overlay dir")
    p.set_defaults(func=cmd_add_capture)

    p = sub.add_parser("add-marker")
    p.add_argument("--overlay", required=True)
    p.add_argument("--target", required=True, help="module:attr to mark without capturing I/O")
    p.add_argument("--marker-file", default="", dest="marker_file")
    p.add_argument("--from", dest="base", default="", help="seed the overlay from this existing overlay dir")
    p.set_defaults(func=cmd_add_marker)

    p = sub.add_parser("check")
    p.add_argument("--module", required=True)
    p.add_argument("--path-only", action="store_true", dest="path_only")
    p.set_defaults(func=cmd_check)

    a = ap.parse_args()
    a.func(a)


def _dispatch_add_module(a):
    # Convert legacy copy-subtree --package/--subpath into a dotted --module if needed.
    if not a.module and a.package and a.subpath:
        sub = a.subpath[:-3] if a.subpath.endswith(".py") else a.subpath
        a.module = a.package + "." + sub.replace(os.sep, ".")
    if not a.module:
        raise SystemExit("add-module requires --module (or legacy --package + --subpath)")
    cmd_add_module(a)


if __name__ == "__main__":
    main()
