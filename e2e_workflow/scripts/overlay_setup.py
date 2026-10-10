#!/usr/bin/env python3
"""Build a reversible, COMPOUNDING overlay for an installed package — without editing site-packages.

Why not "copy a subtree + __init__.py onto PYTHONPATH": a regular package (one with __init__.py) on an
earlier path entry FULLY shadows the install — Python does not merge regular packages across path
entries, so every sibling submodule disappears and `import sglang` breaks. The correct, reversible
mechanism is a `sitecustomize.py` (auto-run by Python at interpreter startup, before anything imports
the target) that either (a) injects a PATCHED submodule file into sys.modules under its dotted name,
or (b) imports the real module and REBINDS one attribute (monkeypatch), or (c) installs a capture
hook, or (d) registers a deferred installer for a multi-seam package. All are driven by a manifest
so multiple overlays COMPOUND (each accepted kernel appends).

Layout produced:
    <overlay>/sitecustomize.py          # generic, manifest-driven (idempotent)
    <overlay>/_overlay_manifest.json    # modules, rebinds, captures, markers, installers
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
  add-installer register a package's deferred import hooks (package files must already be in O)
                --overlay O --id NAME --module PACKAGE [--callable install]
                [--target MODULE:ATTR ...] [--kernel NAME ...]
  merge         compose complete overlays into a fresh directory, rejecting conflicting hooks/files
                --overlay NEW --from BASE --with INCOMING
  check         print where a module resolves from (run with the overlay on PYTHONPATH)
                --module sglang.srt.layers.activation [--path-only]
  Every add-* takes --from BASE to SEED a new overlay from an existing one, so a candidate overlay is
  "the live stack + ONE entry". Stacking by PYTHONPATH does NOT work: only the first sitecustomize on
  sys.path is imported, so a second overlay dir is silently dead. Use one composed directory.

Back-compat aliases: `monkeypatch` == add-rebind, `copy-subtree` == add-module (file granularity).
Stdlib only.
"""
import argparse, importlib, json, os, shutil, subprocess, sys, tempfile

SITECUSTOMIZE = r'''# Auto-generated reversible overlay (e2e_workflow). Drop this dir from PYTHONPATH to revert.
import json, os, sys, importlib, importlib.abc, importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAN = os.path.join(_HERE, "_overlay_manifest.json")
_EMPTY = {"modules": [], "rebinds": [], "markers": [], "captures": []}

# FORK-BOMB GUARD -- do not remove.
# The ROCm/HIP arch probes (rocm_agent_enumerator, rocminfo, offload-arch, hipconfig, ...) are
# themselves python scripts, and they inherit PYTHONPATH from whoever shells out to them. Installing
# an overlay whose hooks import a GPU library (e.g. `aiter`, which probes the arch at import) makes
# that child re-enter THIS file, import the library again, and shell out again -- an unbounded
# recursive process spawn that wedges the whole node (observed: 13.7k processes, load 590, every
# subsequent vLLM launch failing torch.distributed rendezvous with "2/8 clients joined").
# The overlay has no business running inside an arch probe, so no-op there.
_ARGV0 = (sys.argv[0] if sys.argv else "") or ""
_PROBES = {"rocm_agent_enumerator", "rocminfo", "offload-arch", "amdgpu-arch", "hipconfig", "hipcc",
           "hipinfo", "rocm-smi", "rocm_smi.py"}
if os.path.basename(_ARGV0) in _PROBES or os.path.abspath(_ARGV0 or ".").startswith(
        ("/opt/rocm", "/usr/local/rocm")):
    _m = dict(_EMPTY)
else:
    try:
        with open(_MAN) as _fh:
            _m = json.load(_fh)
    except Exception as _e:
        _m = dict(_EMPTY)

# (a) inject patched submodules under their dotted names, through the NORMAL import machinery.
#
# A meta_path FINDER, not an eager exec at interpreter start. Eager exec was wrong twice over:
#   * It executed the patched submodule BEFORE the interpreter had finished starting and before the
#     package's own __init__ ran, so the patched file's imports fired in a different ORDER than a real
#     import would have used. Library state that is built once at first-import (arch probes, tile/config
#     registries, JIT/codegen caches) then gets built from a different entry point. Observed on the
#     aiter FlyDSL MoE seam: the candidate leg (module entry -> eager exec) aborted inside the kernel
#     JIT compiler while the baseline leg (empty manifest -> no eager exec) compiled the same kernel.
#   * It left TWO live module objects for the same dotted name: the package __init__ later imported the
#     real submodule anyway, and any `from a.b import c` binding it made still pointed at the UNPATCHED
#     copy -- so part of the process silently kept calling the original code while the leg was labelled
#     "candidate". That corrupts both parity and speedup results.
# The finder claims the name at the moment something actually imports it: exactly one module object,
# created in the normal order, with the parent attribute bound by Python itself. With an EMPTY manifest
# the finder is still installed and simply claims nothing, so the baseline and candidate legs take
# BYTE-IDENTICAL startup paths -- which is what makes a two-leg timing comparison mean anything.
_MODS = {}
for _e in _m.get("modules", []):
    try:
        _MODS[_e["module"]] = os.path.join(_HERE, _e["file"])
    except Exception as _ex:
        sys.stderr.write("[overlay] bad module entry %r: %r\n" % (_e, _ex))


class _OverlayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, _name, _path=None, _target=None):
        _file = _MODS.get(_name)
        if _file is None:
            return None
        sys.stderr.write("[overlay] injected module %s <- %s\n" % (_name, _file))
        return importlib.util.spec_from_file_location(_name, _file)


sys.meta_path.insert(0, _OverlayFinder())
for _dotted in _MODS:
    if _dotted in sys.modules:   # imported before sitecustomize ran -> the overlay would be a no-op
        sys.stderr.write("[overlay] WARNING %s was already imported before the overlay finder\n" % _dotted)

# Manifest installers register deferred hooks without eagerly importing serving/GPU libraries.
# They run before rebinds, since a rebind can itself import a module the installer must observe.
for _e in _m.get("installers", []):
    try:
        _impl = importlib.import_module(_e["module"])
        getattr(_impl, _e.get("callable", "install"))()
        sys.stderr.write("[overlay] installed %s\n" % _e["id"])
    except Exception as _ex:
        sys.stderr.write("[overlay] installer FAILED %r: %r\n" % (_e, _ex))
        if _e.get("required", True):
            # CPython swallows sitecustomize exceptions and continues. A required installer
            # failure must not quietly turn a candidate into a stock-server measurement.
            sys.stderr.flush()
            os._exit(70)

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


# Previous generic runtime, accepted when upgrading an existing manifest-based overlay.
_INSTALLER_START = "# Manifest installers register deferred hooks"
_INSTALLER_END = "# (b) rebind single attributes (monkeypatch)."
LEGACY_SITECUSTOMIZE = (SITECUSTOMIZE.split(_INSTALLER_START)[0]
                       + _INSTALLER_END + SITECUSTOMIZE.split(_INSTALLER_END)[1])


def cmd_add_installer(a):
    """Register a source package's deferred installer; its files already live in the overlay."""
    man = _ensure_overlay(a.overlay, getattr(a, "base", ""))
    runtime = os.path.join(a.overlay, "sitecustomize.py")
    with open(runtime) as handle:
        if handle.read() not in (SITECUSTOMIZE, LEGACY_SITECUSTOMIZE):
            raise SystemExit("cannot add an installer to a custom sitecustomize; preserve its hooks first")
    m = _load_man(man)
    entry = {"id": a.id, "module": a.module, "callable": a.callable,
             "required": True, "targets": a.target, "kernels": a.kernel}
    old = [e for e in m.get("installers", []) if e.get("id") == a.id]
    if old and old != [entry]:
        raise SystemExit("installer id already has a different definition: " + a.id)
    if not old:
        m.setdefault("installers", []).append(entry)
    _save_man(man, m)
    with open(runtime, "w") as handle:
        handle.write(SITECUSTOMIZE)
    print("OVERLAY_DIR=" + a.overlay)


def _targets_overlap(left, right):
    left_module, _, left_attr = left.partition(":")
    right_module, _, right_attr = right.partition(":")
    if left_module == right_module:
        return (not left_attr or not right_attr or left_attr == right_attr
                or left_attr.startswith(right_attr + ".") or right_attr.startswith(left_attr + "."))
    return ((not left_attr and right_module.startswith(left_module + "."))
            or (not right_attr and left_module.startswith(right_module + ".")))


def merge_overlays(output, sources):
    """Build a fresh overlay without changing its inputs or silently replacing an existing hook."""
    output = os.path.abspath(output)
    if os.path.exists(output) and (not os.path.isdir(output) or os.listdir(output)):
        raise ValueError("merge output must be absent or empty: " + output)
    kinds = {"modules": "module", "rebinds": "target", "captures": "target",
             "markers": "target", "installers": "id"}
    merged, files, targets = {}, {}, {}
    for source in sources:
        if not source:
            continue
        source = os.path.abspath(source)
        if source == output or os.path.commonpath((source, output)) in (source, output):
            raise ValueError("merge inputs and output must not contain one another")
        with open(os.path.join(source, "_overlay_manifest.json")) as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError("overlay manifest must be an object: " + source)
        with open(os.path.join(source, "sitecustomize.py")) as handle:
            if handle.read() not in (SITECUSTOMIZE, LEGACY_SITECUSTOMIZE):
                raise ValueError("custom sitecustomize cannot be merged automatically: " + source)
        for kind, entries in manifest.items():
            if kind not in kinds:
                if kind in merged and merged[kind] != entries:
                    raise ValueError("conflicting manifest metadata: " + kind)
                merged[kind] = entries
                continue
            index = {e[kinds[kind]]: e for e in merged.get(kind, [])}
            for entry in entries:
                key = entry[kinds[kind]]
                if key in index and index[key] != entry:
                    raise ValueError("conflicting overlay entry: " + kind + ":" + key)
                index[key] = entry
                # Installers declare the serving sites they own so another overlay cannot
                # overwrite them by introducing a rebind or whole-module replacement.
                sites = entry.get("targets", []) if kind == "installers" else (
                    [key] if kind in ("modules", "rebinds") else [])
                owner = (kind, key)
                for site in sites:
                    for previous, prev_owner in targets.items():
                        if _targets_overlap(site, previous) and prev_owner != owner:
                            raise ValueError("conflicting overlay target: " + site)
                    targets[site] = owner
            merged[kind] = list(index.values())
        for root, dirs, names in os.walk(source):
            dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".torch_ext", ".git"))
            if any(os.path.islink(os.path.join(root, d)) for d in dirs):
                raise ValueError("overlay directory symlinks must be materialized before merge")
            for name in sorted(names):
                path = os.path.join(root, name)
                rel = os.path.relpath(path, source)
                if rel in ("sitecustomize.py", "_overlay_manifest.json") or name.endswith(".pyc"):
                    continue
                if os.path.islink(path):
                    raise ValueError("overlay file symlinks must be materialized before merge: " + rel)
                if rel in files:
                    with open(files[rel], "rb") as old, open(path, "rb") as new:
                        if old.read() != new.read():
                            raise ValueError("conflicting overlay source file: " + rel)
                files[rel] = path
    parent = os.path.dirname(output)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".overlay_merge_", dir=parent)
    try:
        for rel, source in files.items():
            dest = os.path.join(staging, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(source, dest)
        with open(os.path.join(staging, "_overlay_manifest.json"), "w") as handle:
            json.dump(merged, handle, indent=2)
        with open(os.path.join(staging, "sitecustomize.py"), "w") as handle:
            handle.write(SITECUSTOMIZE)
        os.replace(staging, output)
    finally:
        if os.path.isdir(staging):
            shutil.rmtree(staging)
    return output


def cmd_merge(a):
    try:
        print("OVERLAY_DIR=" + merge_overlays(a.overlay, [a.base, a.incoming]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(str(error))


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

    p = sub.add_parser("add-installer")
    p.add_argument("--overlay", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--module", required=True)
    p.add_argument("--callable", default="install")
    p.add_argument("--target", action="append", default=[])
    p.add_argument("--kernel", action="append", default=[])
    p.add_argument("--from", dest="base", default="")
    p.set_defaults(func=cmd_add_installer)

    p = sub.add_parser("merge")
    p.add_argument("--overlay", required=True)
    p.add_argument("--from", dest="base", default="")
    p.add_argument("--with", dest="incoming", required=True)
    p.set_defaults(func=cmd_merge)

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
