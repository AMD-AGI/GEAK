#!/usr/bin/env python3
"""Build a reversible, COMPOUNDING overlay for an installed package — without editing site-packages.

Why not "copy a subtree + __init__.py onto PYTHONPATH": a regular package (one with __init__.py) on an
earlier path entry FULLY shadows the install — Python does not merge regular packages across path
entries, so every sibling submodule disappears and `import sglang` breaks. The correct, reversible
mechanism is a `sitecustomize.py` (auto-run by Python at interpreter startup, before anything imports
the target) that either (a) injects a PATCHED submodule file into sys.modules under its dotted name,
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
  add-hook      run impl_module.impl_attr() lazily, right AFTER a module finishes importing
                (the ONLY safe kind for callbacks that touch the serving stack -- add-rebind
                imports eagerly at startup, which hangs the TP ranks)
                --overlay O --module vllm.v1.worker.gpu_model_runner
                --impl-module vllm_phase_annotate [--impl-attr install] [--impl-file F]
  add-vllm-semantic-capture
                wire scripts/vllm_semantic_capture.py (+ the capture engine and the shared
                phase rule) so a vLLM run logs per-op shapes. This is the EAGER PROBE that
                supplies decode shapes the cudagraph-replayed production trace cannot.
                Inert unless GEAK_SEMANTICS_CAPTURE=1.   --overlay O
  add-vllm-phase-annotation
                shorthand: wire scripts/vllm_phase_annotate.py so a vLLM trace carries the
                sglang-dialect step[EXTEND|DECODE ...] spans the fusion semantics layer needs.
                Inert unless GEAK_VLLM_PHASE_ANNOTATE=1.   --overlay O
  add-triton-launch-probe
                wire scripts/triton_launch_probe.py so every Triton launch made while a
                torch profiler records carries its tensor arguments in the trace (the only
                shape a Python-launched Triton kernel can get). Inert unless
                GEAK_TRITON_LAUNCH_SHAPES=1.   --overlay O [--from BASE]
  check         print where a module resolves from (run with the overlay on PYTHONPATH)
                --module sglang.srt.layers.activation [--path-only]
  Every add-* takes --from BASE to SEED a new overlay from an existing one, so a candidate overlay is
  "the live stack + ONE entry". Stacking by PYTHONPATH does NOT work: only the first sitecustomize on
  sys.path is imported, so a second overlay dir is silently dead. Seeding by copy is the only way.

Back-compat aliases: `monkeypatch` == add-rebind, `copy-subtree` == add-module (file granularity).
Stdlib only.
"""
import argparse, importlib, json, os, shutil, subprocess, sys

SITECUSTOMIZE = r'''# Auto-generated reversible overlay (e2e_workflow). Drop this dir from PYTHONPATH to revert.
import json, os, sys, importlib, importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAN = os.path.join(_HERE, "_overlay_manifest.json")
try:
    with open(_MAN) as _fh:
        _m = json.load(_fh)
except Exception as _e:
    _m = {"modules": [], "rebinds": [], "markers": [], "captures": [], "hooks": []}

# (a) inject patched submodules under their dotted names BEFORE anything imports them.
for _e in _m.get("modules", []):
    try:
        _dotted, _file = _e["module"], os.path.join(_HERE, _e["file"])
        _spec = importlib.util.spec_from_file_location(_dotted, _file)
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_dotted] = _mod
        _spec.loader.exec_module(_mod)
        # bind as attribute on the parent so both `from a.b import c` and `import a.b; a.b.c` see the patch.
        if "." in _dotted:
            _parent, _child = _dotted.rsplit(".", 1)
            try:
                setattr(importlib.import_module(_parent), _child, _mod)
            except Exception:
                pass
        sys.stderr.write("[overlay] injected module %s <- %s\n" % (_dotted, _file))
    except Exception as _ex:
        sys.stderr.write("[overlay] module inject FAILED %r: %r\n" % (_e, _ex))

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

# (e) LAZY post-import hooks: run <impl_module>.<impl_attr>() right AFTER a target module
# finishes importing. This is the only safe shape for anything that must touch the serving
# stack: section (b) above calls importlib.import_module() EAGERLY at interpreter startup,
# and an eager `import vllm...`/`import sglang...` there runs before the engine has built
# its distributed environment -- which HANGS every TP rank. Here nothing is imported until
# the engine itself imports the target, and a module that is never imported costs nothing.
#
# Mechanism: a meta_path finder that resolves the real spec, then wraps the loader's
# exec_module so the callback fires after the module body has executed. find_spec alone
# cannot do this -- it runs BEFORE the module exists.
_hooks = {}
for _e in _m.get("hooks", []):
    _hooks.setdefault(_e["module"], []).append((_e["impl_module"], _e["impl_attr"]))

if _hooks:
    def _run_hooks(_dotted, _entries):
        for _impl_module, _impl_attr in _entries:
            try:
                getattr(importlib.import_module(_impl_module), _impl_attr)()
            except Exception as _ex:
                sys.stderr.write("[overlay] hook %s.%s on %s FAILED: %r\n"
                                 % (_impl_module, _impl_attr, _dotted, _ex))

    class _GeakPostImportFinder:
        def __init__(self, hooks):
            self._hooks = hooks
            self._busy = set()

        def find_spec(self, fullname, path=None, target=None):
            _entries = self._hooks.get(fullname)
            # _busy guards the re-entrant find_spec below: without it our own lookup
            # would hit this finder again and recurse forever.
            if not _entries or fullname in self._busy:
                return None
            self._busy.add(fullname)
            try:
                _spec = importlib.util.find_spec(fullname)
            except Exception:
                _spec = None
            finally:
                self._busy.discard(fullname)
            if _spec is None or getattr(_spec, "loader", None) is None:
                return None
            _loader = _spec.loader
            _orig_exec = _loader.exec_module

            def _exec_module(_module, __orig=_orig_exec, __e=_entries, __n=fullname):
                __orig(_module)
                _run_hooks(__n, __e)

            try:
                _loader.exec_module = _exec_module
            except Exception:
                return None          # immutable loader -> decline, don't break the import
            return _spec

    sys.meta_path.insert(0, _GeakPostImportFinder(_hooks))
    # A target already imported before sitecustomize ran would never trip the finder.
    for _dotted, _entries in _hooks.items():
        if _dotted in sys.modules:
            _run_hooks(_dotted, _entries)
    sys.stderr.write("[overlay] armed %d post-import hook target(s): %s\n"
                     % (len(_hooks), ", ".join(sorted(_hooks))))
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
            json.dump({"modules": [], "rebinds": [], "markers": [], "captures": [], "hooks": []}, fh, indent=2)
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


def cmd_add_hook(a):
    """Register a lazy post-import hook: after MODULE imports, call IMPL_MODULE.IMPL_ATTR().

    Use this (never add-rebind) whenever the callback has to touch the serving stack, and
    for anything that must run after the engine has built the object it patches.
    """
    man = _ensure_overlay(a.overlay)
    if a.impl_file:
        shutil.copy2(a.impl_file, os.path.join(a.overlay, os.path.basename(a.impl_file)))
    m = _load_man(man)
    m.setdefault("hooks", [])
    m["hooks"] = [e for e in m["hooks"]
                  if not (e["module"] == a.module and e["impl_module"] == a.impl_module)]
    m["hooks"].append({"module": a.module, "impl_module": a.impl_module,
                       "impl_attr": a.impl_attr})
    _save_man(man, m)
    print(f"OVERLAY_DIR={a.overlay}")
    print(f"add-hook after import {a.module} -> {a.impl_module}.{a.impl_attr}()")
    print(f"launch with: PYTHONPATH={a.overlay}:$PYTHONPATH")


def cmd_add_vllm_phase_annotation(a):
    """One-liner for the KernelFusion capture overlay on vLLM.

    Wires scripts/vllm_phase_annotate.py as a post-import hook on the vLLM model runner so
    the trace carries sglang-dialect `step[EXTEND|DECODE ...]` spans. The hook stays inert
    until GEAK_VLLM_PHASE_ANNOTATE=1, so this overlay is safe to leave on a normal run.
    """
    impl = a.impl_file or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "vllm_phase_annotate.py")
    if not os.path.exists(impl):
        raise SystemExit(f"vllm_phase_annotate.py not found at {impl}")
    a.impl_file = impl
    a.impl_module = "vllm_phase_annotate"
    a.impl_attr = "install"
    cmd_add_hook(a)


def cmd_add_vllm_semantic_capture(a):
    """Wire the vLLM eager shape probe (GEAK Semantics 1.2) as a post-import hook.

    Unlike the phase-annotation overlay this needs THREE modules on the path -- the hook
    entry, the shared capture engine it drives, and the phase rule both it and the trace
    annotation read -- so copy them together. A partially-populated overlay fails at
    import time inside the server, where the only symptom is an empty shape log.

    Armed by the capture engine's own switch, GEAK_SEMANTICS_CAPTURE=1 (plus
    GEAK_SEMANTICS_SHAPE_LOG / _LAYERS / _PHASES). Inert otherwise.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    needed = ["vllm_semantic_capture.py", "semantic_runtime_capture.py",
              "vllm_phase_annotate.py"]
    missing = [n for n in needed if not os.path.exists(os.path.join(here, n))]
    if missing:
        raise SystemExit("missing capture module(s) beside overlay_setup.py: %s"
                         % ", ".join(missing))
    _ensure_overlay(a.overlay)
    for name in needed:
        shutil.copy2(os.path.join(here, name), os.path.join(a.overlay, name))
    a.impl_file = ""            # already copied, all three of them
    a.impl_module = "vllm_semantic_capture"
    a.impl_attr = "install"
    cmd_add_hook(a)


def cmd_add_triton_launch_probe(a):
    """Wire scripts/triton_launch_probe.py as a post-import hook on triton.runtime.jit.

    Every Triton launch made while a torch profiler records is then annotated with its
    tensor arguments, which is the only shape evidence a Python-launched Triton kernel
    can have (it has no dispatcher op). Inert unless GEAK_TRITON_LAUNCH_SHAPES=1.
    """
    impl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "triton_launch_probe.py")
    if not os.path.exists(impl):
        raise SystemExit(f"triton_launch_probe.py not found at {impl}")
    _ensure_overlay(a.overlay, a.base)
    a.impl_file = impl
    a.impl_module = "triton_launch_probe"
    a.impl_attr = "install"
    cmd_add_hook(a)


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

    p = sub.add_parser("add-hook")
    p.add_argument("--overlay", required=True)
    p.add_argument("--module", required=True,
                   help="dotted module whose import triggers the hook, "
                        "e.g. vllm.v1.worker.gpu_model_runner")
    p.add_argument("--impl-module", required=True, dest="impl_module")
    p.add_argument("--impl-attr", default="install", dest="impl_attr")
    p.add_argument("--impl-file", default="", dest="impl_file")
    p.set_defaults(func=cmd_add_hook)

    p = sub.add_parser("add-vllm-phase-annotation")
    p.add_argument("--overlay", required=True)
    p.add_argument("--module", default="vllm.v1.worker.gpu_model_runner")
    p.add_argument("--impl-file", default="", dest="impl_file")
    p.set_defaults(func=cmd_add_vllm_phase_annotation)

    p = sub.add_parser("add-vllm-semantic-capture")
    p.add_argument("--overlay", required=True)
    p.add_argument("--module", default="vllm.v1.worker.gpu_model_runner")
    p.set_defaults(func=cmd_add_vllm_semantic_capture)

    p = sub.add_parser("add-triton-launch-probe")
    p.add_argument("--overlay", required=True)
    p.add_argument("--module", default="triton.runtime.jit")
    p.add_argument("--from", dest="base", default="", help="seed the overlay from this existing overlay dir")
    p.set_defaults(func=cmd_add_triton_launch_probe)

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
