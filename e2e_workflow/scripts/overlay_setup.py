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
  check         print where a module resolves from (run with the overlay on PYTHONPATH)
                --module sglang.srt.layers.activation

Back-compat aliases: `monkeypatch` == add-rebind, `copy-subtree` == add-module (file granularity).

NATIVE apply-back (compiled-source kernels: .cu/.hip/.cpp/CK/...) — these CANNOT be deployed via the
non-invasive PYTHONPATH overlay above (they need a recompiled .so in the install). They are applied
IN PLACE to the install and tracked in the same manifest's "natives" section so a single `revert`
undoes both the PYTHONPATH overlay AND the in-place native changes. This helper only does the
safety-critical, framework-agnostic plumbing (back up → byte-exact restore → cache-dir snapshot →
fresh-build verify); it NEVER invents a build command — the caller (e2e_integrator agent) discovers
the install's own incremental build the same way benchmark_engineer does and passes it via --build-cmd.

  add-native    apply a compiled-source patch in place + (optionally) run an incremental rebuild
                --overlay O --target <install_src_file> (--patched-file F | --patch D)
                [--artifact A ...] [--invalidate-cache DIR ...] [--build-cmd "..."] [--build-cwd C]
  verify-native confirm each tracked artifact actually changed (fresh build happened, not a silent no-op)
                --overlay O
  revert        restore EVERYTHING (PYTHONPATH overlay is just dropped; native files/artifacts/caches
                restored byte-exact from backup). Idempotent. --overlay O
  gc-stale      scan a root for native overlays left applied by a crashed run and revert them
                --root R

Stdlib only.
"""
import argparse, glob, hashlib, importlib, json, os, shlex, shutil, subprocess, sys, tarfile

SITECUSTOMIZE = r'''# Auto-generated reversible overlay (e2e_workflow). Drop this dir from PYTHONPATH to revert.
import json, os, sys, importlib, importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAN = os.path.join(_HERE, "_overlay_manifest.json")
try:
    with open(_MAN) as _fh:
        _m = json.load(_fh)
except Exception as _e:
    _m = {"modules": [], "rebinds": [], "captures": []}

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

# (c) capture hooks (shape/IO oracle recording).
for _e in _m.get("captures", []):
    try:
        import capture_shapes
        capture_shapes.install(_e["target"], _e["out"], int(_e.get("max", 5)))
    except Exception as _ex:
        sys.stderr.write("[overlay] capture install FAILED %r: %r\n" % (_e, _ex))
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


def _ensure_overlay(overlay):
    os.makedirs(overlay, exist_ok=True)
    sc = os.path.join(overlay, "sitecustomize.py")
    if not os.path.exists(sc):
        with open(sc, "w") as fh:
            fh.write(SITECUSTOMIZE)
    man = os.path.join(overlay, "_overlay_manifest.json")
    if not os.path.exists(man):
        with open(man, "w") as fh:
            json.dump({"modules": [], "rebinds": [], "captures": [], "natives": []}, fh, indent=2)
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
    man = _ensure_overlay(a.overlay)
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
    man = _ensure_overlay(a.overlay)
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
    man = _ensure_overlay(a.overlay)
    cap = a.capture_file or os.path.join(os.path.dirname(os.path.abspath(__file__)), "capture_shapes.py")
    shutil.copy2(cap, os.path.join(a.overlay, "capture_shapes.py"))
    m = _load_man(man)
    m["captures"] = [e for e in m.get("captures", []) if e["target"] != a.target]
    m["captures"].append({"target": a.target, "out": a.out, "max": a.max})
    _save_man(man, m)
    print(f"OVERLAY_DIR={a.overlay}")
    print(f"add-capture {a.target} -> {a.out}")
    print(f"launch with: PYTHONPATH={a.overlay}:$PYTHONPATH")


def cmd_check(a):
    f = module_file(a.module)
    print(f"{a.module} -> {f}")
    print("OVERLAY_ACTIVE" if os.sep + "_patched" + os.sep in f else
          ("INJECTED" if f.endswith(a.module + ".py") else "INSTALL (overlay not shadowing this module)"))


# ----------------------------------------------------------------------------------------------------
# NATIVE apply-back: in-place, reversible deploy of a COMPILED-source kernel patch.
# Framework-agnostic plumbing only — the caller discovers + passes the incremental build command.
# ----------------------------------------------------------------------------------------------------
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _native_backup_dir(overlay):
    d = os.path.join(overlay, "_native_backup")
    os.makedirs(d, exist_ok=True)
    return d


def cmd_add_native(a):
    """Apply a compiled-source patch in place (reversibly), optionally invalidating caches + rebuilding.

    Crash-safe ordering: every backup is created AND the manifest entry saved BEFORE the install is
    mutated, so a crash at any point leaves a manifest whose backups all exist -> revert/gc-stale can
    always restore byte-exact. Per-item restore is idempotent.
    """
    man = _ensure_overlay(a.overlay)
    bdir = _native_backup_dir(a.overlay)
    m = _load_man(man)
    m.setdefault("natives", [])
    n = len(m["natives"])

    tgt = os.path.abspath(a.target)
    if not os.path.exists(tgt):
        raise SystemExit(f"add-native: target source does not exist: {tgt}")
    if not (a.patched_file or a.patch):
        raise SystemExit("add-native: requires --patched-file or --patch")

    entry = {"sources": [], "artifacts": [], "caches": [],
             "build_cmds": [], "build_cwd": a.build_cwd or "", "verify": {}}

    # 1. back up the source FIRST (so the manifest never references a missing backup)
    src_bak = os.path.join(bdir, f"src_{n}_{os.path.basename(tgt)}")
    shutil.copy2(tgt, src_bak)
    entry["sources"].append({"install_path": tgt,
                             "backup": os.path.relpath(src_bak, a.overlay),
                             "sha256": _sha256(tgt)})

    # 2. back up named artifacts (pre-build); the first existing one is the fresh-build verify anchor
    for i, art in enumerate(a.artifact or []):
        art = os.path.abspath(art)
        if os.path.exists(art):
            art_bak = os.path.join(bdir, f"art_{n}_{i}_{os.path.basename(art)}")
            shutil.copy2(art, art_bak)
            sha = _sha256(art)
            entry["artifacts"].append({"path": art, "backup": os.path.relpath(art_bak, a.overlay),
                                       "sha256": sha, "existed": True})
            if not entry["verify"]:
                entry["verify"] = {"artifact": art, "pre_sha256": sha}
        else:
            entry["artifacts"].append({"path": art, "backup": "", "sha256": "", "existed": False})

    # 3. snapshot (tar) each cache dir we will invalidate, so revert restores it intact
    for i, cd in enumerate(a.invalidate_cache or []):
        cd = os.path.abspath(cd)
        existed = os.path.isdir(cd)
        rel_tar = ""
        if existed:
            tar_bak = os.path.join(bdir, f"cache_{n}_{i}.tar")
            with tarfile.open(tar_bak, "w") as tf:
                tf.add(cd, arcname=os.path.basename(cd))
            rel_tar = os.path.relpath(tar_bak, a.overlay)
        entry["caches"].append({"dir": cd, "backup_tar": rel_tar, "existed": existed})

    if a.build_cmd:
        entry["build_cmds"].append(shlex.split(a.build_cmd))

    # SAVE manifest BEFORE any mutation (the crash-safety point)
    m["natives"].append(entry)
    _save_man(man, m)

    # 4. MUTATE the install: write the patched source in place
    if a.patched_file:
        shutil.copy2(a.patched_file, tgt)
    elif a.patch and not _try_apply(a.patch, target_file=tgt):
        shutil.copy2(src_bak, tgt)  # roll back the one file we touched, then abort
        raise SystemExit(f"add-native: failed to apply patch {a.patch} to {tgt}")

    # delete invalidated caches to force a rebuild
    for c in entry["caches"]:
        if c["existed"] and os.path.isdir(c["dir"]):
            shutil.rmtree(c["dir"])

    # 5. run the caller-supplied incremental build (we never invent it)
    for cmd in entry["build_cmds"]:
        r = subprocess.run(cmd, cwd=(a.build_cwd or None), capture_output=True, text=True)
        sys.stderr.write(r.stdout + r.stderr)
        if r.returncode != 0:
            raise SystemExit(f"add-native: build failed ({' '.join(cmd)}) rc={r.returncode}")

    print(f"OVERLAY_DIR={a.overlay}")
    print(f"add-native {tgt}  (sources=1 artifacts={len(entry['artifacts'])} caches={len(entry['caches'])} build={'yes' if entry['build_cmds'] else 'no'})")


def cmd_verify_native(a):
    """Confirm each tracked artifact actually changed (or was created) — catches a silent no-op build."""
    man = os.path.join(a.overlay, "_overlay_manifest.json")
    if not os.path.exists(man):
        raise SystemExit("verify-native: no manifest")
    m = _load_man(man)
    ok, checked = True, 0
    for entry in m.get("natives", []):
        for art in entry.get("artifacts", []):
            p = art["path"]
            if not os.path.exists(p):
                print(f"FRESH_BUILD_FAIL {'missing-after-build' if art.get('existed') else 'not-created'} {p}")
                ok = False
                continue
            if art.get("existed") and _sha256(p) == art.get("sha256"):
                print(f"FRESH_BUILD_FAIL unchanged {p}")
                ok = False
            else:
                checked += 1
    print(("FRESH_BUILD_OK" if ok else "FRESH_BUILD_FAIL") + f" (checked {checked})")
    if not ok:
        raise SystemExit(2)


def cmd_revert(a):
    """Restore everything. PYTHONPATH overlay reverts by simply not being on PYTHONPATH; native changes
    are restored byte-exact from backup (sources + artifacts + cache dirs), in reverse apply order.
    Idempotent: each item restores from its own backup, and the natives list is cleared only on full success."""
    man = os.path.join(a.overlay, "_overlay_manifest.json")
    if not os.path.exists(man):
        print("revert: no manifest; nothing to do")
        return
    m = _load_man(man)
    for entry in reversed(m.get("natives", [])):
        for c in entry.get("caches", []):
            d = c["dir"]
            if os.path.isdir(d):
                shutil.rmtree(d)
            if c.get("existed") and c.get("backup_tar"):
                tarp = os.path.join(a.overlay, c["backup_tar"])
                if os.path.exists(tarp):
                    with tarfile.open(tarp) as tf:
                        tf.extractall(os.path.dirname(d))
            # existed=False -> candidate-built cache: leave it removed
        for art in entry.get("artifacts", []):
            if art.get("existed") and art.get("backup"):
                bak = os.path.join(a.overlay, art["backup"])
                if os.path.exists(bak):
                    shutil.copy2(bak, art["path"])
            elif not art.get("existed") and os.path.exists(art["path"]):
                os.remove(art["path"])  # candidate created it -> remove
        for s in entry.get("sources", []):
            bak = os.path.join(a.overlay, s["backup"])
            if os.path.exists(bak):
                shutil.copy2(bak, s["install_path"])
    m["natives"] = []
    _save_man(man, m)
    print(f"revert: restored native changes in {a.overlay}")


def cmd_gc_stale(a):
    """Scan a root for native overlays left applied by a crashed run (manifest has a non-empty natives
    list) and revert each — so a fresh session never starts on a dirty install."""
    found = 0
    for man in glob.glob(os.path.join(a.root, "**", "_overlay_manifest.json"), recursive=True):
        try:
            m = _load_man(man)
        except Exception:
            continue
        if m.get("natives"):
            cmd_revert(argparse.Namespace(overlay=os.path.dirname(man)))
            found += 1
    print(f"gc-stale: reverted {found} stale native overlay(s) under {a.root}")


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
        p.set_defaults(func=_dispatch_add_module)

    for name in ("add-rebind", "monkeypatch"):
        p = sub.add_parser(name)
        p.add_argument("--overlay", required=True)
        p.add_argument("--target", required=True, help="module:attr to rebind")
        p.add_argument("--impl-module", required=True, dest="impl_module")
        p.add_argument("--impl-attr", required=True, dest="impl_attr")
        p.add_argument("--impl-file", default="", dest="impl_file")
        p.set_defaults(func=cmd_add_rebind)

    p = sub.add_parser("add-capture")
    p.add_argument("--overlay", required=True)
    p.add_argument("--target", required=True, help="module:attr to hook")
    p.add_argument("--out", required=True, help="task dir to flush reference_io.pt + meta.json into")
    p.add_argument("--max", type=int, default=5)
    p.add_argument("--capture-file", default="", dest="capture_file")
    p.set_defaults(func=cmd_add_capture)

    p = sub.add_parser("check")
    p.add_argument("--module", required=True)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("add-native")
    p.add_argument("--overlay", required=True)
    p.add_argument("--target", required=True, help="absolute path of the install source file to patch")
    p.add_argument("--patched-file", default="", dest="patched_file", help="full replacement source")
    p.add_argument("--patch", default="", help="unified diff to apply to --target")
    p.add_argument("--artifact", action="append", default=[], help="built artifact(s) to snapshot for revert/verify (repeatable)")
    p.add_argument("--invalidate-cache", action="append", default=[], dest="invalidate_cache",
                   help="cache dir(s) to snapshot+delete so only the changed unit rebuilds (repeatable)")
    p.add_argument("--build-cmd", default="", dest="build_cmd", help="caller-discovered incremental build command")
    p.add_argument("--build-cwd", default="", dest="build_cwd")
    p.set_defaults(func=cmd_add_native)

    p = sub.add_parser("verify-native")
    p.add_argument("--overlay", required=True)
    p.set_defaults(func=cmd_verify_native)

    p = sub.add_parser("revert")
    p.add_argument("--overlay", required=True)
    p.set_defaults(func=cmd_revert)

    p = sub.add_parser("gc-stale")
    p.add_argument("--root", required=True)
    p.set_defaults(func=cmd_gc_stale)

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
