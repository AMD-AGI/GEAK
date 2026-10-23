#!/usr/bin/env python3
"""Layer every cold-re-verified kernel into **one** reversible overlay, and verify that it
actually takes effect.

Two things worth stating plainly:

* **The overlays must be chain-seeded, not mounted side by side.** `PYTHONPATH=a:b` only
  imports the first sitecustomize on sys.path, and the second overlay directory silently
  has no effect — you end up with an overlay that installed only one kernel, with nothing
  to show for it. So each added kernel passes `--from` the previous overlay.
* **How a kernel binds back is not guessed here.** The extraction skill already recorded
  `{kind, target|module, impl_module, impl_attr, file}` in meta.json["candidate_bind"] —
  the same descriptor it uses for its own candidate leg. This copies it, replacing only
  `file` with the optimized kernel_src.

Usage:
    python3 build_overlay.py --optimized <optimized.json> --tasks-dir <dir> \
                             --overlay <out_dir> [--manifest <out.json>]
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from source_tree import copy_source_tree, source_files


def find_overlay_setup(geak_root, tasks_dir):
    """Prefer the copy shipped with the task directory (same version as at extraction
    time), falling back to the one in the repository."""
    cands = [os.path.join(tasks_dir, "overlay_setup.py"),
             os.path.join(geak_root, "e2e_workflow", "scripts", "overlay_setup.py")]
    for t in sorted(os.listdir(tasks_dir)) if os.path.isdir(tasks_dir) else []:
        cands.insert(0, os.path.join(tasks_dir, t, "overlay_setup.py"))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def read_check(out):
    """(file, verdict) from `overlay_setup.py check`, which prints a "<module> -> <file>"
    line followed by one of OVERLAY_ACTIVE / INJECTED / INSTALL."""
    where, verdict = "", ""
    for line in out.splitlines():
        line = line.strip()
        if " -> " in line:
            where = line.split(" -> ", 1)[1]
        elif line in ("OVERLAY_ACTIVE", "INJECTED") or line.startswith("INSTALL"):
            verdict = line.split()[0]
    return where, verdict


def editable_src(ksrc_dir):
    """The one editable file in kernel_src/, chosen by what it is.

    Sorting the directory and taking the first entry picks `__pycache__` as soon as one
    attempt has run -- a directory, which the overlay then installs as the kernel."""
    if not os.path.isdir(ksrc_dir):
        return None
    names = [n for n in sorted(os.listdir(ksrc_dir))
             if os.path.isfile(os.path.join(ksrc_dir, n)) and not n.startswith("__")]
    py = [n for n in names if n.endswith(".py")]
    pick = (py or names or [None])[0]
    return os.path.join(ksrc_dir, pick) if pick else None


def add_entry(tool, overlay, base, bind, src):
    if bind.get("kind") == "rebind":
        args = ["add-rebind", "--target", bind["target"],
                "--impl-module", bind["impl_module"], "--impl-attr", bind["impl_attr"],
                "--impl-file", src]
    else:
        args = ["add-module", "--module", bind["module"], "--patched-file", src]
    cmd = [sys.executable, tool] + args + ["--overlay", overlay]
    if base:
        cmd += ["--from", base]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def bundle_module_sources(overlay, task, bind, src, source_env=None):
    """Keep the bound module beside its complete, task-specific computation tree.

    The overlay and reference harness can both be relocated after this step. Relative
    __file__ lookups must stay inside this bundle, never find another task's leaves.
    """
    overlay, task, src = Path(overlay), Path(task), Path(src)
    relative = src.resolve().relative_to(task.resolve())
    bundle = overlay / "_patched" / task.name
    copy_source_tree(task / "kernel_src", bundle / "kernel_src")
    delivered = bundle / relative
    if not delivered.exists():
        delivered.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, delivered)
    manifest_path = overlay / "_overlay_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if bind.get("kind") == "rebind":
        # Re-export the unchanged entry from its complete bundle. This preserves
        # __file__-relative companion lookups when the harness relocates an overlay.
        wrapper = overlay / os.path.basename(src)
        if wrapper.stem != bind["impl_module"]:
            raise ValueError("rebind impl_module must match its entry filename")
        if not bind['impl_module'].isidentifier() or not bind['impl_attr'].isidentifier():
            raise ValueError('rebind implementation names must be plain Python identifiers')
        relative_entry = delivered.relative_to(overlay).as_posix()
        name = "_geak_applyback_" + hashlib.sha256(
            (task.name + ':' + bind['target']).encode()).hexdigest()[:20]
        wrapper.write_text(
            'import importlib.util as _iu, os as _os, sys as _sys\n'
            'from pathlib import Path as _Path\n'
            '_root = _Path(__file__).resolve().parent\n'
            f'_env = {dict(source_env or {})!r}\n'
            'for _key, _relative in _env.items():\n'
            '    _value = (_root / _relative).resolve(strict=True)\n'
            '    if _root not in _value.parents:\n'
            '        raise RuntimeError("source environment escaped accepted overlay")\n'
            '    _os.environ[_key] = str(_value)\n'
            f'_spec = _iu.spec_from_file_location({name!r}, _root / {relative_entry!r})\n'
            '_impl = _iu.module_from_spec(_spec)\n'
            '_sys.modules[_spec.name] = _impl\n'
            '_spec.loader.exec_module(_impl)\n'
            f'{bind["impl_attr"]} = getattr(_impl, {bind["impl_attr"]!r})\n'
        )
        return relative_entry
    matches = [m for m in manifest.get("modules", []) if m.get("module") == bind["module"]]
    if len(matches) != 1:
        raise ValueError("overlay manifest does not uniquely name " + bind["module"])
    old_file = overlay / matches[0]["file"]
    matches[0]["file"] = delivered.relative_to(overlay).as_posix()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if old_file != delivered:
        old_file.unlink()
    return delivered.relative_to(overlay).as_posix()


def source_environment(task, bind, options):
    """Resolve declared source-root hooks into the accepted task bundle."""
    declared = dict(bind.get('requires_env') or {})
    declared.update(options.get('source_env') or {})
    result = {}
    bundle = Path('_patched') / Path(task).name
    for key, value in declared.items():
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) or not isinstance(value, str):
            raise ValueError('source_env requires environment names and path templates')
        relative = value.replace('<task>', '.').replace('{task}', '.').replace('{kernel_src}', 'kernel_src')
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('source_env must remain inside the accepted task bundle')
        if not (Path(task) / path).is_dir():
            raise ValueError('source_env directory is missing: ' + value)
        result[key] = (bundle / path).as_posix()
    return result


def install_source_environment(overlay, applied):
    """Set all relocated source roots before the overlay imports any patched module."""
    values = {key: value for entry in applied
              for key, value in entry.get('source_env', {}).items()}
    if not values:
        return
    path = Path(overlay) / 'sitecustomize.py'
    prelude = (
        '# GEAK applyback: resolve source hooks before patched imports.\n'
        'import os as _geak_source_os\n'
        'from pathlib import Path as _GeakSourcePath\n'
        '_geak_source_root = _GeakSourcePath(__file__).resolve().parent\n'
        f'_geak_source_env = {values!r}\n'
        'for _geak_key, _geak_relative in _geak_source_env.items():\n'
        '    _geak_value = (_geak_source_root / _geak_relative).resolve(strict=True)\n'
        '    if _geak_source_root not in _geak_value.parents or not _geak_value.is_dir():\n'
        '        raise RuntimeError("source environment escaped accepted overlay or is not a directory")\n'
        '    _geak_source_os.environ[_geak_key] = str(_geak_value)\n\n'
    )
    path.write_text(prelude + path.read_text())


def check_rebind(target, overlay, env):
    """Local callable identity check, explicitly not serving-rank source proof."""
    probe = '''import hashlib, importlib, inspect, json, pathlib, sys
module, attribute = sys.argv[1].split(':', 1)
value = importlib.import_module(module)
for part in attribute.split('.'):
    value = getattr(value, part)
seen = set()
while hasattr(value, 'fn') and id(value) not in seen:
    seen.add(id(value)); value = value.fn
file = pathlib.Path(inspect.getfile(value)).resolve()
root = pathlib.Path(sys.argv[2]).resolve()
print(json.dumps({'file':str(file), 'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                  'inside_overlay':root in file.parents, 'callable':callable(value)}))
'''
    process = subprocess.run([sys.executable, '-c', probe, target, overlay],
                             capture_output=True, text=True, env=env)
    identity = {}
    try:
        identity = json.loads([line for line in process.stdout.splitlines() if line.startswith('{')][-1])
    except (ValueError, IndexError):
        pass
    return {'target': target, 'resolves_to': identity.get('file', ''),
            'shadowed': process.returncode == 0 and identity.get('inside_overlay') is True
                        and identity.get('callable') is True,
            'identity': identity, 'import_exit_code': process.returncode,
            'import_stderr_tail': process.stderr[-1500:],
            'scope': 'local callable binding only; serving-rank proof remains required'}


def input_inventory(overlay):
    """Hash the source overlay before any import can create runtime outputs."""
    overlay = Path(overlay)
    return {path.relative_to(overlay).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(overlay.rglob('*')) if path.is_file() and not path.is_symlink()
            and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo')}


def install_proof_producer(overlay, settings, applied):
    """Install a reviewed agent-supplied bootstrap; do not invent runtime proof."""
    config = settings.get('serving_source_proof') or {}
    if not config.get('producer_file'):
        return None
    producer = Path(config['producer_file']).resolve()
    data = producer.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != config.get('producer_sha256'):
        raise ValueError('serving source producer differs from the reviewed hash')
    compile(data, str(producer), 'exec')
    overlay = Path(overlay)
    name = '_geak_serving_source_producer.py'
    (overlay / name).write_bytes(data)
    targets = {'contract': 'serving_source_obligations/v1', 'world_size': config['world_size'],
               'proof_file': config['file'], 'applied': [
                   {key: entry[key] for key in ('kernel', 'target_callable', 'bundled_file',
                       'bundled_source_hashes', 'native_proof_required') if key in entry}
                   for entry in applied]}
    (overlay / '_geak_source_obligations.json').write_text(json.dumps(targets, indent=2) + '\n')
    with (overlay / 'sitecustomize.py').open('a') as stream:
        stream.write('''\n# Reviewed agent-supplied serving observation bootstrap. No producer runs in import checks.
if (os.environ.get("GEAK_SERVING_SOURCE_PROOF_RUN_DIR")
        and "_GEAK_OVERLAY_BOOTSTRAPPING" not in os.environ
        and "_GEAK_SERVING_PROOF_BOOTSTRAPPING" not in os.environ
        and os.path.basename(sys.argv[0]) != "rocm_agent_enumerator"):
    os.environ["_GEAK_SERVING_PROOF_BOOTSTRAPPING"] = "1"
    try:
        import runpy as _geak_runpy
        _geak_runpy.run_path(os.path.join(os.path.dirname(__file__), "_geak_serving_source_producer.py"),
                            run_name="_geak_serving_source_producer")
    except BaseException as _geak_producer_error:
        sys.stderr.write("[overlay] serving proof producer failed: %r\\n" % (_geak_producer_error,))
        raise SystemExit(78)
    finally:
        os.environ.pop("_GEAK_SERVING_PROOF_BOOTSTRAPPING", None)
''')
    return {'file': name, 'sha256': digest, 'scope': 'bootstrap installed; runtime observations still required'}


SITECUSTOMIZE_GUARD = ("# --- GEAK applyback: keep the overlay off sys.path for "
                       "everyone else ---")


def sanitize_sitecustomize(overlay):
    """Keep the overlay invisible to importlib.metadata.

    sitecustomize injects the patched module eagerly, at interpreter startup. That import
    reaches setuptools, which appends site-packages/setuptools/_vendor to sys.path -- a
    directory holding older vendored copies of importlib_metadata, zipp, packaging, tomli,
    platformdirs and typing_extensions. A serving stack that audits its own packages with
    importlib.metadata.distributions() (last wins) then sees the vendored versions and
    refuses to start, so the candidate leg dies before it serves a token while the
    baseline leg -- same packages, no eager import -- starts fine.

    The overlay has no business changing what any other code can import. Snapshot sys.path
    before the injection and drop the vendor entries the injection added; setuptools
    re-appends through its own VendorImporter if it needs them again. Idempotent.
    """
    path = os.path.join(overlay, "sitecustomize.py")
    if not os.path.isfile(path):
        return {"sanitized": False, "reason": "no sitecustomize.py in the overlay"}
    text = open(path).read()
    if SITECUSTOMIZE_GUARD in text:
        return {"sanitized": True, "note": "already present"}
    pre = ("# GEAK applyback: sys.path as the interpreter built it, before any overlay "
           "import.\nimport sys as _geak_sys\n_GEAK_PATH_BEFORE = list(_geak_sys.path)\n")
    post = "\n\n" + SITECUSTOMIZE_GUARD + """
try:
    _geak_before = set(_GEAK_PATH_BEFORE)
    _geak_vendor = os.path.join("setuptools", "_vendor")
    for _geak_p in list(sys.path):
        if _geak_p not in _geak_before and _geak_vendor in _geak_p:
            sys.path.remove(_geak_p)
            sys.stderr.write("[overlay] removed sys.path entry added during injection: "
                             "%s\\n" % _geak_p)
except Exception as _geak_ex:          # never let hygiene break the run
    sys.stderr.write("[overlay] sys.path cleanup FAILED: %r\\n" % (_geak_ex,))
"""
    with open(path, "w") as fh:
        fh.write(pre + text + post)
    return {"sanitized": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimized", required=True)
    ap.add_argument("--tasks-dir", required=True)
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--geak-root", default=os.environ.get("GEAK_ROOT", ""))
    ap.add_argument("--settings", help="stage params JSON containing generic binding_options")
    args = ap.parse_args()
    settings = json.load(open(args.settings)) if args.settings else {}
    binding_options = settings.get('binding_options') or {}

    opt = json.load(open(args.optimized))
    kept = [r for r in opt.get("kernels", []) if r.get("verdict") == "pass"]
    out = {"applied": [], "skipped": [], "overlay": os.path.abspath(args.overlay)}

    if not kept:
        out["skipped"].append({"reason": "no kernel from the optimize stage passed cold "
                                         "re-verification"})
        json.dump(out, open(args.manifest or "/dev/stdout", "w"), indent=1, ensure_ascii=False)
        return 3

    tool = find_overlay_setup(args.geak_root, args.tasks_dir)
    if not tool:
        print("cannot find overlay_setup.py (neither in the task directories nor under "
              "e2e_workflow/scripts)", file=sys.stderr)
        return 2

    if os.path.lexists(args.overlay):
        archive = tempfile.mkdtemp(prefix=Path(args.overlay).name + '.previous-',
                                   dir=str(Path(args.overlay).parent))
        preserved = Path(archive) / Path(args.overlay).name
        os.rename(args.overlay, preserved)
        out['previous_overlay'] = str(preserved)
    base = ""
    used_bindings, used_impl_modules, used_source_env = set(), set(), {}
    for r in kept:
        k = r["kernel"]
        task = os.path.join(args.tasks_dir, k)
        try:
            meta = json.load(open(os.path.join(task, "meta.json")))
        except (OSError, ValueError) as e:
            out["skipped"].append({"kernel": k, "reason": "cannot read meta.json: %s" % e})
            continue
        bind = meta.get("candidate_bind") or {}
        if not bind:
            out["skipped"].append({"kernel": k, "reason":
                "meta.json has no candidate_bind — the extraction skill did not record how "
                "to bind this back, and guessing is not an option; skipped"})
            continue
        identity = bind.get('target') or bind.get('module')
        if identity in used_bindings:
            raise ValueError('duplicate overlay binding: ' + str(identity))
        used_bindings.add(identity)
        if bind.get('kind') == 'rebind':
            if bind.get('impl_module') in used_impl_modules:
                raise ValueError('duplicate rebind implementation module')
            used_impl_modules.add(bind.get('impl_module'))
        # The optimized source. candidate_bind["file"] is authoritative when it names a
        # file that exists in the task: the extraction skill wrote it knowing how this
        # particular leaf binds, and for a leaf that is HIP C++ it points at a delivery
        # module rather than at kernel_src. Overriding it with editable_src() installs the
        # .py of a kernel whose edit lives in a .h -- an overlay that shadows the
        # dispatcher with an identical dispatcher, leaves the image's prebuilt .so serving
        # every token, and still passes the shadow check below. The e2e that follows would
        # be measuring the baseline against itself.
        ksrc = os.path.join(task, "kernel_src")
        named = bind.get("file")
        src, via = None, ""
        if named:
            for cand in (os.path.join(task, named), os.path.join(ksrc, named)):
                if os.path.isfile(cand):
                    src, via = cand, "candidate_bind.file"
                    break
            if not src:
                out["skipped"].append({"kernel": k, "reason":
                    "candidate_bind names file %r and it is not in the task directory; "
                    "installing a different file would bind something the extraction "
                    "skill did not describe" % named})
                continue
        if not src:
            src, via = editable_src(ksrc), "kernel_src (candidate_bind named no file)"
        if not src:
            out["skipped"].append({"kernel": k, "reason": "kernel_src/ holds no editable file"})
            continue
        source_env = source_environment(task, bind, binding_options.get(k) or {})
        for key, value in source_env.items():
            if key in used_source_env and used_source_env[key] != value:
                raise ValueError('conflicting task source environment: ' + key)
            used_source_env[key] = value
        rc, log = add_entry(tool, args.overlay, base, bind, src)
        if rc != 0:
            out["skipped"].append({"kernel": k,
                                   "reason": "overlay install failed: %s" % log[-400:]})
            continue
        try:
            bundled_file = bundle_module_sources(args.overlay, task, bind, src, source_env)
        except (ValueError, OSError) as exc:
            print("cannot bundle source tree for %s: %s" % (k, exc), file=sys.stderr)
            return 2
        base = args.overlay          # the next kernel seeds from this overlay
        entry = {"kernel": k, "bind_kind": bind.get("kind"),
                 "target": bind.get("target") or bind.get("module"),
                 "impl_file": src, "impl_file_from": via, "bundled_file": bundled_file,
                 "target_callable": meta.get('target_callable') or identity,
                 "source_env": source_env,
                 "verified_speedup": r.get("verified_speedup")}
        # A kernel spread over several files is delivered through one module. Record what
        # that module is expected to pull in, so a reader can tell at a glance whether the
        # .cu/.h actually travelled with the .py.
        if os.path.isdir(ksrc):
            files = source_files(ksrc)
            entry["kernel_src_files"] = list(files)
            entry["kernel_src_hashes"] = {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in files.items()}
            entry['bundled_source_hashes'] = {
                (Path('_patched') / k / 'kernel_src' / name).as_posix(): digest
                for name, digest in entry['kernel_src_hashes'].items()}
            entry['native_proof_required'] = any(Path(name).suffix.lower() in
                {'.cu', '.cuh', '.hip', '.cpp', '.cc', '.cxx', '.s'} for name in files)
            if len(entry["kernel_src_files"]) > 1 and via != "candidate_bind.file":
                entry["warning"] = (
                    "kernel_src holds %d files but the overlay installs only %s, and "
                    "candidate_bind named none. If the optimized edit is in one of the "
                    "others it is not in this overlay."
                    % (len(entry["kernel_src_files"]), os.path.basename(src)))
        out["applied"].append(entry)

    install_source_environment(args.overlay, out['applied'])
    out['proof_producer'] = install_proof_producer(args.overlay, settings, out['applied'])
    out["sitecustomize"] = sanitize_sitecustomize(args.overlay)
    out['source_inventory'] = input_inventory(args.overlay)

    # Native loaders can create build outputs during an import. Exercise a real
    # mirror and preserve its diagnostics, leaving the accepted source pristine
    # for independent per-arm copies.
    check_overlay = tempfile.mkdtemp(prefix=Path(args.overlay).name + '.import-check-',
                                     dir=str(Path(args.overlay).parent))
    shutil.copytree(args.overlay, check_overlay, dirs_exist_ok=True)
    out['binding_check_overlay'] = check_overlay

    # --- Verify the overlay really shadows. Measuring without this check means "no gain"
    #     may simply be an overlay that was never loaded.
    checked = []
    env = dict(os.environ)
    env.pop('GEAK_SERVING_SOURCE_PROOF_RUN_DIR', None)
    env["PYTHONPATH"] = check_overlay + os.pathsep + env.get("PYTHONPATH", "")
    for name, leaf in [('TRITON_CACHE_DIR', 'triton'), ('XDG_CACHE_HOME', 'xdg'),
                       ('TORCH_EXTENSIONS_DIR', 'torch_ext'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
        env[name] = str(Path(check_overlay) / 'run' / 'import_cache' / leaf)
    for a in out["applied"]:
        if a['bind_kind'] == 'rebind':
            check = check_rebind(a['target'], check_overlay, env)
            checked.append(check)
            if not check['shadowed']:
                a['warning'] = 'the declared callable/JIT rebind did not resolve into the accepted overlay'
            continue
        mod = (a["target"] or "").split(":")[0]
        if not mod:
            continue
        r = subprocess.run([sys.executable, tool, "check", "--module", mod],
                           capture_output=True, text=True, env=env)
        where, verdict = read_check(r.stdout or "")
        # Trust the tool's own verdict rather than looking for the overlay path in its
        # output. `check` prints "<module> -> <file>" and then a separate verdict line, so
        # taking the last line as the path judged a correctly installed overlay to be
        # missing -- it fails closed, which aborts a run that was in fact fine.
        # find_spec alone can point at a file that raises while importing its missing
        # nested computation. Exercise the actual import before spending hours on E2E.
        imported = subprocess.run(
            [sys.executable, "-c", "import importlib,sys; importlib.import_module(sys.argv[1])", mod],
            capture_output=True, text=True, env=env)
        inside = (r.returncode == 0 and imported.returncode == 0
                  and verdict in ("OVERLAY_ACTIVE", "INJECTED")
                  and os.path.realpath(where).startswith(os.path.realpath(check_overlay) + os.sep))
        checked.append({"module": mod, "resolves_to": where, "verdict": verdict,
                        "shadowed": inside, "import_exit_code": imported.returncode,
                        "import_stderr_tail": imported.stderr[-1500:] if imported.returncode else ""})
        if not inside:
            a["warning"] = ("the overlay does not shadow %s — it resolves to %s (%s). "
                            "Any e2e measured in this state is meaningless."
                            % (mod, where or "<unknown>", verdict or "no verdict"))
    out["checks"] = checked
    out["all_shadowed"] = bool(checked) and not out['skipped'] and all(c["shadowed"] for c in checked)
    out['check_scope'] = 'local imports/bindings only; not actual serving-rank source proof'
    out['serving_rank_source_proof_required'] = True
    if input_inventory(args.overlay) != out['source_inventory']:
        raise RuntimeError('accepted source overlay changed during binding checks')

    with open(args.manifest or os.path.join(args.overlay, "applied.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("installed %d kernels, skipped %d; overlay=%s shadow check=%s"
          % (len(out["applied"]), len(out["skipped"]), args.overlay,
             "passed" if out["all_shadowed"] else "**FAILED**"))
    return 0 if out["all_shadowed"] else 2


if __name__ == "__main__":
    sys.exit(main())
