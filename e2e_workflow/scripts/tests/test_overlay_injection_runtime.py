#!/usr/bin/env python3
"""Runtime tests for the overlay sitecustomize shim -- it is EXECUTED here, not just compiled.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_overlay_injection_runtime.py

test_overlay_setup.py is the unit suite for the installer: it drives the argparse surface and the
manifest bookkeeping with subprocess replaced by a recorder, and it only ever ``compile()``s
SITECUSTOMIZE (test_shim_is_syntactically_valid). Nothing there executes the shim. But the shim's
whole job happens at interpreter start inside a live inference server, so the two properties it
exists for are invisible to a suite that never starts an interpreter:

  A. IMPORT ORDER AND MODULE IDENTITY. A patched submodule must be produced by the NORMAL import
     machinery at the moment something imports it, so the process ends up with exactly one module
     object, created in the normal order, with the parent attribute bound by Python itself.
     Executing it eagerly at interpreter start runs it before the package __init__ and from a
     different entry point, so library state built once at first import (arch probes, tile/config
     registries, JIT caches) is built from the wrong place.
  B. FORK-BOMB GUARD. The ROCm arch probes are python scripts and they inherit PYTHONPATH. If the
     overlay installs inside one and a hook imports a GPU library that probes the arch at import,
     the probe re-enters the shim and shells out again, unbounded (observed: 13.7k processes,
     load 590, every later vLLM launch failing rendezvous with "2/8 clients joined").

Each scenario runs a real child interpreter three ways over IDENTICAL fixtures:

    real   -- no overlay, original submodule. The GOLD REFERENCE: whatever a normal import does is
              by definition the correct order, so the fixed shim is checked against observed
              Python behaviour rather than against someone's belief about it.
    old    -- PRE_FIX_SHIM below, the injection strategy as it stood at 52aa7aa5.
    new    -- the shim in this working tree.

A regression test that cannot fail on the un-fixed code is not a regression test, so the
load-bearing cases assert through assertDiscriminating(): if `old` and `new` agree on the field,
the test FAILS as NON-DISCRIMINATING rather than passing and implying coverage it does not have.

Everything runs in a temp dir with a stdlib-only child; no GPU, no torch, no network.
"""
import importlib.util
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OV = _load("overlay_setup", "overlay_setup.py")
NEW_SHIM = OV.SITECUSTOMIZE

# Frozen copy of the pre-fix injection strategy (overlay_setup.py at 52aa7aa5), kept verbatim so
# the discrimination checks keep working after the fix is committed and `git show HEAD` no longer
# yields the old code. Only the header and section (a) are reproduced: these fixtures exercise
# `modules` only, and the rebind/capture/marker sections were not what changed.
PRE_FIX_SHIM = r'''# Auto-generated reversible overlay (e2e_workflow). Drop this dir from PYTHONPATH to revert.
import json, os, sys, importlib, importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAN = os.path.join(_HERE, "_overlay_manifest.json")
try:
    with open(_MAN) as _fh:
        _m = json.load(_fh)
except Exception as _e:
    _m = {"modules": [], "rebinds": [], "markers": [], "captures": []}

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
'''

# The child prints one JSON line and is byte-identical across all three modes.
CHILD = textwrap.dedent(
    """
    import json, sys
    out = {}
    try:
        import pkg
        out["init_log"] = list(getattr(pkg, "LOG", []))
        out["value"] = getattr(getattr(pkg, "sub", None), "VALUE", None)
        out["sys_modules_value"] = getattr(sys.modules.get("pkg.sub"), "VALUE", None)
        out["one_module_object"] = (getattr(pkg, "sub", None) is sys.modules.get("pkg.sub"))
        out["import_ok"] = True
    except Exception as exc:
        out["import_ok"] = False
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
    out["finder_installed"] = any(type(f).__name__ == "_OverlayFinder" for f in sys.meta_path)
    sys.stdout.write("RESULT:" + json.dumps(out) + "\\n")
    """
)


def _write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


class _Scenario(unittest.TestCase):
    """Builds a real package plus an overlay, then runs a child interpreter against it."""

    # Fixture bodies; subclasses override to change the seam shape.
    PKG_INIT = 'LOG = []\nLOG.append("init-start")\nfrom . import sub\nLOG.append("init-end")\n'
    SUB_REAL = 'import pkg\npkg.LOG.append("sub-exec")\nVALUE = "original"\n'
    SUB_PATCHED = 'import pkg\npkg.LOG.append("sub-exec")\nVALUE = "patched"\n'

    def _build(self, tmp, shim, patched, script_name):
        pkgdir = os.path.join(tmp, "pkg")
        os.makedirs(pkgdir)
        _write(os.path.join(pkgdir, "__init__.py"), self.PKG_INIT)
        _write(os.path.join(pkgdir, "sub.py"), self.SUB_REAL)

        pypath = [tmp]
        if shim is not None:
            overlay = os.path.join(tmp, "overlay")
            os.makedirs(overlay)
            _write(os.path.join(overlay, "sitecustomize.py"), shim)
            _write(os.path.join(overlay, "patched_sub.py"), self.SUB_PATCHED)
            manifest = {"modules": [], "rebinds": [], "markers": [], "captures": []}
            if patched:
                manifest["modules"] = [{"module": "pkg.sub", "file": "patched_sub.py"}]
            _write(os.path.join(overlay, "_overlay_manifest.json"), json.dumps(manifest))
            pypath.insert(0, overlay)

        script = os.path.join(tmp, script_name)
        _write(script, CHILD)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(pypath)
        return script, env

    def run_mode(self, mode, patched=True, script_name="app.py"):
        """mode: 'real' (no overlay) | 'old' (pre-fix shim) | 'new' (working-tree shim)."""
        shim = {"real": None, "old": PRE_FIX_SHIM, "new": NEW_SHIM}[mode]
        tmp = tempfile.mkdtemp()
        try:
            script, env = self._build(tmp, shim, patched, script_name)
            proc = subprocess.run([sys.executable, script], env=env, cwd=tmp,
                                  capture_output=True, text=True, timeout=120)
            lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT:")]
            self.assertTrue(lines, f"child produced no RESULT line.\n"
                                   f"stdout={proc.stdout}\nstderr={proc.stderr}")
            res = json.loads(lines[0][len("RESULT:"):])
            res["_stderr"] = proc.stderr
            return res
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def assertDiscriminating(self, old, new, field):
        if old.get(field) == new.get(field):
            self.fail(f"NON-DISCRIMINATING: {field!r} is {new.get(field)!r} under BOTH the pre-fix "
                      f"and the current shim, so this test does not cover the fix it claims to.")


class TestImportOrderAndIdentity(_Scenario):
    def test_patched_submodule_executes_in_real_import_order(self):
        """The patched file must run WHERE the real one would: inside the parent's __init__."""
        real, new, old = self.run_mode("real"), self.run_mode("new"), self.run_mode("old")
        self.assertEqual(real["init_log"], ["init-start", "sub-exec", "init-end"],
                         "gold reference is wrong, so the fixture is broken")
        self.assertTrue(new["import_ok"], f"shim broke the import: {new.get('error')}")
        self.assertEqual(new["init_log"], real["init_log"])
        self.assertEqual(new["value"], "patched", "shim did not actually inject")
        self.assertDiscriminating(old, new, "init_log")

    def test_exactly_one_module_object_for_the_dotted_name(self):
        """pkg.sub and sys.modules['pkg.sub'] must be the same object.

        If they diverge, every `from pkg import sub` binding made before the fix-up keeps pointing
        at unpatched code while the leg is still labelled "candidate".
        """
        new = self.run_mode("new")
        self.assertTrue(new["one_module_object"])
        self.assertEqual(new["value"], "patched")
        self.assertEqual(new["sys_modules_value"], "patched")


class TestParentReadsSubmoduleAtImportTime(_Scenario):
    """The shape that broke the aiter seam: submodule imports its package, package reads an
    attribute off the submodule while initialising. Eager exec runs the parent __init__ in the
    middle of the submodule's own execution, so __init__ sees a half-built module."""

    PKG_INIT = ('LOG = []\nLOG.append("init-start")\nfrom . import sub\n'
                'NAME = sub.VALUE\nLOG.append("init-end")\n')
    SUB_REAL = 'import pkg\nVALUE = "original"\n'
    SUB_PATCHED = 'import pkg\nVALUE = "patched"\n'

    def test_package_can_read_the_patched_submodule_while_initialising(self):
        real, new, old = self.run_mode("real"), self.run_mode("new"), self.run_mode("old")
        self.assertTrue(real["import_ok"], f"gold reference broke: {real.get('error')}")
        self.assertTrue(new["import_ok"], f"shim cannot inject this seam shape: {new.get('error')}")
        self.assertEqual(new["value"], "patched")
        self.assertDiscriminating(old, new, "import_ok")


class TestEmptyManifestParity(_Scenario):
    """An A/B is only meaningful if the legs differ in the kernel and nothing else."""

    def test_empty_manifest_leaves_originals_intact_but_still_installs_the_finder(self):
        empty, real = self.run_mode("new", patched=False), self.run_mode("real")
        self.assertEqual(empty["value"], "original")
        self.assertEqual(empty["init_log"], real["init_log"])
        self.assertTrue(empty["finder_installed"],
                        "finder must be installed even when it claims nothing, so the baseline leg "
                        "takes the same startup path as the candidate leg")

    def test_baseline_and_candidate_legs_differ_only_in_the_injected_value(self):
        base, cand = self.run_mode("new", patched=False), self.run_mode("new", patched=True)
        self.assertEqual(base["init_log"], cand["init_log"])
        self.assertEqual(base["one_module_object"], cand["one_module_object"])
        self.assertNotEqual(base["value"], cand["value"])


class TestForkBombGuard(_Scenario):
    PROBES = ["rocm_agent_enumerator", "rocminfo", "offload-arch", "amdgpu-arch",
              "hipconfig", "hipcc", "hipinfo", "rocm-smi", "rocm_smi.py"]

    def test_every_known_probe_gets_an_empty_manifest(self):
        for probe in self.PROBES:
            with self.subTest(probe=probe):
                res = self.run_mode("new", script_name=probe)
                self.assertEqual(res["value"], "original",
                                 f"overlay installed inside arch probe {probe!r}: fork-bomb risk")
                self.assertNotIn("injected module", res["_stderr"])

    def test_guard_is_discriminating_against_the_prefix_shim(self):
        old = self.run_mode("old", script_name="rocm_agent_enumerator")
        new = self.run_mode("new", script_name="rocm_agent_enumerator")
        self.assertDiscriminating(old, new, "value")

    def test_probe_still_starts_cleanly_it_is_a_no_op_not_a_crash(self):
        res = self.run_mode("new", script_name="rocminfo")
        self.assertTrue(res["import_ok"], f"probe interpreter failed: {res.get('error')}")
        self.assertTrue(res["finder_installed"])

    def test_a_normal_script_name_is_not_mistaken_for_a_probe(self):
        """The guard must not be so broad that real workloads silently lose their overlay."""
        for name in ["app.py", "vllm", "sglang_launch.py", "worker.py", "rocm_helper.py"]:
            with self.subTest(name=name):
                res = self.run_mode("new", script_name=name)
                self.assertEqual(res["value"], "patched",
                                 f"guard wrongly suppressed the overlay for {name!r}")

    def test_interpreter_launched_from_a_rocm_install_path_is_guarded(self):
        """Probes are also reached by absolute path, under a name the basename list misses."""
        if not os.path.isdir("/opt/rocm") or not os.access("/opt/rocm", os.W_OK):
            self.skipTest("/opt/rocm not present or not writable here")
        root = "/opt/rocm/libexec/geak_overlay_test"
        os.makedirs(root, exist_ok=True)
        try:
            tmp = tempfile.mkdtemp(dir=root)
            script, env = self._build(tmp, NEW_SHIM, True, "launcher.py")
            proc = subprocess.run([sys.executable, script], env=env, cwd=tmp,
                                  capture_output=True, text=True, timeout=120)
            line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT:")][0]
            res = json.loads(line[len("RESULT:"):])
            self.assertEqual(res["value"], "original",
                             "a script under /opt/rocm must be treated as a probe")
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestClassMethodRebind(unittest.TestCase):
    """Real child calls, failure exits and prior-shim composition; no serving/GPU imports."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "sample.py").write_text('''
def module_fn(x): return ["original-module", x]
VALUE = 3
class Base:
    def method(self, x): return ["original-method", self.tag, x]
    @staticmethod
    def static(x): return ["original-static", x]
    @classmethod
    def klass(cls, x): return ["original-class", cls.__name__, x]
    @property
    def prop(self): raise AssertionError("property must not execute")
class Model(Base): tag = "model"
class Sub(Model): pass
singleton = Model()
''')
        self.impl = '''
from sample import Model as EarlyClassAlias
def method(self, x): return ["replacement-method", self.tag, x]
def static(x): return ["replacement-static", x]
def klass(cls, x): return ["replacement-class", cls.__name__, x]
def module_fn(x): return ["replacement-module", x]
VALUE = 99
'''
        self.calls = '''
import json, inspect, sample
from sample import Model as Alias, module_fn as function_alias
obj = Alias()
print(json.dumps([obj.method(1), Alias.method(obj, 2), Alias.static(3), obj.static(4),
    Alias.klass(5), obj.klass(6), sample.Sub.klass(7), function_alias(8), sample.VALUE,
    sample.Base.static(9), type(inspect.getattr_static(Alias, "static")).__name__,
    type(inspect.getattr_static(Alias, "klass")).__name__]))
'''

    def entry(self, target, impl_attr):
        return dict(target="sample:" + target, impl_module="impl", impl_attr=impl_attr)

    def overlay(self, entries, shim=NEW_SHIM):
        overlay = Path(tempfile.mkdtemp(dir=self.root))
        (overlay / "sitecustomize.py").write_text(shim)
        (overlay / "_overlay_manifest.json").write_text(json.dumps({"rebinds": entries}))
        (overlay / "impl.py").write_text(self.impl)
        return overlay

    def child(self, overlay=None, code=None):
        paths = ([str(overlay)] if overlay else []) + [str(self.root)]
        return subprocess.run([sys.executable, "-c", code or self.calls], cwd=self.root,
            env=dict(os.environ, PYTHONPATH=os.pathsep.join(paths), PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True, text=True, timeout=20)

    def test_real_calls_preserve_descriptors_aliases_inheritance_and_reversal(self):
        entries = [self.entry("Model." + name, name) for name in ("method", "static", "klass")]
        entries += [self.entry("module_fn", "module_fn"), self.entry("VALUE", "VALUE")]
        original = self.child()
        old = self.child(self.overlay(entries, OV.PREVIOUS_SITECUSTOMIZE))
        overlay = self.overlay(entries)
        current = self.child(overlay)
        for proc in (original, old, current):
            self.assertEqual(proc.returncode, 0, proc.stderr)
        before, broken, after = [json.loads(proc.stdout) for proc in (original, old, current)]
        self.assertEqual(broken[:7], before[:7], "old dotted rebind must reproduce the no-op")
        self.assertEqual(after, [["replacement-method", "model", 1], ["replacement-method", "model", 2],
            ["replacement-static", 3], ["replacement-static", 4], ["replacement-class", "Model", 5],
            ["replacement-class", "Model", 6], ["replacement-class", "Sub", 7],
            ["replacement-module", 8], 99, ["original-static", 9], "staticmethod", "classmethod"])
        alias = self.child(overlay, "import impl; print(impl.EarlyClassAlias().method(10))")
        self.assertEqual(alias.returncode, 0, alias.stderr)
        self.assertIn("replacement-method", alias.stdout)
        self.assertEqual(self.child().stdout, original.stdout, "separate process without overlay restores original")

    def test_invalid_targets_fail_before_application_and_do_not_invoke_descriptors(self):
        for target, replacement in [("Model.absent", "method"), ("Missing.method", "method"),
                ("singleton.method", "method"), ("Model.prop", "method"), ("Model.method", "VALUE"),
                ("absent", "VALUE")]:
            with self.subTest(target=target, replacement=replacement):
                proc = self.child(self.overlay([self.entry(target, replacement)]), "print('application started')")
                self.assertEqual(proc.returncode, 70, proc.stderr)
                self.assertNotIn("application started", proc.stdout)
                self.assertIn("rebind FAILED", proc.stderr)
                self.assertNotIn("[overlay] rebound", proc.stderr)
                self.assertNotIn("property must not execute", proc.stderr)

    def test_module_lazy_export_rebind_preserves_public_attribute_lookup(self):
        with (self.root / "sample.py").open("a") as handle:
            handle.write('''
def __getattr__(name):
    if name == "lazy_fn":
        return lambda x: ["original-lazy", x]
    raise AttributeError(name)
''')
        original = self.child(code="import sample; assert 'lazy_fn' not in vars(sample); "
            "assert sample.lazy_fn(1) == ['original-lazy', 1]")
        self.assertEqual(original.returncode, 0, original.stderr)
        current = self.child(self.overlay([self.entry("lazy_fn", "module_fn")]),
            "import sample, impl; from sample import lazy_fn; "
            "assert vars(sample)['lazy_fn'] is impl.module_fn; "
            "assert lazy_fn(2) == ['replacement-module', 2]")
        self.assertEqual(current.returncode, 0, current.stderr)
        self.assertIn("[overlay] rebound sample:lazy_fn", current.stderr)
        missing = self.child(self.overlay([self.entry("missing_export", "module_fn")]),
            "print('application started')")
        self.assertEqual(missing.returncode, 70, missing.stderr)
        self.assertNotIn("application started", missing.stdout)
        self.assertIn("rebind FAILED", missing.stderr)

    def test_failed_add_keeps_prior_shim_and_manifest(self):
        overlay = self.overlay([self.entry("module_fn", "module_fn")], OV.PREVIOUS_SITECUSTOMIZE)
        manifest = (overlay / "_overlay_manifest.json").read_bytes()
        with self.assertRaises(FileNotFoundError):
            OV.cmd_add_rebind(SimpleNamespace(overlay=str(overlay), base="", target="sample:Model.method",
                impl_module="impl", impl_attr="method", impl_file=str(self.root / "missing.py")))
        self.assertEqual((overlay / "sitecustomize.py").read_text(), OV.PREVIOUS_SITECUSTOMIZE)
        self.assertEqual((overlay / "_overlay_manifest.json").read_bytes(), manifest)

    def test_prior_generic_shims_upgrade_through_stack_installer_and_merge(self):
        priors = [(OV.PREVIOUS_SITECUSTOMIZE, "efb5d81c8a2b111f90d295f08ad801082ade58e67a5d3a43e695b562c65bb3a4"),
                  (OV.LEGACY_SITECUSTOMIZE, "1fcb7a9835b7369cff8f60f57aa337bcbe7214122279454ec93a17cdc32f9db9")]
        for prior, digest in priors:
            with self.subTest(prior=digest):
                self.assertEqual(hashlib.sha256(prior.encode()).hexdigest(), digest)
                base = self.overlay([self.entry("module_fn", "module_fn")], prior)
                stacked = self.root / ("stacked-" + digest[:8])
                installer = self.overlay([], prior)
                (installer / "fixture_installer.py").write_text("def install():\n    import sys\n    sys.fixture_installed = True\n")
                with contextlib.redirect_stdout(io.StringIO()):
                    OV.cmd_add_rebind(SimpleNamespace(overlay=str(stacked), base=str(base),
                        target="sample:Model.method", impl_module="impl", impl_attr="method", impl_file=""))
                    OV.cmd_add_installer(SimpleNamespace(overlay=str(installer), base="", id="fixture",
                        module="fixture_installer", callable="install", target=[], kernel=[]))
                self.assertEqual((base / "sitecustomize.py").read_text(), prior)
                merged = OV.merge_overlays(str(self.root / ("merged-" + digest[:8])), [str(base), str(stacked), str(installer)])
                proc = self.child(merged, "import sample, sys; assert sys.fixture_installed; "
                    "assert sample.Model().method(1)[0] == 'replacement-method'; "
                    "assert sample.module_fn(2)[0] == 'replacement-module'; print('composed')")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stdout.strip(), "composed")
        custom = self.overlay([], "# custom startup\n")
        with self.assertRaisesRegex(SystemExit, "custom sitecustomize"):
            OV.cmd_add_rebind(SimpleNamespace(overlay=str(custom), base="", target="sample:Model.method",
                impl_module="impl", impl_attr="method", impl_file=""))
        self.assertEqual((custom / "sitecustomize.py").read_text(), "# custom startup\n")
        self.assertEqual(json.loads((custom / "_overlay_manifest.json").read_text())["rebinds"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
