#!/usr/bin/env python3
"""Unit tests for overlay_setup.py -- the reversible PYTHONPATH overlay installer (stdlib only).

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_overlay_setup.py

overlay_setup builds the directory we hand to a live inference server as
`PYTHONPATH=<overlay>:$PYTHONPATH`, so a candidate kernel, a patched submodule or a capture_shapes
hook takes effect inside sglang/vllm WITHOUT editing site-packages. Two properties are load-bearing
and neither is visible until an e2e run is already burning GPU hours:

  1. COMPOUNDING + IDEMPOTENCE. Every accepted kernel appends to one manifest. If a later
     add-rebind/add-module/add-capture silently drops an earlier entry, or re-adding the same
     target duplicates it, the run measures the wrong code. Covered per command: dedupe by
     target/module, distinct targets accumulate, and a re-run never clobbers an existing
     sitecustomize.py or an existing manifest.
  2. REVERSIBILITY. Reversal is "drop the dir from PYTHONPATH", which only holds if nothing is
     ever written outside <overlay> -- in particular a --patch must land on the overlay's private
     copy, never on the install. test_patch_is_applied_to_the_overlay_copy_not_the_install pins the
     exact argv, and a failed patch must abort without recording a manifest entry.

Also covered: the manifest round-trip, module resolution (pkg_root / module_file) against real
temp packages, _try_apply's full attempt ladder with exact argv and missing-tool handling, the
argparse surface driven by explicit argv lists (including the copy-subtree/monkeypatch back-compat
aliases and the legacy --package/--subpath conversion), and error paths: overlay dir missing,
unwritable overlay, overlay path occupied by a file, add-module with no module named.

Two current-behavior bugs are pinned here rather than fixed, and flagged by name:
test_module_file_requires_the_caller_to_have_imported_importlib_util and
test_plain_top_level_install_is_misreported_as_injected.

Every test runs inside its own tempfile.TemporaryDirectory; subprocess is replaced by a recorder,
so nothing is executed and nothing outside the temp dir is written.
"""
import argparse
import contextlib
import importlib
import importlib.util
import io
import json
import os
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ov = _load("overlay_setup", "overlay_setup.py")

EMPTY_MANIFEST = {"modules": [], "rebinds": [], "markers": [], "captures": []}


class _RecordingRun:
    """Stands in for subprocess.run: records argv and cwd, never executes anything.

    succeed_at -- index of the attempt that should report returncode 0 (None: all fail).
    missing    -- argv[0] values that should raise FileNotFoundError, i.e. tool not installed.
    """

    def __init__(self, succeed_at=None, missing=()):
        self.calls = []
        self.succeed_at = succeed_at
        self.missing = tuple(missing)

    def __call__(self, args, cwd=None, capture_output=False, text=False):
        self.calls.append((list(args), cwd))
        if args[0] in self.missing:
            raise FileNotFoundError(2, "No such file or directory", args[0])
        rc = 0 if self.succeed_at == len(self.calls) - 1 else 1
        return types.SimpleNamespace(returncode=rc, stdout="", stderr="")

    @property
    def argvs(self):
        return [argv for argv, _cwd in self.calls]

    @property
    def cwds(self):
        return [cwd for _argv, cwd in self.calls]


class _OverlayCase(unittest.TestCase):
    """Confines every test to a private temp dir; the real env and repo are never written to."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._cleanup, tmp)
        self.tmp = tmp.name
        self.overlay = os.path.join(self.tmp, "overlay")

    def _cleanup(self, tmp):
        # The unwritable-overlay test leaves a mode 0o500 dir, which would defeat rmtree.
        for root, dirs, _files in os.walk(self.tmp):
            for d in dirs:
                os.chmod(os.path.join(root, d), stat.S_IRWXU)
        tmp.cleanup()

    # -- helpers ----------------------------------------------------------- #
    def _write(self, relpath, text):
        path = os.path.join(self.tmp, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    def _read(self, path):
        with open(path) as fh:
            return fh.read()

    def _manifest(self, overlay=None):
        return json.loads(self._read(os.path.join(overlay or self.overlay, "_overlay_manifest.json")))

    def _overlay_entries(self, overlay=None):
        return sorted(os.listdir(overlay or self.overlay))

    def _run(self, fn, *args):
        """Calls a cmd_* function and returns its stdout lines (they are the operator's contract)."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(*args)
        return buf.getvalue().splitlines()

    def _main(self, argv):
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["overlay_setup.py"] + list(argv)):
            with contextlib.redirect_stdout(buf):
                ov.main()
        return buf.getvalue().splitlines()

    def _main_fails(self, argv):
        """argparse writes usage to stderr and exits; swallow both and return the exception."""
        with mock.patch.object(sys, "argv", ["overlay_setup.py"] + list(argv)):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    ov.main()
        return caught.exception

    def _import_dir(self, name="importable"):
        """A fresh dir on sys.path, removed again after the test."""
        path = os.path.join(self.tmp, name)
        os.makedirs(path, exist_ok=True)
        sys.path.insert(0, path)
        importlib.invalidate_caches()
        self.addCleanup(self._drop_import_dir, path)
        return path

    def _drop_import_dir(self, path):
        while path in sys.path:
            sys.path.remove(path)
        importlib.invalidate_caches()

    def _forget(self, *names):
        def drop():
            for n in names:
                sys.modules.pop(n, None)
        self.addCleanup(drop)

    def _ns(self, **kw):
        """An argparse.Namespace with the defaults main() would have supplied."""
        base = {"overlay": self.overlay, "module": "", "package": "", "subpath": "",
                "patched_file": "", "src_file": "", "patch": ""}
        base.update(kw)
        return argparse.Namespace(**base)


# --------------------------------------------------------------------------- #
# _ensure_overlay -- the shim + empty manifest, and its idempotence
# --------------------------------------------------------------------------- #
class TestEnsureOverlay(_OverlayCase):
    def test_creates_missing_dir_with_shim_and_empty_manifest(self):
        self.assertFalse(os.path.exists(self.overlay))
        man = ov._ensure_overlay(self.overlay)

        self.assertEqual(man, os.path.join(self.overlay, "_overlay_manifest.json"))
        self.assertEqual(self._overlay_entries(), ["_overlay_manifest.json", "sitecustomize.py"])
        self.assertEqual(self._read(os.path.join(self.overlay, "sitecustomize.py")), ov.SITECUSTOMIZE)
        self.assertEqual(self._manifest(), EMPTY_MANIFEST)

    def test_missing_intermediate_dirs_are_created(self):
        nested = os.path.join(self.tmp, "runs", "task-7", "overlay")
        ov._ensure_overlay(nested)
        self.assertTrue(os.path.isfile(os.path.join(nested, "sitecustomize.py")))

    def test_shim_is_syntactically_valid_python(self):
        # sitecustomize.py is exec'd by the interpreter before the server imports anything; a
        # SyntaxError here would take down the whole inference process, not just the overlay.
        compile(ov.SITECUSTOMIZE, "sitecustomize.py", "exec")

    def test_shim_consumes_the_three_manifest_sections(self):
        shim = ov.SITECUSTOMIZE
        for section in EMPTY_MANIFEST:
            self.assertIn('_m.get("%s", [])' % section, shim)

    def test_shim_calls_capture_shapes_install_as_capture_shapes_defines_it(self):
        # The shim's only cross-file call. capture_shapes.install(target, out_dir, max_cases=5)
        # is copied in by add-capture, so the 3-positional-arg call site must keep matching.
        self.assertIn('capture_shapes.install(_e["target"], _e["out"], int(_e.get("max", 5)))',
                      ov.SITECUSTOMIZE)
        self.assertIn("def install(target, out_dir, max_cases=5):",
                      self._read(os.path.join(SCRIPTS_DIR, "capture_shapes.py")))

    def test_shim_installs_captures_before_markers(self):
        # A marker on any module that does `from <capture module> import <attr>` imports that module
        # and freezes the alias. If markers ran first, the later capture hook would rebind only the
        # defining module and the live call (through the alias) would never be recorded.
        shim = ov.SITECUSTOMIZE
        self.assertLess(shim.index('_m.get("captures", [])'), shim.index('_m.get("markers", [])'))

    def test_rerun_preserves_an_edited_shim_and_an_existing_manifest(self):
        # Re-running any add-* must not reset an overlay that already carries accepted kernels.
        ov._ensure_overlay(self.overlay)
        shim = os.path.join(self.overlay, "sitecustomize.py")
        with open(shim, "w") as fh:
            fh.write("# hand-edited by the operator\n")
        ov._save_man(os.path.join(self.overlay, "_overlay_manifest.json"),
                     {"modules": [], "rebinds": [{"target": "m:f", "impl_module": "i",
                                                  "impl_attr": "g"}], "captures": []})

        ov._ensure_overlay(self.overlay)

        self.assertEqual(self._read(shim), "# hand-edited by the operator\n")
        self.assertEqual(self._manifest()["rebinds"],
                         [{"target": "m:f", "impl_module": "i", "impl_attr": "g"}])

    @unittest.skipIf(os.geteuid() == 0, "root ignores mode bits, so nothing would be asserted")
    def test_unwritable_overlay_raises_instead_of_reporting_success(self):
        os.makedirs(self.overlay)
        os.chmod(self.overlay, stat.S_IRUSR | stat.S_IXUSR)
        with self.assertRaises(PermissionError):
            ov._ensure_overlay(self.overlay)
        self.assertEqual(self._overlay_entries(), [])

    def test_overlay_path_occupied_by_a_file_raises(self):
        path = self._write("not-a-dir", "")
        with self.assertRaises(FileExistsError):
            ov._ensure_overlay(path)


# --------------------------------------------------------------------------- #
# _load_man / _save_man
# --------------------------------------------------------------------------- #
class TestManifestIO(_OverlayCase):
    def test_round_trip_preserves_all_three_sections(self):
        man = ov._ensure_overlay(self.overlay)
        payload = {"modules": [{"module": "a.b", "file": os.path.join("_patched", "a.b.py")}],
                   "rebinds": [{"target": "a.b:f", "impl_module": "impl", "impl_attr": "g"}],
                   "captures": [{"target": "a.b:f", "out": "/tasks/t1", "max": 3}]}
        ov._save_man(man, payload)
        self.assertEqual(ov._load_man(man), payload)

    def test_save_truncates_rather_than_appending(self):
        man = ov._ensure_overlay(self.overlay)
        ov._save_man(man, {"rebinds": [{"target": "x:%d" % i} for i in range(50)]})
        ov._save_man(man, EMPTY_MANIFEST)
        self.assertEqual(ov._load_man(man), EMPTY_MANIFEST)

    def test_manifest_is_written_indented_for_operator_diffs(self):
        man = ov._ensure_overlay(self.overlay)
        ov._save_man(man, {"modules": [{"module": "a.b"}]})
        self.assertIn('\n  "modules": [', self._read(man))


# --------------------------------------------------------------------------- #
# pkg_root / module_file -- resolution against real temp packages
# --------------------------------------------------------------------------- #
class TestResolution(_OverlayCase):
    def test_pkg_root_of_a_regular_package_is_its_directory(self):
        root = self._import_dir()
        self._write(os.path.join("importable", "ovpkg", "__init__.py"), "")
        self._forget("ovpkg")
        self.assertEqual(ov.pkg_root("ovpkg"), os.path.join(root, "ovpkg"))

    def test_pkg_root_of_a_namespace_package_uses_the_first_path_entry(self):
        # A namespace package has no __file__, so the __path__ fallback is the only way through.
        root = self._import_dir()
        os.makedirs(os.path.join(root, "ovns"))
        self._forget("ovns")
        mod = importlib.import_module("ovns")
        self.assertIsNone(getattr(mod, "__file__", None))
        self.assertEqual(ov.pkg_root("ovns"), os.path.join(root, "ovns"))

    def test_pkg_root_of_a_module_with_neither_file_nor_path_exits(self):
        sys.modules["ovbodiless"] = types.ModuleType("ovbodiless")
        self._forget("ovbodiless")
        with self.assertRaises(SystemExit) as caught:
            ov.pkg_root("ovbodiless")
        self.assertIn("cannot locate package root for ovbodiless", str(caught.exception))

    def test_module_file_returns_the_installed_submodule_path(self):
        self._import_dir()
        self._write(os.path.join("importable", "ovsrt", "__init__.py"), "")
        target = self._write(os.path.join("importable", "ovsrt", "activation.py"), "def f():\n    pass\n")
        self._forget("ovsrt", "ovsrt.activation")
        self.assertEqual(ov.module_file("ovsrt.activation"), target)

    def test_module_file_of_an_absent_module_exits(self):
        with self.assertRaises(SystemExit) as caught:
            ov.module_file("ov_module_that_is_not_installed")
        self.assertIn("cannot find a file for module", str(caught.exception))

    def test_module_file_of_a_namespace_package_exits(self):
        # spec.origin is None for a namespace package: there is no file to copy or patch.
        root = self._import_dir()
        os.makedirs(os.path.join(root, "ovns2"))
        self._forget("ovns2")
        with self.assertRaises(SystemExit) as caught:
            ov.module_file("ovns2")
        self.assertIn("cannot find a file for module ovns2", str(caught.exception))

    def test_module_file_requires_the_caller_to_have_imported_importlib_util(self):
        """KNOWN BUG, pinned not fixed: overlay_setup does `import importlib` only, but
        module_file() calls importlib.util.find_spec. `import importlib` does not bind the .util
        submodule, so in a clean interpreter -- exactly how the CLI runs -- `check` and
        `add-module` without --patched-file/--src-file die with AttributeError instead of doing
        their job. Tests here pass only because this test module imports importlib.util itself.
        """
        stub = types.SimpleNamespace(import_module=importlib.import_module)
        with mock.patch.object(ov, "importlib", stub):
            with self.assertRaises(AttributeError) as caught:
                ov.module_file("json")
        self.assertIn("util", str(caught.exception))


# --------------------------------------------------------------------------- #
# _try_apply -- the patch attempt ladder, argv-exact
# --------------------------------------------------------------------------- #
class TestTryApply(_OverlayCase):
    def setUp(self):
        super().setUp()
        self.patch = self._write("kernel.diff", "--- a\n+++ b\n")
        self.target = self._write(os.path.join("overlay", "_patched", "a.b.py"), "x = 1\n")
        self.repo = os.path.join(self.tmp, "repo")
        os.makedirs(self.repo)

    def test_no_target_and_no_cwd_runs_nothing_and_fails(self):
        run = _RecordingRun()
        with mock.patch.object(ov.subprocess, "run", run):
            self.assertFalse(ov._try_apply(self.patch))
        self.assertEqual(run.calls, [])

    def test_target_file_ladder_is_patch_then_git_apply_unsafe_paths(self):
        run = _RecordingRun()
        with mock.patch.object(ov.subprocess, "run", run):
            self.assertFalse(ov._try_apply(self.patch, target_file=self.target))
        self.assertEqual(run.argvs, [
            ["patch", self.target, "-i", self.patch],
            ["git", "apply", "--unsafe-paths",
             "--directory=%s" % os.path.dirname(self.target), self.patch],
        ])
        self.assertEqual(run.cwds, [None, None])

    def test_cwd_ladder_is_git_apply_then_patch_p1(self):
        run = _RecordingRun()
        with mock.patch.object(ov.subprocess, "run", run):
            self.assertFalse(ov._try_apply(self.patch, cwd=self.repo))
        self.assertEqual(run.argvs, [["git", "apply", self.patch],
                                     ["patch", "-p1", "-i", self.patch]])
        self.assertEqual(run.cwds, [self.repo, self.repo])

    def test_target_attempts_precede_cwd_attempts(self):
        run = _RecordingRun()
        with mock.patch.object(ov.subprocess, "run", run):
            self.assertFalse(ov._try_apply(self.patch, target_file=self.target, cwd=self.repo))
        self.assertEqual([argv[0] for argv in run.argvs], ["patch", "git", "git", "patch"])
        # cwd is handed to every attempt, including the two targeted ones. Harmless only because
        # target_file and patch are absolute by the time cmd_add_module builds them.
        self.assertEqual(run.cwds, [self.repo] * 4)
        self.assertTrue(os.path.isabs(self.target) and os.path.isabs(self.patch))

    def test_first_zero_exit_short_circuits_the_ladder(self):
        run = _RecordingRun(succeed_at=0)
        with mock.patch.object(ov.subprocess, "run", run):
            self.assertTrue(ov._try_apply(self.patch, target_file=self.target, cwd=self.repo))
        self.assertEqual(run.argvs, [["patch", self.target, "-i", self.patch]])

    def test_missing_tool_falls_through_to_the_next_attempt(self):
        # A container without patch(1) must still land the diff via git apply.
        run = _RecordingRun(succeed_at=1, missing=("patch",))
        with mock.patch.object(ov.subprocess, "run", run):
            self.assertTrue(ov._try_apply(self.patch, target_file=self.target))
        self.assertEqual([argv[0] for argv in run.argvs], ["patch", "git"])

    def test_every_tool_missing_is_a_clean_false(self):
        run = _RecordingRun(missing=("patch", "git"))
        with mock.patch.object(ov.subprocess, "run", run):
            self.assertFalse(ov._try_apply(self.patch, target_file=self.target, cwd=self.repo))
        self.assertEqual(len(run.calls), 4)


# --------------------------------------------------------------------------- #
# add-module -- whole-file source swap
# --------------------------------------------------------------------------- #
class TestAddModule(_OverlayCase):
    DOTTED = "sglang.srt.layers.activation"

    def _patched_path(self, dotted=None):
        return os.path.join(self.overlay, "_patched", (dotted or self.DOTTED) + ".py")

    def test_patched_file_is_copied_under_its_dotted_name_and_recorded(self):
        src = self._write("candidate.py", "def silu_and_mul():\n    return 'fast'\n")
        out = self._run(ov.cmd_add_module, self._ns(module=self.DOTTED, patched_file=src))

        dst = self._patched_path()
        self.assertEqual(self._read(dst), "def silu_and_mul():\n    return 'fast'\n")
        self.assertEqual(self._manifest()["modules"],
                         [{"module": self.DOTTED, "file": os.path.join("_patched", self.DOTTED + ".py")}])
        self.assertEqual(out, ["OVERLAY_DIR=%s" % self.overlay,
                               "add-module %s -> %s" % (self.DOTTED, dst),
                               "launch with: PYTHONPATH=%s:$PYTHONPATH" % self.overlay])

    def test_manifest_file_is_overlay_relative_because_the_shim_joins_it_to_its_own_dir(self):
        src = self._write("candidate.py", "x = 1\n")
        self._run(ov.cmd_add_module, self._ns(module=self.DOTTED, patched_file=src))
        recorded = self._manifest()["modules"][0]["file"]
        self.assertFalse(os.path.isabs(recorded))
        self.assertTrue(os.path.isfile(os.path.join(self.overlay, recorded)))

    def test_src_file_is_copied_when_no_patched_file_is_given(self):
        src = self._write("installed_activation.py", "ORIGINAL = True\n")
        self._run(ov.cmd_add_module, self._ns(module=self.DOTTED, src_file=src))
        self.assertEqual(self._read(self._patched_path()), "ORIGINAL = True\n")

    def test_the_install_is_the_default_source(self):
        self._import_dir()
        self._write(os.path.join("importable", "ovsg", "__init__.py"), "")
        self._write(os.path.join("importable", "ovsg", "activation.py"), "INSTALLED = 1\n")
        self._forget("ovsg", "ovsg.activation")

        self._run(ov.cmd_add_module, self._ns(module="ovsg.activation"))

        self.assertEqual(self._read(self._patched_path("ovsg.activation")), "INSTALLED = 1\n")

    def test_patch_is_applied_to_the_overlay_copy_not_the_install(self):
        # The whole point of the overlay: site-packages must come out of this untouched.
        src = self._write("installed_activation.py", "ORIGINAL = True\n")
        diff = self._write("kernel.diff", "--- a\n+++ b\n")
        run = _RecordingRun(succeed_at=0)
        with mock.patch.object(ov.subprocess, "run", run):
            self._run(ov.cmd_add_module, self._ns(module=self.DOTTED, src_file=src, patch=diff))

        self.assertEqual(run.argvs, [["patch", self._patched_path(), "-i", diff]])
        self.assertEqual(self._read(src), "ORIGINAL = True\n")
        self.assertEqual(self._manifest()["modules"][0]["module"], self.DOTTED)

    def test_unappliable_patch_aborts_without_recording_the_module(self):
        src = self._write("installed_activation.py", "ORIGINAL = True\n")
        diff = self._write("kernel.diff", "garbage\n")
        run = _RecordingRun()
        with mock.patch.object(ov.subprocess, "run", run):
            with self.assertRaises(SystemExit) as caught:
                self._run(ov.cmd_add_module, self._ns(module=self.DOTTED, src_file=src, patch=diff))

        self.assertIn("failed to apply patch", str(caught.exception))
        # The unpatched copy is left on disk, but the manifest never references it, so the shim
        # will not inject it -- the overlay stays inert rather than running unpatched code.
        self.assertTrue(os.path.isfile(self._patched_path()))
        self.assertEqual(self._manifest()["modules"], [])

    def test_a_patched_file_is_taken_verbatim_and_never_patched(self):
        src = self._write("candidate.py", "ALREADY_PATCHED = True\n")
        diff = self._write("kernel.diff", "--- a\n+++ b\n")
        run = _RecordingRun()
        with mock.patch.object(ov.subprocess, "run", run):
            self._run(ov.cmd_add_module, self._ns(module=self.DOTTED, patched_file=src, patch=diff))
        self.assertEqual(run.calls, [])
        self.assertEqual(self._read(self._patched_path()), "ALREADY_PATCHED = True\n")

    def test_re_adding_the_same_module_replaces_it_in_place(self):
        first = self._write("v1.py", "V = 1\n")
        second = self._write("v2.py", "V = 2\n")
        self._run(ov.cmd_add_module, self._ns(module=self.DOTTED, patched_file=first))
        self._run(ov.cmd_add_module, self._ns(module=self.DOTTED, patched_file=second))

        self.assertEqual(len(self._manifest()["modules"]), 1)
        self.assertEqual(self._read(self._patched_path()), "V = 2\n")

    def test_distinct_modules_compound(self):
        src = self._write("candidate.py", "x = 1\n")
        self._run(ov.cmd_add_module, self._ns(module="sglang.a", patched_file=src))
        self._run(ov.cmd_add_module, self._ns(module="sglang.b", patched_file=src))
        self.assertEqual([e["module"] for e in self._manifest()["modules"]], ["sglang.a", "sglang.b"])

    def test_add_module_leaves_existing_rebinds_and_captures_alone(self):
        src = self._write("candidate.py", "x = 1\n")
        man = ov._ensure_overlay(self.overlay)
        ov._save_man(man, {"rebinds": [{"target": "m:f", "impl_module": "i", "impl_attr": "g"}],
                           "captures": [{"target": "m:h", "out": "/t", "max": 5}]})
        self._run(ov.cmd_add_module, self._ns(module=self.DOTTED, patched_file=src))

        got = self._manifest()
        self.assertEqual(len(got["rebinds"]), 1)
        self.assertEqual(len(got["captures"]), 1)


# --------------------------------------------------------------------------- #
# add-rebind -- single attribute swap (the default path for an accepted kernel)
# --------------------------------------------------------------------------- #
class TestAddRebind(_OverlayCase):
    TARGET = "sglang.srt.layers.activation:silu_and_mul"

    def _ns_rebind(self, **kw):
        base = {"overlay": self.overlay, "target": self.TARGET, "impl_module": "fast_act",
                "impl_attr": "fast_silu_and_mul", "impl_file": ""}
        base.update(kw)
        return argparse.Namespace(**base)

    def test_records_the_rebind_and_prints_the_launch_line(self):
        out = self._run(ov.cmd_add_rebind, self._ns_rebind())
        self.assertEqual(self._manifest()["rebinds"],
                         [{"target": self.TARGET, "impl_module": "fast_act",
                           "impl_attr": "fast_silu_and_mul"}])
        self.assertEqual(out, ["OVERLAY_DIR=%s" % self.overlay,
                               "add-rebind %s -> fast_act.fast_silu_and_mul" % self.TARGET,
                               "launch with: PYTHONPATH=%s:$PYTHONPATH" % self.overlay])

    def test_without_impl_file_nothing_extra_is_written(self):
        # The impl is expected to be importable already; the overlay must not invent a file.
        self._run(ov.cmd_add_rebind, self._ns_rebind())
        self.assertEqual(self._overlay_entries(), ["_overlay_manifest.json", "sitecustomize.py"])

    def test_impl_file_is_copied_to_the_overlay_root_so_it_is_importable(self):
        # The overlay dir itself is on PYTHONPATH, so `import fast_act` must resolve from its root.
        impl = self._write(os.path.join("candidates", "fast_act.py"),
                           "def fast_silu_and_mul():\n    return 1\n")
        self._run(ov.cmd_add_rebind, self._ns_rebind(impl_file=impl))

        copied = os.path.join(self.overlay, "fast_act.py")
        self.assertEqual(self._read(copied), "def fast_silu_and_mul():\n    return 1\n")
        self.assertEqual(self._overlay_entries(),
                         ["_overlay_manifest.json", "fast_act.py", "sitecustomize.py"])

    def test_re_rebinding_the_same_target_replaces_it(self):
        self._run(ov.cmd_add_rebind, self._ns_rebind())
        self._run(ov.cmd_add_rebind, self._ns_rebind(impl_attr="v2"))
        self.assertEqual(self._manifest()["rebinds"],
                         [{"target": self.TARGET, "impl_module": "fast_act", "impl_attr": "v2"}])

    def test_distinct_targets_compound_in_order(self):
        self._run(ov.cmd_add_rebind, self._ns_rebind(target="m:a"))
        self._run(ov.cmd_add_rebind, self._ns_rebind(target="m:b"))
        self.assertEqual([e["target"] for e in self._manifest()["rebinds"]], ["m:a", "m:b"])


# --------------------------------------------------------------------------- #
# add-capture -- the shape/IO oracle hook
# --------------------------------------------------------------------------- #
class TestAddCapture(_OverlayCase):
    TARGET = "sglang.srt.layers.activation:silu_and_mul"

    def _ns_capture(self, **kw):
        base = {"overlay": self.overlay, "target": self.TARGET,
                "out": os.path.join(self.tmp, "task"), "max": 5, "capture_file": ""}
        base.update(kw)
        return argparse.Namespace(**base)

    def test_records_the_hook_and_copies_the_given_capture_file(self):
        cap = self._write("my_capture.py", "def install(target, out_dir, max_cases=5):\n    pass\n")
        out = self._run(ov.cmd_add_capture, self._ns_capture(capture_file=cap, max=3))

        self.assertEqual(self._manifest()["captures"],
                         [{"target": self.TARGET, "out": os.path.join(self.tmp, "task"), "max": 3}])
        self.assertEqual(self._read(os.path.join(self.overlay, "capture_shapes.py")),
                         "def install(target, out_dir, max_cases=5):\n    pass\n")
        self.assertEqual(out[1], "add-capture %s -> %s" % (self.TARGET, os.path.join(self.tmp, "task")))

    def test_capture_file_is_always_named_capture_shapes_py(self):
        # The shim does a bare `import capture_shapes`, so the basename cannot be preserved.
        cap = self._write("my_capture.py", "def install(target, out_dir, max_cases=5):\n    pass\n")
        self._run(ov.cmd_add_capture, self._ns_capture(capture_file=cap))
        self.assertEqual(self._overlay_entries(),
                         ["_overlay_manifest.json", "capture_shapes.py", "sitecustomize.py"])

    def test_default_capture_file_is_the_repo_copy_beside_this_script(self):
        self._run(ov.cmd_add_capture, self._ns_capture())
        copied = self._read(os.path.join(self.overlay, "capture_shapes.py"))
        self.assertEqual(copied, self._read(os.path.join(SCRIPTS_DIR, "capture_shapes.py")))
        self.assertIn("def install(target, out_dir, max_cases=5):", copied)

    def test_max_is_kept_as_an_int_the_shim_can_cast(self):
        self._run(ov.cmd_add_capture, self._ns_capture(max=1))
        self.assertIsInstance(self._manifest()["captures"][0]["max"], int)

    def test_re_capturing_the_same_target_replaces_it(self):
        self._run(ov.cmd_add_capture, self._ns_capture(max=5))
        self._run(ov.cmd_add_capture, self._ns_capture(max=9))
        self.assertEqual(len(self._manifest()["captures"]), 1)
        self.assertEqual(self._manifest()["captures"][0]["max"], 9)

    def test_distinct_targets_compound(self):
        self._run(ov.cmd_add_capture, self._ns_capture(target="m:a"))
        self._run(ov.cmd_add_capture, self._ns_capture(target="m:b"))
        self.assertEqual([e["target"] for e in self._manifest()["captures"]], ["m:a", "m:b"])


class TestAddMarker(_OverlayCase):
    def test_marker_targets_compound_and_copy_the_probe(self):
        marker = self._write("custom_marker.py", "def install(target):\n    pass\n")
        self._run(ov.cmd_add_marker, self._ns(target="m:outer", marker_file=marker))
        self._run(ov.cmd_add_marker, self._ns(target="m:inner", marker_file=marker))
        self.assertEqual(
            [entry["target"] for entry in self._manifest()["markers"]],
            ["m:outer", "m:inner"],
        )
        self.assertEqual(
            self._read(os.path.join(self.overlay, "seam_trace.py")),
            "def install(target):\n    pass\n",
        )

    def test_readding_a_marker_is_idempotent(self):
        marker = self._write("custom_marker.py", "def install(target):\n    pass\n")
        self._run(ov.cmd_add_marker, self._ns(target="m:inner", marker_file=marker))
        self._run(ov.cmd_add_marker, self._ns(target="m:inner", marker_file=marker))
        self.assertEqual(self._manifest()["markers"], [{"target": "m:inner"}])


# --------------------------------------------------------------------------- #
# check -- "is the overlay actually shadowing this module?"
# --------------------------------------------------------------------------- #
class TestCheck(_OverlayCase):
    def _inject_like_the_shim(self, dotted, path):
        """Reproduces the shim's sys.modules injection so find_spec reports the overlay's file."""
        spec = importlib.util.spec_from_file_location(dotted, path)
        sys.modules[dotted] = importlib.util.module_from_spec(spec)
        self._forget(dotted)

    def test_injected_patched_file_reports_overlay_active(self):
        dotted = "sglang.srt.layers.activation"
        src = self._write("candidate.py", "PATCHED = True\n")
        self._run(ov.cmd_add_module, self._ns(module=dotted, patched_file=src))
        self._inject_like_the_shim(dotted, os.path.join(self.overlay, "_patched", dotted + ".py"))

        out = self._run(ov.cmd_check, argparse.Namespace(module=dotted))

        self.assertEqual(out[0], "%s -> %s" % (dotted, os.path.join(self.overlay, "_patched",
                                                                   dotted + ".py")))
        self.assertEqual(out[1], "OVERLAY_ACTIVE")

    def test_untouched_submodule_reports_install(self):
        self._import_dir()
        self._write(os.path.join("importable", "ovchk", "__init__.py"), "")
        self._write(os.path.join("importable", "ovchk", "activation.py"), "INSTALLED = 1\n")
        self._forget("ovchk", "ovchk.activation")

        out = self._run(ov.cmd_check, argparse.Namespace(module="ovchk.activation"))

        self.assertEqual(out[1], "INSTALL (overlay not shadowing this module)")

    def test_plain_top_level_install_is_misreported_as_injected(self):
        """KNOWN BUG, pinned not fixed: the INJECTED test is `f.endswith(module + '.py')`, which
        every ordinary top-level module satisfies (loose -> .../loose.py). `check --module loose`
        therefore claims the overlay is live when nothing has been overlaid at all -- the exact
        false positive that makes an unattributable e2e result look attributed.
        """
        self._import_dir()
        self._write(os.path.join("importable", "ovloose.py"), "INSTALLED = 1\n")
        self._forget("ovloose")

        out = self._run(ov.cmd_check, argparse.Namespace(module="ovloose"))

        self.assertEqual(out[1], "INJECTED")

    def test_absent_module_exits_rather_than_printing_a_verdict(self):
        with self.assertRaises(SystemExit):
            self._run(ov.cmd_check, argparse.Namespace(module="ov_absent_module_xyz"))


# --------------------------------------------------------------------------- #
# main() / _dispatch_add_module -- the argparse surface, driven by explicit argv
# --------------------------------------------------------------------------- #
class TestMain(_OverlayCase):
    def test_add_rebind_end_to_end(self):
        out = self._main(["add-rebind", "--overlay", self.overlay,
                          "--target", "sglang.srt.layers.activation:silu_and_mul",
                          "--impl-module", "fast_act", "--impl-attr", "fast_silu_and_mul"])
        self.assertEqual(self._manifest()["rebinds"],
                         [{"target": "sglang.srt.layers.activation:silu_and_mul",
                           "impl_module": "fast_act", "impl_attr": "fast_silu_and_mul"}])
        self.assertEqual(out[0], "OVERLAY_DIR=%s" % self.overlay)

    def test_monkeypatch_alias_is_identical_to_add_rebind(self):
        args = ["--target", "m:f", "--impl-module", "impl", "--impl-attr", "g"]
        alias = os.path.join(self.tmp, "alias")
        self._main(["add-rebind", "--overlay", self.overlay] + args)
        self._main(["monkeypatch", "--overlay", alias] + args)
        self.assertEqual(self._manifest(alias), self._manifest())

    def test_add_rebind_requires_target_and_impl(self):
        for argv in (["add-rebind", "--overlay", self.overlay],
                     ["add-rebind", "--overlay", self.overlay, "--target", "m:f"],
                     ["add-rebind", "--target", "m:f", "--impl-module", "i", "--impl-attr", "g"]):
            self.assertEqual(self._main_fails(argv).code, 2)
        self.assertFalse(os.path.exists(self.overlay))

    def test_add_module_via_patched_file(self):
        src = self._write("candidate.py", "PATCHED = True\n")
        self._main(["add-module", "--overlay", self.overlay, "--module", "sglang.a",
                    "--patched-file", src])
        self.assertEqual(self._read(os.path.join(self.overlay, "_patched", "sglang.a.py")),
                         "PATCHED = True\n")

    def test_legacy_copy_subtree_package_and_subpath_become_a_dotted_module(self):
        src = self._write("candidate.py", "PATCHED = True\n")
        subpath = os.path.join("srt", "layers", "activation.py")
        self._main(["copy-subtree", "--overlay", self.overlay, "--package", "sglang",
                    "--subpath", subpath, "--patched-file", src])
        self.assertEqual([e["module"] for e in self._manifest()["modules"]],
                         ["sglang.srt.layers.activation"])

    def test_legacy_subpath_without_py_suffix_is_accepted(self):
        src = self._write("candidate.py", "PATCHED = True\n")
        self._main(["copy-subtree", "--overlay", self.overlay, "--package", "sglang",
                    "--subpath", os.path.join("srt", "layers"), "--patched-file", src])
        self.assertEqual([e["module"] for e in self._manifest()["modules"]], ["sglang.srt.layers"])

    def test_explicit_module_wins_over_the_legacy_pair(self):
        src = self._write("candidate.py", "PATCHED = True\n")
        self._main(["add-module", "--overlay", self.overlay, "--module", "sglang.explicit",
                    "--package", "sglang", "--subpath", "srt/legacy.py", "--patched-file", src])
        self.assertEqual([e["module"] for e in self._manifest()["modules"]], ["sglang.explicit"])

    def test_add_module_with_no_module_named_exits_without_creating_an_overlay(self):
        for argv in (["add-module", "--overlay", self.overlay],
                     ["add-module", "--overlay", self.overlay, "--package", "sglang"],
                     ["add-module", "--overlay", self.overlay, "--subpath", "srt/x.py"]):
            with self.assertRaises(SystemExit) as caught:
                self._main(argv)
            self.assertIn("add-module requires --module", str(caught.exception))
        self.assertFalse(os.path.exists(self.overlay))

    def test_add_capture_parses_max_as_an_int(self):
        cap = self._write("my_capture.py", "def install(target, out_dir, max_cases=5):\n    pass\n")
        self._main(["add-capture", "--overlay", self.overlay, "--target", "m:f",
                    "--out", os.path.join(self.tmp, "task"), "--max", "3", "--capture-file", cap])
        self.assertEqual(self._manifest()["captures"][0]["max"], 3)

    def test_add_capture_defaults_max_to_five(self):
        cap = self._write("my_capture.py", "def install(target, out_dir, max_cases=5):\n    pass\n")
        self._main(["add-capture", "--overlay", self.overlay, "--target", "m:f",
                    "--out", os.path.join(self.tmp, "task"), "--capture-file", cap])
        self.assertEqual(self._manifest()["captures"][0]["max"], 5)

    def test_check_end_to_end(self):
        self._import_dir()
        self._write(os.path.join("importable", "ovmain", "__init__.py"), "")
        target = self._write(os.path.join("importable", "ovmain", "activation.py"), "X = 1\n")
        self._forget("ovmain", "ovmain.activation")

        out = self._main(["check", "--module", "ovmain.activation"])

        self.assertEqual(out[0], "ovmain.activation -> %s" % target)

    def test_a_subcommand_is_mandatory(self):
        self.assertEqual(self._main_fails([]).code, 2)

    def test_unknown_subcommand_is_rejected(self):
        # There is no uninstall verb: reversal is dropping the dir from PYTHONPATH.
        self.assertEqual(self._main_fails(["uninstall", "--overlay", self.overlay]).code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
