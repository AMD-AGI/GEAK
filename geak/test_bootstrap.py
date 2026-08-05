#!/usr/bin/env python3
"""Tests for geak/bootstrap.py — the build-time hook that runs on a user's
machine during `pip install git+https://github.com/AMD-AGI/GEAK`.

CONTRACT under test: the bootstrap must (a) resolve GEAK_HOME / repo URL / ref
from the environment exactly as documented, (b) issue the exact git and
installer argv implied by that environment, (c) never clobber an existing
checkout, and (d) degrade to a warning on every failure it is designed to
handle. There is no re-run command — this code executes once, unattended, at
install time, so a regression here silently breaks first-run onboarding for
every new user.

Nothing here executes a subprocess: every entry point of `subprocess` is
replaced with a guard that fails the test if it is reached, and all writes are
confined to a TemporaryDirectory.

Run: python3 -m pytest GEAK/geak/test_bootstrap.py -v
"""
from __future__ import annotations

import contextlib
import importlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import geak
from geak import bootstrap


@contextlib.contextmanager
def _captured():
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield out, err


class _FakeStdout:
    """Stands in for sys.stdout at import time so the isatty()-driven ANSI
    styling is exercised deterministically instead of depending on how the
    test runner happens to capture output."""

    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty

    def write(self, s):
        return len(s)

    def flush(self):
        pass


def _make_checkout(root: str) -> str:
    """The three markers _is_geak_checkout() looks for."""
    os.makedirs(os.path.join(root, ".git"))
    os.makedirs(os.path.join(root, "kernel_workflow"))
    os.makedirs(os.path.join(root, "geak"))
    with open(os.path.join(root, "geak", "bootstrap.py"), "w") as fh:
        fh.write("# stand-in\n")
    return root


class BootstrapTestCase(unittest.TestCase):
    """Base: hard-blocks real process execution and gives every test a private
    HOME/tmp tree."""

    _SUBPROCESS_ENTRYPOINTS = ("run", "call", "check_call", "check_output", "Popen")

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.home = os.path.join(self.root, "home")
        os.makedirs(self.home)
        self.calls = []
        for name in self._SUBPROCESS_ENTRYPOINTS:
            self._patch_subprocess(name, self._forbid(name))

    def _patch_subprocess(self, name, replacement):
        patcher = mock.patch.object(subprocess, name, replacement)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _forbid(self, name):
        def _blocked(*args, **kwargs):
            self.fail("subprocess.%s reached the OS with %r" % (name, args))

        return _blocked

    def stub_run(self, handler):
        """Replace subprocess.run with a recorder. `handler(cmd)` returns
        (returncode, stdout)."""

        def fake(cmd, **kwargs):
            self.calls.append((cmd, kwargs))
            rc, out = handler(cmd)
            return subprocess.CompletedProcess(cmd, rc, out, "")

        self._patch_subprocess("run", fake)

    def stub_which(self, present):
        """shutil.which() answers True only for the named executables."""
        self._patch_shutil(lambda cmd: "/usr/bin/" + cmd if cmd in present else None)

    def _patch_shutil(self, fn):
        patcher = mock.patch.object(bootstrap.shutil, "which", fn)
        patcher.start()
        self.addCleanup(patcher.stop)

    def reload_bootstrap(self, *, isatty=False, **env):
        """Re-import the module under a controlled environment: every knob is
        read once, at import, into a module-level constant."""
        base = {"HOME": self.home, "PATH": ""}
        base.update({k: v for k, v in env.items() if v is not None})
        self.addCleanup(self._restore_module)
        with mock.patch.dict(os.environ, base, clear=True), \
                mock.patch.object(sys, "stdout", _FakeStdout(isatty)):
            return importlib.reload(bootstrap)

    def _restore_module(self):
        with mock.patch.object(sys, "stdout", _FakeStdout(False)):
            importlib.reload(bootstrap)


# ── env knobs -> module constants (resolved once, at import) ────────────────

class EnvKnobTests(BootstrapTestCase):

    def test_env_falls_back_on_unset_and_on_empty(self):
        """An exported-but-empty knob (`export GEAK_REF=`) must read as unset,
        not as an empty branch name that would poison `git clone --branch`."""
        with mock.patch.dict(os.environ, {"GEAK_REF": ""}, clear=False):
            self.assertEqual(bootstrap._env("GEAK_REF", "fallback"), "fallback")
        with mock.patch.dict(os.environ, {"GEAK_REF": "v4.0.0"}, clear=False):
            self.assertEqual(bootstrap._env("GEAK_REF", "fallback"), "v4.0.0")
        os.environ.pop("GEAK_NOT_A_REAL_KNOB", None)
        self.assertEqual(bootstrap._env("GEAK_NOT_A_REAL_KNOB", "d"), "d")

    def test_defaults_when_nothing_is_set(self):
        work = os.path.join(self.root, "work")
        os.makedirs(work)
        mod = self.reload_bootstrap(PWD=work)
        self.assertEqual(mod.REPO_URL, "https://github.com/AMD-AGI/GEAK.git")
        self.assertEqual(mod.REPO_REF, "")
        self.assertEqual(mod.GEAK_HOME, os.path.join(work, "GEAK"))
        self.assertEqual(mod.CLAUDE_VERSION, "latest")
        self.assertEqual(mod.CLAUDE_BIN_DIR,
                         os.path.join(self.home, ".local", "bin"))

    def test_every_knob_overrides_its_default(self):
        work = os.path.join(self.root, "work")
        os.makedirs(work)
        mod = self.reload_bootstrap(
            PWD=work,
            GEAK_REPO_URL="https://example.invalid/fork.git",
            GEAK_REF="release/v4",
            GEAK_HOME="~/checkouts/geak",
            CLAUDE_VERSION="2.1.177",
            CLAUDE_BIN_DIR="~/bin",
        )
        self.assertEqual(mod.REPO_URL, "https://example.invalid/fork.git")
        self.assertEqual(mod.REPO_REF, "release/v4")
        # GEAK_HOME is expanded AND absolutised, so `~` never reaches git.
        self.assertEqual(mod.GEAK_HOME, os.path.join(self.home, "checkouts", "geak"))
        self.assertEqual(mod.CLAUDE_VERSION, "2.1.177")
        self.assertEqual(mod.CLAUDE_BIN_DIR, os.path.join(self.home, "bin"))

    def test_invocation_dir_prefers_pwd_over_cwd(self):
        """pip chdir's into /tmp/pip-req-build-xxxx before the hook runs, so cwd
        is the wrong answer; the shell's PWD names the user's real directory."""
        work = os.path.join(self.root, "work")
        os.makedirs(work)
        with mock.patch.dict(os.environ, {"PWD": work}, clear=False):
            self.assertEqual(bootstrap._invocation_dir(), os.path.abspath(work))

    def test_invocation_dir_falls_back_to_cwd(self):
        for pwd in ("", os.path.join(self.root, "does-not-exist")):
            with mock.patch.dict(os.environ, {"PWD": pwd}, clear=False):
                self.assertEqual(bootstrap._invocation_dir(), os.getcwd())

    def test_is_geak_checkout_requires_all_three_markers(self):
        chk = _make_checkout(os.path.join(self.root, "GEAK"))
        self.assertTrue(bootstrap._is_geak_checkout(chk))
        # A bare directory, and a git repo that is some OTHER project, are not.
        plain = os.path.join(self.root, "plain")
        os.makedirs(os.path.join(plain, ".git"))
        self.assertFalse(bootstrap._is_geak_checkout(plain))
        os.rmdir(os.path.join(chk, "kernel_workflow"))
        self.assertFalse(bootstrap._is_geak_checkout(chk))

    def test_default_home_uses_the_checkout_in_place(self):
        """`git clone ... && cd GEAK && pip install .` must not nest GEAK/GEAK."""
        chk = _make_checkout(os.path.join(self.root, "GEAK"))
        mod = self.reload_bootstrap(PWD=chk)
        self.assertEqual(mod.GEAK_HOME, chk)

    def test_default_home_nests_under_a_plain_dir(self):
        work = os.path.join(self.root, "elsewhere")
        os.makedirs(work)
        self.assertEqual(
            self.reload_bootstrap(PWD=work).GEAK_HOME, os.path.join(work, "GEAK"))

    def test_ansi_styling_only_on_a_tty(self):
        """Piped install logs must stay free of escape junk."""
        work = os.path.join(self.root, "work")
        os.makedirs(work)
        mod = self.reload_bootstrap(PWD=work, isatty=True)
        self.assertEqual((mod.C_CMD, mod.C_OFF), ("\033[1;32m", "\033[0m"))
        mod = self.reload_bootstrap(PWD=work, isatty=False)
        self.assertEqual((mod.C_CMD, mod.C_OFF), ("", ""))


# ── version parsing ─────────────────────────────────────────────────────────

class VersionTests(BootstrapTestCase):

    def test_ver_tuple_stops_at_the_first_non_digit(self):
        self.assertEqual(bootstrap._ver_tuple("2.1.177"), (2, 1, 177))
        self.assertEqual(bootstrap._ver_tuple("2.1.206-beta.3"), (2, 1, 206, 3))
        # A field with no leading digits contributes 0 rather than raising.
        self.assertEqual(bootstrap._ver_tuple("2.x.1"), (2, 0, 1))
        self.assertEqual(bootstrap._ver_tuple(""), (0,))

    def test_ver_ge_compares_fields_numerically_not_lexically(self):
        self.assertTrue(bootstrap._ver_ge("2.1.177", bootstrap.CLAUDE_MIN_VERSION))
        # The lexical trap: "2.1.99" > "2.1.177" as strings, but is older.
        self.assertFalse(bootstrap._ver_ge("2.1.99", "2.1.177"))
        self.assertTrue(bootstrap._ver_ge("2.1.206", "2.1.177"))
        self.assertTrue(bootstrap._ver_ge("10.0.0", "9.9.9"))

    def test_claude_version_extracts_leading_semver(self):
        self.stub_run(lambda cmd: (0, "2.1.206 (Claude Code)\n"))
        self.assertEqual(bootstrap.claude_version(), "2.1.206")
        cmd, kwargs = self.calls[0]
        self.assertEqual(cmd, ["claude", "--version"])
        self.assertTrue(kwargs["capture_output"] and kwargs["text"])

    def test_claude_version_empty_when_output_has_no_semver(self):
        self.stub_run(lambda cmd: (0, "unknown build\n"))
        self.assertEqual(bootstrap.claude_version(), "")

    def test_claude_version_empty_when_binary_is_missing(self):
        """`claude` not on PATH raises FileNotFoundError inside subprocess.run;
        that must read as 'no version', never propagate into the install."""

        def boom(cmd, **kwargs):
            raise FileNotFoundError(2, "No such file or directory: 'claude'")

        self._patch_subprocess("run", boom)
        self.assertEqual(bootstrap.claude_version(), "")


# ── logging + the _run wrapper ──────────────────────────────────────────────

class RunWrapperTests(BootstrapTestCase):

    def test_log_and_warn_are_tagged_and_split_across_streams(self):
        with _captured() as (out, err):
            bootstrap.log("hello")
            bootstrap.warn("uh oh")
        self.assertEqual(out.getvalue(), "[geak-bootstrap] hello\n")
        self.assertEqual(err.getvalue(), "[geak-bootstrap WARN] uh oh\n")

    def test_has_reports_executable_presence(self):
        self.stub_which({"git"})
        self.assertTrue(bootstrap._has("git"))
        self.assertFalse(bootstrap._has("npm"))

    def test_run_echoes_argv_and_forwards_shell_flag(self):
        self.stub_run(lambda cmd: (0, ""))
        with _captured() as (out, _):
            self.assertEqual(bootstrap._run(["git", "clone", "x"]).returncode, 0)
            bootstrap._run("curl x | bash", shell=True)
        self.assertEqual(self.calls[0], (["git", "clone", "x"], {"shell": False}))
        self.assertEqual(self.calls[1], ("curl x | bash", {"shell": True}))
        # The echoed line is the copy-pasteable command, list or string.
        self.assertIn("[geak-bootstrap] git clone x", out.getvalue())
        self.assertIn("[geak-bootstrap] curl x | bash", out.getvalue())


# ── step 1: clone the repo ──────────────────────────────────────────────────

class CloneRepoTests(BootstrapTestCase):

    def _clone_into(self, dest, *, ref="", url="https://example.invalid/GEAK.git"):
        patches = [
            mock.patch.object(bootstrap, "GEAK_HOME", dest),
            mock.patch.object(bootstrap, "REPO_REF", ref),
            mock.patch.object(bootstrap, "REPO_URL", url),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_clone_argv_and_parent_creation(self):
        dest = os.path.join(self.root, "nested", "GEAK")
        self._clone_into(dest)
        self.stub_which({"git"})
        self.stub_run(lambda cmd: (0, ""))
        with mock.patch.dict(os.environ, {"PWD": self.root}, clear=False), \
                _captured() as (out, err):
            bootstrap.clone_repo()
        self.assertEqual(self.calls[0][0], [
            "git", "clone", "--depth", "1",
            "https://example.invalid/GEAK.git", dest])
        # The parent is created for git, but the clone target itself is not.
        self.assertTrue(os.path.isdir(os.path.dirname(dest)))
        self.assertFalse(os.path.exists(dest))
        self.assertIn("GEAK repo downloaded to %s" % dest, out.getvalue())
        self.assertEqual(err.getvalue(), "")

    def test_clone_pins_the_branch_when_geak_ref_is_set(self):
        dest = os.path.join(self.root, "GEAK")
        self._clone_into(dest, ref="release/v4")
        self.stub_which({"git"})
        self.stub_run(lambda cmd: (0, ""))
        with mock.patch.dict(os.environ, {"PWD": self.root}, clear=False), \
                _captured():
            bootstrap.clone_repo()
        self.assertEqual(self.calls[0][0], [
            "git", "clone", "--depth", "1", "--branch", "release/v4",
            "https://example.invalid/GEAK.git", dest])

    def test_clone_failure_warns_and_does_not_raise(self):
        dest = os.path.join(self.root, "GEAK")
        self._clone_into(dest)
        self.stub_which({"git"})
        self.stub_run(lambda cmd: (128, ""))
        with mock.patch.dict(os.environ, {"PWD": self.root}, clear=False), \
                _captured() as (out, err):
            self.assertIsNone(bootstrap.clone_repo())
        self.assertIn("git clone failed", err.getvalue())
        self.assertNotIn("downloaded", out.getvalue())

    def test_missing_git_warns_and_never_shells_out(self):
        dest = os.path.join(self.root, "GEAK")
        self._clone_into(dest)
        self.stub_which(set())
        with mock.patch.dict(os.environ, {"PWD": self.root}, clear=False), \
                _captured() as (_, err):
            bootstrap.clone_repo()
        self.assertIn("git not found", err.getvalue())
        self.assertIn(dest, err.getvalue())
        self.assertEqual(self.calls, [])

    def test_existing_non_empty_target_is_left_untouched(self):
        """The user's work must survive a re-install: a populated GEAK_HOME is
        never overwritten and never cloned into."""
        dest = os.path.join(self.root, "GEAK")
        os.makedirs(dest)
        keep = os.path.join(dest, "my_kernel.py")
        with open(keep, "w") as fh:
            fh.write("precious\n")
        self._clone_into(dest)
        self.stub_which({"git"})
        with mock.patch.dict(os.environ, {"PWD": self.root}, clear=False), \
                _captured() as (_, err):
            bootstrap.clone_repo()
        self.assertIn("already exists and is not empty", err.getvalue())
        self.assertIn("set GEAK_HOME to another path", err.getvalue())
        self.assertEqual(self.calls, [])
        with open(keep) as fh:
            self.assertEqual(fh.read(), "precious\n")

    def test_existing_empty_target_is_cloned_into(self):
        dest = os.path.join(self.root, "GEAK")
        os.makedirs(dest)
        self._clone_into(dest)
        self.stub_which({"git"})
        self.stub_run(lambda cmd: (0, ""))
        with mock.patch.dict(os.environ, {"PWD": self.root}, clear=False), \
                _captured():
            bootstrap.clone_repo()
        self.assertEqual(len(self.calls), 1)

    def test_running_inside_a_checkout_uses_it_in_place(self):
        """`cd GEAK && pip install .` must respect the user's branch/worktree
        instead of cloning over it."""
        chk = _make_checkout(os.path.join(self.root, "GEAK"))
        self._clone_into(chk)
        self.stub_which({"git"})
        with mock.patch.dict(os.environ, {"PWD": chk}, clear=False), \
                _captured() as (out, err):
            bootstrap.clone_repo()
        self.assertIn("using it in place", out.getvalue())
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(self.calls, [])

    def test_checkout_elsewhere_is_still_cloned_not_reused(self):
        """The in-place shortcut is keyed on GEAK_HOME == the invocation dir; a
        checkout at some OTHER path falls through to the not-empty guard."""
        chk = _make_checkout(os.path.join(self.root, "GEAK"))
        self._clone_into(chk)
        self.stub_which({"git"})
        with mock.patch.dict(os.environ, {"PWD": self.root}, clear=False), \
                _captured() as (_, err):
            bootstrap.clone_repo()
        self.assertIn("already exists and is not empty", err.getvalue())
        self.assertEqual(self.calls, [])

    def test_git_vanishing_after_the_probe_propagates(self):
        """Documented behaviour, not an endorsement: _run() has no try/except, so
        a FileNotFoundError between the shutil.which probe and the exec escapes
        clone_repo(). Only setup.py's blanket except keeps the install alive."""
        dest = os.path.join(self.root, "GEAK")
        self._clone_into(dest)
        self.stub_which({"git"})

        def vanished(cmd, **kwargs):
            raise FileNotFoundError(2, "No such file or directory: 'git'")

        self._patch_subprocess("run", vanished)
        with mock.patch.dict(os.environ, {"PWD": self.root}, clear=False), \
                _captured():
            with self.assertRaises(FileNotFoundError):
                bootstrap.clone_repo()


# ── step 2: Claude Code CLI ─────────────────────────────────────────────────

class InstallClaudeNativeTests(BootstrapTestCase):

    def test_curl_installer_command_carries_the_pinned_version(self):
        self.stub_which({"curl", "npm"})
        self.stub_run(lambda cmd: (0, ""))
        with mock.patch.object(bootstrap, "CLAUDE_VERSION", "2.1.177"), \
                _captured() as (out, _):
            self.assertTrue(bootstrap._install_claude_native())
        cmd, kwargs = self.calls[0]
        self.assertEqual(
            cmd,
            "curl -fsSL --connect-timeout 20 https://claude.ai/install.sh "
            "| bash -s 2.1.177")
        self.assertTrue(kwargs["shell"])
        # npm is available but must NOT run once the native installer succeeded.
        self.assertEqual(len(self.calls), 1)
        self.assertIn("via the native installer", out.getvalue())

    def test_npm_fallback_after_native_installer_fails(self):
        self.stub_which({"curl", "npm"})
        self.stub_run(lambda cmd: (0, "") if cmd[0] == "npm" else (1, ""))
        with _captured() as (_, err):
            self.assertTrue(bootstrap._install_claude_native())
        self.assertEqual(self.calls[1][0],
                         ["npm", "install", "-g", "@anthropic-ai/claude-code"])
        self.assertIn("native installer failed", err.getvalue())
        self.assertIn("falling back to npm", err.getvalue())

    def test_npm_only_host_skips_curl_entirely(self):
        self.stub_which({"npm"})
        self.stub_run(lambda cmd: (0, ""))
        with _captured():
            self.assertTrue(bootstrap._install_claude_native())
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][0][0], "npm")

    def test_both_installers_failing_returns_false_with_manual_pointer(self):
        self.stub_which({"curl", "npm"})
        self.stub_run(lambda cmd: (1, ""))
        with _captured() as (_, err):
            self.assertFalse(bootstrap._install_claude_native())
        self.assertEqual(len(self.calls), 2)
        self.assertIn("could not install Claude Code", err.getvalue())
        self.assertIn("https://code.claude.com/docs/en/setup", err.getvalue())

    def test_no_curl_and_no_npm_bails_out_before_any_subprocess(self):
        self.stub_which(set())
        with _captured() as (_, err):
            self.assertFalse(bootstrap._install_claude_native())
        self.assertEqual(self.calls, [])
        self.assertIn("need curl (native installer) or npm", err.getvalue())


class EnsureClaudeCodeTests(BootstrapTestCase):

    def _versions(self, *sequence):
        """Queue the answers `claude --version` gives, in order."""
        queue = list(sequence)

        def handler(cmd):
            if cmd[:2] == ["claude", "--version"]:
                return 0, (queue.pop(0) if queue else "")
            return 0, ""

        self.stub_run(handler)

    def test_new_enough_install_is_left_alone(self):
        self.stub_which({"curl"})
        self._versions("2.1.206 (Claude Code)")
        with _captured() as (out, err):
            bootstrap.ensure_claude_code()
        self.assertEqual(len(self.calls), 1)
        self.assertIn("Claude Code present (2.1.206) >= 2.1.177", out.getvalue())
        self.assertEqual(err.getvalue(), "")

    def test_outdated_install_is_updated_in_place(self):
        self.stub_which({"curl"})
        self._versions("2.1.100", "2.1.206")
        with _captured() as (out, err):
            bootstrap.ensure_claude_code()
        self.assertEqual([c[0] for c in self.calls], [
            ["claude", "--version"], ["claude", "update"], ["claude", "--version"]])
        self.assertIn("is older than 2.1.177; updating", err.getvalue())
        self.assertIn("Claude Code updated to 2.1.206", out.getvalue())

    def test_failed_update_falls_through_to_a_reinstall(self):
        self.stub_which({"curl"})
        self._versions("2.1.100", "2.1.100", "2.1.206")
        with _captured() as (out, err):
            bootstrap.ensure_claude_code()
        argvs = [c[0] for c in self.calls]
        self.assertIn(["claude", "update"], argvs)
        self.assertTrue(any(isinstance(a, str) and "install.sh" in a for a in argvs),
                        "a stuck `claude update` must escalate to the installer")
        self.assertEqual(err.getvalue().count("WARN"), 1)
        self.assertIn("2.1.100", err.getvalue())
        self.assertNotIn("still <", err.getvalue())
        self.assertNotIn("updated to", out.getvalue())

    def test_missing_cli_is_installed_then_reverified(self):
        self.stub_which({"curl"})
        self._versions("", "2.1.206")
        with _captured() as (_, err):
            bootstrap.ensure_claude_code()
        self.assertIn("Claude Code CLI not found", err.getvalue())
        self.assertTrue(any(isinstance(c[0], str) and "install.sh" in c[0]
                            for c in self.calls))
        # Reverified as good => no PATH / too-old complaint.
        self.assertNotIn("not on your PATH", err.getvalue())
        self.assertNotIn("still <", err.getvalue())

    def test_install_that_lands_off_path_names_the_bin_dir(self):
        self.stub_which({"curl"})
        self._versions("", "")
        bin_dir = os.path.join(self.home, ".local", "bin")
        with mock.patch.object(bootstrap, "CLAUDE_BIN_DIR", bin_dir), \
                _captured() as (_, err):
            bootstrap.ensure_claude_code()
        self.assertIn("claude not on PATH after install", err.getvalue())
        self.assertIn(bin_dir, err.getvalue())

    def test_install_that_lands_too_old_tells_the_user_how_to_pin(self):
        self.stub_which({"curl"})
        self._versions("", "2.1.10")
        with _captured() as (_, err):
            bootstrap.ensure_claude_code()
        self.assertIn("installed Claude Code 2.1.10 is still < 2.1.177",
                      err.getvalue())
        self.assertIn("CLAUDE_VERSION", err.getvalue())


# ── step 3: environment detection ───────────────────────────────────────────

class CheckEnvironmentTests(BootstrapTestCase):

    def test_reports_rocm_profiler_and_backend_when_all_present(self):
        self.stub_which({"rocminfo", "rocprofv3"})
        self.stub_run(lambda cmd: (0, ""))
        with _captured() as (out, err):
            bootstrap.check_environment()
        self.assertIn("ROCm: present", out.getvalue())
        # First hit in the documented preference order wins.
        self.assertIn("profiler: rocprofv3", out.getvalue())
        self.assertIn("serving backend: sglang", out.getvalue())
        self.assertEqual(err.getvalue(), "")
        # Probed with THIS interpreter, and stopped at the first importable one.
        self.assertEqual(self.calls[0][0], [sys.executable, "-c", "import sglang"])
        self.assertEqual(len(self.calls), 1)

    def test_falls_through_to_vllm_when_sglang_is_absent(self):
        self.stub_which({"rocm-smi", "rocprof-compute"})
        self.stub_run(lambda cmd: (1, "") if "sglang" in cmd[-1] else (0, ""))
        with _captured() as (out, _):
            bootstrap.check_environment()
        self.assertEqual([c[0][-1] for c in self.calls],
                         ["import sglang", "import vllm"])
        self.assertIn("serving backend: vllm", out.getvalue())
        self.assertIn("profiler: rocprof-compute", out.getvalue())

    def test_bare_host_warns_on_every_axis_without_failing(self):
        self.stub_which(set())
        self.stub_run(lambda cmd: (1, ""))
        with _captured() as (_, err):
            self.assertIsNone(bootstrap.check_environment())
        self.assertIn("ROCm not detected", err.getvalue())
        self.assertIn("no profiler found", err.getvalue())
        self.assertIn("no serving backend", err.getvalue())


# ── step 4: next steps ──────────────────────────────────────────────────────

class PrintNextStepsTests(BootstrapTestCase):

    def _bin_dir(self, *, with_claude):
        bin_dir = os.path.join(self.home, ".local", "bin")
        os.makedirs(bin_dir)
        if with_claude:
            with open(os.path.join(bin_dir, "claude"), "w") as fh:
                fh.write("#!/bin/sh\n")
        return bin_dir

    def _print(self, bin_dir, path_entries, geak_home):
        patches = [
            mock.patch.object(bootstrap, "CLAUDE_BIN_DIR", bin_dir),
            mock.patch.object(bootstrap, "GEAK_HOME", geak_home),
            mock.patch.object(bootstrap, "C_CMD", ""),
            mock.patch.object(bootstrap, "C_OFF", ""),
            mock.patch.dict(os.environ,
                            {"PATH": os.pathsep.join(path_entries)}, clear=False),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with _captured() as (out, _):
            bootstrap.print_next_steps()
        return out.getvalue()

    def test_warns_when_the_installed_cli_is_off_path(self):
        bin_dir = self._bin_dir(with_claude=True)
        text = self._print(bin_dir, ["/usr/bin"], os.path.join(self.root, "GEAK"))
        self.assertIn("which is not on", text)
        self.assertIn("export PATH=\"%s:$PATH\"" % bin_dir, text)

    def test_no_path_note_when_the_bin_dir_is_already_on_path(self):
        bin_dir = self._bin_dir(with_claude=True)
        # A trailing-slash spelling still matches: PATH entries are absolutised.
        text = self._print(bin_dir, ["/usr/bin", bin_dir + os.sep],
                           os.path.join(self.root, "GEAK"))
        self.assertNotIn("is not on", text)

    def test_no_path_note_when_no_cli_was_installed_there(self):
        bin_dir = self._bin_dir(with_claude=False)
        text = self._print(bin_dir, ["/usr/bin"], os.path.join(self.root, "GEAK"))
        self.assertNotIn("is not on", text)

    def test_next_steps_name_the_three_auth_paths_and_the_launch_dir(self):
        """These lines are the entire onboarding UX — if GEAK_HOME is wrong here
        the user's first `cd` lands nowhere."""
        geak_home = os.path.join(self.root, "GEAK")
        text = self._print(self._bin_dir(with_claude=False), ["/usr/bin"], geak_home)
        self.assertIn("setup complete", text)
        self.assertIn("export ANTHROPIC_API_KEY=", text)
        self.assertIn("export ANTHROPIC_BASE_URL=", text)
        self.assertIn("export ANTHROPIC_AUTH_TOKEN=", text)
        self.assertIn("cd %s" % geak_home, text)
        self.assertIn("IS_SANDBOX=1 claude --dangerously-skip-permissions", text)

    def test_styling_wraps_the_copy_paste_commands_on_a_tty(self):
        with mock.patch.object(bootstrap, "CLAUDE_BIN_DIR", self.root), \
                mock.patch.object(bootstrap, "GEAK_HOME", self.root), \
                mock.patch.object(bootstrap, "C_CMD", "\033[1;32m"), \
                mock.patch.object(bootstrap, "C_OFF", "\033[0m"), \
                _captured() as (out, _):
            bootstrap.print_next_steps()
        self.assertIn("\033[1;32mexport ANTHROPIC_API_KEY=sk-ant-...\033[0m",
                      out.getvalue())


# ── orchestration ───────────────────────────────────────────────────────────

class MainTests(BootstrapTestCase):

    def _spy_steps(self):
        seen = []
        for name in ("clone_repo", "ensure_claude_code", "check_environment",
                     "print_next_steps"):
            p = mock.patch.object(bootstrap, name,
                                  lambda n=name: seen.append(n))
            p.start()
            self.addCleanup(p.stop)
        return seen

    def test_main_runs_the_four_steps_in_order(self):
        seen = self._spy_steps()
        env = dict(os.environ)
        env.pop("GEAK_SKIP_BOOTSTRAP", None)
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(bootstrap, "GEAK_HOME", "/tmp/does/not/matter"), \
                _captured() as (out, _):
            self.assertIsNone(bootstrap.main())
        self.assertEqual(seen, ["clone_repo", "ensure_claude_code",
                                "check_environment", "print_next_steps"])
        self.assertIn("GEAK_HOME=/tmp/does/not/matter", out.getvalue())

    def test_geak_skip_bootstrap_suppresses_every_side_effect(self):
        """The CI / docker-image escape hatch: no clone, no installer, nothing."""
        seen = self._spy_steps()
        with mock.patch.dict(os.environ, {"GEAK_SKIP_BOOTSTRAP": "1"}, clear=False), \
                _captured() as (out, _):
            bootstrap.main()
        self.assertEqual(seen, [])
        self.assertIn("GEAK_SKIP_BOOTSTRAP set", out.getvalue())
        self.assertEqual(self.calls, [])

    def test_main_completes_on_a_host_where_every_step_fails(self):
        """A laptop with no git, no curl/npm, no ROCm: the bootstrap must still
        finish and print next steps rather than abort `pip install`."""
        self.stub_which(set())
        self.stub_run(lambda cmd: (1, ""))
        env = dict(os.environ)
        env.pop("GEAK_SKIP_BOOTSTRAP", None)
        env["PATH"] = "/usr/bin"
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(bootstrap, "GEAK_HOME",
                                  os.path.join(self.root, "GEAK")), \
                mock.patch.object(bootstrap, "CLAUDE_BIN_DIR",
                                  os.path.join(self.home, "bin")), \
                _captured() as (out, err):
            self.assertIsNone(bootstrap.main())
        self.assertIn("git not found", err.getvalue())
        self.assertIn("need curl (native installer) or npm", err.getvalue())
        self.assertIn("setup complete", out.getvalue())
        # Nothing was cloned or installed into the temp tree.
        self.assertEqual(os.listdir(self.root), ["home"])


class PackageTests(unittest.TestCase):

    def test_version_matches_the_packaged_metadata(self):
        """geak/__init__.py must stay import-safe at build time (stdlib only) and
        keep the version pyproject.toml ships."""
        self.assertEqual(geak.__version__, "4.0.0")


if __name__ == "__main__":
    raise SystemExit(unittest.main())
