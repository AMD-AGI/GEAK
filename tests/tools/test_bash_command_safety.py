"""Tests for the bash tool safety layers (L1 timeout + L2 scan-scope firewall).

These layers exist because the bash tool is the only unsupervised entry
point the LLM has, and commands like ``find /`` or ``find /wekafs``
would otherwise stall the agent for hours on NFS-backed mounts.

L1 — wall-clock timeout + SIGKILL the entire process group on expiry
     so grandchildren (e.g. ``find`` spawned from ``timeout``-less
     wrapper scripts) do not survive.

L2 — ancestry-based firewall rejecting recursive scans whose root
     escapes the allowed set (``$GEAK_REPO_ROOT``, ``$GEAK_WORK_DIR``,
     cwd-when-inside-a-root, scratch, common system dirs).
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

from minisweagent.tools.bash_command import (
    BashCommand,
    _allowed_search_roots,
    _check_command_scope,
    _extract_search_paths,
    _is_recursive_scan,
    _resolve_path_token,
    _split_shell_segments,
)


# ---------------------------------------------------------------------------
# Unit tests for the parsing / firewall helpers
# ---------------------------------------------------------------------------


class TestSplitShellSegments:
    def test_simple(self):
        assert _split_shell_segments("echo hi") == ["echo hi"]

    def test_and_or(self):
        assert _split_shell_segments("a && b || c") == ["a", "b", "c"]

    def test_semicolon_and_pipe(self):
        assert _split_shell_segments("ls; grep x | head") == ["ls", "grep x", "head"]


class TestIsRecursiveScan:
    def test_find_always_recursive(self):
        assert _is_recursive_scan(["find", "."]) is True

    def test_rg_always_recursive(self):
        assert _is_recursive_scan(["rg", "pattern"]) is True

    def test_grep_without_r_is_not_scan(self):
        assert _is_recursive_scan(["grep", "pattern", "file.py"]) is False

    def test_grep_with_dash_r(self):
        assert _is_recursive_scan(["grep", "-r", "pattern", "/"]) is True

    def test_grep_with_dash_R(self):
        assert _is_recursive_scan(["grep", "-R", "pattern", "/"]) is True

    def test_grep_combined_short_flag(self):
        # ``grep -rni`` is recursive.
        assert _is_recursive_scan(["grep", "-rni", "pattern", "/"]) is True

    def test_ls_without_R_is_not_scan(self):
        assert _is_recursive_scan(["ls", "-la", "/"]) is False

    def test_ls_with_R(self):
        assert _is_recursive_scan(["ls", "-R", "/"]) is True

    def test_du_without_a_is_not_scan(self):
        assert _is_recursive_scan(["du", "-sh", "/"]) is False

    def test_du_with_a(self):
        assert _is_recursive_scan(["du", "-a", "/"]) is True

    def test_non_scan_binary(self):
        assert _is_recursive_scan(["echo", "hi"]) is False


class TestExtractSearchPaths:
    def test_find_single_path_before_flag(self):
        assert _extract_search_paths(["find", "/wekafs", "-name", "foo"], None) == ["/wekafs"]

    def test_find_multiple_paths_before_flag(self):
        assert _extract_search_paths(["find", "/a", "/b", "-name", "x"], None) == ["/a", "/b"]

    def test_find_no_explicit_path_defaults_to_cwd(self):
        assert _extract_search_paths(["find", "-name", "foo"], "/cwd/here") == ["/cwd/here"]

    def test_find_name_arg_not_treated_as_path(self):
        # ``-name '/foo'`` must NOT be classified as a search root.
        paths = _extract_search_paths(["find", "/tmp", "-name", "/foo"], None)
        assert paths == ["/tmp"]
        assert "/foo" not in paths

    def test_grep_path_anywhere(self):
        # path appears after the pattern
        paths = _extract_search_paths(["grep", "-r", "pattern", "/"], None)
        assert "/" in paths

    def test_grep_dash_e_value_is_not_path(self):
        # ``grep -e PATTERN`` should skip PATTERN as path even if it looks like one.
        paths = _extract_search_paths(["grep", "-r", "-e", "/usr/", "/tmp"], None)
        assert "/usr/" not in paths
        assert "/tmp" in paths


class TestResolvePathToken:
    def test_absolute(self):
        assert _resolve_path_token("/wekafs/x", None) == "/wekafs/x"

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("FOO", "/repo")
        assert _resolve_path_token("$FOO/sub", None) == "/repo/sub"

    def test_tilde(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/user")
        assert _resolve_path_token("~/repo", None) == "/home/user/repo"

    def test_relative_uses_cwd(self):
        assert _resolve_path_token("sub/dir", "/cwd") == "/cwd/sub/dir"


class TestAllowedSearchRoots:
    def test_includes_env_vars(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GEAK_REPO_ROOT", str(tmp_path / "repo"))
        monkeypatch.setenv("GEAK_WORK_DIR", str(tmp_path / "wt"))
        roots = _allowed_search_roots(None)
        assert str(tmp_path / "repo") in roots
        assert str(tmp_path / "wt") in roots

    def test_includes_system_roots(self, monkeypatch):
        monkeypatch.delenv("GEAK_REPO_ROOT", raising=False)
        monkeypatch.delenv("GEAK_WORK_DIR", raising=False)
        roots = _allowed_search_roots(None)
        for sysroot in ("/tmp", "/var/tmp", "/opt", "/usr", "/etc", "/var/lib"):
            assert sysroot in roots

    def test_does_not_include_arbitrary_cwd(self, monkeypatch, tmp_path):
        # cwd is NOT a top-level allowed root: an arbitrary cwd would let a
        # misconfigured agent bypass the firewall.  cwd is only used to
        # resolve relative paths in the command.
        monkeypatch.delenv("GEAK_REPO_ROOT", raising=False)
        monkeypatch.delenv("GEAK_WORK_DIR", raising=False)
        roots = _allowed_search_roots(str(tmp_path))
        assert str(tmp_path) not in roots


# ---------------------------------------------------------------------------
# L2 — _check_command_scope behavioural tests
# ---------------------------------------------------------------------------


class TestScopeFirewallBlocks:
    """Commands whose scan root escapes the allowed set must be rejected."""

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch, tmp_path):
        self.repo = tmp_path / "repo"
        self.work = tmp_path / "worktree"
        self.repo.mkdir()
        self.work.mkdir()
        monkeypatch.setenv("GEAK_REPO_ROOT", str(self.repo))
        monkeypatch.setenv("GEAK_WORK_DIR", str(self.work))

    @pytest.mark.parametrize(
        "cmd",
        [
            "find / -name foo",
            "find /wekafs -name foo",
            "find /home -name foo",
            "find /root -name foo",
            "find /proc -name foo",
            "find /sys -name foo",
            "find /mnt -name foo",
            "find /sgl-workspace -name foo",
            "grep -r 'pattern' /",
            "grep -R pattern /home",
            "rg pattern /",
            "tree /",
            "ls -R /",
            "du -a /",
        ],
    )
    def test_blocks_unbounded_scan(self, cmd):
        err = _check_command_scope(cmd, cwd=None)
        assert err is not None, f"should block: {cmd!r}"
        assert "Blocked" in err

    def test_block_message_includes_env_values(self):
        err = _check_command_scope("find /wekafs -name foo", cwd=None)
        assert err is not None
        assert str(self.repo) in err
        assert str(self.work) in err

    def test_block_message_includes_rewrite_examples(self):
        err = _check_command_scope("find / -name foo", cwd=None)
        assert err is not None
        assert "$GEAK_REPO_ROOT" in err
        assert "$GEAK_WORK_DIR" in err

    def test_composite_command_blocks_inner_bad(self):
        err = _check_command_scope("cd /tmp && find / -name foo", cwd=None)
        assert err is not None

    def test_cwd_outside_any_root_is_blocked_for_relative_find(self, monkeypatch):
        # cwd is not inside any allowed root → ``find .`` resolves to a
        # path outside the allowlist and must be rejected.
        monkeypatch.delenv("GEAK_REPO_ROOT", raising=False)
        monkeypatch.delenv("GEAK_WORK_DIR", raising=False)
        err = _check_command_scope("find . -name foo", cwd="/some/random/dir")
        assert err is not None


class TestScopeFirewallAllows:
    """Commands whose scan root is inside an allowed root must pass."""

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch, tmp_path):
        self.repo = tmp_path / "repo"
        self.work = tmp_path / "worktree"
        self.repo.mkdir()
        self.work.mkdir()
        (self.repo / "sub").mkdir()
        monkeypatch.setenv("GEAK_REPO_ROOT", str(self.repo))
        monkeypatch.setenv("GEAK_WORK_DIR", str(self.work))

    def test_repo_root_allowed(self):
        cmd = f"find {self.repo} -name '*.py'"
        assert _check_command_scope(cmd, cwd=None) is None

    def test_repo_subdir_allowed(self):
        cmd = f"find {self.repo}/sub -name '*.py'"
        assert _check_command_scope(cmd, cwd=None) is None

    def test_worktree_allowed(self):
        cmd = f"find {self.work} -name '*.py'"
        assert _check_command_scope(cmd, cwd=None) is None

    def test_repo_root_via_env_var(self):
        cmd = "find $GEAK_REPO_ROOT -name '*.py'"
        assert _check_command_scope(cmd, cwd=None) is None

    def test_worktree_via_env_var(self):
        cmd = "find $GEAK_WORK_DIR -name '*.py'"
        assert _check_command_scope(cmd, cwd=None) is None

    def test_dot_when_cwd_inside_repo(self):
        cmd = "find . -name '*.py'"
        assert _check_command_scope(cmd, cwd=str(self.repo)) is None

    @pytest.mark.parametrize(
        "cmd",
        [
            "find /tmp -name foo",
            "find /var/tmp -name foo",
            "find /opt/rocm -name '*.h'",
            "find /opt -maxdepth 3 -name 'rocm*'",
            "find /usr/include -name '*.h'",
            "find /etc -name 'os-release'",
            "find /var/lib/dpkg -name status",
        ],
    )
    def test_system_dirs_allowed(self, cmd):
        assert _check_command_scope(cmd, cwd=None) is None

    @pytest.mark.parametrize(
        "cmd",
        [
            "echo hello",
            "ls -la /tmp",
            "cat /etc/os-release",
            "pip install -e .",
            "python -c 'print(1)'",
            "grep pattern file.py",
        ],
    )
    def test_non_scan_commands_pass(self, cmd):
        # Non-scan or non-recursive commands are not subject to L2.
        assert _check_command_scope(cmd, cwd=None) is None


# ---------------------------------------------------------------------------
# L1 — wall-clock timeout + process-group SIGKILL
# ---------------------------------------------------------------------------


class TestTimeoutAndKillpg:
    @pytest.fixture(autouse=True)
    def _short_timeout(self, monkeypatch):
        # 2 seconds is short enough to make the test fast but long enough
        # to clear startup jitter on slow CI.
        monkeypatch.setenv("GEAK_BASH_TIMEOUT_SEC", "2")

    def test_normal_command_completes(self):
        bash = BashCommand()
        result = bash(command="echo hello")
        assert result["returncode"] == 0
        assert "hello" in result["output"]

    def test_sleep_command_times_out(self):
        bash = BashCommand()
        start = time.monotonic()
        result = bash(command="sleep 30")
        elapsed = time.monotonic() - start
        assert elapsed < 10, "timeout did not fire within budget"
        assert result["returncode"] != 0
        assert "timed out" in result["output"].lower()

    def test_timeout_kills_grandchildren(self, tmp_path):
        # Spawn a child shell that backgrounds a long sleep; without
        # ``start_new_session=True`` + ``killpg`` the grandchild would
        # survive the timeout and leak.  We assert the grandchild dies
        # within a few seconds of timeout firing by checking that no
        # process exists with the marker after kill.
        marker = tmp_path / "marker"
        # Child writes its PID to ``marker`` then sleeps in foreground.
        # ``exec`` ensures the sleep replaces the shell so it inherits
        # the process group.
        script = (
            f"echo $$ > {marker} && exec sleep 30"
        )
        bash = BashCommand()
        result = bash(command=script)
        assert result["returncode"] != 0
        # Marker file should exist (child wrote its PID before sleeping).
        assert marker.exists()
        pid = int(marker.read_text().strip())
        # After SIGKILL on the process group the PID must be gone.
        # Give the kernel a moment to reap.
        time.sleep(0.2)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


# ---------------------------------------------------------------------------
# Integration — BashCommand.__call__
# ---------------------------------------------------------------------------


class TestBashCommandCallIntegration:
    def test_blocked_scan_returns_descriptive_error(self, monkeypatch, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        monkeypatch.setenv("GEAK_REPO_ROOT", str(repo))
        monkeypatch.delenv("GEAK_WORK_DIR", raising=False)
        bash = BashCommand()
        result = bash(command="find / -name foo")
        assert result["returncode"] == 1
        assert "Blocked" in result["output"]
        assert str(repo) in result["output"]

    def test_allowed_scan_runs(self, monkeypatch, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("")
        monkeypatch.setenv("GEAK_REPO_ROOT", str(repo))
        monkeypatch.setenv("GEAK_BASH_TIMEOUT_SEC", "10")
        bash = BashCommand()
        result = bash(command=f"find {repo} -name '*.py'")
        assert result["returncode"] == 0
        assert "a.py" in result["output"]

    def test_existing_blocklist_still_works(self):
        bash = BashCommand()
        result = bash(command="vim")
        assert result["returncode"] == 1
        assert "Blocked" in result["output"]


class TestBlocklistDirection:
    """Regression tests for the blocklist prefix-direction fix.

    Before the fix, ``BashCommand.__call__`` used ``entry.startswith(
    command)`` so longer commands like ``vim foo.txt`` bypassed the
    editor blocklist while single-char commands like ``v`` were
    over-blocked.  After the fix, ``command.startswith(entry + " ")`` is
    used (whitespace-aware), with exact-equality fallback.
    """

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        monkeypatch.setenv("GEAK_REPO_ROOT", str(repo))
        monkeypatch.setenv("GEAK_BASH_TIMEOUT_SEC", "5")
        self.bash = BashCommand()

    @pytest.mark.parametrize(
        "cmd",
        [
            "vim foo.txt",
            "nano /tmp/x",
            "emacs -nw file.py",
            "less /var/log/x",
            "tail -f /tmp/log.txt",
            "python -m venv /tmp/v",
            "gdb --args ./a.out",
            "make all",
        ],
    )
    def test_blocked_with_arguments(self, cmd):
        blocked, by = self.bash._is_blocked(cmd)
        assert blocked, f"should block: {cmd!r}"
        assert by is not None
        assert cmd.startswith(by)

    @pytest.mark.parametrize(
        "cmd",
        [
            "vim",
            "vi",
            "emacs",
            "nano",
            "python",
            "python3",
            "ipython",
            "bash",
            "sh",
            "rm -rf /",
        ],
    )
    def test_blocked_standalone(self, cmd):
        blocked, _ = self.bash._is_blocked(cmd)
        assert blocked, f"should block: {cmd!r}"

    @pytest.mark.parametrize(
        "cmd",
        [
            # Single chars must NOT trigger the blocklist (previously
            # over-blocked because ``vim``/``vi`` started with ``v``).
            "v",
            "n",
            "m",
            # Substring/prefix without whitespace boundary must NOT block.
            "vimrun --version",
            "makefile",
            "make_dist",
            "nanoseconds",
            "lesspipe.sh",
            # ``python script.py`` is the canonical legitimate usage; only
            # bare ``python`` (interactive REPL) is in standalone list.
            "python script.py",
            "python3 -c 'print(1)'",
            "bash run.sh",
            # ``tail -n 10 file`` is fine; only ``tail -f`` is blocked.
            "tail -n 10 /tmp/log.txt",
            "tail /tmp/log.txt",
        ],
    )
    def test_not_blocked(self, cmd):
        blocked, by = self.bash._is_blocked(cmd)
        assert not blocked, f"should NOT block: {cmd!r} (matched {by!r})"

    def test_block_message_includes_matched_rule(self, tmp_path):
        # Helpful for debugging which rule fired.
        result = self.bash(command="vim foo.txt")
        assert result["returncode"] == 1
        assert "vim" in result["output"]
        assert "Blocked" in result["output"]


# ---------------------------------------------------------------------------
# extra_env override — parallel-agent slots inject ``GEAK_*`` via
# ``BashCommand._env_override`` instead of ``os.environ`` (the latter
# would race across threads).  These tests pin the contract.
# ---------------------------------------------------------------------------


class TestExtraEnvScopeCheck:
    """``_check_command_scope(extra_env=...)`` must consult the dict
    *before* falling back to ``os.environ``."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch, tmp_path):
        # Wipe the globals so we can prove extra_env is doing the work.
        monkeypatch.delenv("GEAK_REPO_ROOT", raising=False)
        monkeypatch.delenv("GEAK_WORK_DIR", raising=False)
        self.repo = tmp_path / "repo"
        self.work = tmp_path / "wt"
        self.repo.mkdir()
        self.work.mkdir()
        self.extra_env = {
            "GEAK_REPO_ROOT": str(self.repo),
            "GEAK_WORK_DIR": str(self.work),
        }

    def test_resolve_path_token_uses_extra_env(self):
        resolved = _resolve_path_token("$GEAK_REPO_ROOT/sub", None, extra_env=self.extra_env)
        assert resolved == str(self.repo / "sub")

    def test_allowed_roots_uses_extra_env(self):
        roots = _allowed_search_roots(None, extra_env=self.extra_env)
        assert str(self.repo) in roots
        assert str(self.work) in roots

    def test_extra_env_takes_precedence_over_os_environ(self, monkeypatch, tmp_path):
        # os.environ says one thing, extra_env another → extra_env wins.
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("GEAK_REPO_ROOT", str(other))
        roots = _allowed_search_roots(None, extra_env=self.extra_env)
        assert str(self.repo) in roots
        assert str(other) not in roots

    def test_check_scope_passes_with_extra_env_only(self):
        # The regression: command references $GEAK_REPO_ROOT, env var is
        # absent from os.environ, present only in extra_env.  Must pass.
        cmd = "find $GEAK_REPO_ROOT/sgl-kernel -maxdepth 5 -type f -name '*.py' | head -60"
        assert _check_command_scope(cmd, cwd=None, extra_env=self.extra_env) is None

    def test_check_scope_blocks_unrelated_path_even_with_extra_env(self):
        # Sanity: extra_env does not magically allow scans rooted elsewhere.
        err = _check_command_scope("find /wekafs -name foo", cwd=None, extra_env=self.extra_env)
        assert err is not None
        assert str(self.repo) in err  # error message reflects the override

    def test_block_message_uses_extra_env_values(self):
        # The user-facing rejection message must show the values the
        # subprocess would actually see, not "<unset>" from os.environ.
        err = _check_command_scope("find / -name foo", cwd=None, extra_env=self.extra_env)
        assert err is not None
        assert str(self.repo) in err
        assert str(self.work) in err
        assert "<unset>" not in err

    def test_export_then_find_export_does_not_hide_scan(self):
        # Newline-separated multi-segment command: the firewall must
        # still see the ``find`` segment and reject it on its own merits.
        cmd = "export FOO=bar\nfind /wekafs -name foo"
        err = _check_command_scope(cmd, cwd=None, extra_env=self.extra_env)
        assert err is not None


class TestSplitShellSegmentsNewline:
    """``_split_shell_segments`` must treat ``\\n`` as a separator so an
    LLM can't hide a ``find /`` behind a leading ``export`` line."""

    def test_newline_splits(self):
        segs = _split_shell_segments("export FOO=bar\nfind / -name foo")
        assert "export FOO=bar" in segs
        assert "find / -name foo" in segs

    def test_crlf_splits(self):
        segs = _split_shell_segments("a\r\nb")
        assert segs == ["a", "b"]


class TestBashInstanceUsesEnvOverride:
    """End-to-end: a parallel-agent slot wires ``_env_override`` and the
    scope check then accepts ``$GEAK_REPO_ROOT``-rooted scans even when
    those vars are absent from the global environment."""

    def test_env_override_unblocks_scan(self, monkeypatch, tmp_path):
        # Prove the bug: the global env is empty, so without
        # _env_override the scan would be rejected.
        monkeypatch.delenv("GEAK_REPO_ROOT", raising=False)
        monkeypatch.delenv("GEAK_WORK_DIR", raising=False)
        monkeypatch.setenv("GEAK_BASH_TIMEOUT_SEC", "10")
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "marker.py").write_text("")

        bash = BashCommand()
        # 1) Without override: rejected (current behavior pre-fix).
        rejected = bash(command="find $GEAK_REPO_ROOT -name '*.py'")
        assert rejected["returncode"] == 1
        assert "Blocked" in rejected["output"]

        # 2) With override (mirrors what tools_runtime.py installs): runs.
        bash._env_override = {"GEAK_REPO_ROOT": str(repo)}
        ok = bash(command="find $GEAK_REPO_ROOT -name '*.py'")
        assert ok["returncode"] == 0
        assert "marker.py" in ok["output"]

    def test_env_override_block_message_has_real_values(self, monkeypatch, tmp_path):
        # Wipe os.environ so any leakage would show as "<unset>".
        monkeypatch.delenv("GEAK_REPO_ROOT", raising=False)
        monkeypatch.delenv("GEAK_WORK_DIR", raising=False)
        repo = tmp_path / "repo"
        work = tmp_path / "wt"
        repo.mkdir()
        work.mkdir()
        bash = BashCommand()
        bash._env_override = {"GEAK_REPO_ROOT": str(repo), "GEAK_WORK_DIR": str(work)}
        result = bash(command="find /wekafs -name foo")
        assert result["returncode"] == 1
        # Both override values must appear; neither should fall back to <unset>.
        head = result["output"].split("System dirs")[0]
        assert str(repo) in head
        assert str(work) in head
        assert "<unset>" not in head
