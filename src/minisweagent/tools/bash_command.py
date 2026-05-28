import locale
import logging
import os
import re
import shlex
import signal
import subprocess
from pathlib import Path

_OUTPUT_UNREADABLE = (
    "The combined command output could not be decoded as a whole using the "
    "process locale encoding. Part of the command (e.g. one stage such as "
    '"cat" of a binary file) may have produced invalid or non-text bytes, so '
    "none of the captured stdout is shown. Run text-producing steps separately "
    "or use a tool suited for binary data."
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wall-clock timeout (L1) and filesystem-scan scope firewall (L2).
#
# These exist because the bash tool is the *only* unsupervised entry point
# the LLM has, and commands like ``find /`` or ``find /wekafs`` would
# otherwise stall the agent for hours on NFS-backed mounts.
# ---------------------------------------------------------------------------

#: Default per-command wall-clock timeout in seconds.  Overridable via the
#: ``GEAK_BASH_TIMEOUT_SEC`` environment variable.  Chosen to be long enough
#: for legitimate build/install steps (``pip install -e .`` against a HIP
#: kernel can take 10+ minutes) but short enough to bound the worst case.
_DEFAULT_BASH_TIMEOUT_SEC = 600

#: System directories that contain headers, libraries, and configuration
#: relevant to HIP / ROCm / kernel work.  Scans rooted at these paths are
#: allowed even though they fall outside the per-run repo/worktree.
_SYSTEM_ROOTS: tuple[str, ...] = (
    "/tmp",
    "/var/tmp",
    "/opt",
    "/usr",
    "/etc",
    "/var/lib",
)

#: Binaries that perform recursive / unbounded filesystem traversal.
#: ``grep``/``ls``/``du`` only count when their recursive flag is present
#: (see :func:`_is_recursive_scan`).
_SCAN_BINARIES: frozenset[str] = frozenset({"find", "rg", "grep", "tree", "du", "ls"})


def _bash_timeout_sec() -> float:
    raw = os.environ.get("GEAK_BASH_TIMEOUT_SEC")
    if not raw:
        return float(_DEFAULT_BASH_TIMEOUT_SEC)
    try:
        v = float(raw)
        return v if v > 0 else float(_DEFAULT_BASH_TIMEOUT_SEC)
    except ValueError:
        return float(_DEFAULT_BASH_TIMEOUT_SEC)


# Pattern matching ``$VAR`` and ``${VAR}``.  Mirrors what
# ``os.path.expandvars`` accepts on POSIX (alpha-or-underscore start, then
# alnum/underscore).  Used by :func:`_expand_env_vars` so the scope check
# can resolve variables from an *injected* env dict (the bash tool's
# ``_env_override``) rather than only from ``os.environ``.
_ENV_VAR_RE = re.compile(r"\$(\{)?([A-Za-z_][A-Za-z0-9_]*)(?(1)\})")


def _lookup_env(name: str, extra_env: dict[str, str] | None) -> str | None:
    """Return the value of env var *name*, preferring ``extra_env`` when
    present (per-instance override) and falling back to ``os.environ``.

    Returns ``None`` when the name is unset in both layers.  Empty-string
    values are treated as "set but empty" and returned verbatim, matching
    standard shell semantics.
    """
    if extra_env is not None and name in extra_env:
        return extra_env[name]
    return os.environ.get(name)


def _expand_env_vars(tok: str, extra_env: dict[str, str] | None) -> str:
    """Like :func:`os.path.expandvars` but resolves variables against
    ``extra_env`` first, ``os.environ`` second.

    Why we don't just call ``os.path.expandvars`` then patch up: the
    bash tool runs in a multi-thread parallel-agent model where each
    slot needs a *different* ``GEAK_WORK_DIR``.  Mutating ``os.environ``
    would race; this helper keeps everything per-call.

    Unset variables are left literal (``$FOO`` stays ``$FOO``), matching
    ``os.path.expandvars``'s behavior so any downstream ancestry check
    still rejects them rather than silently treating them as empty.
    """
    if "$" not in tok:
        return tok

    def _sub(match: re.Match[str]) -> str:
        name = match.group(2)
        value = _lookup_env(name, extra_env)
        return value if value is not None else match.group(0)

    return _ENV_VAR_RE.sub(_sub, tok)


def _allowed_search_roots(
    cwd: str | None, extra_env: dict[str, str] | None = None
) -> list[str]:
    """Roots under which recursive filesystem scans are permitted.

    Order matters only for the diagnostic message; ancestry checks are
    set-based.  ``$GEAK_REPO_ROOT`` is listed first because it is the most
    commonly-needed scope and we want the LLM to learn that pattern.

    Note: ``cwd`` is intentionally NOT added as a top-level allowed root.
    Adding an arbitrary cwd would let a misconfigured agent (e.g. with
    ``_cwd = "/"``) bypass the firewall entirely.  Instead, ``cwd`` is
    only used downstream to *resolve* relative paths (``.``, ``./foo``);
    the resolved absolute path is then checked against the
    REPO_ROOT/WORK_DIR/system roots in the usual way.  This matches the
    documented contract "cwd is allowed *when it is inside a
    repo/worktree*".

    ``extra_env`` (when provided) is consulted before ``os.environ`` so
    parallel-agent slots can each have their own ``GEAK_REPO_ROOT`` /
    ``GEAK_WORK_DIR`` without racing on the global environment.
    """
    roots: list[str] = []
    for env_var in ("GEAK_REPO_ROOT", "GEAK_WORK_DIR"):
        v = (_lookup_env(env_var, extra_env) or "").strip()
        if v:
            roots.append(os.path.normpath(v))
    roots.extend(_SYSTEM_ROOTS)
    return roots


def _is_under_any(path: str, roots: list[str]) -> bool:
    p = os.path.normpath(path)
    for r in roots:
        if p == r or p.startswith(r + os.sep):
            return True
    return False


def _resolve_path_token(
    tok: str, cwd: str | None, extra_env: dict[str, str] | None = None
) -> str:
    """Resolve a path-like shell token to an absolute, normalized path.

    Handles ``~``, ``$VAR``, and relative paths.  Does *not* follow
    symlinks (``realpath``) because the caller only needs ancestry, and
    symlink resolution would be a syscall per check.

    ``extra_env`` is consulted before ``os.environ`` for ``$VAR`` /
    ``${VAR}`` substitutions.  ``~`` still uses ``os.environ['HOME']``
    via :func:`os.path.expanduser`; we don't override that because the
    parallel-agent flow doesn't differ on HOME and the bug we're fixing
    is specifically about ``GEAK_*``.
    """
    expanded = _expand_env_vars(os.path.expanduser(tok), extra_env)
    if not os.path.isabs(expanded):
        base = cwd or os.getcwd()
        expanded = os.path.join(base, expanded)
    return os.path.normpath(expanded)


# Shell control operators that separate commands.  ``shlex`` does not
# split on these, so we pre-segment the command string before tokenizing.
# ``\n`` is included so a multi-line command body
# (``export FOO=bar\nfind $FOO -name x``) is also segmented; without
# this an LLM trick that put ``export`` on the first line would hide
# the ``find`` segment from the scope check entirely (the whole thing
# would tokenize as starting with ``export``, which is not a scan
# binary, so the firewall would no-op).
_SHELL_SEP_RE = re.compile(r"(?:&&|\|\||;|\||\r?\n)")


def _split_shell_segments(cmd: str) -> list[str]:
    """Split a command line on top-level shell separators (``;``, ``&&``,
    ``||``, ``|``, newline).  Not aware of quoting/heredocs; segments
    inside a heredoc that contain a separator may be split incorrectly,
    but the scope check is best-effort and L1 timeout is the safety net.
    """
    return [s.strip() for s in _SHELL_SEP_RE.split(cmd) if s.strip()]


def _is_recursive_scan(tokens: list[str]) -> bool:
    """Whether the token list represents a recursive filesystem traversal.

    ``find``/``rg``/``tree`` are always recursive.  ``grep``/``ls``/``du``
    are only included when their recursive flag is present so plain
    ``ls -la dir`` and ``grep pattern file.py`` are not affected.
    """
    if not tokens:
        return False
    name = tokens[0]
    if name in {"find", "rg", "tree"}:
        return True
    flag_args = tokens[1:]
    if name == "grep":
        for t in flag_args:
            if t in {"-r", "-R", "--recursive", "--dereference-recursive"}:
                return True
            if t.startswith("-") and not t.startswith("--") and ("r" in t or "R" in t):
                return True
        return False
    if name == "ls":
        for t in flag_args:
            if t == "-R" or (t.startswith("-") and not t.startswith("--") and "R" in t):
                return True
        return False
    if name == "du":
        for t in flag_args:
            if t in {"-a", "--all"}:
                return True
            if t.startswith("-") and not t.startswith("--") and "a" in t:
                return True
        return False
    return False


def _extract_search_paths(tokens: list[str], cwd: str | None) -> list[str]:
    """Return path-like arguments that act as scan roots for ``tokens[0]``.

    For ``find``/``tree`` the search roots precede any ``-flag``; once a
    ``-flag`` is seen, the remainder is ``-name foo``/``-type f`` style
    arguments that must not be treated as paths.

    For ``grep``/``rg``/``ls``/``du`` the path can appear anywhere, so
    we return every token that *looks* like a path (absolute, env-var,
    home-relative, or ``.``/``..``).  This may include the search
    pattern if it happens to look path-shaped (e.g. ``grep -r '/usr/'``),
    but ancestry against allowed roots makes false positives rare:
    ``/usr/`` is allowed because ``/usr`` is a system root.
    """
    if not tokens:
        return []
    name = tokens[0]
    if name in {"find", "tree"}:
        paths: list[str] = []
        for t in tokens[1:]:
            if t.startswith("-"):
                break
            paths.append(t)
        return paths or [cwd or "."]
    if name in {"grep", "rg", "ls", "du"}:
        paths = []
        skip_next = False
        for t in tokens[1:]:
            if skip_next:
                skip_next = False
                continue
            if t.startswith("-"):
                # ``grep -e PATTERN`` / ``grep -f FILE`` style flags
                if name in {"grep", "rg"} and t in {"-e", "--regexp", "-f", "--file"}:
                    skip_next = True
                continue
            if t.startswith("/") or t.startswith("$") or t.startswith("~") or t in {".", ".."} or t.startswith("./") or t.startswith("../"):
                paths.append(t)
        return paths
    return []


def _format_scope_block(
    bad_token: str,
    resolved: str,
    roots: list[str],
    extra_env: dict[str, str] | None = None,
) -> str:
    """Build a helpful, machine-greppable rejection message for the LLM.

    Reads the GEAK roots from ``extra_env`` first (per-instance override)
    so the rejection message reports the values the agent actually has,
    not whatever happens to be (or not be) in the global ``os.environ``.
    Reporting ``<unset>`` while the bash subprocess actually inherits
    the variables would mislead the LLM into trying ineffective
    workarounds (e.g. ``export`` on its own line).
    """
    repo = _lookup_env("GEAK_REPO_ROOT", extra_env) or "<unset>"
    work = _lookup_env("GEAK_WORK_DIR", extra_env) or "<unset>"
    return (
        f"Blocked: unbounded filesystem scan rooted at {bad_token!r} "
        f"(resolved to {resolved!r}). Scans outside the project scope can "
        f"take hours on NFS.\n\n"
        f"Allowed search roots in this run:\n"
        f"  $GEAK_REPO_ROOT = {repo}\n"
        f"  $GEAK_WORK_DIR  = {work}\n"
        f"  System dirs    = /tmp, /var/tmp, /opt, /usr, /etc, /var/lib\n\n"
        f"Rewrite your command, for example:\n"
        f"  find $GEAK_REPO_ROOT -maxdepth 3 -name 'pattern'\n"
        f"  find $GEAK_WORK_DIR -name 'pattern'\n"
        f"  rg 'pattern' $GEAK_REPO_ROOT\n"
        f"For system directories prefer -maxdepth 3 to keep latency low."
    )


def _check_command_scope(
    cmd: str,
    cwd: str | None,
    extra_env: dict[str, str] | None = None,
) -> str | None:
    """Return a rejection message if *cmd* contains a recursive filesystem
    scan rooted outside any allowed root; otherwise ``None``.

    Best-effort: complex shell constructs (heredocs, deeply nested
    ``$(...)``) may bypass this check, in which case the wall-clock
    timeout is the second line of defense.

    ``extra_env`` (when provided) is consulted before ``os.environ`` for
    every variable lookup performed by this function and its helpers,
    so the parallel-agent caller can pass its own ``_env_override``
    dict and avoid the thread-unsafe pattern of mutating
    ``os.environ`` to publish per-slot ``GEAK_WORK_DIR``.
    """
    if not cmd:
        return None
    for segment in _split_shell_segments(cmd):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            continue
        if not tokens or tokens[0] not in _SCAN_BINARIES:
            continue
        if not _is_recursive_scan(tokens):
            continue
        paths = _extract_search_paths(tokens, cwd)
        if not paths:
            continue
        roots = _allowed_search_roots(cwd, extra_env=extra_env)
        for p in paths:
            resolved = _resolve_path_token(p, cwd, extra_env=extra_env)
            if resolved == "/" or not _is_under_any(resolved, roots):
                return _format_scope_block(p, resolved, roots, extra_env=extra_env)
    return None


def _process_stream_encoding() -> str:
    try:
        return locale.getencoding()
    except AttributeError:
        return locale.getpreferredencoding(False) or "utf-8"


def _decode_captured_output(stdout_b: bytes | None, stderr_b: bytes | None) -> str:
    """Decode subprocess bytes with the locale encoding and strict errors.

    If the chosen stream is non-empty but not valid for that encoding, return
    ``_OUTPUT_UNREADABLE`` instead of partial or replacement-character output.
    """
    enc = _process_stream_encoding()
    out = (stdout_b or b"").strip()
    err = (stderr_b or b"").strip()
    if out:
        try:
            return out.decode(enc, "strict")
        except UnicodeDecodeError:
            return _OUTPUT_UNREADABLE
    if err:
        try:
            return err.decode(enc, "strict")
        except UnicodeDecodeError:
            return _OUTPUT_UNREADABLE
    return ""


# Matches shell redirect / heredoc patterns that write to COMMANDMENT.md,
# e.g. ``cat > path/COMMANDMENT.md``, ``tee path/COMMANDMENT.md``,
# ``> path/COMMANDMENT.md << 'EOF'``.
_COMMANDMENT_WRITE_RE = re.compile(
    r"""(?:cat\s+>|>\s*|tee\s+)"""
    r"""\s*([^\s<|&]+COMMANDMENT\.md)"""
    r"""|"""
    r"""(?:>\s*|\s+)([^\s<|&]+COMMANDMENT\.md)\s*<<""",
    re.VERBOSE,
)


class BashCommand:
    def __init__(self):
        self._env_override: dict[str, str] = {}
        self._cwd: str | None = None
        self.blocklist: list[str] = [
            "vim",
            "vi",
            "emacs",
            "nano",
            "nohup",
            "gdb",
            "less",
            "tail -f",
            "python -m venv",
            "make",
        ]
        self.blocklist_standalone: list[str] = [
            "python",
            "python3",
            "ipython",
            "bash",
            "sh",
            "/bin/bash",
            "/bin/sh",
            "nohup",
            "vi",
            "vim",
            "emacs",
            "nano",
            "su",
            "reboot",
            "shutdown",
            "mkfs",
            "rm -rf /",
        ]

    def _is_blocked(self, command: str) -> tuple[bool, str | None]:
        """Return ``(blocked, reason)`` for *command* against both blocklists.

        ``blocklist`` matches when *command* starts with an entry followed
        by whitespace (or equals it exactly).  This catches ``vim foo.txt``
        (starts with ``vim``) and ``tail -f log`` (starts with ``tail -f``)
        but not ``vimrun`` (no whitespace boundary) or ``makefile`` (a
        substring of ``make``).

        ``blocklist_standalone`` matches only on exact equality, so
        ``python`` alone is blocked (would drop to an interactive REPL
        and hang the agent) but ``python script.py`` is allowed.

        Historical note: an earlier version checked ``entry.startswith(
        command)`` instead of ``command.startswith(entry)``, which silently
        let ``vim foo.txt`` / ``nano file.py`` through while also
        over-blocking single-character commands such as ``v`` (because
        ``vim``/``vi`` start with ``v``).  This method restores the
        intended semantics.
        """
        for entry in self.blocklist:
            if command == entry or command.startswith(entry + " ") or command.startswith(entry + "\t"):
                return True, entry
        if command in self.blocklist_standalone:
            return True, command
        return False, None

    @staticmethod
    def _sandbox_command(command: str) -> str:
        """Rewrite absolute paths in the command that target the original repo.

        Agents in worktrees must never write to the original repo
        (``GEAK_REPO_ROOT``).  Replace occurrences of the repo root with
        the agent's worktree (``GEAK_WORK_DIR``) so that ``cat >``,
        ``cp``, ``cd``, and similar commands land in the worktree.

        Safe because every legitimate repo-root reference in agent bash
        commands (``cd``, ``python -c``, ``cp``) works identically with
        the worktree path. The PYTHONPATH and COMMANDMENT ``run.sh``
        scripts read ``$GEAK_REPO_ROOT`` at shell-expansion time, not
        from the command string, so they are unaffected.
        """
        repo_root = os.environ.get("GEAK_REPO_ROOT", "")
        work_dir = os.environ.get("GEAK_WORK_DIR", "")
        if not repo_root or not work_dir or repo_root == work_dir:
            return command
        if repo_root in command:
            rewritten = command.replace(repo_root, work_dir)
            logger.debug("bash_command: rewrote repo_root paths in command")
            return rewritten
        return command

    def __call__(
        self,
        *,
        command: str,
        **kwargs,
    ):
        if not command:
            return {
                "output": "bash tool call need a command argument, it must not be empty.",
                "returncode": 1,
            }
        blocked, blocked_by = self._is_blocked(command)
        if blocked:
            return {
                "output": f"Blocked dangerous command (matched rule: {blocked_by!r}): {command}",
                "returncode": 1,
            }

        command = self._sandbox_command(command)
        env = os.environ | self._env_override if self._env_override else None
        cwd = self._cwd if self._cwd and Path(self._cwd).is_dir() else None

        # Pass the per-instance env override into the scope check so it
        # resolves $GEAK_REPO_ROOT / $GEAK_WORK_DIR from the same dict
        # that the subprocess will see.  Without this, parallel-agent
        # slots silently fail every recursive scan because each slot's
        # GEAK_WORK_DIR lives only in the bash tool's _env_override
        # (NOT in os.environ — that would race across threads).
        scope_err = _check_command_scope(command, cwd, extra_env=self._env_override or None)
        if scope_err is not None:
            logger.info("bash_command: rejected unbounded scan: %s", command[:200])
            return {"output": scope_err, "returncode": 1}

        returncode, stdout_b, stderr_b, timed_out = self._run_with_timeout(command, env, cwd)
        output_text = _decode_captured_output(stdout_b, stderr_b)

        if timed_out:
            timeout_sec = _bash_timeout_sec()
            timeout_msg = (
                f"Command timed out after {timeout_sec:.0f}s and was killed "
                f"(SIGKILL to process group). Partial output above (if any) "
                f"may be incomplete.\n\n"
                f"Tip: scope filesystem scans to $GEAK_REPO_ROOT or "
                f"$GEAK_WORK_DIR and use -maxdepth 3 for unknown layouts."
            )
            output_text = f"{output_text}\n\n{timeout_msg}" if output_text else timeout_msg

        if "COMMANDMENT.md" in command:
            output_text = self._maybe_validate_commandment(command, output_text)

        if returncode != 0 and not output_text:
            output_text = "Command failed with no output."

        return {
            "output": output_text or "Bash command executed successfully.",
            "returncode": returncode,
        }

    @staticmethod
    def _run_with_timeout(
        command: str,
        env: dict[str, str] | None,
        cwd: str | None,
    ) -> tuple[int, bytes, bytes, bool]:
        """Run *command* in a new session and SIGKILL its process group on
        timeout.

        Returns ``(returncode, stdout, stderr, timed_out)``.  When
        ``timed_out`` is True, ``returncode`` is ``-signal.SIGKILL``
        (negative per ``subprocess`` convention for signal terminations).

        ``start_new_session=True`` puts the child in its own process group
        so ``os.killpg`` reaches grandchildren (e.g. ``find`` spawned
        from a wrapper script).  Without this, a hung ``find`` survives a
        plain ``proc.kill()`` because the shell's exit does not propagate
        SIGTERM to descendants on every platform.
        """
        timeout_sec = _bash_timeout_sec()
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        try:
            stdout_b, stderr_b = proc.communicate(timeout=timeout_sec)
            return proc.returncode, stdout_b, stderr_b, False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                logger.debug("bash_command: killpg failed; falling back to proc.kill", exc_info=True)
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                stdout_b, stderr_b = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout_b, stderr_b = b"", b""
            returncode = proc.returncode if proc.returncode is not None else -signal.SIGKILL
            return returncode, stdout_b or b"", stderr_b or b"", True

    @staticmethod
    def _maybe_validate_commandment(command: str, output_text: str) -> str:
        """Validate COMMANDMENT.md if the bash command wrote one.

        COMMANDMENT.md is the evaluation contract between sub-agents and the
        orchestrator.  Sub-agents must not silently produce an invalid one, so
        every bash command that touches the file is validated on the spot and
        any errors are appended to the command output as immediate feedback.
        """
        path_str: str | None = None

        m = _COMMANDMENT_WRITE_RE.search(command)
        if m:
            path_str = m.group(1) or m.group(2)
        else:
            for token in command.split():
                if token.endswith("COMMANDMENT.md") and "/" in token:
                    path_str = token
                    break

        if path_str:
            p = Path(path_str)
            if p.exists():
                try:
                    from minisweagent.tools.validate_commandment import (  # pylint: disable=no-name-in-module
                        format_validation_message,
                        validate_commandment,
                    )

                    result = validate_commandment(p.read_text())
                    msg = format_validation_message(result)
                    if msg:
                        output_text += f"\n\n{msg}"
                except Exception:
                    logger.debug("COMMANDMENT validation failed", exc_info=True)

        return output_text
