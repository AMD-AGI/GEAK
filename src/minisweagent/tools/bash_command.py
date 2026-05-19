import locale
import logging
import os
import re
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
        if any(f.startswith(command) for f in self.blocklist) or command in self.blocklist_standalone:
            return {
                "output": f"Blocked dangerous command: {command}",
                "returncode": 1,
            }
        else:
            command = self._sandbox_command(command)
            env = os.environ | self._env_override if self._env_override else None
            cwd = self._cwd if self._cwd and Path(self._cwd).is_dir() else None
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=False,
                env=env,
                cwd=cwd,
            )
            output_text = _decode_captured_output(result.stdout, result.stderr)

            if "COMMANDMENT.md" in command:
                output_text = self._maybe_validate_commandment(command, output_text)

            if result.returncode != 0:
                output_text = output_text or "Command failed with no output."
            return {
                "output": output_text or "Bash command executed successfully.",
                "returncode": result.returncode,
            }

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
