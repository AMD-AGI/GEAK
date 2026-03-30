import os
import re
import subprocess
from pathlib import Path

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
            env = os.environ | self._env_override if self._env_override else None
            cwd = self._cwd if self._cwd and Path(self._cwd).is_dir() else None
            result = subprocess.run(command, shell=True, capture_output=True, text=True, env=env, cwd=cwd)
            output_text = result.stdout.strip() or result.stderr.strip()

            if "COMMANDMENT.md" in command:
                output_text = self._maybe_validate_commandment(command, output_text)

            return {
                "output": output_text,
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
                    pass

        return output_text
