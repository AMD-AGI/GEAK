import os
import re
import subprocess
from pathlib import Path

# Regex to extract the path of a COMMANDMENT.md file being written by a bash command.
# Matches patterns like:
#   cat > /path/to/COMMANDMENT.md
#   > /path/to/COMMANDMENT.md
#   tee /path/to/COMMANDMENT.md
#   ... > /path/to/COMMANDMENT.md (redirect)
_COMMANDMENT_WRITE_RE = re.compile(
    r"""(?:cat\s+>|>\s*|tee\s+)"""  # write indicators
    r"""\s*([^\s<|&]+COMMANDMENT\.md)"""  # capture the file path
    r"""|"""  # OR
    r"""(?:>\s*|\s+)([^\s<|&]+COMMANDMENT\.md)\s*<<""",  # heredoc target
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
            cwd = self._cwd if self._cwd and os.path.isdir(self._cwd) else None
            result = subprocess.run(command, shell=True, capture_output=True, text=True, env=env, cwd=cwd)
            output_text = result.stdout.strip() or result.stderr.strip()

            # Auto-validate COMMANDMENT.md if the command wrote one
            if "COMMANDMENT.md" in command:
                output_text = self._maybe_validate_commandment(command, output_text)

            return {
                "output": output_text,
                "returncode": result.returncode,
            }

    @staticmethod
    def _maybe_validate_commandment(command: str, output_text: str) -> str:
        """If the bash command wrote a COMMANDMENT.md, validate it."""
        # Try regex first, fall back to scanning for any COMMANDMENT.md path
        paths: list[str] = []
        for m in _COMMANDMENT_WRITE_RE.finditer(command):
            p = m.group(1) or m.group(2)
            if p:
                paths.append(p)

        # Fallback: if no regex match but command contains COMMANDMENT.md and
        # a write indicator, try to find the path in the command tokens
        if not paths:
            for token in command.split():
                if token.endswith("COMMANDMENT.md") and "/" in token:
                    paths.append(token)
                    break

        for path_str in paths:
            p = Path(path_str)
            if p.exists():
                try:
                    from minisweagent.tools.validate_commandment import (
                        format_validation_message,
                        validate_commandment,
                    )

                    content = p.read_text()
                    result = validate_commandment(content)
                    msg = format_validation_message(result)
                    if msg:
                        output_text += f"\n\n{msg}"
                except Exception:
                    pass
                break

        return output_text
