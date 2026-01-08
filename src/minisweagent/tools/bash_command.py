import re
import subprocess

class BashCommand:
    def __init__(self):
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
        if any(f.startswith(command) for f in self.blocklist) or command in self.blocklist_standalone:
            return {
                "output": f"Blocked dangerous command: {command}",
                "returncode": 1,
            }
        else:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return {
                "output": result.stdout.strip() or result.stderr.strip(),
                "returncode": result.returncode,
            }