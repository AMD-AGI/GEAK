# Modifications Copyright(C)[2026] Advanced Micro Devices, Inc. All rights reserved.

import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

from minisweagent.environments.protected_files import (
    blocked_message,
    would_write_protected_file,
)


@dataclass
class LocalEnvironmentConfig:
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    protected_files: list[str] = field(default_factory=list)
    """Basename(s) of files that must not be modified (e.g. ['kernel.py'])."""


class LocalEnvironment:
    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        """This class executes bash commands directly on the local machine."""
        self.config = config_class(**kwargs)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None):
        """Execute a command in the local environment and return the result as a dict."""
        cwd = cwd or self.config.cwd or os.getcwd()
        protected = getattr(self.config, "protected_files", None) or []
        if protected:
            blocked = would_write_protected_file(command, protected)
            if blocked:
                return {"output": blocked_message(blocked), "returncode": 1}
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            cwd=cwd,
            env=os.environ | self.config.env,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | platform.uname()._asdict() | os.environ
