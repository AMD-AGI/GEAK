"""Tool: resolve_kernel_url -- resolve a GitHub URL to a local kernel path.

Clones the repo (if needed) and returns the local file path, line number,
and kernel function name. Used by the agent when given a --kernel-url.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from minisweagent.tools.resolve_kernel_url_impl import get_kernel_name_at_line, resolve_kernel_url


class ResolveKernelUrlTool:
    """ToolRuntime-compatible callable for resolve_kernel_url."""

    def __call__(self, url: str, workspace: str | None = None) -> dict[str, Any]:
        """Resolve a GitHub kernel URL to a local file path.

        Args:
            url: GitHub URL (e.g. https://github.com/org/repo/blob/main/kernel.py#L106)
            workspace: Directory to clone into (default: cwd)

        Returns:
            {output: str, returncode: int}
        """

        clone_into = Path(workspace) if workspace else Path.cwd()
        try:
            resolved = resolve_kernel_url(url, clone_into=clone_into)
        except Exception as e:
            return {"output": f"Failed to resolve URL: {e}", "returncode": 1}

        if resolved.get("error"):
            return {"output": f"Resolve error: {resolved['error']}", "returncode": 1}

        path = resolved["local_file_path"]
        line_num = resolved.get("line_number")
        kernel_name = None
        if line_num:
            try:
                kernel_name = get_kernel_name_at_line(path, line_num)
            except Exception:
                pass

        parts = [f"local_file_path: {path}"]
        if line_num:
            parts.append(f"line_number: {line_num}")
        if kernel_name:
            parts.append(f"kernel_name: {kernel_name}")

        return {"output": "\n".join(parts), "returncode": 0}
