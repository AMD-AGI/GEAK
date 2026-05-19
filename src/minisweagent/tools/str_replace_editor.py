from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

logger = logging.getLogger(__name__)


class str_replace_editor:
    def __init__(self) -> None:
        self.tool_py = Path(__file__).parent / "editor_tool.py"
        self._env_override: dict[str, str] = {}

    def _sandbox_path(self, path: str) -> str:
        """Rewrite paths that point into the original repo to the agent's worktree.

        Agents running in worktrees should never write to the original repo
        (``GEAK_REPO_ROOT``). If the LLM passes an absolute path inside the
        original repo, silently redirect it to the equivalent path inside
        ``GEAK_WORK_DIR`` so the original stays pristine.
        """
        env = os.environ | self._env_override if self._env_override else os.environ
        repo_root = env.get("GEAK_REPO_ROOT", "")
        work_dir = env.get("GEAK_WORK_DIR", "")
        if not repo_root or not work_dir or repo_root == work_dir:
            return path
        if path.startswith(repo_root + "/") or path == repo_root:
            rewritten = work_dir + path[len(repo_root) :]
            logger.debug("str_replace_editor: redirected %s -> %s", path, rewritten)
            return rewritten
        return path

    def __call__(
        self,
        *,
        command: str,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        **kwargs: object,
    ) -> dict[str, str | int]:
        path = self._sandbox_path(path)
        cmd: list[str] = [sys.executable, str(self.tool_py), command, path]

        file_text_path: str | None = None
        if file_text is not None:
            with NamedTemporaryFile("w", delete=False, encoding="utf-8") as f_txt:
                f_txt.write(file_text)
                file_text_path = f_txt.name
            cmd.extend(["--file_text_path", file_text_path])

        if view_range is not None:
            cmd.extend(["--view_range", json.dumps(view_range)])

        old_file: str | None = None
        new_file: str | None = None
        if old_str is not None:
            with NamedTemporaryFile("w", delete=False, encoding="utf-8") as f_old:
                f_old.write(old_str)
                old_file = f_old.name
            cmd.extend(["--old_str", old_file])
        if new_str is not None:
            with NamedTemporaryFile("w", delete=False, encoding="utf-8") as f_new:
                f_new.write(new_str)
                new_file = f_new.name
            cmd.extend(["--new_str", new_file])

        if insert_line is not None:
            cmd.extend(["--insert_line", str(insert_line)])

        # Child imports minisweagent (via editor_tool → registry); package __init__
        # prints a startup banner to stdout unless silenced.
        subprocess_env = os.environ.copy()
        subprocess_env.update(self._env_override)
        subprocess_env["MSWEA_SILENT_STARTUP"] = "1"

        try:
            result = subprocess.run(
                cmd,
                shell=False,
                capture_output=True,
                text=True,
                timeout=3600,
                env=subprocess_env,
            )
            out = (result.stdout or "").strip() or (result.stderr or "").strip()
            return {"output": out, "returncode": result.returncode}
        finally:
            if file_text_path and Path(file_text_path).exists():
                os.remove(file_text_path)
            if old_file and Path(old_file).exists():
                os.remove(old_file)
            if new_file and Path(new_file).exists():
                os.remove(new_file)


if __name__ == "__main__":
    print("Import str_replace_editor and use str_replace_editor() as the tool callable.")
