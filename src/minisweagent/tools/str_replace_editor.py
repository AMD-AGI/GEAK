# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile


class str_replace_editor:
    def __init__(self):
        self.tool_py = Path(__file__).parent / "editor_tool.py"

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
        **kwargs,
    ):
        if view_range:
            view_range = f'"{str(view_range)}"'
        if file_text:
            file_text = f'"{str(file_text)}"'
        old_file = None
        new_file = None
        if old_str:
            with NamedTemporaryFile("w", delete=False) as f_old:
                f_old.write(old_str)
                old_file = f_old.name
        if new_str:
            with NamedTemporaryFile("w", delete=False) as f_old:
                f_old.write(new_str)
                new_file = f_old.name
        make_cmd = [
            f"python {str(self.tool_py)} {command} {path} --file_text {file_text} --view_range {view_range} --old_str {old_file} --new_str {new_file} --insert_line {insert_line}"
        ]
        result = subprocess.run(make_cmd, shell=True, capture_output=True, text=True, timeout=3600)
        if old_file and Path(old_file).exists():
            os.remove(old_file)
        if new_file and Path(new_file).exists():
            os.remove(new_file)

        output_text = result.stdout.strip() or result.stderr.strip()

        # Auto-validate COMMANDMENT.md files after create/edit
        if path.endswith("COMMANDMENT.md") and command in ("create", "str_replace", "insert"):
            output_text = self._validate_commandment_file(path, output_text)

        return {
            "output": output_text,
            "returncode": result.returncode,
        }

    @staticmethod
    def _validate_commandment_file(path: str, output_text: str) -> str:
        """Run COMMANDMENT validation and append results to editor output."""
        try:
            from minisweagent.tools.validate_commandment import (
                format_validation_message,
                validate_commandment,
            )

            content = Path(path).read_text()
            result = validate_commandment(content)
            msg = format_validation_message(result)
            if msg:
                output_text += f"\n\n{msg}"
        except Exception:
            pass  # Don't break the editor if validation fails
        return output_text


if __name__ == "__main__":
    command = "str_replace"  # str_replace insert veiw
    path = "/mcp/rocPRIM_device_binary_search/benchmark/benchmark_device_binary_search.cpp"
    insert_line = 10
    old_str = "in the Software without restriction, including without limitation the rights"
    new_str = "sssssssssssss"
    view_range = [10, 200]
    edit_tool = str_replace_editor()
    response = edit_tool(
        command=command, path=path, view_range=view_range, insert_line=insert_line, new_str=new_str, old_str=old_str
    )
    print(response["output"])
