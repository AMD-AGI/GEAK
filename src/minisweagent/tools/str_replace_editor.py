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
        make_cmd = [f"python {str(self.tool_py)} {command} {path} --file_text {file_text} --view_range {view_range} --old_str {old_file} --new_str {new_file} --insert_line {insert_line}"]
        result = subprocess.run(make_cmd, shell=True, capture_output=True, text=True, timeout=3600)
        if old_file and os.path.exists(old_file):
            os.remove(old_file)
        if new_file and os.path.exists(new_file):
            os.remove(new_file)
        return {
            "output": result.stdout.strip() or result.stderr.strip(),
            "returncode": result.returncode,
        }
    

if __name__ == "__main__":
    command= "str_replace" # str_replace insert veiw
    path= "/mcp/rocPRIM_device_binary_search/benchmark/benchmark_device_binary_search.cpp"
    insert_line = 10
    old_str = "in the Software without restriction, including without limitation the rights"
    new_str = "sssssssssssss"
    view_range = [10, 200]
    edit_tool = str_replace_editor()
    response = edit_tool(command=command, path=path, view_range=view_range, insert_line=insert_line, new_str=new_str, old_str=old_str)
    print(response["output"])