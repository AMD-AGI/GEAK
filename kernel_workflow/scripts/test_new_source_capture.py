#!/usr/bin/env python3
"""One real-Git regression for extracted-task scope and inherited-source custody."""
from pathlib import Path
import subprocess
import tempfile

CAPTURE = Path(__file__).with_name("save_candidate_patch.sh")
with tempfile.TemporaryDirectory(prefix="geak-new-source-") as temp:
    root = Path(temp)
    workspace = root / "workspace"
    workspace.mkdir()
    def git(*args, cwd=workspace):
        return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()
    git("init", "-q")
    (workspace / "kernel_src").mkdir()
    (workspace / "kernel_src/start.py").write_text("original\n")
    (workspace / "cases.py").write_text("frozen oracle\n")
    git("add", ".")
    git("-c", "user.name=fixture", "-c", "user.email=fixture@localhost", "commit", "-qm", "original")
    original = git("rev-parse", "HEAD")
    (workspace / "kernel_src/parent.py").write_text("inherited parent source\n")
    git("add", "kernel_src/parent.py")
    git("-c", "user.name=fixture", "-c", "user.email=fixture@localhost", "commit", "-qm", "parent")
    (workspace / "kernel_src/new.hip").write_text("new HIP source\n")
    (workspace / "kernel_src/new.s").write_text("new assembly source\n")
    (workspace / "test_harness.py").write_text("unrelated harness\n")
    (workspace / "cases.py").write_text("must never enter a source patch\n")
    patch = root / "best.diff"
    subprocess.check_call(["bash", str(CAPTURE), str(workspace), str(patch), original])
    text = patch.read_text()
    assert "b/kernel_src/parent.py" in text
    assert "b/kernel_src/new.hip" in text and "b/kernel_src/new.s" in text
    assert "cases.py" not in text and "test_harness.py" not in text
    verify = root / "verify"
    subprocess.check_call(["git", "clone", "-q", str(workspace), str(verify)])
    git("reset", "--hard", original, cwd=verify)
    git("apply", "--index", "--", str(patch), cwd=verify)
    assert (verify / "kernel_src/parent.py").read_text() == "inherited parent source\n"
    assert (verify / "kernel_src/new.hip").read_text() == "new HIP source\n"
    assert (verify / "kernel_src/new.s").read_text() == "new assembly source\n"
    assert (verify / "cases.py").read_text() == "frozen oracle\n"
    assert not (verify / "test_harness.py").exists()
print("PASS: extracted scope retains new backend files and the inherited parent against ORIGINAL; harness edits stay out.")
