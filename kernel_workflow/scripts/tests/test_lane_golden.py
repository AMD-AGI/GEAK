"""kernel_lane.js optimize/author runs are pinned: every agent prompt, label and the result of each scripted
scenario (lane_mockrun.mjs) must match fixtures/lane_golden.json. Regenerate only for an intended change:
    GEAK_REGEN_LANE_GOLDEN=1 python -m pytest kernel_workflow/scripts/tests/test_lane_golden.py
"""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "fixtures" / "lane_golden.json"
SCENARIOS = ("optimize", "author", "multigpu", "stop", "deadline")
NODE = os.environ.get("GEAK_NODE_BIN") or shutil.which("node")
pytestmark = pytest.mark.skipif(not NODE, reason="node not available (set GEAK_NODE_BIN)")


def trace(scenario):
    p = subprocess.run([NODE, str(HERE / "lane_mockrun.mjs"), scenario], capture_output=True, text=True,
                       timeout=120, check=False)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    return {"calls": [[c["label"], c["phase"], hashlib.sha256(c["prompt"].encode()).hexdigest()]
                      for c in out["calls"]],
            "result": hashlib.sha256(json.dumps(out["result"], sort_keys=True).encode()).hexdigest()}


def test_optimize_and_author_runs_are_unchanged():
    got = {s: trace(s) for s in SCENARIOS}
    if os.environ.get("GEAK_REGEN_LANE_GOLDEN"):
        GOLDEN.write_text(json.dumps(got, indent=1) + "\n")
    want = json.loads(GOLDEN.read_text())
    for s in SCENARIOS:
        assert [c[0] for c in got[s]["calls"]] == [c[0] for c in want[s]["calls"]], f"{s}: agent sequence changed"
        changed = [c[0] for c, w in zip(got[s]["calls"], want[s]["calls"]) if c != w]
        assert not changed, f"{s}: prompts changed for {changed}"
        assert got[s]["result"] == want[s]["result"], f"{s}: result changed"


def test_the_lane_script_stays_under_the_workflow_size_cap():
    # The Workflow runtime rejects a script over 512 KiB; keep room for the next change.
    size = (HERE.parents[1] / "kernel_lane.js").stat().st_size
    assert size < 512 * 1024 - 16 * 1024, f"kernel_lane.js is {size} bytes"
