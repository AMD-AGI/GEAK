import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

WORKLOAD = Path(__file__).resolve().parents[1]
ENTRY = WORKLOAD / "stages/profile/select_candidates.py"
SKILL = WORKLOAD / "stages/profile/kernel-extract"
spec = importlib.util.spec_from_file_location("selection_invocation", ENTRY)
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


@pytest.mark.parametrize("params", [{"selection_top_k": 0}, {"selection_top_k": True},
    {"selection_overall_pct_gt": float("nan")}, {"selection_overall_pct_gt": -1},
    {"selection_phases": []}, {"selection_phases": ["DECODE", "DECODE"]}])
def test_invalid_scope_cannot_fall_back_to_default(params):
    with pytest.raises(ValueError):
        selection.selector_args(params)


def test_upper_call_expands_candidates_and_resumes_without_overwriting(tmp_path):
    rows = [{"id": "k%d" % i, "selection_pct_gpu_time": 12 - i,
             "share_of_phase_pct": {"PREFILL": 12 - i, "DECODE": 12 - i}}
            for i in range(10)]
    ranking = tmp_path / "kernel_ranking.json"
    ranking.write_text(json.dumps({"ranking": rows}))
    params = tmp_path / "params.json"
    params.write_text("{}")
    env = {**os.environ, "OUT": str(tmp_path), "PARAMS": str(params), "SKILL_ROOT": str(SKILL)}
    cmd = [sys.executable, str(ENTRY)]
    first = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    output = tmp_path / "selection_candidates.json"
    before = output.read_bytes()
    assert json.loads(before)["target_count"] == 10
    repeated = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert repeated.returncode == 0 and "Reusing" in repeated.stdout
    assert output.read_bytes() == before
    params.write_text('{"selection_top_k": 5, "selection_overall_pct_gt": 3}')
    changed = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert changed.returncode != 0
    assert output.read_bytes() == before
    old = tmp_path / "narrow.json"
    narrowed = subprocess.run(cmd + ["--output", str(old)], env=env, capture_output=True, text=True)
    assert narrowed.returncode == 0
    assert json.loads(old.read_text())["target_count"] == 5
