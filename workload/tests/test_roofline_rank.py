"""Keep unsupported analysis distinct from missing inputs and from whole-task gains."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
RANK = ROOT / "workload/stages/roofline/rank.py"
SKILL = ROOT / "workload/stages/roofline/kernel-roofline"
TASK_ORDER = ["moe", "attention", "allreduce", "norm", "quickreduce", "draft", "routing", "kv"]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, names=TASK_ORDER, legacy_plan=False):
    inp, out = tmp_path / "profile", tmp_path / "roofline"
    out.mkdir()
    tasks, selected = [], []
    for i, name in enumerate(names):
        source = inp / "tasks" / name / "kernel_src/leaf.py"
        source.parent.mkdir(parents=True)
        source.write_text("# CPU fixture source\n")
        cases = [{"sig": "first", "regime": "decode", "effective_calls": 10},
                 {"sig": "second", "regime": "decode", "effective_calls": 5},
                 {"sig": "diagnostic", "regime": "prefill", "effective_calls": 0}]
        write(source.parent.parent / "meta.json", {"workload": {"cases": cases}})
        leaf = "leaf_" + name
        tasks.append({"name": name, "selected_identities": [leaf]} if legacy_plan else
                     {"task": name, "selected_device_kernels": [{"device_kernel": leaf}]})
        selected.append({"id": leaf, "name": "alias_" + name, "overall_pct": float(10 - i)})
    write(inp / "task_plan.json", {"tasks": tasks})
    write(inp / "selection_targets.json", {"targets": selected})
    write(out / "params.json", {"roofline_mode": "tl-only"})
    return inp, out


def rows_for(names, supported=False):
    rows = []
    for name in names:
        for case, weight in [("first", 10), ("second", 5)]:
            rows.append({"kernel": name, "regime": "decode", "case": case,
                         "case_weight": weight, "measured_us": 100.0,
                         "roofline_us": 50.0 if supported else None,
                         "bound": "MEMORY_BOUND" if supported else None,
                         "reason": None if supported else "Native model does not represent this complete boundary"})
    return rows


def run_rank(inp, out, rows, roof_key=None):
    write(out / "rows.json", {"rows": rows})
    env = dict(os.environ, OUT=str(out), IN=str(inp), SKILL_ROOT=str(SKILL),
               PARAMS=str(out / "params.json"), STAGE="roofline", STAGE_SKILL="kernel-roofline")
    command = [sys.executable, str(RANK)]
    if roof_key:
        command += ["--roof-key", roof_key]
    process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    return process, json.loads((out / "roofline.json").read_text()), json.loads((out / "result.json").read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_partial_native_fold_never_becomes_whole_task_gain_and_keeps_all_tasks(tmp_path):
    inp, out = fixture(tmp_path)
    rows = rows_for(TASK_ORDER)
    for row in rows:
        row["ranking_admissible"] = False
        if row["kernel"] == "attention" and row["case"] == "first":
            row.update(roofline_us=10.0, bound="MEMORY_BOUND", reason=None)
    process, result, status = run_rank(inp, out, rows)
    assert process.returncode == 0, process.stdout + process.stderr
    assert status["status"] == "ok" and result["ranked_count"] == 0
    assert result["task_count"] == 8 and result["sum_expected_e2e_gain_pct"] is None
    assert [r["task"] for r in result["kernels"]] == TASK_ORDER
    assert all(r["expected_e2e_gain_pct"] is None and r["attainable_speedup"] is None
               and r["roofline_pct"] is None for r in result["kernels"])
    folded = json.loads((out / "folded.json").read_text())
    assert folded["attention/decode"]["speedup"] == 10.0
    scheduler = load("roofline_order_scheduler", ROOT / "workload/scripts/scheduler.py")
    order = scheduler.discover_tasks(str(inp / "tasks"), str(out / "roofline.json"))
    assert order == TASK_ORDER
    groups = load("roofline_order_groups", ROOT / "workload/stages/optimize/geak-kernel-gpu-groups/run_groups.py")
    requirements = {name: 8 if name in ("allreduce", "quickreduce") else 1 for name in order}
    plan = groups.plan_groups(requirements, list(range(8)), 43200, 200000, 1.5, True)
    assert (plan[0]["required_gpus"], plan[0]["slots"], plan[0]["batch"]) == (1, 6, 0)
    assert (plan[1]["required_gpus"], plan[1]["slots"], plan[1]["batch"]) == (8, 1, 1)


@pytest.mark.parametrize("leaf_keyed,legacy_plan", [(False, False), (True, False), (True, True)])
def test_complete_task_and_leaf_keyed_inputs_preserve_real_ranking(tmp_path, leaf_keyed, legacy_plan):
    inp, out = fixture(tmp_path, ["one"], legacy_plan)
    rows = rows_for(["one"], supported=True)
    if leaf_keyed:
        for row in rows:
            row["kernel"] = "alias_one"
    process, result, status = run_rank(inp, out, rows)
    assert process.returncode == 0, process.stdout + process.stderr
    row = result["kernels"][0]
    assert row["task"] == "one" and row["task_dir"] == "tasks/one"
    assert row["attainable_speedup"] == 2.0 and row["expected_e2e_gain_pct"] == 5.0
    assert status["status"] == "ok" and result["ranked_count"] == 1


def test_task_share_counts_each_owned_selected_identity_once(tmp_path):
    inp, out = fixture(tmp_path, ["one"])
    write(inp / "task_plan.json", {"tasks": [{"task": "one", "selected_identities": ["a", "b", "alias_a"]}]})
    write(inp / "selection_targets.json", {"targets": [
        {"id": "a", "name": "alias_a", "overall_pct": 7.0},
        {"id": "b", "name": "alias_b", "overall_pct": 3.0}]})
    process, result, _ = run_rank(inp, out, rows_for(["one"], True))
    assert process.returncode == 0
    assert result["kernels"][0]["pct_gpu_time"] == 10.0
    assert result["kernels"][0]["expected_e2e_gain_pct"] == 5.0


@pytest.mark.parametrize("missing", ["task", "case", "zeroed_positive_case", "explanation", "qualification"])
def test_missing_or_unprocessed_inputs_stay_visible_and_blocked(tmp_path, missing):
    inp, out = fixture(tmp_path)
    rows = rows_for(TASK_ORDER)
    if missing == "task":
        rows = [r for r in rows if r["kernel"] != "attention"]
    elif missing == "case":
        rows = [r for r in rows if not (r["kernel"] == "attention" and r["case"] == "second")]
    elif missing == "explanation":
        next(r for r in rows if r["kernel"] == "attention")["reason"] = None
    elif missing == "zeroed_positive_case":
        next(r for r in rows if r["kernel"] == "attention")["case_weight"] = 0
    else:
        next(r for r in rows if r["kernel"] == "attention")["provenance"] = {"qualification_refresh_required": True}
    process, result, status = run_rank(inp, out, rows)
    assert process.returncode != 0 and status["status"] == "blocked"
    assert {r["task"] for r in result["kernels"]} == set(TASK_ORDER)
    row = next(r for r in result["kernels"] if r["task"] == "attention")
    assert row["analysis"]["status"] == "incomplete_inputs"
    assert row["attainable_speedup"] is None and row["expected_e2e_gain_pct"] is None


def test_zero_weight_diagnostic_does_not_spend_mass_or_invalidate_coverage(tmp_path):
    inp, out = fixture(tmp_path, ["one"])
    rows = rows_for(["one"], True)
    rows.append({"kernel": "one", "regime": "prefill", "case": "diagnostic", "case_weight": 0,
                 "measured_us": 1e9, "roofline_us": None, "ranking_admissible": False, "bound": "COMPUTE_BOUND"})
    process, result, _ = run_rank(inp, out, rows)
    assert process.returncode == 0
    row = result["kernels"][0]
    assert row["expected_e2e_gain_pct"] == 5.0 and row["attainable_speedup"] == 2.0
    decode = next(r for r in row["regimes"] if r["regime"] == "decode")
    assert decode["time_share_within_kernel"] == 1.0
    assert next(r for r in row["regimes"] if r["regime"] == "prefill")["time_share_within_kernel"] == 0.0


@pytest.mark.parametrize("reason", ["explicit", "suspect", "partial_coverage"])
def test_inadmissible_case_blocks_whole_task_gain(tmp_path, reason):
    inp, out = fixture(tmp_path, ["one"])
    rows = rows_for(["one"], True)
    if reason == "explicit":
        rows[0]["ranking_admissible"] = False
    elif reason == "suspect":
        rows[0]["roofline_us"] = 110.0
    else:
        rows[0]["coverage"] = 0.5
    process, result, status = run_rank(inp, out, rows)
    assert process.returncode == 0 and status["status"] == "ok"
    assert result["kernels"][0]["attainable_speedup"] is None
    if reason == "suspect":
        assert result["kernels"][0]["suspect"] is True


def test_custom_roof_key_is_honored(tmp_path):
    inp, out = fixture(tmp_path, ["one"])
    rows = rows_for(["one"], True)
    for row in rows:
        row["native_roof"] = row.pop("roofline_us")
    process, result, _ = run_rank(inp, out, rows, "native_roof")
    assert process.returncode == 0
    assert result["kernels"][0]["expected_e2e_gain_pct"] == 5.0
