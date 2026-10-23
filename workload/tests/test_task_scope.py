import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


WORKLOAD = Path(__file__).resolve().parents[1]
ENTRY = WORKLOAD / "stages/optimize/check_scope.py"
spec = importlib.util.spec_from_file_location("task_scope", ENTRY)
scope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scope)


@pytest.mark.parametrize("limit", [None, True, 0, -1, "8"])
def test_invalid_capacity_is_not_an_unlimited_campaign(limit):
    with pytest.raises(ValueError, match="positive integer"):
        scope.check_limit(["a"], limit)


def test_all_eight_tasks_are_accepted_but_nine_are_not_truncated():
    tasks = ["kernel_%d" % i for i in range(8)]
    scope.check_dispatch(tasks, list(reversed(tasks)), 8)
    with pytest.raises(ValueError, match="exactly once"):
        scope.check_dispatch(tasks + ["ninth"], tasks, 8)
    with pytest.raises(ValueError, match="9 tasks are scheduled.*task_limit=8"):
        scope.check_dispatch(tasks + ["ninth"], tasks + ["ninth"], 8)


def test_the_ninth_task_is_deferred_with_a_reason_not_dropped():
    tasks = ["kernel_%d" % i for i in range(8)] + ["ninth"]
    deferred = [{"kernel": "ninth", "roofline_rank": 9, "reason": "beyond task_limit=8"}]
    result = scope.check_report(tasks, {**_report(), "tasks_total": 9,
                                        "kernels": [{"kernel": n, "attempts": 1}
                                                    for n in tasks[:8]],
                                        "kept": [], "capacity_deferred": deferred}, 8)
    assert result["tasks_capacity_deferred"] == 1
    assert result["capacity_deferred"] == ["ninth"]
    assert result["unattempted_tasks"] == ["ninth"]
    assert result["tasks_attempted"] == 8
    assert not result["all_tasks_attempted"]
    with pytest.raises(ValueError, match="needs a stated reason"):
        scope.check_dispatch(tasks, tasks[:8], 8, [{"kernel": "ninth"}])



def test_completed_report_can_span_more_tasks_than_inflight_capacity(tmp_path):
    tasks = ["kernel_%d" % i for i in range(10)]
    report = {"tasks_total": len(tasks), "kept": [], "kernels": [
        {"kernel": name, "attempts": 4, "credited_seconds": 172800} for name in tasks]}
    work = tmp_path / "work"
    (work / "state").mkdir(parents=True)
    for name in tasks:
        (work / "state" / (name + ".status.json")).write_text(
            '{"status":"done","credited_seconds":172800}')
    result = scope.check_report(tasks, report, 8, required_seconds=172800, work_root=work)
    assert result["tasks_attempted"] == result["tasks_reported"] == 10
    assert result["all_tasks_attempted"] and not result["capacity_deferred"]


def test_deferral_cannot_satisfy_a_full_budget_or_be_kept():
    tasks = ["a", "b"]
    report = {"tasks_total": 2, "kept": [],
              "kernels": [{"kernel": "a", "attempts": 1, "credited_seconds": 172800}],
              "capacity_deferred": [{"kernel": "b", "reason": "waiting for next allocation"}]}
    with pytest.raises(ValueError, match="deferred.*required 172800s"):
        scope.check_report(tasks, report, 1, required_seconds=172800)
    report["kept"] = [{"kernel": "b"}]
    with pytest.raises(ValueError, match="kept"):
        scope.check_report(tasks, report, 1)


def test_duplicate_deferrals_cannot_fake_exact_coverage():
    report = {"tasks_total": 2, "kept": [], "kernels": [{"kernel": "a", "attempts": 1}],
              "capacity_deferred": [{"kernel": "b", "reason": "capacity"}] * 2}
    with pytest.raises(ValueError, match="exactly once"):
        scope.check_report(["a", "b"], report, 1)

def test_capacity_cannot_excuse_leaving_a_slot_empty():
    tasks = ["a", "b", "c"]
    with pytest.raises(ValueError, match="slots.*unused"):
        scope.check_dispatch(tasks, ["a"], 8,
                             [{"kernel": "b", "reason": "capacity"},
                              {"kernel": "c", "reason": "capacity"}])


@pytest.mark.parametrize("selected", [["a"], ["a", "b", "unknown"], ["a", "a", "b"]])
def test_dispatch_must_cover_the_exact_input_set(selected):
    with pytest.raises(ValueError, match="exactly once"):
        scope.check_dispatch(["a", "b"], selected, 8)


def _report():
    return {"tasks_total": 3, "kept": [{"kernel": "a"}, {"kernel": "b"}],
            "kernels": [{"kernel": "a", "attempts": 2},
                        {"kernel": "b", "attempts": 3},
                        {"kernel": "c", "attempts": 3}]}


def test_two_keeps_do_not_mean_only_two_tasks_were_attempted():
    result = scope.check_report(["a", "b", "c"], _report(), 8)
    assert result == {"tasks_extracted": 3, "tasks_reported": 3,
                      "tasks_attempted": 3, "tasks_kept": 2,
                      "unattempted_tasks": [], "all_tasks_attempted": True,
                      "tasks_capacity_deferred": 0, "capacity_deferred": []}


def test_success_only_report_is_rejected():
    report = _report()
    report["kernels"].pop()
    report["tasks_total"] = 2
    with pytest.raises(ValueError, match="unaccounted=.*c"):
        scope.check_report(["a", "b", "c"], report, 8)


def test_zero_keeps_can_still_cover_all_extracted_tasks():
    report = _report()
    report["kept"] = []
    result = scope.check_report(["a", "b", "c"], report, 8)
    assert result["all_tasks_attempted"] and result["tasks_kept"] == 0


def test_unattempted_task_is_visible_and_needs_an_explanation():
    report = _report()
    report["kernels"][2]["attempts"] = 0
    with pytest.raises(ValueError, match="notes"):
        scope.check_report(["a", "b", "c"], report, 8)
    report["kernels"][2]["notes"] = ["Source prerequisite absent; recorded as blocked"]
    result = scope.check_report(["a", "b", "c"], report, 8)
    assert result["tasks_attempted"] == 2
    assert result["unattempted_tasks"] == ["c"]
    assert not result["all_tasks_attempted"]


def test_fixed_budget_cannot_be_relabelled_complete_after_a_short_window(tmp_path):
    report = _report()
    for row in report['kernels']:row['credited_seconds'] = 2000
    with pytest.raises(ValueError, match='required 43200s'):
        scope.check_report(['a','b','c'],report,8,required_seconds=43200)
    work = tmp_path/'work'
    (work/'state').mkdir(parents=True)
    for row in report['kernels']:
        row['credited_seconds'] = 43200
        (work/'state'/(row['kernel']+'.status.json')).write_text('{"status":"running","credited_seconds":2000}')
    with pytest.raises(ValueError, match='worker state'):
        scope.check_report(['a','b','c'],report,8,required_seconds=43200,work_root=work)
    for p in (work/'state').glob('*.json'):
        p.write_text('{"status":"done","credited_seconds":43200}')
    assert scope.check_report(['a','b','c'],report,8,required_seconds=43200,work_root=work)['all_tasks_attempted']


# The scope `select_candidates.selector_args` passes when $PARAMS says nothing. The slot
# now verifies that it is what the selector actually ran at, so every fixture records it.
POLICY = {"top_k_per_phase": 10, "overall_pct_gt": 0.5, "phases": ["PREFILL", "DECODE"]}


def _profile(tmp_path, count):
    out = tmp_path / "profile"
    out.mkdir()
    inventory = {"ranking": [{"id": "k%d" % n} for n in range(count)],
                 "selection_scope": {"policy": "explicit test scope",
                                     "selector": {"args": ["--top-k", "10",
                                                           "--overall-pct-gt", "0.5",
                                                           "--phases", "PREFILL", "DECODE"]}},
                 "denominators": {"total_device_ms": 1000.0},
                 "candidate_inventory": [{"id": "k%d" % n} for n in range(count)],
                 "coverage_summary": {"inventoried": count}}
    results = {"identities": [{"id": "k%d" % n, "task": "k%d" % n, "status": "delivered"}
                              for n in range(count)],
               "tasks": [{"task": "k%d" % n} for n in range(count)],
               "coverage": {"delivered_declared_pct_gpu_time": float(count),
                            "delivered_achieved_pct_gpu_time": float(count)}}
    for name, data in {"kernel_workload.json": {"total_gpu_time_ms": 1000.0},
                       "kernel_ranking.json": inventory, "results.json": results,
                       "selection_targets.json": {"policy": POLICY, "targets": [
                           {"id": "k%d" % n, "overall_pct": 1} for n in range(count)]}}.items():
        (out / name).write_text(json.dumps(data))
    for n in range(count):
        task = out / "tasks" / ("k%d" % n)
        (task / "kernel_src").mkdir(parents=True)
        (task / "run").mkdir()
        for name in ("unittest.py", "cases.py", "meta.json"):
            (task / name).write_text("{}")
        (task / "run/ut_result.json").write_text('{"ok": true, "failed_gates": []}')
        (task / "run/RETURN.json").write_text(json.dumps({"short_name": "k%d" % n}))
    return out


def _retarget(out, targets):
    """Replace the selected set, and the coverage claim that must stay consistent with it.

    These targets carry no `overall_pct`, so the delivery owns 0% of GPU time and may
    claim no more than that."""
    (out / "selection_targets.json").write_text(json.dumps({"policy": POLICY,
                                                            "targets": targets}))
    results = json.loads((out / "results.json").read_text())
    results.pop("identities")
    results["coverage"] = {"delivered_declared_pct_gpu_time": 0.0,
                           "delivered_achieved_pct_gpu_time": 0.0}
    (out / "results.json").write_text(json.dumps(results))


@pytest.mark.parametrize("count, deferred", [(8, 0), (9, 1)])
def test_profile_capacity_defers_but_never_truncates_extraction(tmp_path, count, deferred):
    out = _profile(tmp_path, count)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"stages": {"optimize": {"params": {"task_limit": 8}}}}))
    env = {**os.environ, "OUT": str(out), "CONFIG": str(config),
           "STAGE": "profile", "STAGE_SKILL": "test-profile"}
    p = subprocess.run([sys.executable, str(WORKLOAD / "stages/profile/finalize.py")],
                       env=env, capture_output=True, text=True)
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "ok", p.stdout + p.stderr
    assert len(list((out / "tasks").iterdir())) == count
    assert result["metrics"]["tasks_built"] == count
    assert result["metrics"]["tasks_capacity_deferred"] == deferred
    assert result["metrics"]["tasks_within_capacity"] == min(count, 8)
    if deferred:
        assert "task_limit=8" in result["notes"] and "later wave" in result["notes"]


def test_joint_boundary_mapping_preserves_identity_coverage(tmp_path):
    out = _profile(tmp_path, 1)
    _retarget(out, [{"id": "branch_2d"}, {"id": "branch_3d"}])
    (out / "task_plan.json").write_text(json.dumps({"tasks": [
        {"name": "k0", "selected_identities": ["branch_2d", "branch_3d"]}]}))
    config = tmp_path / "config.json"
    config.write_text("{}")
    env = {**os.environ, "OUT": str(out), "CONFIG": str(config),
           "STAGE": "profile", "STAGE_SKILL": "test-profile"}
    p = subprocess.run([sys.executable, str(WORKLOAD / "stages/profile/finalize.py")],
                       env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    metrics = json.loads((out / "result.json").read_text())["metrics"]
    assert metrics["targets_selected"] == metrics["identities_owned_by_tasks"] == 2
    assert metrics["tasks_built"] == 1


@pytest.mark.parametrize('kind', ['missing_owner', 'duplicate_owner'])
def test_profile_plan_cannot_claim_phantom_or_duplicate_coverage(tmp_path, kind):
    out = _profile(tmp_path, 2)
    _retarget(out, [{'id': 'chosen'}])
    owners = ['absent'] if kind == 'missing_owner' else ['k0', 'k1']
    (out / 'task_plan.json').write_text(json.dumps({'tasks': [
        {'name': name, 'selected_identities': ['chosen']} for name in owners]}))
    config = tmp_path / 'config.json'
    config.write_text('{}')
    env = {**os.environ, 'OUT': str(out), 'CONFIG': str(config),
           'STAGE': 'profile', 'STAGE_SKILL': 'test-profile'}
    p = subprocess.run([sys.executable, str(WORKLOAD / 'stages/profile/finalize.py')],
                       env=env, capture_output=True, text=True)
    result = json.loads((out / 'result.json').read_text())
    assert p.returncode != 0 and result['status'] == 'blocked'
    assert 'chosen' in result['notes']


def test_profile_cannot_claim_inventory_complete_from_only_selected_tasks(tmp_path):
    out = _profile(tmp_path, 1)
    (out / 'kernel_ranking.json').write_text('{"ranking": [{"id": "k0"}]}')
    config = tmp_path / 'config.json'
    config.write_text('{}')
    env = {**os.environ, 'OUT': str(out), 'CONFIG': str(config),
           'STAGE': 'profile', 'STAGE_SKILL': 'test-profile'}
    p = subprocess.run([sys.executable, str(WORKLOAD / 'stages/profile/finalize.py')],
                       env=env, capture_output=True, text=True)
    result = json.loads((out / 'result.json').read_text())
    assert p.returncode != 0 and result['status'] == 'blocked'
    assert 'candidate_inventory' in result['notes']


def test_standalone_plan_shape_keeps_real_owners_and_explicit_source_blockers(tmp_path):
    out = _profile(tmp_path, 1)
    _retarget(out, [{'id': 'selected'}, {'id': 'opaque'}])
    plan = {'tasks':[{'task':'k0','selected_device_kernels':[{'device_kernel':'selected'}]}],
            'blocked':[{'reason':'native source unavailable','selected_device_kernels':[{'device_kernel':'opaque'}]}]}
    (out / 'task_plan.json').write_text(json.dumps(plan))
    config = tmp_path / 'config.json'; config.write_text('{}')
    env = {**os.environ,'OUT':str(out),'CONFIG':str(config),'STAGE':'profile','STAGE_SKILL':'test-profile'}
    p = subprocess.run([sys.executable,str(WORKLOAD/'stages/profile/finalize.py')],env=env,capture_output=True,text=True)
    result = json.loads((out/'result.json').read_text())
    assert p.returncode == 0, p.stdout+p.stderr
    assert result['metrics']['identities_owned_by_tasks'] == 1
    assert result['metrics']['identities_blocked'] == 1
    sys.path.insert(0,str(WORKLOAD/'stages/roofline'))
    import rank
    assert rank.task_dir_index(plan) == {'selected':'k0'}
