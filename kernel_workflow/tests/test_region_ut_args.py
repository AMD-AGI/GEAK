"""A region UT (args.region_ut, a GEAK UT task dir) supplies the mode=mega args."""

import importlib.util
import json
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "scripts" / "mega" / "region_ut_args.py"
SPEC = importlib.util.spec_from_file_location("region_ut_args", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

UNITTEST = '''import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--self-noise", action="store_true")
ap.add_argument("--iters", type=int, default=50)
'''


def _task(tmp_path, noise=None, unittest=UNITTEST, **meta_extra):
    meta = {"target_callable": "pkg.models.m:MoE.forward", "required_gpu_count": 4,
            "candidate_bind": {"kind": "rebind", "target": "pkg.models.m:MoE.forward", "impl_module": "moe_ar",
                               "impl_attr": "forward", "file": "kernel_src/moe_ar.py"},
            "acceptance": {"min_speedup": 1.0, "exact_outputs": [], "noise_bound": 0.01, "tol": 0.1},
            "cases": [{"sig": "pre_r77", "regime": "eager", "weight": 0.1, "effective_calls": 10},
                      {"sig": "verify_r240", "regime": "graph", "weight": 0.6, "effective_calls": 60},
                      {"sig": "draft_r40", "regime": "graph", "weight": 0.3, "effective_calls": 30}]}
    meta.update(meta_extra)
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "unittest.py").write_text(unittest)
    if noise is not None:
        (tmp_path / "run").mkdir()
        (tmp_path / "run" / "self_noise.json").write_text(json.dumps(
            {"per_case": [{"sig": s, "ratio": r} for s, r in noise.items()]}))
    return tmp_path


def test_the_ut_fixes_benchmark_guards_accuracy_and_module(tmp_path):
    a = MODULE.region_ut_args(_task(tmp_path))
    assert a["benchmark_harness"] == str(tmp_path / "unittest.py")
    assert a["candidate_import_modules"] == ["moe_ar"]
    assert a["required_accuracy_cases"] == ["pre_r77", "verify_r240", "draft_r40"]
    assert (a["target_guards"], a["regression_guards"]) == (["verify_r240"], ["pre_r77", "draft_r40"])
    assert a["regression_floor_by_guard"] == {"pre_r77": 0.98, "draft_r40": 0.98}
    assert (a["accuracy_metric"], a["accuracy_threshold"]) == ("ut_max_err", 0.1)
    assert a["required_replays"] == 50 and "region_ut_warning" not in a
    assert a["positive_control"]["guard"] == "verify_r240" and a["positive_control"]["abort_on_fail"] is False
    assert a["require_graph_capture"] == "false" and a["region_ut_format"] == "acceptance"


def test_a_noisy_case_gets_a_wider_floor(tmp_path):
    a = MODULE.region_ut_args(_task(tmp_path, noise={"pre_r77": 0.945, "draft_r40": 1.004, "verify_r240": 0.9}))
    assert a["regression_floor_by_guard"] == {"pre_r77": 0.89, "draft_r40": 0.98}


def test_the_replay_count_is_the_uts_own_and_no_graph_case_gives_none(tmp_path):
    a = MODULE.region_ut_args(_task(tmp_path, unittest=UNITTEST.replace("default=50", "default=10")))
    assert a["required_replays"] == 10 and "region_ut_warning" not in a
    b = MODULE.region_ut_args(_task(tmp_path, cases=[{"sig": "pre_r77", "regime": "eager", "weight": 1}]))
    assert b["target_guards"] == ["pre_r77"] and b["regression_guards"] == [] and "region_ut_warning" in b


def test_a_ut_without_cases_binds_or_tolerance_is_refused(tmp_path, capsys):
    assert MODULE.main([str(_task(tmp_path, cases=[]))]) == 1
    assert "no cases" in json.loads(capsys.readouterr().out)["error"]
    assert MODULE.main([str(_task(tmp_path, candidate_bind=None))]) == 1
    assert "binds no module" in json.loads(capsys.readouterr().out)["error"]
    assert MODULE.main([str(_task(tmp_path, acceptance={}))]) == 1
    assert "acceptance.tol" in json.loads(capsys.readouterr().out)["error"]
    assert MODULE.main([str(tmp_path / "missing")]) == 1


def test_an_e2e_kernel_ut_reads_its_workload_cases_and_tol(tmp_path):
    meta = {"target_callable": "pkg.ops.attn:decode", "tol": 0.001, "regime": {"cuda_graph": True},
            "candidate_bind": {"kind": "module", "module": "pkg.ops.attn", "file": "kernel_src/attn.py"},
            "phase_regimes": {"decode": {"cuda_graph": True}, "prefill": {"cuda_graph": False}},
            "workload": {"kind": "agentx_trace_replay", "cases": [
                {"sig": "dec_r96", "regime": "decode", "effective_calls": 2500},
                {"sig": "pre_r4096", "regime": "prefill", "effective_calls": 80}]}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "unittest.py").write_text("print('{\"UT\": \"pass\"}')\n")
    a = MODULE.region_ut_args(tmp_path)
    assert a["region_ut_format"] == "kernel_ut" and a["candidate_import_modules"] == ["pkg.ops.attn"]
    assert (a["target_guards"], a["regression_guards"]) == (["dec_r96"], ["pre_r4096"])
    assert a["accuracy_threshold"] == 0.001 and a["required_replays"] == 1 and "region_ut_warning" not in a
