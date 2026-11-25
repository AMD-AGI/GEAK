"""A region UT (args.region_ut) supplies the mode=mega args: its unittest.py, cases, tolerances, bound modules."""

import importlib.util
import json
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools" / "region_ut_args.py"
SPEC = importlib.util.spec_from_file_location("region_ut_args", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _task(tmp_path, **meta_extra):
    meta = {"target_callable": "pkg.mega.op:MegaMoE", "world_size": 8, "correct_metric": "rel_l2", "tol": 0.05,
            "candidate_bind": [{"kind": "module", "module": "pkg.mega.op", "file": "kernel_src/pkg/mega/op.py"},
                               {"kind": "module", "module": "pkg.mega.gemm", "file": "kernel_src/pkg/mega/gemm.py"}],
            "cases": [{"sig": "c64", "tol": 0.10, "weight": 0.7}, {"sig": "c8192", "tol": 0.02, "weight": 0.3}]}
    meta.update(meta_extra)
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "unittest.py").write_text("print('CORRECTNESS: PASS')\n")
    return tmp_path


def test_the_ut_fixes_benchmark_guards_accuracy_and_modules(tmp_path):
    a = MODULE.region_ut_args(_task(tmp_path))
    assert a["benchmark_harness"] == str(tmp_path / "unittest.py")
    assert a["candidate_import_modules"] == ["pkg.mega.op", "pkg.mega.gemm"]
    assert a["target_guards"] == a["required_accuracy_cases"] == ["c64", "c8192"]
    assert a["regression_guards"] == []
    assert (a["accuracy_metric"], a["accuracy_threshold"]) == ("relL2", 0.10)
    assert a["positive_control"]["guard"] == "c64" and a["positive_control"]["abort_on_fail"] is False
    assert a["require_graph_capture"] == "false" and a["region_ut_world_size"] == 8


def test_a_single_bind_and_meta_tolerance(tmp_path):
    a = MODULE.region_ut_args(_task(tmp_path, candidate_bind={"kind": "rebind", "target": "pkg.m:f",
                                                              "impl_module": "fast", "file": "kernel_src/fast.py"},
                                    cases=[{"sig": "only"}], correct_metric=None))
    assert a["candidate_import_modules"] == ["fast"]
    assert (a["accuracy_metric"], a["accuracy_threshold"]) == ("elementwise", 0.05)


def test_a_ut_without_cases_or_binds_is_refused(tmp_path, capsys):
    assert MODULE.main([str(_task(tmp_path, cases=[]))]) == 1
    assert "no cases" in json.loads(capsys.readouterr().out)["error"]
    assert MODULE.main([str(_task(tmp_path, candidate_bind=[]))]) == 1
    assert "binds no module" in json.loads(capsys.readouterr().out)["error"]
    assert MODULE.main([str(tmp_path / "missing")]) == 1
