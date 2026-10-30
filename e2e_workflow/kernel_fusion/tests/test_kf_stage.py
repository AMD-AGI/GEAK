"""kf_stage.py: the switch, the arg mapping, the fold into the main workflow, and the run_e2e hook."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import kf_stage  # noqa: E402

GEAK_ROOT = HERE.parents[2]
RUN_E2E = GEAK_ROOT / "interface" / "run_e2e.py"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("GEAK_KF_") or k == "GEAK_KERNEL_FUSION":
            monkeypatch.delenv(k)


def test_switch_on_by_default_and_precedence(monkeypatch):
    assert kf_stage.config({})["enabled"]                                         # external call => on
    assert kf_stage.config({"kernel_fusion": False}) is None
    assert kf_stage.config({"kernel_fusion": True})["enabled"]
    assert kf_stage.config({"kernel_fusion": {"prompt": "/p"}})["prompt"] == "/p"   # object => on
    assert kf_stage.config({"kernel_fusion": {"enabled": False}}) is None
    monkeypatch.setenv("GEAK_KERNEL_FUSION", "0")
    assert kf_stage.config({"kernel_fusion": True}) is None                        # env beats handoff
    assert kf_stage.config({"kernel_fusion": True}, {"--kernel-fusion"})["enabled"]  # CLI beats env
    monkeypatch.setenv("GEAK_KERNEL_FUSION", "1")
    assert kf_stage.config({}, {"--no-kernel-fusion"}) is None
    monkeypatch.setenv("GEAK_KF_STEPS", "topk")
    assert kf_stage.config({})["steps"] == "topk"


def ps_args(tmp_path):
    return {"model_path": "/m", "workflow_dir": "/w", "eval_dir": str(tmp_path / "e2e_M_20261030T0_ab12Z"),
            "initial_overlay_pythonpath": "/base", "time_budget_s": 36000, "phases": "all", "state": {"x": 1},
            "tracelens": {}, "apply_to_original": "true", "tp": 8}


def test_several_prompt_files(tmp_path):
    a, b = tmp_path / "prompt.txt", tmp_path / "deploy.txt"
    a.write_text("x"); b.write_text("y")
    args = kf_stage.build_args({"prompt": f"{a}, {b}"}, ps_args(tmp_path), 60)
    assert args["kf_prompt"] == f"{a},{b}" and kf_stage.validate(args) == []
    args["kf_prompt"] += ",/no/such.txt"
    assert kf_stage.validate(args) == ["kf_prompt does not exist: /no/such.txt"]


def test_build_args_drops_main_only_keys_and_maps_knobs(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("x")
    a = kf_stage.build_args({"prompt": str(prompt), "steps": "topk", "author_k": 2}, ps_args(tmp_path), 3600)
    for k in ("eval_dir", "phases", "state", "tracelens", "apply_to_original"):
        assert k not in a
    assert a["tp"] == 8 and a["kf_prompt"] == str(prompt) and a["kf_steps"] == "topk" and a["kf_author_k"] == 2
    assert a["time_budget_s"] == 3600
    assert a["kf_dir"] == str(tmp_path / "kf_M_20261030T0_ab12Z")
    assert kf_stage.validate(a) == []
    assert kf_stage.validate({**a, "kf_trace_dir": "/nope"}) == ["kf_trace_dir does not exist: /nope"]
    assert kf_stage.stage_budget_s({}, 30000) == 10000
    assert kf_stage.stage_budget_s({}, 10 ** 6) == kf_stage.DEFAULT_BUDGET_S


def fake_R(ret=None, exc=None):
    seen = {}

    def invoke_workflow(prompt, timeout_s, eval_dir, ps_args=None):
        seen.update(prompt=prompt, timeout_s=timeout_s, eval_dir=eval_dir, args=ps_args, script=G["E2E_SCRIPT"])
        if exc:
            raise exc
        return ret
    G = {"E2E_SCRIPT": "/main.js", "PROCESS_SAFETY": "SAFE\n", "invoke_workflow": invoke_workflow}
    return G, seen


def test_run_stage_accepts_and_folds_overlay(tmp_path):
    pa = ps_args(tmp_path)
    R, seen = fake_R({"status": "accepted", "final_overlay": "/kf/overlay/cand_c1", "throughput_speedup": 1.03,
                      "accepted": [{"id": "c1"}], "eval_dir": "/kf"})
    kf = kf_stage.run_stage({"kernel_fusion": {"budget_s": 600}}, pa, 36000, R)
    assert seen["script"] == kf_stage.KF_SCRIPT and R["E2E_SCRIPT"] == "/main.js"   # swapped only for the call
    assert str(kf_stage.KF_SCRIPT) in seen["prompt"] and seen["prompt"].startswith("SAFE")
    assert seen["args"]["time_budget_s"] == 600 and seen["eval_dir"] == kf["kf_dir"]
    assert kf["status"] == "accepted"
    assert pa["initial_overlay_pythonpath"] == "/kf/overlay/cand_c1"
    assert pa["time_budget_s"] == kf["main_timeout_s"] and kf["main_timeout_s"] <= 36000
    saved = json.loads((Path(kf["kf_dir"]) / "kf_return.json").read_text())
    assert saved["workflow_return"]["status"] == "accepted"
    # a resumed run reuses the finished stage instead of running it again
    R2, seen2 = fake_R(exc=AssertionError("must not run"))
    pa2 = ps_args(tmp_path)
    kf2 = kf_stage.run_stage({"kernel_fusion": True}, pa2, 36000, R2)
    assert kf2["reused"] and not seen2 and pa2["initial_overlay_pythonpath"] == "/kf/overlay/cand_c1"


def test_analysis_reports_are_written_by_the_stage(tmp_path):
    pa = ps_args(tmp_path)
    kf_dir = kf_stage.kf_dir_for({}, pa)
    rep = str(kf_dir) + "/analysis/REPORT.md"
    R, _ = fake_R({"status": "analysis_only", "eval_dir": str(kf_dir),
                   "analysis_files": {rep: "# r", "/etc/evil.md": "x"}})
    kf_stage.run_stage({"kernel_fusion": True}, pa, 36000, R)
    assert Path(rep).read_text() == "# r" and not Path("/etc/evil.md").exists()
    saved = json.loads((kf_dir / "kf_return.json").read_text())["workflow_return"]
    assert "analysis_files" not in saved


def test_run_stage_no_gain_or_failure_leaves_main_args_alone(tmp_path):
    for ret, exc in (({"status": "no_gain", "final_overlay": "/base"}, None), (None, RuntimeError("boom"))):
        pa = ps_args(tmp_path / str(bool(exc)))
        R, _ = fake_R(ret, exc)
        kf = kf_stage.run_stage({"kernel_fusion": True}, pa, 36000, R)
        assert pa["initial_overlay_pythonpath"] == "/base"
        assert kf["status"] == ("no_gain" if ret else "error")
        if exc:
            assert "boom" in kf["error"]


def test_envelope_on_disk_is_unwrapped(tmp_path):
    """The driver's stdout had no JSON; the Workflow tool's own envelope on disk carries the return."""
    pa = ps_args(tmp_path)
    kf_dir = kf_stage.kf_dir_for({}, pa)
    kf_dir.mkdir(parents=True)
    (kf_dir / "workflow_return.json").write_text(json.dumps(
        {"summary": "x", "logs": [], "result": {"eval_dir": str(kf_dir), "status": "analysis_only",
                                                "analysis_files": {str(kf_dir) + "/analysis/REPORT.md": "# r"}}}))
    R, _ = fake_R(exc=RuntimeError("WorkflowParseError"))
    kf = kf_stage.run_stage({"kernel_fusion": True}, pa, 36000, R)
    assert kf["status"] == "analysis_only" and (kf_dir / "analysis" / "REPORT.md").read_text() == "# r"


def test_sigterm_self_stop_propagates(tmp_path):
    R, _ = fake_R(exc=TimeoutError("signal 15: self-stop to flush interface files"))
    with pytest.raises(TimeoutError):
        kf_stage.run_stage({"kernel_fusion": True}, ps_args(tmp_path), 36000, R)


def test_bad_prompt_path_is_an_error_without_running(tmp_path):
    R, seen = fake_R(exc=AssertionError("must not run"))
    kf = kf_stage.run_stage({"kernel_fusion": {"prompt": "/no/such/prompt.txt"}}, ps_args(tmp_path), 36000, R)
    assert kf["status"] == "error" and "kf_prompt does not exist" in kf["error"] and not seen


def test_attach(tmp_path):
    rp = tmp_path / "result.json"
    out = kf_stage.attach(rp, {"status": "ok", "throughput_speedup": 1.1}, {"throughput_speedup": 1.05})
    assert out["kernel_fusion"]["combined_speedup_vs_original"] == pytest.approx(1.155)
    assert json.loads(rp.read_text())["kernel_fusion"]["throughput_speedup"] == 1.05


def _dry_run(tmp_path, extra_handoff, flags=(), env=None):
    h = {"model_path": "/models/M", "framework": "sglang", "tp": 2, "exp_root": str(tmp_path), **extra_handoff}
    hp = tmp_path / "handoff.json"
    hp.write_text(json.dumps(h))
    e = {k: v for k, v in os.environ.items() if not k.startswith(("GEAK_KF_", "GEAK_KERNEL_FUSION"))}
    e.update(env or {})
    p = subprocess.run([sys.executable, str(RUN_E2E), str(hp), str(tmp_path / "r.json"), "--dry-run", *flags],
                       capture_output=True, text=True, env=e, timeout=120)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def test_run_e2e_dry_run_reports_the_switch(tmp_path):
    assert _dry_run(tmp_path, {}, ("--no-kernel-fusion",))["kernel_fusion"] == {"enabled": False}
    assert _dry_run(tmp_path, {"kernel_fusion": False})["kernel_fusion"] == {"enabled": False}
    on = _dry_run(tmp_path, {})["kernel_fusion"]                                  # default on
    assert on["enabled"] and on["script"].endswith("kernel_fusion/kf_workflow.js")
    assert on["args"]["tp"] == 2 and on["args"]["kf_dir"].startswith(str(tmp_path / "kf_M_"))
    assert _dry_run(tmp_path, {"kernel_fusion": True}, env={"GEAK_KERNEL_FUSION": "0"})["kernel_fusion"] == {"enabled": False}
