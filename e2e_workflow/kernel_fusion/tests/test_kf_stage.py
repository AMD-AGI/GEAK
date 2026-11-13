"""kf_stage.py: the switch, the arg mapping, the fold into the main workflow, and the run_e2e hook."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import kf_stage

GEAK_ROOT = HERE.parents[2]
RUN_E2E = GEAK_ROOT / "interface" / "run_e2e.py"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("GEAK_KF_") or k == "GEAK_KERNEL_FUSION":
            monkeypatch.delenv(k)


def test_switch_off_by_default_and_precedence(monkeypatch):
    assert kf_stage.config({}) is None                                            # nothing said => off
    assert kf_stage.config({}, {"--kernel-fusion"})["enabled"]                    # CLI switches it on
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


def make_overlay(kf_dir, initial=None, site="import _kf_c1\n", extra=None):
    """An accepted overlay as the integrator leaves it: sources + the integrate_result.json of its A/B."""
    ov = Path(kf_dir) / "overlay" / "cand_kf_stack"
    ov.mkdir(parents=True, exist_ok=True)
    if initial:
        for rel, text in initial.items():
            (ov / rel).parent.mkdir(parents=True, exist_ok=True)
            (ov / rel).write_text(text)
    (ov / "_kf_c1.py").write_text("X = 1\n")
    (ov / "sitecustomize.py").write_text(site)
    for rel, text in (extra or {}).items():
        (ov / rel).write_text(text)
    (ov / "cand").mkdir(exist_ok=True)
    (ov / "cand" / "bench_summary.json").write_text("{}")
    (Path(kf_dir) / "integrate_kf_stack").mkdir(parents=True, exist_ok=True)
    (Path(kf_dir) / "integrate_kf_stack" / "integrate_result.json").write_text(json.dumps(
        {"ab_complete": True, "accepted_overlay": str(ov),
         "measurement_evidence": {"cand_summary": str(ov / "cand" / "bench_summary.json")}}))
    return str(ov)


def test_several_prompt_files(tmp_path):
    a, b = tmp_path / "prompt.txt", tmp_path / "deploy.txt"
    a.write_text("x"); b.write_text("y")
    args = kf_stage.build_args({"prompt": f"{a}, {b}"}, ps_args(tmp_path), 60)
    assert args["kf_prompt"] == f"{kf_stage.FUSION_STRATEGY},{a},{b}" and kf_stage.validate(args) == []
    args["kf_prompt"] += ",/no/such.txt"
    assert kf_stage.validate(args) == ["kf_prompt does not exist: /no/such.txt"]


def test_build_args_drops_main_only_keys_and_maps_knobs(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("x")
    a = kf_stage.build_args({"prompt": str(prompt), "steps": "topk", "author_k": 2}, ps_args(tmp_path), 3600)
    for k in ("eval_dir", "phases", "state", "tracelens", "apply_to_original"):
        assert k not in a
    assert a["tp"] == 8 and a["kf_prompt"] == f"{kf_stage.FUSION_STRATEGY},{prompt}" and a["kf_steps"] == "topk" and a["kf_author_k"] == 2
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
    ov = make_overlay(kf_stage.kf_dir_for({}, pa))
    R, seen = fake_R({"status": "accepted", "final_overlay": ov, "throughput_speedup": 1.03,
                      "accepted": [{"id": "c1"}], "eval_dir": "/kf"})
    kf = kf_stage.run_stage({"kernel_fusion": {"budget_s": 7200}}, pa, 36000, R)
    assert seen["script"] == kf_stage.KF_SCRIPT and R["E2E_SCRIPT"] == "/main.js"   # swapped only for the call
    assert str(kf_stage.KF_SCRIPT) in seen["prompt"] and seen["prompt"].startswith("SAFE")
    assert seen["args"]["time_budget_s"] == 7200 and seen["eval_dir"] == kf["kf_dir"]
    assert kf["status"] == "accepted" and kf["overlay_check"]["ok"]
    assert pa["initial_overlay_pythonpath"] == ov
    assert pa["time_budget_s"] == kf["main_timeout_s"] and kf["main_timeout_s"] <= 36000
    saved = json.loads((Path(kf["kf_dir"]) / "kf_return.json").read_text())
    assert saved["workflow_return"]["status"] == "accepted"
    # a resumed run reuses the finished stage instead of running it again
    R2, seen2 = fake_R(exc=AssertionError("must not run"))
    pa2 = ps_args(tmp_path)
    kf2 = kf_stage.run_stage({"kernel_fusion": True}, pa2, 36000, R2)
    assert kf2["reused"] and not seen2 and pa2["initial_overlay_pythonpath"] == ov


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
                       capture_output=True, text=True, env=e, timeout=120, check=False)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def test_run_e2e_dry_run_reports_the_switch(tmp_path):
    assert _dry_run(tmp_path, {}, ("--no-kernel-fusion",))["kernel_fusion"] == {"enabled": False}
    assert _dry_run(tmp_path, {"kernel_fusion": False})["kernel_fusion"] == {"enabled": False}
    assert _dry_run(tmp_path, {})["kernel_fusion"] == {"enabled": False}           # default off
    assert _dry_run(tmp_path, {}, env={"GEAK_KERNEL_FUSION": "1"})["kernel_fusion"]["enabled"]
    assert _dry_run(tmp_path, {}, ("--kernel-fusion",))["kernel_fusion"]["enabled"]
    on = _dry_run(tmp_path, {"kernel_fusion": True})["kernel_fusion"]
    assert on["enabled"] and on["script"].endswith("kernel_fusion/kf_workflow.js")
    assert on["args"]["tp"] == 2 and on["args"]["kf_dir"].startswith(str(tmp_path / "kf_M_"))
    assert _dry_run(tmp_path, {"kernel_fusion": True}, env={"GEAK_KERNEL_FUSION": "0"})["kernel_fusion"] == {"enabled": False}


def test_a_given_topk_is_passed_in_whole(tmp_path):
    t = tmp_path / "topk.json"
    t.write_text(json.dumps({"candidates": [{"id": "a"}], "recommended": ["a"]}))
    a = kf_stage.build_args({"topk_json": str(t)}, {"eval_dir": str(tmp_path / "e2e_x")}, 100)
    assert a["kf_topk"] == {"candidates": [{"id": "a"}], "recommended": ["a"]} and not kf_stage.validate(a)
    t.write_text(json.dumps({"candidates": [], "recommended": ["a"]}))
    a = kf_stage.build_args({"topk_json": str(t)}, {"eval_dir": str(tmp_path / "e2e_x")}, 100)
    assert any("no candidates" in p for p in kf_stage.validate(a))


def test_reuse_journal_hands_over_finished_prepare_and_author_results(tmp_path):
    j = tmp_path / "journal.jsonl"
    rows = [{"type": "started", "key": "k1", "label": "kf:author c1"},
            {"type": "result", "key": "k1", "result": {"final_patch": "p", "isolated_speedup": 1.2}},
            {"type": "started", "key": "k2", "label": "kf:kernel-extract prepare c1"},
            {"type": "result", "key": "k2", "result": {"delivery_path": "d", "prepared": {}}},
            {"type": "started", "key": "k3", "label": "kf:integrate kf_stack"},
            {"type": "result", "key": "k3", "result": {"gate": "rejected"}},
            {"type": "started", "key": "k4", "label": "kf:author c2"}]          # never finished
    j.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    a = kf_stage.build_args({"reuse_journal": str(j)}, {"eval_dir": str(tmp_path / "e2e_x")}, 100)
    assert sorted(a["kf_reuse"]) == ["kf:author c1", "kf:kernel-extract prepare c1"]


def test_reused_paths_follow_an_archived_eval_dir(tmp_path):
    arch = tmp_path / "attempt1"
    (arch / "exp" / "kf_x").mkdir(parents=True)
    rows = [{"type": "started", "key": "s", "label": "kf:director:setup"},
            {"type": "result", "key": "s", "result": {"eval_dir": "/gone/exp/kf_x"}},
            {"type": "started", "key": "a", "label": "kf:author c1"},
            {"type": "result", "key": "a", "result": {"overlay_dir": "/gone/exp/kf_x/kernels/c1/overlay"}}]
    (arch / "journal.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    r = kf_stage.reused_results(str(arch / "journal.jsonl"))
    assert r["kf:author c1"]["overlay_dir"] == f"{arch}/exp/kf_x/kernels/c1/overlay"


def test_args_carry_only_what_the_stage_reads(tmp_path):
    t = tmp_path / "topk.json"
    cands = [{"id": f"c{i}", "evidence": "x" * 5000, "seam": {"target_callable": "m:f"}} for i in range(15)]
    cands[0]["fallback"] = "c9"
    t.write_text(json.dumps({"candidates": cands, "recommended": ["c0", "c1", "c2", "c3"]}))
    a = kf_stage.build_args({"topk_json": str(t), "author_k": 2}, {"eval_dir": str(tmp_path / "e2e_x")}, 100)
    assert a["kf_topk"]["recommended"] == ["c0", "c1"]
    assert sorted(c["id"] for c in a["kf_topk"]["candidates"]) == ["c0", "c1", "c9"]
    assert all("evidence" not in c for c in a["kf_topk"]["candidates"])
    # no cap (the default): the Feasibility Review sees every candidate
    a = kf_stage.build_args({"topk_json": str(t)}, {"eval_dir": str(tmp_path / "e2e_x")}, 100)
    assert len(a["kf_topk"]["candidates"]) == 15 and a["kf_topk"]["recommended"] == ["c0", "c1", "c2", "c3"]
    prep = {"delivery_path": "d", "prepared": {"kernel_candidates": [{"short_name": "c0", "extra": "y" * 9000,
            "prepared_ut": {"phase": "extract", "result": {"task_dir": "/t", "baseline_frozen": True, "unittest_smoke": "pass",
                                                           "editable": True, "log": "z" * 9000}}}]}}
    slim = kf_stage.slim_result(prep)
    assert slim["prepared"]["kernel_candidates"][0]["prepared_ut"]["result"]["task_dir"] == "/t" and len(json.dumps(slim)) < 600
    au = kf_stage.slim_result({"final_patch": "p", "overlay_dir": "o", "isolated_speedup": 1.2, "correctness": "pass", "notes": "n" * 900})
    assert au["overlay_dir"] == "o" and len(au["notes"]) == 300


def test_several_reuse_journals_merge_and_the_screen_is_reused(tmp_path):
    def journal(name, rows):
        p = tmp_path / name
        p.write_text("\n".join(json.dumps(r) for r in rows))
        return str(p)
    j1 = journal("a.jsonl", [{"type": "started", "key": "a", "label": "kf:author c1"},
                             {"type": "result", "key": "a", "result": {"overlay_dir": "o1", "isolated_speedup": 1.1}}])
    j2 = journal("b.jsonl", [{"type": "started", "key": "s", "label": "kf:integrate screen"},
                             {"type": "result", "key": "s", "result": {"members": [{"short_name": "c1", "correctness": "pass", "reason": "r" * 900}]}}])
    a = kf_stage.build_args({"reuse_journal": f"{j1},{j2}"}, {"eval_dir": str(tmp_path / "e2e_x")}, 100)
    assert a["kf_reuse"]["kf:author c1"]["overlay_dir"] == "o1"
    assert a["kf_reuse"]["kf:integrate screen"]["members"][0]["correctness"] == "pass"
    assert len(a["kf_reuse"]["kf:integrate screen"]["members"][0]["reason"]) == 200


def test_the_fusion_strategy_is_read_first_and_can_be_replaced_or_dropped(tmp_path):
    deploy = tmp_path / "deploy.txt"
    deploy.write_text("x")
    assert kf_stage.FUSION_STRATEGY.is_file()
    assert kf_stage.build_args({}, ps_args(tmp_path), 60)["kf_prompt"] == str(kf_stage.FUSION_STRATEGY)
    a = kf_stage.build_args({"prompt": f"{deploy},{kf_stage.FUSION_STRATEGY}"}, ps_args(tmp_path), 60)
    assert a["kf_prompt"] == f"{kf_stage.FUSION_STRATEGY},{deploy}"                    # each file once, strategy first
    own = tmp_path / "my_strategy.md"
    own.write_text("x")
    assert kf_stage.build_args({"strategy": str(own), "prompt": str(deploy)}, ps_args(tmp_path), 60)["kf_prompt"] == f"{own},{deploy}"
    assert kf_stage.build_args({"strategy": False, "prompt": str(deploy)}, ps_args(tmp_path), 60)["kf_prompt"] == str(deploy)


def test_the_stage_stops_what_it_left_running_whatever_happened(tmp_path):
    calls = {}

    class FakeReaper:
        TERM_GRACE_S, KILL_WAIT_S = 1.0, 1.0

        @staticmethod
        def eval_dir_roots(d):
            calls["root"] = d
            return (d,)

        @staticmethod
        def protected_pids():
            return [os.getpid(), 1]

        @staticmethod
        def _stat(pid):
            return (1, 12345, False)

        @staticmethod
        def snapshot(mark, roots, since):
            calls.update(mark=mark, roots=roots, since=since)
            return {"p": "leg"}

        @staticmethod
        def select(procs, mode, keep):
            calls.update(mode=mode, keep=keep)
            return [("leg", "run")]

        @staticmethod
        def stop(chosen, term_grace_s, kill_wait_s):
            return [{"pid": 77, "name": "bench_e2e.sh", "why": "run"}], []

    for i, exc in enumerate((None, RuntimeError("workflow timed out"))):   # own dir each: no resumed stage
        R, _ = fake_R({"status": "no_gain", "eval_dir": "/kf"}, exc=exc)
        R["_run_reaper"] = lambda: FakeReaper
        kf = kf_stage.run_stage({"kernel_fusion": {"budget_s": 7200}}, ps_args(tmp_path / str(i)), 36000, R)
        assert kf["leftovers"]["stopped"][0]["pid"] == 77
        # only the stage dir, no run marker (that would select the whole run), the runner's ancestors kept
        assert calls["root"] == kf["kf_dir"] and calls["mark"] == "" and calls["since"] == 12345
        assert calls["mode"] == "exit" and os.getpid() in calls["keep"]
    R, _ = fake_R({"status": "no_gain", "eval_dir": "/kf"})
    kf = kf_stage.run_stage({"kernel_fusion": {"budget_s": 7200}}, ps_args(tmp_path / "b"), 36000, R)
    assert "nothing stopped" in kf["leftovers"]["note"]


def test_the_driver_prints_a_pointer_and_the_return_is_read_from_disk(tmp_path):
    # the driver no longer echoes the (40 KB) return on stdout -- echoing it is what failed to parse
    pa = ps_args(tmp_path)
    kf_dir = Path(kf_stage.kf_dir_for({}, pa))
    ov = make_overlay(kf_dir)
    R, seen = fake_R({"eval_dir": str(kf_dir), "workflow_return_json": str(kf_dir / "workflow_return.json"),
                      "status": "accepted"})
    orig = R["invoke_workflow"]

    def invoke(prompt, timeout_s, eval_dir, ps_args=None):
        Path(eval_dir).mkdir(parents=True, exist_ok=True)
        (Path(eval_dir) / "workflow_return.json").write_text(json.dumps({"summary": "x", "result": {
            "status": "accepted", "eval_dir": eval_dir, "final_overlay": ov, "throughput_speedup": 1.03,
            "accepted": [{"id": "c1"}]}}))
        return orig(prompt, timeout_s, eval_dir, ps_args)
    R["invoke_workflow"] = invoke
    kf = kf_stage.run_stage({"kernel_fusion": {"budget_s": 7200}}, pa, 36000, R)
    assert "never print it" in seen["prompt"] and '"workflow_return_json"' in seen["prompt"]
    assert kf["status"] == "accepted" and kf["throughput_speedup"] == 1.03 and "error" not in kf
    assert kf["accepted"] == [{"id": "c1"}] and pa["initial_overlay_pythonpath"] == ov


def test_a_parse_error_with_a_complete_return_on_disk_is_a_note_not_an_error(tmp_path):
    class WorkflowParseError(RuntimeError):
        pass
    for i, (on_disk, is_error) in enumerate(((True, False), (False, True))):
        pa = ps_args(tmp_path / str(i))
        kf_dir = Path(kf_stage.kf_dir_for({}, pa))

        def invoke(prompt, timeout_s, eval_dir, ps_args=None, on_disk=on_disk):
            Path(eval_dir).mkdir(parents=True, exist_ok=True)
            if on_disk:
                (Path(eval_dir) / "workflow_return.json").write_text(json.dumps(
                    {"summary": "x", "result": {"status": "no_gain", "eval_dir": eval_dir}}))
            raise WorkflowParseError("Could not parse a JSON workflow return")
        G = {"E2E_SCRIPT": "/main.js", "PROCESS_SAFETY": "SAFE\n", "invoke_workflow": invoke}
        kf = kf_stage.run_stage({"kernel_fusion": {"budget_s": 7200}}, pa, 36000, G)
        assert ("error" in kf) is is_error and (("note" in kf) is (not is_error))
        if on_disk:
            assert kf["status"] == "no_gain" and str(kf_dir) in kf["kf_dir"]


# --- failure / timeout isolation -------------------------------------------------------------------------

def test_a_failure_anywhere_in_the_stage_is_an_error_and_main_args_are_untouched(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    for i, where in enumerate(("build_args", "stop_leftovers", "validate_overlay")):
        with monkeypatch.context() as m:
            m.setattr(kf_stage, where, boom)
            pa = ps_args(tmp_path / str(i))
            before = json.loads(json.dumps(pa))
            ov = make_overlay(kf_stage.kf_dir_for({}, pa))
            R, _ = fake_R({"status": "accepted", "final_overlay": ov, "eval_dir": "/kf"})
            kf = kf_stage.run_stage({"kernel_fusion": True}, pa, 36000, R)
        assert kf["status"] == "error" and "disk full" in kf["error"], where
        assert {k: v for k, v in pa.items() if k != "time_budget_s"} == \
            {k: v for k, v in before.items() if k != "time_budget_s"}, where
        assert pa["initial_overlay_pythonpath"] == "/base"
        assert 36000 - 5 <= kf["main_timeout_s"] <= 36000 and pa["time_budget_s"] == kf["main_timeout_s"]


def test_a_stage_timeout_is_an_error_and_the_main_run_gets_the_rest(tmp_path):
    R, seen = fake_R(exc=TimeoutError("workflow exceeded 16740s"))
    pa = ps_args(tmp_path)
    kf = kf_stage.run_stage({"kernel_fusion": True}, pa, 43200, R)
    assert kf["status"] == "error" and "16740" in kf["error"] and pa["initial_overlay_pythonpath"] == "/base"
    assert seen["timeout_s"] <= 0.4 * 43200                      # the hard stop stays within 40% of the run
    assert 43200 - 5 <= kf["main_timeout_s"] <= 43200


def test_the_stage_budget_never_takes_more_than_its_share_of_the_run():
    for t in (7200, 21600, 43200, 86400, 172800):
        for cfg in ({}, {"budget_s": 10 ** 6}):
            b = kf_stage.stage_budget_s(cfg, t)
            assert kf_stage.hard_timeout_s(b) <= kf_stage.MAX_RUN_SHARE * t + 1, (t, cfg)
    assert kf_stage.stage_budget_s({}, 43200) == 14400                    # the default is unchanged here


def test_a_run_too_short_for_the_stage_skips_it(tmp_path):
    R, seen = fake_R(exc=AssertionError("must not run"))
    pa = ps_args(tmp_path)
    kf = kf_stage.run_stage({"kernel_fusion": True}, pa, 7200, R)
    assert kf["status"] == "skipped" and not seen and pa["initial_overlay_pythonpath"] == "/base"
    assert kf["main_timeout_s"] >= 7200 - 5


# --- overlay validation ------------------------------------------------------------------------------------

def _initial(tmp_path, files):
    d = tmp_path / "initial_overlay"
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
    return str(d)


def test_validate_overlay_accepts_a_stack_that_keeps_the_initial_overlay(tmp_path):
    kf_dir = tmp_path / "kf_x"
    init = _initial(tmp_path, {"sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n", "pkg/a.py": "A = 1\n"})
    ov = make_overlay(kf_dir, initial={"_initial_sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n",
                                       "pkg/a.py": "A = 1\n"},
                      site="import _initial_sitecustomize\nimport _kf_c1\n")
    check = kf_stage.validate_overlay(ov, init, kf_dir)
    assert check["ok"], check["problems"]
    assert len(check["manifest_sha256"]) == 64


def test_validate_overlay_rejects_what_would_lose_or_break_an_optimization(tmp_path):
    init_files = {"sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n"}
    kept = {"_initial_sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n"}
    good_site = "import _initial_sitecustomize\nimport _kf_c1\n"
    cases = {
        "missing": ({"initial": {"_initial_sitecustomize.py": "import tuned\n"}, "site": good_site}, "tuned.py"),
        "changed": ({"initial": {**kept, "tuned.py": "T = 3\n"}, "site": good_site}, "was changed"),
        "not imported": ({"initial": kept, "site": "import _kf_c1\n"}, "does not import it"),
        "dropped site": ({"initial": {"tuned.py": "T = 2\n"}, "site": good_site}, "is not kept"),
        "syntax": ({"initial": kept, "site": good_site, "extra": {"_kf_bad.py": "def (:\n"}}, "does not compile"),
        "dangling kernel": ({"initial": kept, "site": good_site,
                                 "extra": {"_kf_k.py": "K = '{kf}/kernels/c1/kernel'\n"}}, "does not exist"),
    }
    for i, (name, (kw, needle)) in enumerate(cases.items()):
        root = tmp_path / str(i)
        kf_dir = root / "kf_x"
        if "extra" in kw:
            kw["extra"] = {k: v.replace("{kf}", str(kf_dir)) for k, v in kw["extra"].items()}
        ov = make_overlay(kf_dir, **kw)
        check = kf_stage.validate_overlay(ov, _initial(root, init_files), kf_dir)
        assert not check["ok"] and any(needle in p for p in check["problems"]), (name, check["problems"])


def test_validate_overlay_ties_the_overlay_to_the_ab_that_accepted_it(tmp_path):
    kf_dir = tmp_path / "kf_x"
    ov = make_overlay(kf_dir)
    assert kf_stage.validate_overlay(ov, "", kf_dir)["ok"]
    first = kf_stage.validate_overlay(ov, "", kf_dir)["manifest_sha256"]
    # the integrator's A/B records land in the candidate dir after the A/B (DSv4.1-Flash c64: 4 min later):
    # they are records, not overlay content -- neither a change nor part of the manifest
    later = os.path.getmtime(Path(ov) / "cand" / "bench_summary.json") + 240
    for rec in ("e2e_validation.json", "manifests.json"):
        (Path(ov) / rec).write_text("{}")
        os.utime(Path(ov) / rec, (later, later))
    check = kf_stage.validate_overlay(ov, "", kf_dir)
    assert check["ok"], check["problems"]
    assert check["manifest_sha256"] == first
    os.utime(Path(ov) / "_kf_c1.py", (later, later))                       # code edited after the A/B
    assert any("after the A/B" in p for p in kf_stage.validate_overlay(ov, "", kf_dir)["problems"])
    (kf_dir / "integrate_kf_stack" / "integrate_result.json").unlink()
    assert any("no integrate_result.json" in p for p in kf_stage.validate_overlay(ov, "", kf_dir)["problems"])
    elsewhere = make_overlay(tmp_path / "other_kf")
    assert any("outside the stage dir" in p for p in kf_stage.validate_overlay(elsewhere, "", kf_dir)["problems"])


def test_an_accepted_overlay_that_fails_validation_is_not_handed_on(tmp_path):
    pa = ps_args(tmp_path)
    ov = make_overlay(kf_stage.kf_dir_for({}, pa), extra={"_kf_bad.py": "def (:\n"})
    R, _ = fake_R({"status": "accepted", "final_overlay": ov, "throughput_speedup": 1.03, "eval_dir": "/kf"})
    kf = kf_stage.run_stage({"kernel_fusion": True}, pa, 36000, R)
    assert kf["status"] == "rejected_overlay" and "does not compile" in kf["reason"]
    assert pa["initial_overlay_pythonpath"] == "/base" and "main_initial_overlay" not in kf
