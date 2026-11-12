"""The KernelFusion switch on the run_e2e side: kernel_fusion_args.py (switch + knobs -> workflow args), the
--dry-run mapping, and the additive ``kernel_fusion`` block in result.json."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from interface import kernel_fusion_args as kfa

RUN_E2E = Path(__file__).resolve().parent / "run_e2e.py"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("GEAK_KF_") or k == "GEAK_KERNEL_FUSION":
            monkeypatch.delenv(k)


def test_switch_off_by_default_and_precedence(monkeypatch):
    assert kfa.config({}) is None                                            # nothing said => off
    assert kfa.config({}, {"--kernel-fusion"})["enabled"]                    # CLI switches it on
    assert kfa.config({"kernel_fusion": False}) is None
    assert kfa.config({"kernel_fusion": True})["enabled"]
    assert kfa.config({"kernel_fusion": {"prompt": "/p"}})["prompt"] == "/p"   # object => on
    assert kfa.config({"kernel_fusion": {"enabled": False}}) is None
    monkeypatch.setenv("GEAK_KERNEL_FUSION", "0")
    assert kfa.config({"kernel_fusion": True}) is None                        # env beats handoff
    assert kfa.config({"kernel_fusion": True}, {"--kernel-fusion"})["enabled"]  # CLI beats env
    monkeypatch.setenv("GEAK_KERNEL_FUSION", "1")
    assert kfa.config({}, {"--no-kernel-fusion"}) is None
    monkeypatch.setenv("GEAK_KF_IMPLEMENTABLE_LEVELS", "L1")
    assert kfa.config({})["implementable_levels"] == "L1"


def test_off_adds_no_workflow_arg():
    assert kfa.workflow_args({}) == {}
    assert kfa.workflow_args({"kernel_fusion": False}) == {}
    assert kfa.workflow_args({"kernel_fusion": {"enabled": False, "implementable_levels": "L1"}}) == {}
    assert kfa.workflow_args({"kernel_fusion": True}, {"--no-kernel-fusion"}) == {}


def test_on_forwards_the_switch_and_the_knobs(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("x")
    a = kfa.workflow_args({"kernel_fusion": {"prompt": str(prompt), "implementable_levels": "L1,L2",
                                                 "unsupported_patterns": "MegaMoE", "task": "keep fp8",
                                                 "source_dirs": "/src/sglang"}})
    assert a["kernel_fusion"] == "true" and a["kf_prompt"] == f"{kfa.FUSION_STRATEGY},{prompt}"
    assert (a["kf_implementable_levels"], a["kf_unsupported_patterns"], a["kf_task"], a["kf_source_dirs"]) == (
        "L1,L2", "MegaMoE", "keep fp8", "/src/sglang")
    assert "kf_config_errors" not in a
    # the fusion profile reads the run's own trace, and the run's own kernel track optimizes and measures
    # each region: nothing of a separate stage run, and no step, window or A/B knob of its own
    gone = {"steps": "topk", "author_k": 2, "budget_s": 7200, "min_window_s": 60, "validation_mode": "x",
            "validation_replicas": 2, "accuracy_confirm_limit": 5, "min_gain_pct": 1}
    a = kfa.workflow_args({"kernel_fusion": gone})
    assert a == {"kernel_fusion": "true", "kf_prompt": str(kfa.FUSION_STRATEGY)}
    assert not {"kf_dir", "kf_baseline_dir", "kf_baseline_mode", "kf_reuse", "time_budget_s", "eval_dir"} & set(a)


def test_several_prompt_files(tmp_path):
    a, b = tmp_path / "prompt.txt", tmp_path / "deploy.txt"
    a.write_text("x"); b.write_text("y")
    args = kfa.build_args({"prompt": f"{a}, {b}"})
    assert args["kf_prompt"] == f"{kfa.FUSION_STRATEGY},{a},{b}" and kfa.validate(args) == []
    args["kf_prompt"] += ",/no/such.txt"
    assert kfa.validate(args) == ["kf_prompt does not exist: /no/such.txt"]


def test_a_knob_that_cannot_be_used_reaches_the_workflow_as_a_config_error(monkeypatch):
    a = kfa.workflow_args({"kernel_fusion": {"trace_dir": "/nope"}})
    assert a["kernel_fusion"] == "true" and a["kf_config_errors"] == "kf_trace_dir does not exist: /nope"
    monkeypatch.setenv("GEAK_KF_TRACE_DIR", "~no_such_user_kf549/traces")   # unresolvable: never raises
    a = kfa.workflow_args({"kernel_fusion": True})
    assert a == {"kernel_fusion": "true", "kf_config_errors": a["kf_config_errors"]}
    assert a["kf_config_errors"].startswith("RuntimeError")


def test_a_given_topk_is_passed_in_whole(tmp_path):
    t = tmp_path / "topk.json"
    t.write_text(json.dumps({"candidates": [{"id": "a"}], "recommended": ["a"]}))
    a = kfa.build_args({"topk_json": str(t)})
    assert a["kf_topk"] == {"candidates": [{"id": "a"}], "recommended": ["a"]} and not kfa.validate(a)
    t.write_text(json.dumps({"candidates": [], "recommended": ["a"]}))
    a = kfa.build_args({"topk_json": str(t)})
    assert any("no candidates" in p for p in kfa.validate(a))
    t.write_text('{"candidates": [')                                         # a truncated file is not a crash
    a = kfa.build_args({"topk_json": str(t)})
    assert any("no candidates" in p for p in kfa.validate(a))


def test_args_carry_every_candidate_and_only_what_the_workflow_reads(tmp_path):
    t = tmp_path / "topk.json"
    cands = [{"id": f"c{i}", "evidence": "x" * 5000, "seam": {"target_callable": "m:f"}} for i in range(15)]
    cands[0]["fallback"] = "c9"
    t.write_text(json.dumps({"candidates": cands, "recommended": ["c0", "c1", "c2", "c3"]}))
    a = kfa.build_args({"topk_json": str(t)})
    # the System Architect judges every candidate, so none is cut
    assert len(a["kf_topk"]["candidates"]) == 15 and a["kf_topk"]["recommended"] == ["c0", "c1", "c2", "c3"]
    assert all("evidence" not in c for c in a["kf_topk"]["candidates"])
    assert a["kf_topk"]["candidates"][0] == {"id": "c0", "seam": {"target_callable": "m:f"}, "fallback": "c9"}


def test_the_fusion_strategy_is_read_first_and_can_be_replaced_or_dropped(tmp_path):
    deploy = tmp_path / "deploy.txt"
    deploy.write_text("x")
    assert kfa.FUSION_STRATEGY.is_file()
    assert kfa.build_args({})["kf_prompt"] == str(kfa.FUSION_STRATEGY)
    a = kfa.build_args({"prompt": f"{deploy},{kfa.FUSION_STRATEGY}"})
    assert a["kf_prompt"] == f"{kfa.FUSION_STRATEGY},{deploy}"                    # each file once, strategy first
    own = tmp_path / "my_strategy.md"
    own.write_text("x")
    assert kfa.build_args({"strategy": str(own), "prompt": str(deploy)})["kf_prompt"] == f"{own},{deploy}"
    assert kfa.build_args({"strategy": False, "prompt": str(deploy)})["kf_prompt"] == str(deploy)


def _dry_run(tmp_path, extra_handoff, flags=(), env=None):
    h = {"model_path": "/models/M", "framework": "sglang", "tp": 2, "exp_root": str(tmp_path), **extra_handoff}
    hp = tmp_path / "handoff.json"
    hp.write_text(json.dumps(h))
    e = {k: v for k, v in os.environ.items() if not k.startswith(("GEAK_KF_", "GEAK_KERNEL_FUSION"))}
    e.update(env or {})
    p = subprocess.run([sys.executable, str(RUN_E2E), str(hp), str(tmp_path / "r.json"), "--dry-run", *flags],
                       capture_output=True, text=True, env=e, timeout=120, check=False)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)["mapped_args"]


def _kf_keys(mapped):
    return {k for k in mapped if k == "kernel_fusion" or k.startswith("kf_")}


def _stable(mapped):
    return {k: v for k, v in mapped.items() if k not in ("eval_dir", "geak_launch_nonce")}   # new per launch


def test_run_e2e_forwards_the_switch_and_nothing_when_off(tmp_path):
    base = _dry_run(tmp_path, {})
    assert not _kf_keys(base)                                                     # default off
    assert not _kf_keys(_dry_run(tmp_path, {"kernel_fusion": False}))
    assert not _kf_keys(_dry_run(tmp_path, {"kernel_fusion": True}, ("--no-kernel-fusion",)))
    assert not _kf_keys(_dry_run(tmp_path, {"kernel_fusion": True}, env={"GEAK_KERNEL_FUSION": "0"}))
    off = _dry_run(tmp_path, {"kernel_fusion": False})
    assert set(off) == set(base) and _stable(off) == _stable(base)                # off: the args are unchanged
    assert _dry_run(tmp_path, {}, env={"GEAK_KERNEL_FUSION": "1"})["kernel_fusion"] == "true"
    on = _dry_run(tmp_path, {"kernel_fusion": {"implementable_levels": "L1,L2"}}, ("--kernel-fusion",))
    assert on["kernel_fusion"] == "true" and on["kf_implementable_levels"] == "L1,L2"
    assert on["kf_prompt"] == str(kfa.FUSION_STRATEGY) and on["tp"] == base["tp"]
    assert set(on) - set(base) == _kf_keys(on)                                    # on: only kf args are added


def _run_e2e():
    spec = importlib.util.spec_from_file_location("run_e2e_kf", RUN_E2E)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_kernel_fusion_record_is_added_to_the_result_and_nothing_else_changes(tmp_path):
    rx = _run_e2e()
    wf = {"baseline_throughput_tok_s": 1000.0, "final_throughput_tok_s": 1200.0, "throughput_speedup": 1.2,
          "output_parity": "pass", "accepted_config": {"flags": "--foo", "env": "BAR=1"}, "eval_dir": str(tmp_path)}
    plain = rx.normalize_result({}, dict(wf))
    assert "kernel_fusion" not in plain
    record = {"status": "accepted", "routed": ["kf_c1"], "accepted": [{"short_name": "kf_c1", "fusion_id": "c1"}],
              "soft_parity": True, "outcomes": [{"short_name": "kf_c1", "verdict": "confirmed"}]}
    with_kf = rx.normalize_result({}, dict(wf, kernel_fusion=record))
    assert with_kf["kernel_fusion"] == record
    assert {k: v for k, v in with_kf.items() if k != "kernel_fusion"} == plain
