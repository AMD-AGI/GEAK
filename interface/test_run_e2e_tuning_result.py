#!/usr/bin/env python3
"""Tests for the ADDITIVE ``tuning_skillset`` block in result.json.

CONTRACT under test: result.json is the artifact Hyperloom and every downstream consumer read. The
standalone tuning phase may ADD to it and may not otherwise touch it.

  1. A run without the tuning phase produces a result.json that is byte-identical to one from a build
     without the feature — the key is absent, not null, not empty.
  2. A run WITH the tuning phase changes exactly one thing: a new top-level ``tuning_skillset`` key.
     Every pre-existing key keeps its name, type and value.
  3. The block says how the win reaches production, because a tuned DATA artifact does not ride the
     PYTHONPATH overlay and a caller reproducing the bundle by hand needs the deploy step.

Run: python3 -m pytest GEAK/interface/test_run_e2e_tuning_result.py -v
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

from e2e_workflow.scripts.runtime_csv import build_runtime_csv
from e2e_workflow.scripts.tuning_acceptance import tuning_accepted

_HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("run_e2e_tuning", _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load()


def _wf(**extra) -> dict:
    """A minimal workflow return that normalize_result is happy with."""
    wf = {
        "baseline_throughput_tok_s": 1000.0,
        "final_throughput_tok_s": 1200.0,
        "throughput_speedup": 1.2,
        "output_parity": "pass",
        "accepted_config": {"flags": "--foo", "env": "BAR=1"},
    }
    wf.update(extra)
    return wf


def _tuning(**extra) -> dict:
    t = {
        "enabled": True,
        "ran": True,
        "gate": "accepted",
        "mode": "derived",
        "skills_used": ["tuning-core", "tuning-gemm"],
        "ops_tuned": [{"op": "gemm_a8w8_bpreshuffle", "engaged": True}],
        "pre_tune_throughput_tok_s": 1000.0,
        "post_tune_throughput_tok_s": 1080.0,
        "tuning_delta_pct": 8.0,
        "tuning_speedup": 1.08,
        "share_of_total_gain_pct": 40.0,
        "engagement_verified": True,
        "engagement_evidence": "kernel_gemm_0 dispatched (CK symbol gone)",
        "correctness_gate": "pass",
        "ab_interleaved": True,
        "ab_complete": True,
        "deploy_bundle": "/eval/tuning/deploy",
        "deploy_verified": True,
        "cache_invalidation": ["rm -rf /tmp/aiter_configs"],
        "live_tree_files": ["aiter/configs/model_configs/tuned_gemm_qwen3_8b.csv"],
        "apply_overlay": "/eval/tuning/overlay",
        "apply_env": "AITER_CONFIG_GEMM_BF16=/eval/tuning/tuned.csv",
        "artifacts": ["/eval/tuning/tuned.csv"],
        "in_final_bundle": True,
    }
    t.update(extra)
    return t


def _norm(tmp_path: Path, wf: dict) -> dict:
    return rx.normalize_result({}, dict(wf, eval_dir=str(tmp_path)))


# --------------------------------------------------------------------------- absence


def test_absent_when_phase_did_not_run(tmp_path):
    """No tuning in the workflow return => no key at all. Not null, not {}."""
    assert "tuning_skillset" not in _norm(tmp_path, _wf())


def test_absent_when_phase_disabled(tmp_path):
    """`tuning_skillset:"false"` still emits a disabled stub on the workflow return; result.json
    must stay clean so the feature is invisible to consumers when switched off."""
    out = _norm(tmp_path, _wf(tuning_skillset={"enabled": False, "ran": False}))
    assert "tuning_skillset" not in out


# --------------------------------------------------------------------------- additivity


def test_adds_exactly_one_key_and_changes_nothing_else(tmp_path):
    """The load-bearing test. Same workflow return, with and without the tuning block: the diff must
    be exactly one added key. This is what "do not change result.json" means operationally."""
    without = _norm(tmp_path, _wf())
    with_tuning = _norm(tmp_path, _wf(tuning_skillset=_tuning()))

    added = set(with_tuning) - set(without)
    assert added == {"tuning_skillset"}
    assert not set(without) - set(with_tuning), "no pre-existing key may be dropped"
    for key, value in without.items():
        assert with_tuning[key] == value, f"pre-existing key {key!r} was modified"


def test_headline_is_not_inflated_by_tuning(tmp_path):
    """The tuning gain is already inside the headline (later phases measure on the tuned stack), so the
    block must attribute, never add. A consumer summing the two would double-count."""
    out = _norm(tmp_path, _wf(tuning_skillset=_tuning()))
    assert out["throughput_speedup"] == 1.2
    assert out["final_throughput_tok_s"] == 1200.0
    assert "part of the headline" in out["tuning_skillset"]["explanation"]


def test_verified_runtime_table_is_bound_to_the_accepted_environment(tmp_path):
    baseline = tmp_path / "stock.csv"
    baseline.write_text("M,kernel\n64,stock\n128,retained\n")
    tuned = tmp_path / "tuned.csv"
    tuned.write_text("M,kernel\n64,candidate\n")
    output = tmp_path / "final/tuning/runtime/gemm"
    table = build_runtime_csv(baseline, [tuned], output, "AITER_CONFIG_GEMM_BF16", ["M"])
    tuning = _tuning(
        runtime_csv_manifests=[str(output / "runtime_csv.json")],
        apply_env=table["apply_env"], live_tree_files=[], cache_invalidation=[], deploy_bundle="",
    )
    wf = _wf(accepted_config={"flags": "", "env": table["apply_env"]}, tuning_skillset=tuning)
    result = _norm(tmp_path, wf)
    assert result["tuning_skillset"]["runtime_csvs"] == [table]
    assert result["accepted_config"]["env_map"] == {table["env_name"]: table["candidate"]["path"]}
    assert result["tuning_skillset"]["reaches_production_via"]["final_launch_runs_deploy"] is False
    wf["accepted_config"]["env"] = table["baseline_env"]
    with pytest.raises(ValueError, match="does not select"):
        _norm(tmp_path, wf)


# --------------------------------------------------------------------------- content


def test_accepted_block_carries_attribution_and_evidence(tmp_path):
    t = _norm(tmp_path, _wf(tuning_skillset=_tuning()))["tuning_skillset"]
    assert t["gate"] == "accepted"
    assert t["pre_tune_throughput_tok_s"] == 1000.0
    assert t["post_tune_throughput_tok_s"] == 1080.0
    assert t["share_of_total_gain_pct"] == 40.0
    assert t["engagement_verified"] is True
    assert t["engagement_evidence"]
    assert "1000.0 -> 1080.0" in t["explanation"]
    assert "40.0%" in t["explanation"]


def test_accepted_block_says_how_it_reaches_production(tmp_path):
    """A tuned config table is data, not code, so it cannot ride final_overlay. The block must point at
    the handles that DO carry it, or a caller reproducing the bundle silently loses the tuning."""
    t = _norm(tmp_path, _wf(tuning_skillset=_tuning()))["tuning_skillset"]
    prod = t["reaches_production_via"]
    assert prod["final_patch_includes_tuning"] is True
    assert prod["final_launch_runs_deploy"] is True
    assert prod["deploy_script"] == "/eval/tuning/deploy/deploy.sh"
    assert "final_patch" in prod["note"] and "deploy.sh" in prod["note"]
    assert t["cache_invalidation"] == ["rm -rf /tmp/aiter_configs"]
    assert t["deploy_bundle"] == "/eval/tuning/deploy"
    assert t["live_tree_files"] == ["aiter/configs/model_configs/tuned_gemm_qwen3_8b.csv"]
    # The code half of a tuning win (a routing switch that makes the tuned artifact bind) ships in the
    # accepted overlay, so it reaches production through the pre-existing final_overlay key.
    assert t["apply_overlay"] == "/eval/tuning/overlay"
    assert "apply_overlay" in prod["note"]


def test_deploy_script_falls_back_into_the_final_bundle(tmp_path):
    """With no bundle path recorded, point at where Finalize copies it, not at a dangling path."""
    t = _norm(tmp_path, _wf(tuning_skillset=_tuning(deploy_bundle="")))["tuning_skillset"]
    assert t["reaches_production_via"]["deploy_script"] == str(tmp_path / "final" / "tuning" / "deploy.sh")


# --------------------------------------------------------------------------- non-accepted


def test_no_win_block_omits_deploy_fields(tmp_path):
    """A phase that ran and won nothing must not advertise a deploy path — there is nothing to deploy,
    and an empty bundle in result.json would read as a shipped artifact."""
    t = _norm(tmp_path, _wf(tuning_skillset=_tuning(
        gate="no_win", reason="no candidate cleared the noise floor")))["tuning_skillset"]
    assert t["gate"] == "no_win"
    assert "did not bank a win" in t["explanation"]
    assert "no candidate cleared the noise floor" in t["explanation"]
    for key in ("deploy_bundle", "reaches_production_via", "apply_env", "artifacts",
                "live_tree_files", "apply_overlay"):
        assert key not in t
    # Attribution fields still present: a measured negative result is a result.
    assert t["pre_tune_throughput_tok_s"] == 1000.0
    assert t["engagement_verified"] is True


def test_rejected_block_reports_the_correctness_failure(tmp_path):
    t = _norm(tmp_path, _wf(tuning_skillset=_tuning(
        gate="rejected", correctness_gate="fail", reason="gsm8k dropped 0.93 -> 0.71")))["tuning_skillset"]
    assert t["gate"] == "rejected"
    assert t["correctness_gate"] == "fail"
    assert "gsm8k dropped" in t["explanation"]
    assert "reaches_production_via" not in t


def test_enabled_but_not_run(tmp_path):
    """A phase-scoped invocation that skipped tuning must not look like a measured no-win."""
    t = _norm(tmp_path, _wf(tuning_skillset={"enabled": True, "ran": False}))["tuning_skillset"]
    assert t["ran"] is False
    assert t["gate"] == "not_run"
    assert "did not run" in t["explanation"]
    assert "pre_tune_throughput_tok_s" not in t


# Strict phase acceptance must survive every report/recovery path.
def _valid_complete_pair():
    return _tuning(pre_tune_throughput_tok_s=1000.,post_tune_throughput_tok_s=1033.48,
                   tuning_delta_pct=3.348,tuning_speedup=1.03348,correctness_gate='none')


@pytest.mark.parametrize('field,value',[
    ('ab_complete',None),('ab_complete',False),('ab_complete',1),('ab_complete','true'),
    ('pre_tune_throughput_tok_s',None),('pre_tune_throughput_tok_s',0),('pre_tune_throughput_tok_s',-1),
    ('pre_tune_throughput_tok_s',True),('pre_tune_throughput_tok_s','1000'),
    ('pre_tune_throughput_tok_s',math.nan),('pre_tune_throughput_tok_s',math.inf),
    ('post_tune_throughput_tok_s',None),('post_tune_throughput_tok_s',0),('post_tune_throughput_tok_s',-1),
    ('post_tune_throughput_tok_s',math.nan),('post_tune_throughput_tok_s',math.inf),
    ('correctness_gate',None),('correctness_gate','unknown'),('correctness_gate','fail'),('ran',False),
    ('enabled',False),
])
def test_incomplete_or_invalid_phase_cannot_be_reported_as_accepted(tmp_path,field,value):
    tuning=_valid_complete_pair()
    if value is None:
        tuning.pop(field, None)
    else:
        tuning[field] = value
    assert not tuning_accepted(tuning)
    workflow = _wf(tuning_skillset=tuning)
    result = rx._tuning_skillset_section(workflow, tmp_path)
    if result is not None:
        assert result['gate']!='accepted'
        assert 'artifacts' not in result and 'apply_env' not in result
        if result['ran']:
            assert result['share_of_total_gain_pct'] is None
    with pytest.raises(ValueError, match="Workflow banks tuning"):
        _norm(tmp_path, workflow)


def test_complete_none_gate_preserves_the_3348_percent_object(tmp_path):
    tuning = _valid_complete_pair()
    before = copy.deepcopy(tuning)
    assert tuning_accepted(tuning)
    result=_norm(tmp_path,_wf(tuning_skillset=tuning))['tuning_skillset']
    assert result['gate']=='accepted' and result['ab_complete'] is True
    assert result['tuning_delta_pct']==3.348 and result['tuning_speedup']==1.03348
    assert tuning==before
    assert not tuning_accepted(tuning,accuracy_gate='gsm8k')
    assert not tuning_accepted({**tuning,'accuracy_gate':'gsm8k'})


@pytest.mark.parametrize('complete',[None,False])
def test_raw_claim_cannot_be_filed_by_kb_or_recovered_via_report(tmp_path,monkeypatch,complete):
    directory = tmp_path / 'tuning'
    directory.mkdir()
    tuning = _valid_complete_pair()
    tuning.pop('ab_complete')
    if complete is False:
        tuning['ab_complete'] = False
    (directory/'tuning_result.json').write_text(json.dumps(tuning))
    (directory/'tuning_report.md').write_text('Outcome: accepted (1000 -> 1033.48 tok/s)')
    monkeypatch.setattr(rx.subprocess,'run',lambda *a,**k:pytest.fail('An incomplete phase attempted a writer'))
    assert rx._kb_write_tuned_ops(tmp_path)['skipped'] is True
    assert rx._recover_tuning_result(tmp_path) is None
    assert rx._recover_tuning_report(tmp_path) is None


def test_report_alone_cannot_invent_explicit_completion(tmp_path):
    (tmp_path/'tuning').mkdir()
    (tmp_path/'tuning/tuning_report.md').write_text('Outcome: accepted (1000 -> 1033.48 tok/s)')
    assert rx._recover_tuning_report(tmp_path) is None


def test_none_tuning_is_not_accepted_when_handoff_requires_accuracy(tmp_path):
    with pytest.raises(ValueError, match="Workflow banks tuning"):
        rx.normalize_result({"accuracy_gate": "gsm8k"}, _wf(eval_dir=str(tmp_path), tuning_skillset=_valid_complete_pair()))


def test_env_only_tuning_claim_requires_a_complete_pair(tmp_path):
    tuning = _valid_complete_pair()
    tuning.pop("ab_complete")
    workflow = _wf(tuning_skillset=tuning,
                   accepted_config={"flags": "", "env": tuning["apply_env"]})
    with pytest.raises(ValueError, match="Workflow banks tuning"):
        _norm(tmp_path, workflow)


@pytest.mark.parametrize("carrier", ["accepted_kernels", "accepted_heads"])
@pytest.mark.parametrize("state", ["missing", "disabled", "incomplete", "disabled_complete"])
def test_tagged_tuning_requires_an_accepted_section(tmp_path, carrier, state):
    tuning = {}
    if state == "disabled":
        tuning["tuning_skillset"] = {"enabled": False, "ran": False}
    elif state in ("incomplete", "disabled_complete"):
        tuning["tuning_skillset"] = _valid_complete_pair()
        if state == "incomplete":
            tuning["tuning_skillset"].pop("ab_complete")
        else:
            tuning["tuning_skillset"]["enabled"] = False
    with pytest.raises(ValueError, match="Workflow banks tuning"):
        _norm(tmp_path, _wf(**tuning, **{carrier: [{
            "short_name": "unproven_tuned_op", "from_tuning_skillset": True,
        }]}))


@pytest.mark.parametrize("carrier", ["accepted_kernels", "accepted_heads"])
@pytest.mark.parametrize("tuning", [None, {"enabled": False, "ran": False}])
def test_independent_kernel_win_survives_without_tuning(tmp_path, carrier, tuning):
    kernels = [{"short_name": "independent_op", "e2e_delta_pct": 20.0}]
    result = _norm(tmp_path, _wf(tuning_skillset=tuning, **{carrier: kernels}))
    assert result[carrier] == kernels
    assert result["throughput_speedup"] == 1.2
    assert "tuning_skillset" not in result


@pytest.mark.parametrize("carrier", ["accepted_kernels", "accepted_heads"])
def test_complete_real_value_pair_keeps_tagged_tuning(tmp_path, carrier):
    tuning = _tuning(
        pre_tune_throughput_tok_s=146069.366,
        post_tune_throughput_tok_s=150959.985,
        tuning_speedup=150959.985 / 146069.366,
        tuning_delta_pct=(150959.985 / 146069.366 - 1) * 100,
        correctness_gate="none",
    )
    kernels = [{"short_name": "proven_tuned_op", "from_tuning_skillset": True}]
    result = _norm(tmp_path, _wf(tuning_skillset=tuning, **{carrier: kernels}))
    assert result[carrier] == kernels
    assert result["tuning_skillset"]["gate"] == "accepted"
    assert result["tuning_skillset"]["pre_tune_throughput_tok_s"] == 146069.366
    assert result["tuning_skillset"]["post_tune_throughput_tok_s"] == 150959.985
