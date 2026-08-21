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

import importlib.util
from pathlib import Path

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
