#!/usr/bin/env python3
"""Tests for run_e2e's guaranteed interface-file emission + intermediate-win
recovery.

CONTRACT under test: as long as PerfSkills produced ANY measured E2E effect on
disk, result.json (+ kernel_journey.json) MUST be written — no termination,
timeout, signal, or exception may leave the interface files missing.

Run: python3 -m pytest GEAK/interface/test_run_e2e_recovery.py -v
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("run_e2e", _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load()


def _make_eval_dir(tmp_path: Path, *, accepted: bool = True,
                   with_validation: bool = False) -> Path:
    """Build a fake eval_dir with a bench_e2e.sh + an accepted intermediate."""
    eval_dir = tmp_path / "e2e_fake"
    (eval_dir / "overlay" / "cand_fused_moe_kernel_gptq_awq").mkdir(parents=True)
    (eval_dir / "bench_e2e.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    ir = {
        "short_name": "fused_moe_kernel_gptq_awq",
        "winner_kind": "env",
        "apply_env": "VLLM_TUNED_CONFIG_FOLDER=/x/config/integrate_moe_tuned",
        "apply_flags": "--max-num-batched-tokens 16384",
        "isolated_speedup": 1.5902,
        "ref_med": 461.314, "cand_med": 535.352,
        "e2e_throughput_tok_s": 535.352, "e2e_delta_pct": 16.049,
        "output_parity": "pass",
        "gate": "accepted" if accepted else "rejected",
        "serving_config": {"backend": "vllm", "tp": 8, "gpu": "0,1,2,3,4,5,6,7"},
    }
    (eval_dir / "overlay" / "cand_fused_moe_kernel_gptq_awq"
     / "integrate_result.json").write_text(json.dumps(ir), encoding="utf-8")
    if with_validation:
        (eval_dir / "director_e2e_validation.json").write_text(json.dumps({
            "baseline_throughput_tok_s": 461.314,
            "director_verified_throughput_tok_s": 535.352,
            "throughput_speedup": 1.16, "output_parity": "pass",
            "serving_config": {"final_flags": "--max-num-batched-tokens 16384"},
        }), encoding="utf-8")
    return eval_dir


def _handoff(eval_dir: Path) -> dict:
    return {
        "schema_version": 1, "model_path": "/models/fake", "framework": "vllm",
        "tp": 8, "workload": {"isl": 8192, "osl": 1024, "conc": 64},
        "exp_root": str(eval_dir.parent), "eval_dir": str(eval_dir),
    }


# ── intermediate-win recovery ───────────────────────────────────────────────

def test_recover_best_intermediate_win_config(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, accepted=True)
    wf = rx._recover_best_intermediate_win(eval_dir)
    assert wf is not None
    assert wf["recovered_intermediate"] is True
    assert wf["final_throughput_tok_s"] == pytest.approx(535.352)
    assert wf["throughput_speedup"] == pytest.approx(535.352 / 461.314)
    assert wf["accepted_config"]["flags"] == "--max-num-batched-tokens 16384"
    assert "VLLM_TUNED_CONFIG_FOLDER" in wf["accepted_config"]["env"]
    # winner_kind == "env" => config-only, not an authored kernel.
    assert wf["accepted_kernels"] == []


def test_recover_skips_rejected(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, accepted=False)
    assert rx._recover_best_intermediate_win(eval_dir) is None


def test_recover_workflow_return_falls_back_to_intermediate(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, accepted=True, with_validation=False)
    wf = rx._recover_workflow_return(eval_dir.parent)
    assert wf is not None and wf.get("recovered_intermediate") is True


def test_recover_workflow_return_prefers_validation(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, accepted=True, with_validation=True)
    wf = rx._recover_workflow_return(eval_dir.parent)
    assert wf is not None
    # The director path does NOT tag recovered_intermediate.
    assert not wf.get("recovered_intermediate")


# ── guaranteed emit in main() ───────────────────────────────────────────────

def _run_main(monkeypatch, tmp_path, eval_dir, *, invoke):
    monkeypatch.setattr(rx, "invoke_workflow", invoke)
    monkeypatch.setattr(rx, "apply_bench_client", lambda h: "native")
    monkeypatch.setattr(rx, "apply_bench_protocol", lambda h: {})
    hp = tmp_path / "handoff.json"
    rp = tmp_path / "out" / "result.json"
    hp.write_text(json.dumps(_handoff(eval_dir)), encoding="utf-8")
    rc = rx.main([str(hp), str(rp)])
    return rc, rp


def test_emit_on_success(monkeypatch, tmp_path):
    eval_dir = _make_eval_dir(tmp_path, with_validation=True)

    def ok_invoke(prompt, t, ed):
        return {"eval_dir": str(eval_dir), "throughput_speedup": 1.16,
                "final_throughput_tok_s": 535.352,
                "baseline_throughput_tok_s": 461.314}

    rc, rp = _run_main(monkeypatch, tmp_path, eval_dir, invoke=ok_invoke)
    assert rp.is_file(), "result.json MUST exist on success"
    out = json.loads(rp.read_text())
    assert out["status"] == "ok"
    assert (eval_dir / "kernel_journey.json").is_file()


def test_emit_when_workflow_raises_but_disk_has_intermediate(monkeypatch, tmp_path):
    """The killer case: workflow dies before Validate, but an accepted
    intermediate is on disk -> result.json MUST still be ok (not discarded)."""
    eval_dir = _make_eval_dir(tmp_path, accepted=True, with_validation=False)

    def boom(prompt, t, ed):
        raise TimeoutError("budget expired before Validate")

    rc, rp = _run_main(monkeypatch, tmp_path, eval_dir, invoke=boom)
    assert rp.is_file(), "result.json MUST exist even when workflow raised"
    out = json.loads(rp.read_text())
    assert out["status"] == "ok"
    assert out.get("recovered_from_disk") is True
    assert out["final_throughput_tok_s"] == pytest.approx(535.352)
    assert (eval_dir / "kernel_journey.json").is_file()


def test_emit_error_when_nothing_on_disk(monkeypatch, tmp_path):
    """No measured effect at all -> still MUST emit a parseable error file."""
    eval_dir = tmp_path / "e2e_empty"
    eval_dir.mkdir()

    def boom(prompt, t, ed):
        raise RuntimeError("crashed immediately")

    rc, rp = _run_main(monkeypatch, tmp_path, eval_dir, invoke=boom)
    assert rp.is_file(), "result.json MUST exist even with nothing to recover"
    out = json.loads(rp.read_text())
    assert out["status"] in ("error", "timeout")
    assert rc == 1


def test_emit_is_atomic_and_parseable(monkeypatch, tmp_path):
    """No .tmp residue; the emitted file always parses as JSON."""
    eval_dir = _make_eval_dir(tmp_path, with_validation=True)

    def ok_invoke(prompt, t, ed):
        return {"eval_dir": str(eval_dir), "throughput_speedup": 1.16,
                "final_throughput_tok_s": 535.352}

    rc, rp = _run_main(monkeypatch, tmp_path, eval_dir, invoke=ok_invoke)
    assert rp.is_file()
    json.loads(rp.read_text())  # parseable
    assert not (rp.parent / (rp.name + ".tmp")).exists(), "no .tmp residue"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
