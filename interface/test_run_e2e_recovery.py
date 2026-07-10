#!/usr/bin/env python3
"""Tests for run_e2e's guaranteed interface-file emission + intermediate-win
recovery.

CONTRACT under test: as long as GEAK produced ANY measured E2E effect on
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


def test_recover_intermediate_nested_schema(tmp_path):
    """The Kimi-K2.6 20260621T151617Z shape: a real accepted win whose numbers are
    NESTED under e2e/accepted_config (not flat). Must still recover (+14.74%)."""
    eval_dir = tmp_path / "e2e_nested"
    (eval_dir / "overlay" / "cand_int4_fused_moe_grouped_gemm").mkdir(parents=True)
    (eval_dir / "bench_e2e.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    ir = {
        "short_name": "int4_fused_moe_grouped_gemm",
        "winner_kind": "env", "isolated_speedup": 1.631, "pct_gpu_time": 50,
        "gate": "accepted", "output_parity": "pass",
        "e2e": {
            "ref_median_tok_s": 504.189, "cand_median_tok_s": 578.502,
            "ref_max_tok_s": 504.426, "cand_min_tok_s": 578.033,
            "delta_pct": 14.739, "non_overlapping": True,
            "ttft_ms_ref": 3294.168, "ttft_ms_cand": 5131.849,
            "tpot_ms_ref": 121.448, "tpot_ms_cand": 103.456,
        },
        "accepted_config": {
            "apply_env": "VLLM_TUNED_CONFIG_FOLDER=/x/config/moe_tuned",
            "apply_flags": "--max-num-batched-tokens 16384",
        },
    }
    (eval_dir / "overlay" / "cand_int4_fused_moe_grouped_gemm"
     / "integrate_result.json").write_text(json.dumps(ir), encoding="utf-8")
    wf = rx._recover_best_intermediate_win(eval_dir)
    assert wf is not None, "nested-schema accepted win must NOT be skipped"
    assert wf["baseline_throughput_tok_s"] == pytest.approx(504.189)
    assert wf["final_throughput_tok_s"] == pytest.approx(578.502)
    assert wf["throughput_speedup"] == pytest.approx(578.502 / 504.189)
    assert wf["accepted_config"]["flags"] == "--max-num-batched-tokens 16384"
    assert "VLLM_TUNED_CONFIG_FOLDER" in wf["accepted_config"]["env"]
    out = rx.normalize_result(_handoff(eval_dir), wf)
    assert out["status"] == "ok"
    assert out["result_source"] == "disk_intermediate_win"
    # Latency is carried from the candidate (accepted) A/B leg, not fabricated.
    assert out["ttft_ms"] == pytest.approx(5131.849)
    assert out["tpot_ms"] == pytest.approx(103.456)


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


# ── completed-but-no-gain recovery (the Kimi-K2.6 20260621 failure) ──────────
def _make_no_gain_eval_dir(tmp_path: Path) -> Path:
    """Mimic a run that COMPLETED but accepted nothing: a measured baseline,
    a REJECTED head (do-no-harm), validation benches that ran, an empty overlay,
    and NO director_e2e_validation.json. This is the exact shape that used to be
    misreported as workflow_parse_error."""
    eval_dir = tmp_path / "e2e_nogain"
    (eval_dir / "baseline").mkdir(parents=True)
    (eval_dir / "final").mkdir(parents=True)
    (eval_dir / "validation" / "final").mkdir(parents=True)
    (eval_dir / "overlay" / "cand_mla_decode_fwd").mkdir(parents=True)
    (eval_dir / "bench_e2e.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (eval_dir / "final" / "final_launch.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (eval_dir / "baseline" / "baseline_official.json").write_text(json.dumps({
        "baseline_throughput_tok_s": 604.8,
        "plateau_median_tok_s": 604.8,
        "server_flags": "--trust-remote-code --kv-cache-dtype fp8_e4m3",
        "server_env": "",
        "serving_config": {"backend": "vllm", "tp": 8},
    }), encoding="utf-8")
    (eval_dir / "baseline" / "bench_summary.json").write_text(json.dumps({
        "output_throughput_tok_s_median": 601.786, "ttft_ms_median": 1958.57,
        "tpot_ms_median": 103.976,
    }), encoding="utf-8")
    (eval_dir / "validation" / "final" / "bench_summary.json").write_text(json.dumps({
        "output_throughput_tok_s_median": 594.712, "ttft_ms_median": 1892.889,
        "tpot_ms_median": 105.142,
    }), encoding="utf-8")
    # A REJECTED head (do-no-harm) — must NOT be salvaged as a win.
    (eval_dir / "overlay" / "cand_mla_decode_fwd" / "integrate_result.json").write_text(
        json.dumps({"short_name": "mla_decode_fwd", "gate": "rejected",
                    "isolated_speedup": 13.1452, "output_parity": "not_measured"}),
        encoding="utf-8")
    return eval_dir


def test_recover_completed_no_gain_synthesizes_no_gain(tmp_path):
    eval_dir = _make_no_gain_eval_dir(tmp_path)
    wf = rx._recover_completed_no_gain(eval_dir)
    assert wf is not None
    assert wf["recovered_no_gain"] is True
    assert wf["baseline_throughput_tok_s"] == pytest.approx(604.8)
    # Empty overlay => served path unchanged => final == baseline, speedup 1.0.
    assert wf["final_throughput_tok_s"] == pytest.approx(604.8)
    assert wf["throughput_speedup"] == pytest.approx(1.0)
    assert wf["accepted_kernels"] == []


def test_recover_workflow_return_no_gain_not_parse_error(tmp_path):
    """The killer regression: a completed run with a REJECTED candidate and no
    director json must recover as no_gain — never None (workflow_parse_error)."""
    eval_dir = _make_no_gain_eval_dir(tmp_path)
    wf = rx._recover_workflow_return(eval_dir.parent)
    assert wf is not None, "completed no-gain run must NOT recover as None"
    assert wf.get("recovered_no_gain") is True
    out = rx.normalize_result(_handoff(eval_dir), wf)
    assert out["status"] == "no_gain"
    assert out["baseline_throughput_tok_s"] == pytest.approx(604.8)
    # ttft/tpot come from the same-session validation bench summary.
    assert out["ttft_ms"] == pytest.approx(1892.889)
    # final_launch_script falls back to the real on-disk bundle.
    assert out["final_launch_script"].endswith("final/final_launch.sh")


def test_no_baseline_still_errors(tmp_path):
    """No measured baseline at all => genuinely nothing => None (-> error)."""
    eval_dir = tmp_path / "e2e_bare"
    (eval_dir / "overlay").mkdir(parents=True)
    assert rx._recover_completed_no_gain(eval_dir) is None


def test_workflow_done_marker_ignores_final_launch(tmp_path):
    """final/final_launch.sh (Finalize, pre-Validate) must NOT count as done;
    only the post-Validate terminal markers (director_e2e_validation.json /
    workflow_return.json) do."""
    eval_dir = tmp_path / "e2e_marker"
    (eval_dir / "final").mkdir(parents=True)
    (eval_dir / "final" / "final_launch.sh").write_text("x", encoding="utf-8")
    assert rx._workflow_done_on_disk(str(eval_dir)) is False
    (eval_dir / "director_e2e_validation.json").write_text("{}", encoding="utf-8")
    assert rx._workflow_done_on_disk(str(eval_dir)) is True


def test_workflow_done_marker_accepts_canonical_return(tmp_path):
    """The workflow's canonical handoff (workflow_return.json) is its LAST act and
    is itself a terminal done-marker, even without director_e2e_validation.json."""
    eval_dir = tmp_path / "e2e_canon_marker"
    eval_dir.mkdir(parents=True)
    assert rx._workflow_done_on_disk(str(eval_dir)) is False
    (eval_dir / rx.WORKFLOW_RETURN_FILE).write_text(
        json.dumps({"schema_version": 1, "eval_dir": str(eval_dir)}), encoding="utf-8")
    assert rx._workflow_done_on_disk(str(eval_dir)) is True


# ── canonical-artifact contract (the perfect-cooperation handoff) ───────────

def test_recover_trusts_workflow_written_canonical(tmp_path):
    """A workflow-WRITTEN canonical workflow_return.json (schema_version, NO
    recovery markers) is the source of truth: returned verbatim, not re-derived."""
    eval_dir = _make_eval_dir(tmp_path, accepted=True, with_validation=False)
    canonical = {
        "schema_version": 1, "mode": "e2e", "eval_dir": str(eval_dir),
        "baseline_throughput_tok_s": 461.314, "final_throughput_tok_s": 600.0,
        "throughput_speedup": 1.30, "output_parity": "pass",
        "accepted_kernels": [{"short_name": "fused_moe_kernel_gptq_awq"}],
    }
    (eval_dir / rx.WORKFLOW_RETURN_FILE).write_text(json.dumps(canonical), encoding="utf-8")
    wf = rx._recover_workflow_return(eval_dir.parent)
    # Returned verbatim — NOT re-derived from the (lower, 535.352) intermediate.
    assert wf["final_throughput_tok_s"] == pytest.approx(600.0)
    assert not wf.get("recovered_intermediate")
    out = rx.normalize_result(_handoff(eval_dir), wf)
    assert out["result_source"] == "workflow_return"


def test_recover_redrives_our_own_recovered_persist(tmp_path):
    """A workflow_return.json WE previously wrote from disk recovery (carries a
    recovered_* flag) must NOT shadow fresh recovery — it is re-derived so later
    recovery improvements (e.g. newly-extracted latency) take effect."""
    eval_dir = _make_eval_dir(tmp_path, accepted=True, with_validation=False)
    stale = {
        "schema_version": 1, "eval_dir": str(eval_dir),
        "final_throughput_tok_s": 535.352, "throughput_speedup": 1.16,
        "recovered_intermediate": True,  # written by our OWN recovery
    }
    (eval_dir / rx.WORKFLOW_RETURN_FILE).write_text(json.dumps(stale), encoding="utf-8")
    wf = rx._recover_workflow_return(eval_dir.parent)
    # Re-derived fresh from the intermediate (the fixture carries apply_* config).
    assert wf.get("recovered_intermediate") is True
    assert wf["accepted_config"]["flags"] == "--max-num-batched-tokens 16384"


def test_normalize_reconciles_crashed_validate_with_accepted_win(tmp_path):
    """The Kimi-K2.6 20260625T130314Z bug: the workflow ACCEPTED a head (A/B
    +18.93%) but the final Validate bench CRASHED, so the live return carried
    final_throughput_tok_s=0 / throughput_speedup=0. result.json must NOT report
    no_gain — it reconciles from the on-disk accepted integrate A/B."""
    eval_dir = _make_eval_dir(tmp_path, accepted=True, with_validation=False)
    # Live return: a real accepted head, but degenerate final/speedup (crash).
    wf = {
        "eval_dir": str(eval_dir),
        "baseline_throughput_tok_s": 255.049,
        "final_throughput_tok_s": 0,
        "throughput_speedup": 0,
        "validation_status": "flagged_no_number_used_carried_ab",
        "accepted_heads": [{
            "short_name": "fused_moe_kernel_gptq_awq",
            "op_kind": "gemm", "backend": "triton", "kind": "env",
            "e2e_delta_pct": 16.049, "isolated": 1.5902,
        }],
        "accepted_kernels": [],
    }
    out = rx.normalize_result(_handoff(eval_dir), wf)
    assert out["status"] == "ok", "an accepted same-session win must never read as no_gain"
    assert out["throughput_speedup"] == pytest.approx(535.352 / 461.314)
    assert out["final_throughput_tok_s"] == pytest.approx(535.352)
    # Provenance is honest: the number came from the disk intermediate A/B.
    assert out["result_source"] == "disk_intermediate_win"
    # The accepted head metadata from the live return is preserved.
    assert out["accepted_heads"][0]["short_name"] == "fused_moe_kernel_gptq_awq"


def test_normalize_does_not_reconcile_genuine_no_gain(tmp_path):
    """A return that accepted NOTHING (empty heads/kernels) with speedup 1.0 is a
    legitimate no_gain — the reconciliation guard must leave it untouched."""
    eval_dir = _make_eval_dir(tmp_path, accepted=True, with_validation=False)
    wf = {
        "eval_dir": str(eval_dir),
        "baseline_throughput_tok_s": 255.049,
        "final_throughput_tok_s": 255.049,
        "throughput_speedup": 1.0,
        "accepted_heads": [],
        "accepted_kernels": [],
    }
    out = rx.normalize_result(_handoff(eval_dir), wf)
    assert out["status"] == "no_gain"
    assert out["result_source"] == "workflow_return"


def test_result_source_no_gain(tmp_path):
    eval_dir = _make_no_gain_eval_dir(tmp_path)
    wf = rx._recover_workflow_return(eval_dir.parent)
    out = rx.normalize_result(_handoff(eval_dir), wf)
    assert out["result_source"] == "disk_no_gain_synthesis"


def test_result_source_director_validation(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, accepted=True, with_validation=True)
    wf = rx._recover_workflow_return(eval_dir.parent)
    out = rx.normalize_result(_handoff(eval_dir), wf)
    assert out["result_source"] == "disk_director_validation"


def test_result_source_live_workflow_return(tmp_path):
    """A live (scraped) workflow return — no recovery flags — is the canonical
    source and stamps result_source=workflow_return."""
    eval_dir = _make_eval_dir(tmp_path, with_validation=True)
    wf = {"eval_dir": str(eval_dir), "throughput_speedup": 1.16,
          "final_throughput_tok_s": 535.352, "baseline_throughput_tok_s": 461.314}
    out = rx.normalize_result(_handoff(eval_dir), wf)
    assert out["result_source"] == "workflow_return"


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
    """No measured effect at all -> still MUST emit a parseable error file AND an
    honest (kernels-empty) kernel_journey.json that carries the failure status."""
    eval_dir = tmp_path / "e2e_empty"
    eval_dir.mkdir()

    def boom(prompt, t, ed):
        raise RuntimeError("crashed immediately")

    rc, rp = _run_main(monkeypatch, tmp_path, eval_dir, invoke=boom)
    assert rp.is_file(), "result.json MUST exist even with nothing to recover"
    out = json.loads(rp.read_text())
    assert out["status"] in ("error", "timeout")
    assert rc == 1
    # kernel_journey.json is a GUARANTEED file too — present even on a pure error,
    # with empty kernels (never fabricated) and the run status recorded.
    kj = eval_dir / "kernel_journey.json"
    assert kj.is_file(), "kernel_journey.json MUST exist even on a pure error"
    journey = json.loads(kj.read_text())
    assert journey["kernels"] == []
    assert journey["status"] in ("error", "timeout")
    assert out["kernel_journey_path"] == str(kj)


# ── kernel_journey guaranteed-emit + failure surfacing ──────────────────────

def test_write_kernel_journey_empty_on_no_wf(tmp_path):
    """wf is None -> a valid empty-kernels journey is still written (not dropped)."""
    eval_dir = tmp_path / "e2e_kj_none"
    eval_dir.mkdir()
    out = {"status": "error", "error_class": "runner_error", "error": "boom"}
    path = rx._write_kernel_journey(eval_dir, None, out)
    journey = json.loads(Path(path).read_text())
    assert journey["kernels"] == []
    assert journey["discovery_runs"] == []
    assert journey["status"] == "error"
    assert journey["versions"]["geak"]["tool"] == "geak"


def test_write_kernel_journey_falls_back_when_build_raises(tmp_path, monkeypatch):
    """If the FULL build raises, we degrade to a valid empty journey rather than
    dropping the file (and never fabricate kernels)."""
    eval_dir = tmp_path / "e2e_kj_buildfail"
    eval_dir.mkdir()
    monkeypatch.setattr(rx, "build_kernel_journey",
                        lambda wf, n: (_ for _ in ()).throw(ValueError("bad wf")))
    out = {"status": "ok", "eval_dir": str(eval_dir)}
    path = rx._write_kernel_journey(eval_dir, {"eval_dir": str(eval_dir)}, out)
    assert Path(path).is_file()
    assert json.loads(Path(path).read_text())["kernels"] == []


def test_write_kernel_journey_is_atomic(tmp_path):
    eval_dir = tmp_path / "e2e_kj_atomic"
    eval_dir.mkdir()
    rx._write_kernel_journey(eval_dir, None, {"status": "no_gain"})
    assert not (eval_dir / "kernel_journey.json.tmp").exists(), "no .tmp residue"


def test_emit_timeout_still_writes_journey(monkeypatch, tmp_path):
    """A TIMEOUT (the SIGTERM self-stop path raises TimeoutError) with nothing
    recoverable on disk MUST still leave result.json (status=timeout) AND an
    honest empty kernel_journey.json — a timeout must never drop the journey."""
    eval_dir = tmp_path / "e2e_to"
    eval_dir.mkdir()

    def boom(prompt, t, ed):
        raise TimeoutError("signal 15: self-stop to flush interface files")

    rc, rp = _run_main(monkeypatch, tmp_path, eval_dir, invoke=boom)
    out = json.loads(rp.read_text())
    assert out["status"] == "timeout"
    kj = eval_dir / "kernel_journey.json"
    assert kj.is_file()
    journey = json.loads(kj.read_text())
    assert journey["kernels"] == [] and journey["status"] == "timeout"


def test_emit_surfaces_journey_write_failure(monkeypatch, tmp_path):
    """A journey WRITE failure must be surfaced into result.json (not silently
    dropped) — result.json itself MUST still be emitted."""
    eval_dir = _make_eval_dir(tmp_path, with_validation=True)
    monkeypatch.setattr(rx, "_write_kernel_journey",
                        lambda ed, wf, n: (_ for _ in ()).throw(OSError("disk full")))

    def ok_invoke(prompt, t, ed):
        return {"eval_dir": str(eval_dir), "throughput_speedup": 1.16,
                "final_throughput_tok_s": 535.352,
                "baseline_throughput_tok_s": 461.314}

    rc, rp = _run_main(monkeypatch, tmp_path, eval_dir, invoke=ok_invoke)
    assert rp.is_file(), "result.json MUST exist even when journey write failed"
    out = json.loads(rp.read_text())
    assert out["status"] == "ok"
    assert "kernel_journey_error" in out
    assert "disk full" in out["kernel_journey_error"]


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


# ── kernel_journey reconstruction (discovery + config-only win) ─────────────

def test_journey_recovers_discovery_and_config_win(tmp_path):
    """A CONFIG-only win on a discovered hot kernel must NOT yield an empty journey:
    discovery_runs come from profile_topN.json, and the integrated win is a kernels[]
    entry whose tuned flags land in e2e.extra_server_args (not dropped)."""
    eval_dir = tmp_path / "e2e_journey"
    (eval_dir / "profile" / "round_0").mkdir(parents=True)
    (eval_dir / "overlay" / "cand_fused_moe_kernel_gptq_awq").mkdir(parents=True)
    (eval_dir / "overlay" / "cand_fwd_grouped_kernel_stage1").mkdir(parents=True)  # incomplete A/B
    (eval_dir / "profile" / "round_0" / "profile_topN.json").write_text(json.dumps({
        "source": "rocprofv3", "num_distinct_kernels": 213,
        "top_kernels": [
            {"rank": 1, "short_name": "fused_moe_kernel_gptq_awq",
             "name": "fused_moe_kernel_gptq_awq", "pct_gpu_time": 50.64,
             "total_ms": 161920.7, "classification": "triton", "editable": True},
            {"rank": 2, "short_name": "_fwd_grouped_kernel_stage1",
             "name": "_fwd_grouped_kernel_stage1", "pct_gpu_time": 15.55,
             "total_ms": 49709.0, "classification": "triton", "editable": True},
            {"rank": 3, "short_name": "cross_device_reduce_2stage",
             "name": "cross_device_reduce_2stage", "pct_gpu_time": 6.96,
             "classification": "comm", "editable": False},
        ],
    }), encoding="utf-8")
    (eval_dir / "overlay" / "cand_fused_moe_kernel_gptq_awq" / "integrate_result.json").write_text(
        json.dumps({
            "short_name": "fused_moe_kernel_gptq_awq", "winner_kind": "env",
            "apply_flags": "--max-num-batched-tokens 16384",
            "apply_env": "VLLM_TUNED_CONFIG_FOLDER=/x/moe",
            "tuned_config_file": "/x/moe/E=384.json",
            "isolated_speedup": 1.5902, "pct_gpu_time": 50.64,
            "ref_med": 461.314, "cand_med": 535.352,
            "e2e_throughput_tok_s": 535.352,
            "e2e_delta_pct": 16.049, "output_parity": "pass", "gate": "accepted",
        }), encoding="utf-8")

    wf = rx._recover_best_intermediate_win(eval_dir)
    out = rx.normalize_result(_handoff(eval_dir), wf)
    j = rx.build_kernel_journey(wf, out)

    # discovery_runs from the profiler, with selection marked.
    assert len(j["discovery_runs"]) == 1
    disc = j["discovery_runs"][0]
    assert disc["source"] == "bypass" and disc["hot_kernel_count"] == 3
    hot = {h["kernel_id"]: h for h in disc["hot_kernels"]}
    assert hot["fused_moe_kernel_gptq_awq"]["selected_for_optimization"] is True
    # discovery emits the CANONICAL id (leading underscore stripped) so it folds
    # with the overlay-derived kernels[] entry; the raw spelling stays in ``name``.
    assert hot["fwd_grouped_kernel_stage1"]["selected_for_optimization"] is True
    assert hot["fwd_grouped_kernel_stage1"]["name"] == "_fwd_grouped_kernel_stage1"
    assert hot["cross_device_reduce_2stage"]["selected_for_optimization"] is False

    by_id = {k["kernel_id"]: k for k in j["kernels"]}
    # The config win is recorded (NOT dropped), flags in extra_server_args.
    win = by_id["fused_moe_kernel_gptq_awq"]
    assert win["e2e"]["integrated"] is True and win["e2e"]["decision"] == "KEEP"
    assert win["e2e"]["extra_server_args"] == "--max-num-batched-tokens 16384"
    assert win["e2e"]["target_file"] == "/x/moe/E=384.json"
    assert win["backend_result"]["attempts"][0]["correctness_passed"] is True
    # The incomplete A/B is dispatch-only: no fabricated KEEP/FAIL e2e.
    inc = by_id["fwd_grouped_kernel_stage1"]
    assert "e2e" not in inc
    assert inc["dispatch"]["task_group"] == "ab_incomplete"
    assert inc["backend_result"]["attempts"] == []
    # name is UNIFIED with discovery: the kernels[] entry adopts the profiler's
    # real symbol (underscore intact), not the underscore-stripped overlay dir name.
    assert inc["name"] == "_fwd_grouped_kernel_stage1"
    assert inc["name"] == hot["fwd_grouped_kernel_stage1"]["name"]

    # Folding contract: every kernels[] id has a matching discovery hot-kernel id
    # (and vice-versa for the optimized ones) so the orchestrator's assembler
    # produces ONE entry per kernel — never a split discovered/adopted pair.
    assert set(by_id) <= set(hot)


def test_journey_kernel_id_consistent_across_substreams(tmp_path):
    """Regression for the discovery-vs-kernels[] kernel_id split: a profiler hot
    kernel with a leading underscore (``_fwd_grouped_kernel_stage1``) and its
    optimization overlay (``cand_fwd_grouped_kernel_stage1``) MUST surface under
    ONE identical kernel_id in both substreams, or the orchestrator's assembler
    folds them into two journey entries for one kernel."""
    eval_dir = tmp_path / "e2e_kid"
    (eval_dir / "profile" / "round_0").mkdir(parents=True)
    (eval_dir / "overlay" / "cand_fwd_grouped_kernel_stage1").mkdir(parents=True)
    (eval_dir / "profile" / "round_0" / "profile_topN.json").write_text(json.dumps({
        "source": "rocprofv3", "num_distinct_kernels": 7,
        "top_kernels": [
            {"rank": 1, "short_name": "_fwd_grouped_kernel_stage1",
             "name": "_fwd_grouped_kernel_stage1", "pct_gpu_time": 22.0,
             "total_ms": 1234.0, "classification": "triton", "editable": True},
        ],
    }), encoding="utf-8")
    (eval_dir / "overlay" / "cand_fwd_grouped_kernel_stage1"
     / "integrate_result.json").write_text(json.dumps({
        "short_name": "fwd_grouped_kernel_stage1", "winner_kind": "authored",
        "isolated_speedup": 1.4, "e2e_delta_pct": 8.0, "output_parity": "pass",
        "gate": "accepted", "final_patch": "/x/p.diff", "target_callable": "mod.fn",
    }), encoding="utf-8")

    j = rx.build_kernel_journey({"eval_dir": str(eval_dir)}, {"eval_dir": str(eval_dir)})
    disc = j["discovery_runs"][0]["hot_kernels"]
    disc_ids = {h["kernel_id"] for h in disc}
    kernel_ids = {k["kernel_id"] for k in j["kernels"]}
    assert disc_ids == {"fwd_grouped_kernel_stage1"}
    assert kernel_ids == {"fwd_grouped_kernel_stage1"}
    # The single fold key is shared by BOTH substreams (no underscore variant).
    assert disc_ids == kernel_ids
    assert all(not kid.startswith("_") for kid in disc_ids | kernel_ids)
    # name is ALSO unified: both substreams carry the profiler's real symbol
    # (leading underscore intact), while kernel_id stays the stripped fold key.
    disc_names = {h["name"] for h in disc}
    kernel_names = {k["name"] for k in j["kernels"]}
    assert disc_names == {"_fwd_grouped_kernel_stage1"}
    assert kernel_names == {"_fwd_grouped_kernel_stage1"}


def test_journey_rejected_overlay_is_reverted(tmp_path):
    """A rejected (do-no-harm) overlay is recorded as REVERT/REJECTED, not KEEP."""
    eval_dir = tmp_path / "e2e_journey_rej"
    (eval_dir / "overlay" / "cand_mla_decode_fwd").mkdir(parents=True)
    (eval_dir / "overlay" / "cand_mla_decode_fwd" / "integrate_result.json").write_text(
        json.dumps({"short_name": "mla_decode_fwd", "gate": "rejected",
                    "isolated_speedup": 13.1, "e2e_delta_pct": -2.0,
                    "output_parity": "pass", "winner_kind": "authored"}),
        encoding="utf-8")
    j = rx.build_kernel_journey({"eval_dir": str(eval_dir)}, {"eval_dir": str(eval_dir)})
    k = j["kernels"][0]
    assert k["e2e"]["integrated"] is False and k["e2e"]["decision"] == "REJECTED"
    assert k["backend_result"]["attempts"][0]["decision"] == "REVERT"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
