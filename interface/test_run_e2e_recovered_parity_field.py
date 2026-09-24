#!/usr/bin/env python3
"""A recovered tuning result must not report its op-level verdict as e2e output parity.

WHY: GEAK carries two different correctness bars that were both being written to the SAME field.

  * the integrate lane's `output_parity` — e2e byte-exact greedy parity against a proven-deterministic
    no-overlay baseline, per prompt, at temp=0;
  * the tuning lane's `correctness_gate` — a relative OP-LEVEL numeric check on the tuned op alone.

`_tuning_recovery_return` used to copy the second into the first. The op check is a real check, but it
is not the one the field's name promises: no baseline determinism control, no greedy decode, no
per-prompt comparison. A recovered +8.06% run therefore shipped `output_parity: "pass"` with no
validation artifacts behind it, and `e2e_store` marked it decided on the strength of that string.

The consequence worth spelling out: because recovered runs were the ones supplying "pass", the
byte-exact bar had never actually been exercised on a kernel/BLAS swap on any platform. When it finally
ran on gfx1151 and rejected a genuine +14.38% win, it looked like a platform quirk. It was this.

Run: python3 -m pytest GEAK/interface/test_run_e2e_recovered_parity_field.py -v
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("run_e2e_recovered_parity", _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load()


def _scaffold(tmp_path: Path) -> None:
    """The two files `_write_tuning_recovery_launcher` refuses to proceed without."""
    (tmp_path / "tuning" / "deploy").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tuning" / "deploy" / "deploy.sh").write_text("#!/usr/bin/env bash\n")
    (tmp_path / "bench_e2e.sh").write_text("#!/usr/bin/env bash\n")
    (tmp_path / "tuning" / "bf160.csv").write_text("x\n")


def _ret(tmp_path: Path, **tuning_extra):
    tuning = {
        "enabled": True,
        "ran": True,
        "gate": "accepted",
        "correctness_gate": "pass",
        "apply_env": "PYTORCH_TUNABLEOP_ENABLED=1",
        "apply_flags": "",
        "tuning_artifact": str(tmp_path / "tuning" / "bf160.csv"),
    }
    tuning.update(tuning_extra)
    _scaffold(tmp_path)
    return rx._tuning_recovery_return(
        tmp_path, tuning, pre=100.0, post=108.06, source="disk_provisional"
    )


def test_op_level_verdict_is_not_reported_as_output_parity(tmp_path):
    out = _ret(tmp_path)
    assert out is not None
    # The op-level verdict survives -- under its own name.
    assert out["correctness_gate"] == "pass"
    # ...and does NOT masquerade as an e2e parity result.
    assert out["output_parity"] != "pass"
    assert not out["output_parity"]


def test_parity_kind_labels_the_bar_that_actually_ran(tmp_path):
    assert _ret(tmp_path)["parity_kind"] == "op_tolerance"
    assert _ret(tmp_path, correctness_gate="fail")["parity_kind"] == "none"
    assert _ret(tmp_path, correctness_gate="unknown")["parity_kind"] == "none"


def test_a_failing_op_gate_is_still_reported_verbatim(tmp_path):
    out = _ret(tmp_path, correctness_gate="fail")
    assert out["correctness_gate"] == "fail"
    assert not out["output_parity"]


def test_missing_op_gate_stays_unknown_not_silently_clean(tmp_path):
    tuning = {"enabled": True, "ran": True, "gate": "accepted",
              "apply_env": "X=1", "tuning_artifact": str(tmp_path / "tuning" / "bf160.csv")}
    _scaffold(tmp_path)
    out = rx._tuning_recovery_return(tmp_path, tuning, pre=100.0, post=108.0, source="disk")
    assert out["correctness_gate"] == "unknown"
    assert out["parity_kind"] == "none"


def test_downstream_parity_readers_are_unchanged(tmp_path):
    """The `output_parity or correctness_gate` fallback readers must see the same string as before.

    run_e2e.py resolves the displayed parity as `summary.get("output_parity") or
    summary.get("correctness_gate") or ""`. An EMPTY output_parity (not "n/a", not "none") is what
    keeps those two call sites byte-identical to their pre-change output.
    """
    out = _ret(tmp_path)
    resolved = out.get("output_parity") or out.get("correctness_gate") or ""
    assert resolved == "pass"


def test_no_other_top_level_key_changed(tmp_path):
    """De-conflation is additive: it renames nothing and drops nothing."""
    out = _ret(tmp_path)
    for k in ("eval_dir", "throughput_speedup", "baseline_throughput_tok_s",
              "final_throughput_tok_s", "final_overlay", "final_launch_script",
              "accepted_config", "accepted_kernels", "accepted_heads", "tuning_skillset"):
        assert k in out, k
    assert out["accepted_config"]["env"] == "PYTORCH_TUNABLEOP_ENABLED=1"
    assert out["throughput_speedup"] == 108.06 / 100.0
