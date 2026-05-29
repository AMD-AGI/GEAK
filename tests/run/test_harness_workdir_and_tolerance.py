"""Pins for the two regressions found in hip_act_and_mul_20260528_0919.

Fix B — ``_build_env`` injects ``GEAK_WORK_DIR``
------------------------------------------------
The harness pipeline routes worktree paths through ``GEAK_WORK_DIR``
(see ``hip_act_and_mul_20260528_0919/_preprocess_subagent_worktree/
sgl-kernel/_geak_harness/act_and_mul_jit.py:30-35``).  Two
``_build_env`` implementations build the subprocess env that runs the
harness:

  * ``minisweagent.run.preprocess.run_harness._build_env``
  * ``minisweagent.run.preprocess_v3.baseline._build_env``

Before this change, neither set ``GEAK_WORK_DIR``; harnesses with an
``os.environ.setdefault("GEAK_WORK_DIR", "/original/repo")`` fallback
would silently evaluate the un-patched repo instead of the worktree.
The pins below assert the env var is set whenever a worktree path is
supplied, and is omitted when one isn't.

Fix D — tolerance hard cap in ``validate_harness``
--------------------------------------------------
The HarnessBuilder's universal-contract validator now rejects any
harness whose correctness tolerances exceed
``GEAK_HARNESS_MAX_TOLERANCE`` (default 2e-2).  This catches the
``hip_act_and_mul_20260528`` failure mode where the LLM wrote
``assert_close(..., atol=5e-2, rtol=5e-2)`` and let broken kernels
silently pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from minisweagent.kernel_languages.contract import (
    ContractViolation,
    _scan_tolerances,
    validate_harness,
)


# ---------------------------------------------------------------------------
# Fix B — GEAK_WORK_DIR injection
# ---------------------------------------------------------------------------


class TestRunHarnessBuildEnv:
    """``run/preprocess/run_harness._build_env`` must set ``GEAK_WORK_DIR``."""

    def test_workdir_set_when_repo_root_present(self):
        from minisweagent.run.preprocess.run_harness import _build_env

        env = _build_env("/path/to/worktree", gpu_id=0, env_overrides=None)
        assert env["GEAK_WORK_DIR"] == "/path/to/worktree"
        # PYTHONPATH must include the worktree (pre-existing behaviour) —
        # pinning this so a future cleanup doesn't accidentally drop one
        # without the other.  The compile-mode bootstrap dir is prepended
        # AHEAD of the worktree (it only ships sitecustomize.py and shadows
        # no kernel package), so we assert membership rather than position.
        entries = env["PYTHONPATH"].split(os.pathsep)
        assert "/path/to/worktree" in entries
        # The bootstrap dir, when present, must sort before the worktree so
        # `site` auto-loads sitecustomize before any kernel import.
        if any(e.endswith("_compile_bootstrap") for e in entries):
            boot_idx = next(i for i, e in enumerate(entries) if e.endswith("_compile_bootstrap"))
            assert boot_idx < entries.index("/path/to/worktree")

    def test_workdir_unset_when_no_repo_root(self, monkeypatch):
        from minisweagent.run.preprocess import run_harness

        # Strip any inherited GEAK_WORK_DIR from the process env so we
        # can prove _build_env doesn't synthesize one from thin air.
        monkeypatch.delenv("GEAK_WORK_DIR", raising=False)
        env = run_harness._build_env(None, gpu_id=0, env_overrides=None)
        assert "GEAK_WORK_DIR" not in env

    def test_env_overrides_can_replace_workdir(self):
        from minisweagent.run.preprocess.run_harness import _build_env

        env = _build_env(
            "/path/to/worktree",
            gpu_id=0,
            env_overrides={"GEAK_WORK_DIR": "/explicit/override"},
        )
        # Explicit overrides win over the worktree-derived default.
        assert env["GEAK_WORK_DIR"] == "/explicit/override"


class TestBaselineV3BuildEnv:
    """``preprocess_v3.baseline._build_env`` must set ``GEAK_WORK_DIR``."""

    def test_workdir_set_when_work_dir_present(self):
        from minisweagent.run.preprocess_v3.baseline import _build_env

        env = _build_env(Path("/path/to/worktree"), gpu_id=0)
        assert env["GEAK_WORK_DIR"] == "/path/to/worktree"
        assert env["PYTHONPATH"].startswith("/path/to/worktree")

    def test_workdir_unset_when_no_work_dir(self, monkeypatch):
        from minisweagent.run.preprocess_v3 import baseline

        monkeypatch.delenv("GEAK_WORK_DIR", raising=False)
        env = baseline._build_env(None, gpu_id=0)
        assert "GEAK_WORK_DIR" not in env

    def test_extra_overrides_workdir(self):
        from minisweagent.run.preprocess_v3.baseline import _build_env

        env = _build_env(
            Path("/path/to/worktree"),
            gpu_id=0,
            extra={"GEAK_WORK_DIR": "/explicit/override"},
        )
        assert env["GEAK_WORK_DIR"] == "/explicit/override"


# ---------------------------------------------------------------------------
# Fix D — tolerance hard cap
# ---------------------------------------------------------------------------


# Minimal "universal-contract-complete" prelude that we splice
# tolerance literals into; each test only varies the assert_close call.
_HARNESS_PREAMBLE = """\
#!/usr/bin/env python3
import argparse, sys
p = argparse.ArgumentParser()
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--correctness", action="store_true")
g.add_argument("--benchmark", action="store_true")
g.add_argument("--full-benchmark", action="store_true")
g.add_argument("--profile", action="store_true")
args = p.parse_args()
import torch
x = torch.randn(8)
print("GEAK_RESULT_LATENCY_MS=1.0")
print("GEAK_RESULT_SPEEDUP=1.0")
"""


def _write_harness(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "harness.py"
    p.write_text(_HARNESS_PREAMBLE + body, encoding="utf-8")
    return p


class TestScanTolerances:
    def test_finds_atol_and_rtol_with_lineno(self, tmp_path: Path):
        p = _write_harness(tmp_path, "torch.testing.assert_close(x, x, atol=5e-2, rtol=1e-3)\n")
        findings = _scan_tolerances(p.read_text())
        kinds = {k: v for k, v, _ in findings}
        assert kinds == {"atol": 5e-2, "rtol": 1e-3}
        # Linenos are 1-based.
        assert all(ln > 0 for _, _, ln in findings)

    def test_supports_decimal_and_scientific_notation(self, tmp_path: Path):
        p = _write_harness(tmp_path,
            "torch.testing.assert_close(x, x, atol=0.05, rtol=.001)\n"
            "torch.allclose(x, x, atol=1e-4, rtol=1E-4)\n")
        values = sorted(v for _, v, _ in _scan_tolerances(p.read_text()))
        assert values == pytest.approx([1e-4, 1e-4, 1e-3, 0.05])

    def test_no_tolerance_literals(self, tmp_path: Path):
        # A harness with no assert_close / allclose at all is fine.
        p = _write_harness(tmp_path, "")
        assert _scan_tolerances(p.read_text()) == []


class TestValidateHarnessTolerance:
    def test_rejects_5e_2(self, tmp_path: Path):
        # Exactly the literal from the buggy hip_act_and_mul harness.
        p = _write_harness(tmp_path,
            "torch.testing.assert_close(x, x, atol=5e-2, rtol=5e-2)\n")
        with pytest.raises(ContractViolation) as ei:
            validate_harness(p)
        msg = str(ei.value)
        assert "tolerance" in msg.lower()
        assert "atol=0.05" in msg
        assert "rtol=0.05" in msg
        # Error message must give the LLM a concrete recovery hint.
        assert "fp16" in msg or "bf16" in msg

    def test_rejects_when_only_one_exceeds_cap(self, tmp_path: Path):
        # Tight atol, loose rtol — still rejected.
        p = _write_harness(tmp_path,
            "torch.testing.assert_close(x, x, atol=1e-4, rtol=0.1)\n")
        with pytest.raises(ContractViolation) as ei:
            validate_harness(p)
        assert "rtol=0.1" in str(ei.value)
        assert "atol=" not in str(ei.value) or "atol=0.1" not in str(ei.value)

    def test_reports_every_offender_not_just_the_first(self, tmp_path: Path):
        # If the harness has both a tight default and a wide override,
        # both are reported so the retry prompt can fix them in one go.
        p = _write_harness(tmp_path,
            "torch.testing.assert_close(x, x, atol=1e-4, rtol=1e-4)\n"
            "torch.testing.assert_close(x, x, atol=1.0, rtol=1.0)\n")
        with pytest.raises(ContractViolation) as ei:
            validate_harness(p)
        # Both offenders mentioned, NEITHER of the tight ones is.
        msg = str(ei.value)
        assert msg.count("atol=1") >= 1
        assert msg.count("rtol=1") >= 1
        assert "atol=0.0001" not in msg
        assert "rtol=0.0001" not in msg

    def test_accepts_tight_tolerances(self, tmp_path: Path):
        p = _write_harness(tmp_path,
            "torch.testing.assert_close(x, x, atol=1e-3, rtol=1e-3)\n")
        validate_harness(p)  # must not raise

    def test_accepts_at_the_cap(self, tmp_path: Path):
        # The hard cap is inclusive on the safe side: 2e-2 itself
        # passes (we reject ``> 2e-2``).
        p = _write_harness(tmp_path,
            "torch.testing.assert_close(x, x, atol=2e-2, rtol=2e-2)\n")
        validate_harness(p)  # must not raise

    def test_env_var_override(self, tmp_path: Path, monkeypatch):
        # GEAK_HARNESS_MAX_TOLERANCE is the per-run escape hatch for
        # fp8 / accumulation-heavy kernels that genuinely need wider
        # tolerances.  Must be honoured.
        monkeypatch.setenv("GEAK_HARNESS_MAX_TOLERANCE", "1e-1")
        # Reload the contract module so it picks up the new env value.
        import importlib

        import minisweagent.kernel_languages.contract as contract_mod
        importlib.reload(contract_mod)
        try:
            p = _write_harness(tmp_path,
                "torch.testing.assert_close(x, x, atol=5e-2, rtol=5e-2)\n")
            # 5e-2 is now under the 1e-1 cap → must pass.
            contract_mod.validate_harness(p)
        finally:
            # Restore default for sibling tests.
            monkeypatch.delenv("GEAK_HARNESS_MAX_TOLERANCE", raising=False)
            importlib.reload(contract_mod)


# ---------------------------------------------------------------------------
# Fix — worktree-bypass gate: hardcoded absolute sys.path.insert
# ---------------------------------------------------------------------------
#
# Pins rotary_embedding_kernel_202605290819: the harness did
# ``sys.path.insert(0, "/sgl-workspace/sglang/python")`` which shadowed the
# GEAK worktree, so every optimization round imported the baseline kernel
# and reported ~1.00x.  Both harness validators must reject this.


from minisweagent.kernel_languages.contract import find_hardcoded_syspath_inserts


class TestFindHardcodedSyspathInserts:
    def test_detects_absolute_literal_insert_with_lineno(self):
        text = (
            "import sys\n"
            'sys.path.insert(0, "/sgl-workspace/sglang/python")\n'
        )
        findings = find_hardcoded_syspath_inserts(text)
        assert findings == [(2, "/sgl-workspace/sglang/python")]

    def test_detects_single_quotes_and_nonzero_index(self):
        text = "sys.path.insert(1, '/opt/baseline/python')\n"
        assert find_hardcoded_syspath_inserts(text) == [(1, "/opt/baseline/python")]

    def test_ignores_env_derived_inserts(self):
        # The blessed pattern: derive from GEAK_WORK_DIR, not a literal.
        text = (
            "import os, sys\n"
            'sys.path.insert(0, os.environ["GEAK_WORK_DIR"])\n'
            "sys.path.insert(0, os.path.dirname(__file__))\n"
        )
        assert find_hardcoded_syspath_inserts(text) == []

    def test_ignores_relative_literal(self):
        # A relative path literal is not the baseline-pinning anti-pattern
        # (and is unusual); only POSIX-absolute literals are flagged.
        text = 'sys.path.insert(0, "python")\n'
        assert find_hardcoded_syspath_inserts(text) == []


class TestValidateHarnessSyspathContract:
    """contract.validate_harness must raise on hardcoded inserts (fail-loud
    path consumed by HarnessBuilder to trigger regeneration)."""

    def test_rejects_hardcoded_baseline_insert(self, tmp_path: Path):
        p = _write_harness(
            tmp_path,
            'import sys\nsys.path.insert(0, "/sgl-workspace/sglang/python")\n',
        )
        with pytest.raises(ContractViolation) as ei:
            validate_harness(p)
        msg = str(ei.value)
        assert "/sgl-workspace/sglang/python" in msg
        assert "GEAK_WORK_DIR" in msg  # actionable recovery hint
        assert "1.00x" in msg or "baseline-vs-baseline" in msg

    def test_accepts_env_derived_insert(self, tmp_path: Path):
        p = _write_harness(
            tmp_path,
            'import os, sys\nsys.path.insert(0, os.environ["GEAK_WORK_DIR"])\n',
        )
        validate_harness(p)  # must not raise


class TestHarnessUtilsValidateSyspath:
    """harness_utils.validate_harness must also reject hardcoded inserts
    (path consumed by phases + the Path-A short-circuit to drop the harness)."""

    def test_returns_invalid_for_hardcoded_insert(self, tmp_path: Path):
        from minisweagent.run.preprocess.harness_utils import (
            validate_harness as static_validate,
        )

        p = _write_harness(
            tmp_path,
            'import sys\nsys.path.insert(0, "/sgl-workspace/sglang/python")\n',
        )
        valid, messages = static_validate(str(p))
        assert valid is False
        assert any("/sgl-workspace/sglang/python" in m for m in messages)

    def test_commented_insert_does_not_trip(self, tmp_path: Path):
        # The static validator strips comments first, so a documented
        # example in a comment must NOT fail validation.
        from minisweagent.run.preprocess.harness_utils import (
            validate_harness as static_validate,
        )

        p = _write_harness(
            tmp_path,
            '# do NOT do: sys.path.insert(0, "/sgl-workspace/sglang/python")\n',
        )
        valid, _ = static_validate(str(p))
        assert valid is True
