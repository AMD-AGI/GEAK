"""Translation-run strict-correctness gating in postprocess evaluation.

A pytorch->flydsl (or any) *translation* run sets ``GEAK_TRANSLATION_RUN=1``.
On such runs the op-aware scaled-tolerance translation harness is the
correctness judge, so the stricter fixed-tolerance COMMANDMENT CORRECTNESS
gate must be skipped -- while the benchmark phase still runs.

These tests pin that behavior deterministically (no GPU / no orchestrator):
  - ``run_correctness_and_benchmark`` records ``correctness={skipped:True}``
    and runs *no* CORRECTNESS subprocess when the flag is set;
  - it still enforces the gate (builds + runs SETUP+CORRECTNESS) when unset;
  - ``preflight_commandment_contract`` smoke-tests SETUP only when the flag is
    set, SETUP+CORRECTNESS otherwise.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _setup_path():
    import sys

    repo = Path(__file__).resolve().parent.parent.parent
    if str(repo / "src") not in sys.path:
        sys.path.insert(0, str(repo / "src"))


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["bash"], returncode=returncode, stdout=stdout, stderr=stderr)


_COMMANDMENT = """## SETUP
```bash
echo setup
```

## CORRECTNESS
```bash
python harness.py --correctness
```

## FULL_BENCHMARK
```bash
python harness.py --full-benchmark
```
"""


def _write_commandment(tmp_path: Path) -> Path:
    p = tmp_path / "COMMANDMENT.md"
    p.write_text(_COMMANDMENT)
    return p


# ---------------------------------------------------------------------------
# run_correctness_and_benchmark
# ---------------------------------------------------------------------------
class TestTranslationGateSkip:
    def test_translation_run_skips_strict_correctness_and_runs_no_subprocess(self, tmp_path, monkeypatch):
        from minisweagent.run.postprocess import evaluation

        monkeypatch.setenv("GEAK_TRANSLATION_RUN", "1")
        commandment = _write_commandment(tmp_path)
        round_eval: dict = {}

        # No baseline files in pp_dir + repo_root=None => benchmark loop finds
        # nothing and returns; the skip path must never shell out.
        with patch.object(evaluation, "build_eval_script") as mock_build, patch.object(
            evaluation.subprocess, "run"
        ) as mock_run:
            evaluation.run_correctness_and_benchmark(
                eval_worktree=tmp_path,
                eval_env={},
                commandment_path=commandment,
                pp_dir=tmp_path,
                round_eval=round_eval,
                round_num=1,
                repo_root=None,
            )

        assert round_eval["correctness"] == {"skipped": True, "reason": "translation_run"}
        assert round_eval.get("status") != "correctness_failed"
        # Strict gate skipped => CORRECTNESS script never built and nothing run.
        mock_build.assert_not_called()
        mock_run.assert_not_called()

    def test_non_translation_run_enforces_strict_correctness_gate(self, tmp_path, monkeypatch):
        from minisweagent.run.postprocess import evaluation

        monkeypatch.delenv("GEAK_TRANSLATION_RUN", raising=False)
        commandment = _write_commandment(tmp_path)
        round_eval: dict = {}

        fake_script = str(tmp_path / "corr.sh")
        with patch.object(evaluation, "build_eval_script", return_value=fake_script) as mock_build, patch.object(
            evaluation.subprocess, "run", return_value=_completed(0, stdout="ok")
        ) as mock_run:
            evaluation.run_correctness_and_benchmark(
                eval_worktree=tmp_path,
                eval_env={},
                commandment_path=commandment,
                pp_dir=tmp_path,
                round_eval=round_eval,
                round_num=1,
                repo_root=None,
            )

        # Gate enforced: built SETUP+CORRECTNESS and actually ran it.
        assert mock_build.call_args_list[0].args[1] == ["SETUP", "CORRECTNESS"]
        assert mock_run.called
        assert round_eval["correctness"] == {"returncode": 0, "success": True}
        assert "skipped" not in round_eval["correctness"]


# ---------------------------------------------------------------------------
# preflight_commandment_contract
# ---------------------------------------------------------------------------
class TestPreflightSectionSelection:
    @pytest.mark.parametrize(
        "flag,expected_sections",
        [("1", ["SETUP"]), (None, ["SETUP", "CORRECTNESS"])],
    )
    def test_preflight_smoke_sections(self, tmp_path, monkeypatch, flag, expected_sections):
        from minisweagent.run.postprocess import evaluation

        if flag is None:
            monkeypatch.delenv("GEAK_TRANSLATION_RUN", raising=False)
        else:
            monkeypatch.setenv("GEAK_TRANSLATION_RUN", flag)
        monkeypatch.delenv("GEAK_SKIP_COMMANDMENT_PREFLIGHT", raising=False)
        commandment = _write_commandment(tmp_path)

        # Returning None makes preflight log "no <sections>" and return before
        # shelling out, so we only assert the section selection.
        with patch.object(evaluation, "build_eval_script", return_value=None) as mock_build:
            evaluation.preflight_commandment_contract(
                commandment,
                str(tmp_path),
                "",
                [0],
            )

        assert mock_build.call_count == 1
        assert mock_build.call_args.args[1] == expected_sections
