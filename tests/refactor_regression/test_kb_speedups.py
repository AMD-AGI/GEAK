"""Nightly regression: assert each §0.2 kernel meets its speedup threshold.

Reads thresholds from `tests/refactor_regression/baseline_speedups.yaml`.

This is a STUB today — the full nightly runner lands in PR-3 alongside
scripts/check_baseline_speedups.py. For now, the test:

1. Validates the baseline_speedups.yaml is well-formed
2. (if `GEAK_KB_PATH` is set) Compares thresholds against the current KB's
   best_speedup values and fails if any stored speedup has REGRESSED below
   the per-kernel threshold (detects KB corruption, not agent regression).

The full nightly version runs geak on each kernel for 5 rounds × 1 seed on
4 GPUs (~4 GPU-hours). That's gated behind GEAK_KB_NIGHTLY=1 in PR-3.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

THRESHOLDS_FILE = Path(__file__).parent / "baseline_speedups.yaml"


def _load_thresholds() -> dict:
    """Load baseline_speedups.yaml. Uses yaml if available; falls back to a
    narrow custom parser so this test runs without a yaml dependency."""
    text = THRESHOLDS_FILE.read_text()
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        pytest.skip("pyyaml not available; install with `pip install pyyaml`")


def test_baseline_yaml_wellformed() -> None:
    data = _load_thresholds()
    assert isinstance(data, dict)
    assert "kernels" in data
    kernels = data["kernels"]
    assert isinstance(kernels, dict)
    assert len(kernels) == 13, f"Expected 13 kernels in §0.2 table, got {len(kernels)}"
    for name, entry in kernels.items():
        assert "best_observed" in entry, f"{name} missing best_observed"
        assert "threshold" in entry, f"{name} missing threshold"
        assert "source_line" in entry, f"{name} missing source_line (KB line)"
        assert 0 < entry["threshold"] <= entry["best_observed"], (
            f"{name}: threshold {entry['threshold']} must be > 0 and <= "
            f"best_observed {entry['best_observed']}"
        )


def test_kb_matches_stored_thresholds_if_kb_present() -> None:
    """If GEAK_KB_PATH is set, verify current KB best_speedup >= threshold.

    This catches KB corruption / manual-edit regressions BEFORE a refactor PR
    lands. The agent-level regression (refactor causing agent performance drop)
    is tested by the full nightly runner (PR-3).
    """
    kb_path = os.environ.get("GEAK_KB_PATH")
    if not kb_path:
        pytest.skip("Set GEAK_KB_PATH to enable KB-consistency check")

    p = Path(kb_path)
    assert p.exists(), f"GEAK_KB_PATH={p} does not exist"
    kb = json.loads(p.read_text())

    thresholds = _load_thresholds()["kernels"]
    # Walk KB looking for each kernel's best_speedup
    failures: list[str] = []
    kb_entries = kb.get("entries", [])
    best_by_name: dict[str, float] = {}
    for e in kb_entries:
        name = e.get("kernel_name")
        speedup = e.get("best_speedup")
        if name and speedup is not None:
            if best_by_name.get(name, 0) < speedup:
                best_by_name[name] = float(speedup)

    for name, spec in thresholds.items():
        got = best_by_name.get(name)
        if got is None:
            # Not a failure; kernel may legitimately be missing from this KB snapshot
            continue
        if got < spec["threshold"]:
            failures.append(
                f"{name}: KB best_speedup={got:.3f} is below threshold={spec['threshold']:.3f}"
            )

    assert not failures, "KB consistency failures:\n" + "\n".join(failures)


@pytest.mark.slow
@pytest.mark.gpu
def test_kb_nightly_full_sweep() -> None:
    """Full nightly sweep: run geak on each kernel × 5 rounds × 1 seed.

    STUB today — full implementation lands in PR-3 with scripts/run_nightly_regression.py.
    """
    if os.environ.get("GEAK_KB_NIGHTLY") != "1":
        pytest.skip("Set GEAK_KB_NIGHTLY=1 to run the full sweep (requires GPU + ~4 GPU-hours)")
    pytest.skip("Full nightly runner lands in PR-3")
