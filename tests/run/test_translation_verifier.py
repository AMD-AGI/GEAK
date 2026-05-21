"""Tests for Workstream D4 — TranslationPhase verifier + performance thresholds.

Pins:
  - ``validate_translation_performance`` threshold semantics
  - Structural + syntactic verifier layers (entry-point preservation,
    target-language token presence)
  - Pair-hint loading via KernelLanguage.translation_hints_for
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.run.preprocess.phases.base import PhaseContext
from minisweagent.run.preprocess.phases.translation import (
    PerformanceReport,
    TranslationPhase,
    _extract_entry_points,
    validate_translation_performance,
)


# ──────────────────────────────────────────────────────────────────────
# Performance thresholds
# ──────────────────────────────────────────────────────────────────────


class TestValidatePerformance:
    def test_ok_when_near_parity(self) -> None:
        r = validate_translation_performance(100.0, 110.0)
        assert r.status == "ok"
        assert r.ratio == pytest.approx(1.1)
        assert "threshold" in r.message.lower() or "source" in r.message.lower()

    def test_ok_when_target_faster(self) -> None:
        # Target is 20% faster, still within fail (0.5x) gate
        r = validate_translation_performance(100.0, 80.0)
        assert r.status == "ok"

    def test_warn_on_mild_regression(self) -> None:
        # 1.3x slower -> warn (> 1/0.8 = 1.25x), but not fail
        r = validate_translation_performance(100.0, 130.0)
        assert r.status == "warn"
        assert "regression" in r.message.lower()

    def test_fail_on_severe_regression(self) -> None:
        # 3x slower — exceeds 1/0.5 = 2x fail threshold
        r = validate_translation_performance(100.0, 300.0)
        assert r.status == "fail"
        assert "regression" in r.message.lower()

    def test_fail_on_suspicious_speedup(self) -> None:
        # Target is 0.3x — <0.5x fail threshold — likely correctness bug
        r = validate_translation_performance(100.0, 30.0)
        assert r.status == "fail"
        assert "suspicious" in r.message.lower() or "correctness" in r.message.lower()

    def test_invalid_latencies_raise(self) -> None:
        with pytest.raises(ValueError):
            validate_translation_performance(0.0, 100.0)
        with pytest.raises(ValueError):
            validate_translation_performance(100.0, 0.0)
        with pytest.raises(ValueError):
            validate_translation_performance(-1.0, 100.0)

    def test_invalid_thresholds_raise(self) -> None:
        with pytest.raises(ValueError):
            validate_translation_performance(
                100.0, 100.0, fail_threshold=0.9, warn_threshold=0.5
            )
        with pytest.raises(ValueError):
            validate_translation_performance(
                100.0, 100.0, fail_threshold=1.2, warn_threshold=0.8
            )

    def test_custom_thresholds(self) -> None:
        # Tighter gate — 30% regression is now fail (because we're
        # saying fail at <0.6x or >1.67x)
        r = validate_translation_performance(
            100.0, 200.0, fail_threshold=0.6, warn_threshold=0.8
        )
        assert r.status == "fail"

    def test_report_contains_latencies_and_ratio(self) -> None:
        r = validate_translation_performance(50.0, 100.0)
        assert r.source_latency_ms == 50.0
        assert r.target_latency_ms == 100.0
        assert r.ratio == pytest.approx(2.0)
        assert isinstance(r, PerformanceReport)


# ──────────────────────────────────────────────────────────────────────
# Entry-point extraction
# ──────────────────────────────────────────────────────────────────────


class TestEntryPointExtraction:
    def test_python_def_extraction(self) -> None:
        src = """
def kernel_a(x, y): pass

    def not_top_level(): pass  # indented, should NOT be picked up

@triton.jit
def kernel_b(x): pass
"""
        assert _extract_entry_points(src, "triton") == {"kernel_a", "kernel_b"}

    def test_hip_global_extraction(self) -> None:
        src = """
__global__ void kernel_a(float* x) {}
__global__ void kernel_b(float* y) {}
"""
        assert _extract_entry_points(src, "hip") == {"kernel_a", "kernel_b"}

    def test_unknown_language_tries_both(self) -> None:
        """For unrecognised target languages, extraction falls back
        to both heuristics and returns the union."""
        src = """
def python_entry(x): pass
__global__ void c_entry(float* y) {}
"""
        result = _extract_entry_points(src, "wat_lang")
        assert "python_entry" in result
        assert "c_entry" in result

    def test_returns_empty_set_when_no_matches(self) -> None:
        assert _extract_entry_points("// just a comment", "triton") == set()


# ──────────────────────────────────────────────────────────────────────
# Verifier layers
# ──────────────────────────────────────────────────────────────────────


def _make_phase_with_ctx(target: str, source_code: str, source_lang: str) -> tuple[TranslationPhase, Any]:
    """Build a TranslationPhase + verify_fn for testing."""
    phase = TranslationPhase()
    ctx = PhaseContext()
    verify_fn = phase._build_verify_fn(
        ctx=ctx, target=target, source_code=source_code, source_language=source_lang
    )
    return phase, verify_fn


class TestVerifierLayers:
    def test_empty_candidate_rejected(self) -> None:
        _, verify = _make_phase_with_ctx("hip", "def kernel(x): pass", "triton")
        ok, reason = verify("")
        assert not ok
        assert "empty" in reason.lower()

        ok, reason = verify("   \n\n   ")
        assert not ok

    def test_whitespace_only_rejected(self) -> None:
        _, verify = _make_phase_with_ctx("hip", "def kernel(x): pass", "triton")
        ok, _ = verify("\n\n\t  ")
        assert not ok

    def test_hip_candidate_missing_hip_tokens_rejected(self) -> None:
        _, verify = _make_phase_with_ctx("hip", "def kernel(x): pass", "triton")
        # Python-looking candidate for a HIP target -> should fail the
        # syntactic layer.
        ok, reason = verify("def not_hip(): pass")
        assert not ok
        assert "hip" in reason.lower()

    def test_hip_candidate_with_global_accepted(self) -> None:
        _, verify = _make_phase_with_ctx(
            "hip",
            "def kernel(x, y): return x + y",
            "triton",
        )
        candidate = """
__global__ void kernel(float* x, float* y, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = x[idx] + y[idx];
}
"""
        ok, reason = verify(candidate)
        assert ok, reason

    def test_triton_candidate_with_tl_token_accepted(self) -> None:
        _, verify = _make_phase_with_ctx(
            "triton",
            "__global__ void kernel(float* x) {}",
            "hip",
        )
        candidate = """
@triton.jit
def kernel(x_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    tl.store(x_ptr + pid, 0.0)
"""
        ok, reason = verify(candidate)
        assert ok, reason

    def test_cuda_candidate_with_global_accepted(self) -> None:
        _, verify = _make_phase_with_ctx(
            "cuda",
            "def foo(): pass",
            "triton",
        )
        candidate = "__global__ void foo(float* p) {}"
        ok, _ = verify(candidate)
        assert ok

    def test_structural_entry_point_check_rejects_rename(self) -> None:
        """Translation that renames ALL source entry points fails."""
        _, verify = _make_phase_with_ctx(
            "hip",
            "def my_original_kernel(x, y): return x + y",
            "triton",
        )
        # Candidate has HIP tokens but renames the entry point
        candidate = "__global__ void completely_different_name(float* x) {}"
        ok, reason = verify(candidate)
        assert not ok
        assert "entry" in reason.lower()

    def test_structural_check_accepts_preserved_entry_point(self) -> None:
        _, verify = _make_phase_with_ctx(
            "hip",
            "def my_kernel(x, y): return x + y",
            "triton",
        )
        candidate = "__global__ void my_kernel(float* x, float* y, float* out) {}"
        ok, _ = verify(candidate)
        assert ok

    def test_structural_check_skipped_when_source_has_no_entry_points(self) -> None:
        """If source has no detectable entry points, the layer is
        skipped (don't false-reject)."""
        _, verify = _make_phase_with_ctx(
            "hip",
            "# just comments, no function defs",
            "triton",
        )
        candidate = "__global__ void whatever(float* x) {}"
        ok, _ = verify(candidate)
        assert ok


# ──────────────────────────────────────────────────────────────────────
# Pair-hint loading
# ──────────────────────────────────────────────────────────────────────


class TestPairHintsLoading:
    def test_loads_pair_specific_hints(self, tmp_path: Path) -> None:
        src_path = tmp_path / "kernel.py"
        src_path.write_text("def k(): pass")

        # Registry-backed lookup; triton -> hip pair should exist in our
        # real bundle.
        hints = TranslationPhase._load_pair_hints(src_path, "hip", "triton")
        assert hints, "Expected triton -> hip hints to load"
        assert "hip" in hints.lower() or "translation" in hints.lower()

    def test_falls_back_when_pair_missing(self, tmp_path: Path) -> None:
        src_path = tmp_path / "kernel.py"
        src_path.write_text("def k(): pass")

        # triton -> some_future_language should fall back to _fallback.md
        hints = TranslationPhase._load_pair_hints(src_path, "some_future_language", "triton")
        assert hints, "Expected _fallback.md to provide generic guidance"

    def test_returns_empty_for_unknown_source_language(self, tmp_path: Path) -> None:
        src_path = tmp_path / "kernel.xyz"
        src_path.write_text("foo")

        hints = TranslationPhase._load_pair_hints(src_path, "hip", "nonexistent_lang")
        assert hints == ""


# ──────────────────────────────────────────────────────────────────────
# Config YAML presence
# ──────────────────────────────────────────────────────────────────────


class TestTranslatorConfig:
    def test_config_yaml_exists(self) -> None:
        from minisweagent.pipeline_workers import translation

        mod_dir = Path(translation.__file__).parent
        config_path = mod_dir / "configs" / "translator.yaml"
        assert config_path.exists(), f"Missing config: {config_path}"

    def test_config_yaml_is_loadable(self) -> None:
        import yaml

        from minisweagent.pipeline_workers import translation

        mod_dir = Path(translation.__file__).parent
        data = yaml.safe_load((mod_dir / "configs" / "translator.yaml").read_text())
        assert data["name"] == "translation"
        assert "system_template" in data
        assert "step_limit" in data
        assert "cost_limit" in data
        # perf thresholds live under extra
        assert "perf_fail_threshold" in data.get("extra", {})
        assert "perf_warn_threshold" in data.get("extra", {})
