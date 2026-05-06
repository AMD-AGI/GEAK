"""Tests for ``agents/heterogeneous/workload_guidance.py`` detection routing.

Per plan §I3 (row 16 from §13.2-C): literal ``kernel_type == "triton"``
and ``kernel_type == "hip"`` checks have been replaced by calls into
``kernel_languages.registry``.  This test suite pins that wiring so
any accidental revert is caught.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from minisweagent.agents.heterogeneous.workload_guidance import (
    _detect_backend,
    _is_hip_like_kernel,
    _is_triton_like_kernel,
    _resolved_language_name,
)


class TestResolvedLanguageName:
    """``_resolved_language_name`` routes every lookup through the registry."""

    def test_explicit_triton_goes_through_registry(self) -> None:
        fake_lang = MagicMock(name="fake_triton")
        fake_lang.name = "triton"
        with patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best_by_name",
            return_value=fake_lang,
        ) as mock_get:
            assert _resolved_language_name({"kernel_type": "triton"}) == "triton"
            mock_get.assert_called_once_with("triton")

    def test_explicit_hip_goes_through_registry(self) -> None:
        fake_lang = MagicMock(name="fake_hip")
        fake_lang.name = "hip"
        with patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best_by_name",
            return_value=fake_lang,
        ):
            assert _resolved_language_name({"kernel_type": "hip"}) == "hip"

    def test_rocm_alias_routed_through_registry(self) -> None:
        """'rocm' should be normalised via the registry (not handled
        inline)."""
        fake_hip = MagicMock(name="fake_hip")
        fake_hip.name = "hip"
        with patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best_by_name",
            return_value=fake_hip,
        ) as mock_get:
            assert _resolved_language_name({"kernel_type": "rocm"}) == "hip"
            mock_get.assert_called_once_with("rocm")

    def test_non_hip_kernel_names_short_circuit(self) -> None:
        """Legacy ``ck`` / ``asm`` names must short-circuit and NOT
        touch the registry — they have no KernelLanguage bundle."""
        with patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best_by_name",
        ) as mock_get:
            assert _resolved_language_name({"kernel_type": "ck"}) == "ck"
            assert _resolved_language_name({"kernel_type": "asm"}) == "asm"
            mock_get.assert_not_called()

    def test_path_fallback_when_type_not_recognised(self, tmp_path) -> None:
        """When ``kernel_type`` is missing/unknown, fall back to
        registry.detect_best on the file path."""
        kfile = tmp_path / "kernel.py"
        kfile.write_text("@triton.jit\ndef foo(): pass")

        fake_lang = MagicMock()
        fake_lang.name = "triton"
        with patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best_by_name",
            return_value=None,
        ), patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best",
            return_value=fake_lang,
        ):
            assert _resolved_language_name({"file_path": str(kfile)}) == "triton"

    def test_empty_when_both_routes_fail(self) -> None:
        with patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best_by_name",
            return_value=None,
        ), patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best",
            return_value=None,
        ):
            assert _resolved_language_name({"kernel_type": "alien", "file_path": "/tmp/x.xyz"}) == ""


class TestDetectors:
    """``_is_hip_like_kernel`` / ``_is_triton_like_kernel`` / ``_detect_backend``
    must all agree with the registry-resolved name."""

    def test_is_hip_like_via_registry(self) -> None:
        fake_hip = MagicMock()
        fake_hip.name = "hip"
        with patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best_by_name",
            return_value=fake_hip,
        ):
            assert _is_hip_like_kernel({"kernel_type": "hip"}) is True
            assert _is_triton_like_kernel({"kernel_type": "hip"}) is False

    def test_is_triton_like_via_registry(self) -> None:
        fake_triton = MagicMock()
        fake_triton.name = "triton"
        with patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best_by_name",
            return_value=fake_triton,
        ):
            assert _is_triton_like_kernel({"kernel_type": "triton"}) is True
            assert _is_hip_like_kernel({"kernel_type": "triton"}) is False

    def test_ck_kernel_is_neither(self) -> None:
        # ck is in _NON_HIP_KERNEL_NAMES — short-circuits.
        assert _is_hip_like_kernel({"kernel_type": "ck"}) is False
        assert _is_triton_like_kernel({"kernel_type": "ck"}) is False

    def test_path_heuristic_for_hip_when_registry_fails(self) -> None:
        """Partial-discovery fallback: .hip file with a ROCm token."""
        with patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best_by_name",
            return_value=None,
        ), patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best",
            return_value=None,
        ):
            assert (
                _is_hip_like_kernel({"kernel_type": "", "file_path": "/tmp/hip_kernel.hip"})
                is True
            )
            # Not ROCm-ish enough → no heuristic match
            assert (
                _is_hip_like_kernel({"kernel_type": "", "file_path": "/tmp/plain.cpp"}) is False
            )

    def test_path_heuristic_for_triton_when_registry_fails(self) -> None:
        with patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best_by_name",
            return_value=None,
        ), patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best",
            return_value=None,
        ):
            assert (
                _is_triton_like_kernel({"kernel_type": "", "file_path": "/tmp/triton_mul.py"})
                is True
            )
            assert (
                _is_triton_like_kernel({"kernel_type": "", "file_path": "/tmp/vanilla.py"})
                is False
            )

    @pytest.mark.parametrize(
        "kernel_type,expected",
        [("triton", "triton"), ("hip", "hip")],
    )
    def test_detect_backend_returns_registry_name(
        self,
        kernel_type: str,
        expected: str,
    ) -> None:
        fake_lang = MagicMock()
        fake_lang.name = expected
        with patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best_by_name",
            return_value=fake_lang,
        ):
            assert _detect_backend({"kernel_type": kernel_type}) == expected

    def test_detect_backend_generic_fallback(self) -> None:
        with patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best_by_name",
            return_value=None,
        ), patch(
            "minisweagent.agents.heterogeneous.workload_guidance.registry.detect_best",
            return_value=None,
        ):
            assert _detect_backend({"kernel_type": "", "file_path": "/tmp/x.wat"}) == "generic"


class TestNoLiteralEqualityRegression:
    """Lock in the fact that no literal language-equality check remains.

    Complements ``check_language_leaks.py``, which runs at CI time;
    this test runs in pytest so contributors catch violations
    immediately.
    """

    def test_module_has_no_literal_language_equality(self) -> None:
        import inspect

        from minisweagent.agents.heterogeneous import workload_guidance

        source = inspect.getsource(workload_guidance)
        for forbidden in (
            'kernel_type == "triton"',
            "kernel_type == 'triton'",
            'kernel_type == "hip"',
            "kernel_type == 'hip'",
            'kernel_language == "triton"',
            'kernel_language == "hip"',
        ):
            assert forbidden not in source, (
                f"workload_guidance.py contains forbidden literal "
                f"language-equality check: {forbidden!r}"
            )
