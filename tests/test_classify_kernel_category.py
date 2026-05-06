"""Tests pinning the canonical location of ``classify_kernel_category``.

Per plan §13.2-D row 25: ``classify_kernel_category`` now lives at
``minisweagent.memory.cross_session``; the legacy module
``minisweagent.memory.cross_session_memory`` is kept as a deprecation
shim for one release.  These tests lock in both the new canonical
import path and the shim's backwards-compat behavior so any future
accidental revert is caught.
"""

from __future__ import annotations

import warnings

import pytest


class TestCanonicalImport:
    """New canonical home: ``minisweagent.memory.cross_session``."""

    def test_import_from_canonical_location(self) -> None:
        from minisweagent.memory.cross_session import classify_kernel_category

        assert callable(classify_kernel_category)

    @pytest.mark.parametrize(
        "path,expected_category",
        [
            ("/kernels/gemm_a16wfp4.py", "gemm"),
            ("/kernels/matmul_bf16.hip", "gemm"),
            ("/kernels/mm_grouped.py", "gemm"),
            ("/kernels/attention_backward.py", "attention"),
            ("/kernels/fused_qkv_rope.py", "positional_encoding"),
            ("/kernels/fused_rms_fp8.py", "normalization"),
            ("/kernels/fused_mxfp4_quant_moe_sort.py", "moe"),
            ("/kernels/nearest_neighbor_2d.py", "spatial_search"),
            ("/kernels/flash_atten.py", "attention"),
            ("/kernels/something_unrelated.py", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_classification_preserves_legacy_semantics(
        self,
        path: str,
        expected_category: str,
    ) -> None:
        """Every legacy case must map to the same category after the move."""
        from minisweagent.memory.cross_session import classify_kernel_category

        assert classify_kernel_category(path) == expected_category


class TestDeprecationShim:
    """Legacy module still works but emits DeprecationWarning."""

    def test_shim_still_exports_function(self) -> None:
        from minisweagent.memory import cross_session_memory

        assert hasattr(cross_session_memory, "classify_kernel_category")

    def test_shim_emits_deprecation_warning(self) -> None:
        from minisweagent.memory.cross_session_memory import classify_kernel_category

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = classify_kernel_category("/kernels/gemm.py")

        assert result == "gemm"
        deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecation_warnings, "Expected DeprecationWarning to be emitted"
        assert "cross_session_memory" in str(deprecation_warnings[0].message)
        assert "cross_session" in str(deprecation_warnings[0].message)

    def test_shim_returns_identical_output_to_canonical(self) -> None:
        """Both paths must return byte-identical results."""
        from minisweagent.memory import cross_session_memory
        from minisweagent.memory.cross_session import classify_kernel_category as canonical

        test_paths = [
            "/kernels/gemm_a16wfp4.py",
            "/kernels/fused_qkv_rope.py",
            "/kernels/nothing_here.py",
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for path in test_paths:
                assert cross_session_memory.classify_kernel_category(path) == canonical(path)


class TestNoStaleImports:
    """CI guard: new code in src/ must not import classify_kernel_category
    from the legacy shim location.
    """

    def test_src_has_no_stale_imports(self) -> None:
        import subprocess
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        src_dir = repo_root / "src"

        # grep for any import of classify_kernel_category from the shim
        result = subprocess.run(
            [
                "rg",
                "--no-heading",
                "--no-line-number",
                r"from minisweagent\.memory\.cross_session_memory import",
                str(src_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        # Only the shim file itself is permitted to mention its own name;
        # and it does so in a docstring + warning string, not an import.
        # So we expect zero matches.
        stale_hits = [
            line
            for line in result.stdout.splitlines()
            if line.strip()
            # exclude the shim file itself if it were to show up
            and "memory/cross_session_memory.py" not in line
        ]
        assert not stale_hits, f"Found stale imports from the deprecation shim:\n{chr(10).join(stale_hits)}"
