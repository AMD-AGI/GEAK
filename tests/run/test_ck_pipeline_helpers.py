# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Tests for CK harness validation in pipeline_helpers."""

import textwrap
from pathlib import Path

import pytest

from minisweagent.run.pipeline_helpers import validate_harness


# ---------------------------------------------------------------------------
# CK-style harness validation
# ---------------------------------------------------------------------------


class TestValidateHarnessCK:
    """Validate that CK subprocess-based harnesses pass static checks."""

    CK_HARNESS = textwrap.dedent("""\
        import argparse
        import subprocess
        import numpy as np

        ORIGINAL_BINARY = "/path/to/original"

        def main():
            parser = argparse.ArgumentParser()
            mode = parser.add_mutually_exclusive_group(required=True)
            mode.add_argument("--correctness", action="store_true")
            mode.add_argument("--profile", action="store_true")
            mode.add_argument("--benchmark", action="store_true")
            mode.add_argument("--full-benchmark", action="store_true")
            args = parser.parse_args()

        if __name__ == "__main__":
            main()
    """)

    def test_accepts_ck_harness(self, tmp_path):
        harness = tmp_path / "test_harness.py"
        harness.write_text(self.CK_HARNESS)
        valid, errors = validate_harness(str(harness))
        assert valid, f"CK harness should pass: {errors}"

    def test_ck_harness_skips_gpu_alloc_check(self, tmp_path):
        """CK harnesses run binaries via subprocess; no GPU alloc to check."""
        harness_code = self.CK_HARNESS + textwrap.dedent("""\
            def run_profile():
                x = torch.randn(10, device='cuda')
        """)
        harness = tmp_path / "test_harness.py"
        harness.write_text(harness_code)
        valid, errors = validate_harness(str(harness))
        assert valid, f"CK harness should skip GPU alloc check: {errors}"


class TestValidateHarnessNonCK:
    """Validate that non-CK harnesses still get GPU alloc checks."""

    PYTHON_HARNESS_WITH_GPU_ALLOC = textwrap.dedent("""\
        import argparse

        def run_profile():
            import torch
            x = torch.randn(10, device='cuda')
            return x

        def main():
            parser = argparse.ArgumentParser()
            mode = parser.add_mutually_exclusive_group(required=True)
            mode.add_argument("--correctness", action="store_true")
            mode.add_argument("--profile", action="store_true")
            mode.add_argument("--benchmark", action="store_true")
            mode.add_argument("--full-benchmark", action="store_true")
            args = parser.parse_args()

        if __name__ == "__main__":
            main()
    """)

    def test_gpu_alloc_warning_for_non_ck(self, tmp_path):
        harness = tmp_path / "test_harness.py"
        harness.write_text(self.PYTHON_HARNESS_WITH_GPU_ALLOC)
        valid, errors = validate_harness(str(harness))
        assert not valid
        assert any("GPU tensor allocation" in e for e in errors)


class TestValidateHarnessMissingFlags:
    """Validate that harnesses missing flags are rejected."""

    INCOMPLETE_HARNESS = textwrap.dedent("""\
        import argparse

        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--correctness", action="store_true")

        if __name__ == "__main__":
            main()
    """)

    def test_rejects_missing_flags(self, tmp_path):
        harness = tmp_path / "test_harness.py"
        harness.write_text(self.INCOMPLETE_HARNESS)
        valid, errors = validate_harness(str(harness))
        assert not valid
        assert any("--profile" in e for e in errors)
