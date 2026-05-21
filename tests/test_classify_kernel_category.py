from __future__ import annotations

import pytest

from minisweagent.memory.cross_session import classify_kernel_category


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/repo/gemm_kernel.py", "gemm"),
        ("/repo/attention_mla.py", "attention"),
        ("/repo/rms_norm.py", "normalization"),
        ("/repo/moe_expert.hip", "moe"),
        ("/repo/rope_rotary.py", "positional_encoding"),
        ("/repo/nearest_neighbor.cu", "spatial_search"),
        ("/repo/unknown.py", "unknown"),
    ],
)
def test_classify_kernel_category(path: str, expected: str) -> None:
    assert classify_kernel_category(path) == expected


def test_legacy_import_still_works() -> None:
    from minisweagent.memory.cross_session_memory import classify_kernel_category as legacy

    with pytest.warns(DeprecationWarning):
        assert legacy("/repo/matmul.py") == "gemm"

