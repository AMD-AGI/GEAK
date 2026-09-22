# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Search cutoff survives result normalization and is independent of performance."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("run_e2e_search_termination", Path(__file__).with_name("run_e2e.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("speedup", [1.0, 1.1])
@pytest.mark.parametrize("termination", [None, {"reason": "dispatch_cutoff", "remaining_s": 5400}])
def test_search_termination_is_preserved(tmp_path, speedup, termination):
    wf = {
        "eval_dir": str(tmp_path), "baseline_throughput_tok_s": 100,
        "final_throughput_tok_s": 100 * speedup, "throughput_speedup": speedup,
        "output_parity": "pass", "search_termination": termination,
    }
    result = runner.normalize_result({}, wf)
    assert result["search_termination"] == (termination or {"reason": "unknown"})
