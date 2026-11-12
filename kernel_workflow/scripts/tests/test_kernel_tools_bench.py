# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the search-time measurement tools in kernel_workflow/scripts/kernel_tools.

ab_bench.py (same-window interleaved A/B screening), create_harness.py (+ harness_stub_env.py, the
stub importer its selftest runs a generated skeleton under) and parse_correctness.py are GEAK shared
kernel tools. They never time or judge a kernel themselves: every sample comes from
e2e_workflow/scripts/harness_lib.py, and a missing harness_lib is a refusal, never a fallback timer.
No GPU, no network, no torch.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
KT = REPO / "kernel_workflow" / "scripts" / "kernel_tools"
HARNESS_LIB = REPO / "e2e_workflow" / "scripts" / "harness_lib.py"
TOOLS = ["ab_bench", "create_harness", "parse_correctness"]


def run(*args, env=None, cwd=None, timeout=300):
    full_env = dict(os.environ)
    full_env.update(env or {})
    return subprocess.run([str(a) for a in args], text=True, capture_output=True, check=False,
                          env=full_env, cwd=cwd or "/tmp", timeout=timeout)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("tool", TOOLS)
def test_selftest(tool):
    result = run(sys.executable, KT / f"{tool}.py", "--selftest")
    out = result.stdout + result.stderr
    assert result.returncode == 0, out[-3000:]
    assert "SELFTEST PASS" in out, out[-3000:]


def test_ab_bench_times_through_harness_lib():
    ab = _load("ab_bench_kt", KT / "ab_bench.py")
    assert Path(ab.harness_lib_path()).resolve() == HARNESS_LIB.resolve()
    assert ab.load_harness_lib().cache_policy()["mode"] == "read-evict"


def test_ab_bench_selftest_uses_the_real_instrument():
    """From kernel_tools the repo's harness_lib is reachable, so the real-instrument check must RUN."""
    out = run(sys.executable, KT / "ab_bench.py", "--selftest").stdout
    assert "real-instrument check skipped" not in out, out[-2000:]


def test_ab_bench_refuses_missing_harness_lib(tmp_path):
    adapter = tmp_path / "bm.py"
    adapter.write_text("def variants():\n    return {'a': lambda: None}\n")
    result = run(sys.executable, KT / "ab_bench.py", "--module", adapter,
                 env={"GEAK_HARNESS_LIB": str(tmp_path / "missing")})
    assert result.returncode != 0 and "harness_lib" in result.stderr


def test_create_harness_bakes_in_the_repo_harness_lib(tmp_path):
    ch = _load("create_harness_kt", KT / "create_harness.py")
    assert Path(ch.default_harness_lib()).resolve() == HARNESS_LIB.resolve()
    out = tmp_path / "harness.py"
    (tmp_path / "k.py").write_text("import triton\n\n@triton.jit\ndef k(x):\n    pass\n")
    result = run(sys.executable, KT / "create_harness.py", "--kernel-path", tmp_path / "k.py",
                 "--kernel-name", "k", "--output", out)
    assert result.returncode == 0, result.stderr[-2000:]
    assert str(HARNESS_LIB.resolve()) in out.read_text()


def test_parse_correctness_geak_exit_codes(tmp_path):
    log = tmp_path / "ut.out"
    for rc, text, want in ((0, '{"case": "a", "correct": true, "max_rel_err": 0.001}\n', 0),
                           (1, '{"case": "a", "correct": false, "max_rel_err": 0.4}\n', 1),
                           (2, "ImportError\n", 2),
                           (3, "UT_HARNESS_INCOMPLETE: no replay bundle\n", 3)):
        log.write_text(text)
        result = run(sys.executable, KT / "parse_correctness.py", "--log", log,
                     "--parse", "geak", "--exit-code", str(rc))
        assert result.returncode == want, (rc, result.stdout, result.stderr)
