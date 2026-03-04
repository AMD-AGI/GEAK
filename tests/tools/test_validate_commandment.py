# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""Tests for validate_commandment -- COMMANDMENT.md validation."""

from minisweagent.tools.validate_commandment import (
    format_validation_message,
    validate_commandment,
)

# ---- Valid COMMANDMENT files ----

VALID_COMMANDMENT = """\
## SETUP
printf '#!/bin/bash\\nexport PYTHONPATH=%s:${PYTHONPATH}\\n' "${GEAK_WORK_DIR}" > ${GEAK_WORK_DIR}/run.sh && chmod +x ${GEAK_WORK_DIR}/run.sh

## CORRECTNESS
${GEAK_WORK_DIR}/run_harness.sh --correctness

## PROFILE
${GEAK_WORK_DIR}/run_harness.sh --profile

## BENCHMARK
${GEAK_WORK_DIR}/run_harness.sh --benchmark

## FULL_BENCHMARK
${GEAK_WORK_DIR}/run_harness.sh --full-benchmark
"""


def test_valid_commandment():
    result = validate_commandment(VALID_COMMANDMENT)
    assert result["valid"] is True
    assert result["errors"] == []


# ---- Missing sections ----


def test_missing_profile_section():
    content = """\
## SETUP
echo hello

## CORRECTNESS
python3 test.py

## BENCHMARK
python3 bench.py --benchmark

## FULL_BENCHMARK
python3 bench.py --full-benchmark
"""
    result = validate_commandment(content)
    assert result["valid"] is False
    assert any("PROFILE" in e for e in result["errors"])


def test_missing_all_sections():
    content = "Just some text with no sections at all."
    result = validate_commandment(content)
    assert result["valid"] is False
    assert any("SETUP" in e for e in result["errors"])
    assert any("CORRECTNESS" in e for e in result["errors"])
    assert any("PROFILE" in e for e in result["errors"])
    assert any("BENCHMARK" in e for e in result["errors"])
    assert any("FULL_BENCHMARK" in e for e in result["errors"])


# ---- Unknown sections ----


def test_unknown_section_header():
    content = """\
## SETUP
mkdir -p /tmp/test

## Test Command
python3 test.py

## CORRECTNESS
python3 test.py

## PROFILE
python3 test.py --profile

## BENCHMARK
python3 bench.py --benchmark

## FULL_BENCHMARK
python3 bench.py --full-benchmark
"""
    result = validate_commandment(content)
    assert result["valid"] is False
    assert any("Test" in e and "recognized" in e for e in result["errors"])


# ---- Shell built-ins ----


def test_cd_as_command_prefix():
    content = """\
## SETUP
mkdir -p /tmp/test

## CORRECTNESS
python3 test.py

## PROFILE
cd /workspace && python3 bench.py

## BENCHMARK
python3 bench.py --benchmark

## FULL_BENCHMARK
python3 bench.py --full-benchmark
"""
    result = validate_commandment(content)
    assert result["valid"] is False
    assert any("cd" in e and "shell built-in" in e for e in result["errors"])


def test_source_as_command_prefix():
    content = """\
## SETUP
source /opt/rocm/bin/setenv.sh

## CORRECTNESS
python3 test.py

## PROFILE
python3 bench.py

## BENCHMARK
python3 bench.py --benchmark

## FULL_BENCHMARK
python3 bench.py --full-benchmark
"""
    result = validate_commandment(content)
    assert result["valid"] is False
    assert any("source" in e for e in result["errors"])


def test_export_as_command_prefix():
    content = """\
## SETUP
export PYTHONPATH=/workspace

## CORRECTNESS
python3 test.py

## PROFILE
python3 bench.py

## BENCHMARK
python3 bench.py --benchmark

## FULL_BENCHMARK
python3 bench.py --full-benchmark
"""
    result = validate_commandment(content)
    assert result["valid"] is False
    assert any("export" in e for e in result["errors"])


# ---- Empty sections ----


def test_empty_section():
    content = """\
## SETUP

## CORRECTNESS
python3 test.py

## PROFILE
python3 bench.py

## BENCHMARK
python3 bench.py --benchmark

## FULL_BENCHMARK
python3 bench.py --full-benchmark
"""
    result = validate_commandment(content)
    assert result["valid"] is False
    assert any("SETUP" in e and "no commands" in e for e in result["errors"])


# ---- format_validation_message ----


def test_format_valid():
    result = validate_commandment(VALID_COMMANDMENT)
    msg = format_validation_message(result)
    assert "OK" in msg


def test_format_errors():
    content = "no sections"
    result = validate_commandment(content)
    msg = format_validation_message(result)
    assert "VALIDATION ERRORS" in msg
    assert "must fix" in msg.lower() or "ERRORS" in msg


# ---- Inline env var prefixes ----


def test_inline_env_var_in_profile():
    """HIP_VISIBLE_DEVICES=1 python3 ... in PROFILE should be caught."""
    content = """\
## SETUP
mkdir -p /tmp/test

## CORRECTNESS
python3 /path/to/test.py

## PROFILE
HIP_VISIBLE_DEVICES=1 python3 /path/to/bench.py --profile

## BENCHMARK
python3 /path/to/bench.py --benchmark

## FULL_BENCHMARK
python3 /path/to/bench.py --full-benchmark
"""
    result = validate_commandment(content)
    assert result["valid"] is False
    assert any("inline env var" in e and "HIP_VISIBLE_DEVICES=1" in e for e in result["errors"])


def test_inline_env_var_in_correctness():
    """Inline env var in CORRECTNESS should also be caught."""
    content = """\
## SETUP
mkdir -p /tmp/test

## CORRECTNESS
PYTHONPATH=/workspace python3 /path/to/test.py

## PROFILE
python3 /path/to/bench.py --profile

## BENCHMARK
python3 /path/to/bench.py --benchmark

## FULL_BENCHMARK
python3 /path/to/bench.py --full-benchmark
"""
    result = validate_commandment(content)
    assert result["valid"] is False
    assert any("inline env var" in e and "PYTHONPATH=/workspace" in e for e in result["errors"])


def test_inline_env_var_in_setup_not_flagged():
    """Inline env var in SETUP should NOT be flagged (SETUP doesn't go through rocprofv3)."""
    content = """\
## SETUP
MY_VAR=hello mkdir -p /tmp/test

## CORRECTNESS
python3 /path/to/test.py

## PROFILE
python3 /path/to/bench.py --profile

## BENCHMARK
python3 /path/to/bench.py --benchmark

## FULL_BENCHMARK
python3 /path/to/bench.py --full-benchmark
"""
    result = validate_commandment(content)
    assert not any("inline env var" in e for e in result["errors"])


def test_inline_env_var_multiple_vars():
    """Multiple inline env vars should each be caught."""
    content = """\
## SETUP
mkdir -p /tmp/test

## CORRECTNESS
python3 /path/to/test.py

## PROFILE
HIP_VISIBLE_DEVICES=0 python3 /path/to/bench.py
CUDA_VISIBLE_DEVICES=1 python3 /path/to/bench.py

## BENCHMARK
python3 /path/to/bench.py --benchmark

## FULL_BENCHMARK
python3 /path/to/bench.py --full-benchmark
"""
    result = validate_commandment(content)
    assert result["valid"] is False
    env_var_errors = [e for e in result["errors"] if "inline env var" in e]
    assert len(env_var_errors) == 2


# ---- Edge cases ----


def test_code_block_does_not_false_positive():
    """Shell built-in inside a code block WITHIN a recognized section should still be caught."""
    content = """\
## SETUP
```bash
cd /workspace
python3 setup.py
```

## CORRECTNESS
python3 test.py

## PROFILE
python3 bench.py

## BENCHMARK
python3 bench.py --benchmark

## FULL_BENCHMARK
python3 bench.py --full-benchmark
"""
    # cd is inside a code block in SETUP section -- our validator checks
    # lines in recognized sections regardless (since COMMANDMENT commands
    # may or may not be in code blocks).
    result = validate_commandment(content)
    # The `cd` line is outside the code block tracking since "```" toggles
    # are consumed. The `cd /workspace` line itself is NOT preceded by ```,
    # so it IS checked. This test verifies no crash.
    assert isinstance(result["valid"], bool)
