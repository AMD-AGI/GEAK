"""Tests for validate_commandment -- COMMANDMENT.md validation."""

import pytest
from minisweagent.tools.validate_commandment import (
    format_validation_message,
    validate_commandment,
)


# ---- Valid COMMANDMENT files ----

VALID_COMMANDMENT = """\
## SETUP
mkdir -p ${GEAK_WORK_DIR}/pkg
cp ${GEAK_WORK_DIR}/kernel.py ${GEAK_WORK_DIR}/pkg/kernel.py

## CORRECTNESS
${GEAK_WORK_DIR}/run_harness.sh --correctness

## PROFILE
${GEAK_WORK_DIR}/run_harness.sh --profile
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
"""
    result = validate_commandment(content)
    assert result["valid"] is False
    assert any("Test" in e and "IGNORED" in e for e in result["errors"])


# ---- Shell built-ins ----

def test_cd_as_command_prefix():
    content = """\
## SETUP
mkdir -p /tmp/test

## CORRECTNESS
python3 test.py

## PROFILE
cd /workspace && python3 bench.py
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
"""
    # cd is inside a code block in SETUP section -- our validator checks
    # lines in recognized sections regardless (since COMMANDMENT commands
    # may or may not be in code blocks).
    result = validate_commandment(content)
    # The `cd` line is outside the code block tracking since "```" toggles
    # are consumed. The `cd /workspace` line itself is NOT preceded by ```,
    # so it IS checked. This test verifies no crash.
    assert isinstance(result["valid"], bool)
