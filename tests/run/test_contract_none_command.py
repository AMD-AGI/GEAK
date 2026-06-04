"""Regression tests for ``validate_commandment``'s bare-``None`` command guard.

A Jinja ``{{ cmd | default(...) }}`` where ``cmd`` is Python ``None`` (the
single-arg ``default`` filter does not fire on a defined-but-None value)
renders the literal string ``None``. The resulting COMMANDMENT runs
``... && None`` and fails at runtime with rc=127 ("None: command not found").
``validate_commandment`` now rejects such a bare ``None`` command token at
generation time so the failure is caught loudly instead of deep in a run.
"""

from __future__ import annotations

import pytest

from minisweagent.kernel_languages.contract import ContractViolation, validate_commandment

_VALID_COMMANDMENT = """# Commandment

## Setup

```bash
export GEAK_WORK_DIR="${GEAK_WORK_DIR:-/x}"
```

## Correctness

```bash
cd "${GEAK_WORK_DIR}" && python3 harness.py --correctness
```

## Benchmark

```bash
cd "${GEAK_WORK_DIR}" && python3 harness.py --benchmark
```

## Full Benchmark

```bash
cd "${GEAK_WORK_DIR}" && python3 harness.py --full-benchmark
```

## Profile

```bash
kernel-profile "cd ${GEAK_WORK_DIR} && python3 harness.py --profile" --replays 3
```
"""


def test_validate_commandment_rejects_bare_none(tmp_path):
    leaked = _VALID_COMMANDMENT.replace("python3 harness.py --correctness", "None")
    path = tmp_path / "COMMANDMENT.md"
    path.write_text(leaked, encoding="utf-8")
    with pytest.raises(ContractViolation):
        validate_commandment(path)


def test_validate_commandment_accepts_valid(tmp_path):
    path = tmp_path / "COMMANDMENT.md"
    path.write_text(_VALID_COMMANDMENT, encoding="utf-8")
    # Should not raise — no bare ``None`` command token, all sections present.
    validate_commandment(path)


def test_validate_commandment_ignores_none_in_arg_value_context(tmp_path):
    # ``None`` embedded in a token (not a standalone command word) must NOT trip
    # the guard — e.g. a filename or assignment, which are legitimate.
    safe = _VALID_COMMANDMENT.replace(
        "python3 harness.py --correctness",
        "python3 harness.py --correctness --tag None_baseline",
    )
    path = tmp_path / "COMMANDMENT.md"
    path.write_text(safe, encoding="utf-8")
    validate_commandment(path)  # ``None_baseline`` is not a bare ``None`` token
