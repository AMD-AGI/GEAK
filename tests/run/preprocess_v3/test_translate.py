"""Tests for ``minisweagent.run.preprocess_v3.translate``.

The wrapper has no logic of its own beyond projection + timing; the tests
mock :func:`run_translation` to inject canonical success/failure dicts and
assert the projection contract.

The "regression guard" test asserts the wrapper passes
``target_language="flydsl"`` exactly — this is the contract that lets the
v3 orchestrator hard-code FlyDSL routing without worrying about misrouted
translation pairs.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from minisweagent.run.preprocess_v3 import translate as translate_mod
from minisweagent.run.preprocess_v3.translate import (
    TARGET_LANGUAGE,
    TranslationResult,
    translate_to_flydsl,
)


def _success_dict(translated_path: str = "/tmp/out/foo.flydsl.py") -> dict:
    """Canonical success-shape dict produced by ``run_translation``."""
    return {
        "translation_success": True,
        "translation_source_language": "pytorch",
        "translation_target_language": "flydsl",
        "translation_kernel_path": translated_path,
        "translation_best_attempt_path": None,
        "translation_rounds_used": 2,
        "translation_pytorch_latency_ms": 5.4,
        "translation_flydsl_latency_ms": 2.1,
        "translation_speedup": 2.57,
        "translation_self_review": "passed",
        "translation_errors": [],
        "translation_elapsed_s": 12.3,
    }


def _failure_dict() -> dict:
    """Canonical failure-shape dict (e.g. exhausted retries)."""
    return {
        "translation_success": False,
        "translation_source_language": "pytorch",
        "translation_target_language": "flydsl",
        "translation_kernel_path": None,
        "translation_best_attempt_path": "/tmp/out/best_attempt.flydsl.py",
        "translation_rounds_used": 3,
        "translation_pytorch_latency_ms": None,
        "translation_flydsl_latency_ms": None,
        "translation_errors": [
            "Round 1: correctness failed",
            "Round 2: performance regression",
            "Round 3: review found REPLACE issues",
        ],
    }


# ---------------------------------------------------------------------------
# success-path projection
# ---------------------------------------------------------------------------


def test_success_path_projection_promotes_every_field(tmp_path: Path) -> None:
    """Every promoted field is read from the legacy dict and typed correctly."""
    raw = _success_dict(str(tmp_path / "kernel.flydsl.py"))

    with mock.patch.object(translate_mod, "run_translation", return_value=raw) as mocked:
        result = translate_to_flydsl(
            source_path=tmp_path / "kernel.py",
            output_dir=tmp_path / "out",
        )

    mocked.assert_called_once()
    assert isinstance(result, TranslationResult)
    assert result.success is True
    assert result.target_language == "flydsl"
    assert result.translated_kernel_path == Path(raw["translation_kernel_path"])
    assert result.speedup == pytest.approx(2.57)
    assert result.self_review == "passed"
    assert result.errors == []
    assert result.elapsed_s >= 0.0
    # The full legacy dict survives on .raw so callers can get at non-promoted keys.
    assert result.raw["translation_pytorch_latency_ms"] == pytest.approx(5.4)
    assert result.raw["translation_rounds_used"] == 2


def test_success_path_projection_is_immutable(tmp_path: Path) -> None:
    """``TranslationResult`` is frozen — the orchestrator can pass it around safely."""
    raw = _success_dict(str(tmp_path / "kernel.flydsl.py"))

    with mock.patch.object(translate_mod, "run_translation", return_value=raw):
        result = translate_to_flydsl(
            source_path=tmp_path / "kernel.py",
            output_dir=tmp_path / "out",
        )

    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        result.success = False  # type: ignore[misc]


def test_success_path_raw_is_decoupled(tmp_path: Path) -> None:
    """Mutating ``.raw`` after the fact must not back-mutate the underlying dict.

    The wrapper copies the raw dict into the dataclass so a downstream
    consumer that mutates ``.raw`` (e.g. to redact secrets before logging)
    can't leak into another caller's copy.
    """
    raw = _success_dict(str(tmp_path / "kernel.flydsl.py"))

    with mock.patch.object(translate_mod, "run_translation", return_value=raw):
        result = translate_to_flydsl(
            source_path=tmp_path / "kernel.py",
            output_dir=tmp_path / "out",
        )

    result.raw["translation_rounds_used"] = 999
    assert raw["translation_rounds_used"] == 2


# ---------------------------------------------------------------------------
# failure-path projection
# ---------------------------------------------------------------------------


def test_failure_path_projection(tmp_path: Path) -> None:
    """Failure dicts project with success=False, populated errors, no path."""
    raw = _failure_dict()

    with mock.patch.object(translate_mod, "run_translation", return_value=raw):
        result = translate_to_flydsl(
            source_path=tmp_path / "kernel.py",
            output_dir=tmp_path / "out",
        )

    assert result.success is False
    assert result.target_language == "flydsl"
    assert result.translated_kernel_path is None
    assert result.speedup is None
    assert result.self_review == ""
    assert len(result.errors) == 3
    assert "correctness failed" in result.errors[0]
    # raw should still be present so the caller can recover the
    # best_attempt_path the projection doesn't promote.
    assert result.raw["translation_best_attempt_path"] == "/tmp/out/best_attempt.flydsl.py"


def test_missing_self_review_defaults_to_empty(tmp_path: Path) -> None:
    """The legacy dict can omit ``translation_self_review`` entirely.

    The success-path keeps the field optional in the legacy code (it's only
    written when self-review actually runs). The wrapper must default to
    an empty string rather than ``None`` so consumers can use it as a
    plain ``str``.
    """
    raw = _success_dict()
    raw.pop("translation_self_review")

    with mock.patch.object(translate_mod, "run_translation", return_value=raw):
        result = translate_to_flydsl(
            source_path=tmp_path / "kernel.py",
            output_dir=tmp_path / "out",
        )

    assert result.self_review == ""


# ---------------------------------------------------------------------------
# regression guard: target_language pinning
# ---------------------------------------------------------------------------


def test_wrapper_pins_target_language_to_flydsl(tmp_path: Path) -> None:
    """The wrapper must always forward ``target_language="flydsl"``.

    This is the contract that lets the v3 orchestrator skip translation
    pair detection. If a future refactor accidentally drops the kwarg or
    routes another language through this wrapper, the legacy
    ``run_translation`` will silently re-detect from the file extension
    and may translate to the wrong target.
    """
    raw = _success_dict()

    with mock.patch.object(translate_mod, "run_translation", return_value=raw) as mocked:
        translate_to_flydsl(
            source_path=tmp_path / "kernel.py",
            output_dir=tmp_path / "out",
        )

    _, kwargs = mocked.call_args
    assert kwargs.get("target_language") == "flydsl"
    assert kwargs.get("target_language") == TARGET_LANGUAGE


def test_wrapper_forwards_optional_kwargs(tmp_path: Path) -> None:
    """``model``, ``model_factory``, ``repo``, ``flydsl_repo``, ``console`` reach the legacy call."""
    raw = _success_dict()
    sentinel_model = object()
    sentinel_factory = object()
    sentinel_console = object()
    repo = tmp_path / "repo"
    flydsl_repo = tmp_path / "flydsl"

    with mock.patch.object(translate_mod, "run_translation", return_value=raw) as mocked:
        translate_to_flydsl(
            source_path=tmp_path / "kernel.py",
            output_dir=tmp_path / "out",
            gpu_id=2,
            model=sentinel_model,
            model_factory=sentinel_factory,
            repo=repo,
            flydsl_repo=flydsl_repo,
            console=sentinel_console,
        )

    _, kwargs = mocked.call_args
    assert kwargs["gpu_id"] == 2
    assert kwargs["model"] is sentinel_model
    assert kwargs["model_factory"] is sentinel_factory
    assert kwargs["repo"] == repo
    assert kwargs["flydsl_repo"] == flydsl_repo
    assert kwargs["console"] is sentinel_console


def test_non_dict_return_raises_typeerror(tmp_path: Path) -> None:
    """If the legacy contract is ever violated, surface it loudly."""
    with mock.patch.object(translate_mod, "run_translation", return_value="not a dict"):
        with pytest.raises(TypeError, match="legacy contract violated"):
            translate_to_flydsl(
                source_path=tmp_path / "kernel.py",
                output_dir=tmp_path / "out",
            )
