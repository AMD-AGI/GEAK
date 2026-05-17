"""Tests for ``minisweagent.run.preprocess_v3.commandment``.

Cover both render paths:

* **Jinja path** — synthesize a tiny Jinja template under
  ``tmp_path``, build a fresh ``KernelLanguage`` pointing at it
  (the canonical ``KernelLanguage`` is frozen, so we construct a
  new one rather than mutating the registered Triton/HIP entries),
  and assert the substitution worked.

* **Legacy fallback path** — drive a ``KernelLanguage`` with
  ``commandment_template_path=None`` and assert the legacy Python
  generators produce a valid 5-section commandment via the
  harness-path and eval-command flows.

We never spawn a subprocess and we never call out to an LLM —
``render_commandment`` is pure rendering plus optional file write.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minisweagent.kernel_languages.base import KernelLanguage
from minisweagent.run.preprocess_v3.commandment import (
    CommandmentContext,
    render_commandment,
)


def _make_lang_without_template(name: str = "fakeling") -> KernelLanguage:
    """Build a minimal ``KernelLanguage`` with no Jinja template path.

    Drives :func:`render_commandment` straight to the legacy Python
    fallback. Frozen-dataclass constructor; we can't mutate an
    existing registry entry without unfreezing it.
    """
    return KernelLanguage(
        name=name,
        file_extensions=frozenset({".py"}),
        kb_namespace=name,
        commandment_template_path=None,
    )


def _make_lang_with_template(template_path: Path, *, name: str = "fakeling-jinja") -> KernelLanguage:
    return KernelLanguage(
        name=name,
        file_extensions=frozenset({".py"}),
        kb_namespace=name,
        commandment_template_path=template_path,
    )


def _make_kernel_and_harness(tmp_path: Path) -> tuple[Path, Path]:
    """Build the minimal on-disk artifacts the legacy generator validates."""
    kernel = tmp_path / "kernel.py"
    kernel.write_text("# kernel\n", encoding="utf-8")
    harness = tmp_path / "test_harness.py"
    # Legacy ``validate_commandment`` does NOT inspect harness contents
    # by default, but ``_validate_and_fix`` does check section headers,
    # PYTHONPATH, and shell-builtin usage in the rendered commandment.
    harness.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--correctness', action='store_true')\n"
        "p.add_argument('--profile', action='store_true')\n"
        "p.add_argument('--benchmark', action='store_true')\n"
        "p.add_argument('--full-benchmark', action='store_true')\n"
        "p.add_argument('--iterations', type=int, default=1)\n",
        encoding="utf-8",
    )
    return kernel, harness


# ---------------------------------------------------------------------------
# Jinja path
# ---------------------------------------------------------------------------


_SYNTHETIC_TEMPLATE = """\
## Setup
KERNEL={{ kernel_path }}
HARNESS={{ harness_path }}
REPLAYS={{ profile_replays }}
INNER={{ inner_kernel }}

## Correctness
{{ correctness_command or 'python3 ' + harness_path + ' --correctness' }}

## Benchmark
{{ performance_command or 'python3 ' + harness_path + ' --benchmark' }}

## Full Benchmark
{{ performance_command or 'python3 ' + harness_path + ' --full-benchmark' }}

## Profile
kernel-profile "python3 {{ harness_path }} --profile"
"""


def test_jinja_render_substitutes_template_variables(tmp_path: Path) -> None:
    """Render via Jinja, asserting the template substitutions all hit."""
    template_path = tmp_path / "commandment.j2"
    template_path.write_text(_SYNTHETIC_TEMPLATE, encoding="utf-8")
    lang = _make_lang_with_template(template_path)
    kernel, harness = _make_kernel_and_harness(tmp_path)

    text = render_commandment(
        lang,
        CommandmentContext(
            kernel_path=kernel,
            harness_path=harness,
            repo_root=tmp_path,
            profile_replays=7,
        ),
    )

    assert f"KERNEL={kernel}" in text
    assert f"HARNESS={harness}" in text
    assert "REPLAYS=7" in text
    assert "INNER=False" in text
    # All five sections must be present in the rendered output.
    for section in ("## Setup", "## Correctness", "## Benchmark", "## Full Benchmark", "## Profile"):
        assert section in text


def test_jinja_render_uses_correctness_and_performance_commands(tmp_path: Path) -> None:
    """Caller-supplied commands replace the ``or`` defaults inside the template."""
    template_path = tmp_path / "commandment.j2"
    template_path.write_text(_SYNTHETIC_TEMPLATE, encoding="utf-8")
    lang = _make_lang_with_template(template_path)
    kernel, harness = _make_kernel_and_harness(tmp_path)

    text = render_commandment(
        lang,
        CommandmentContext(
            kernel_path=kernel,
            harness_path=harness,
            correctness_command="my_check.sh",
            performance_command=["build && run", "show_metrics"],
        ),
    )

    assert "my_check.sh" in text
    # List form is joined with " && ".
    assert "build && run && show_metrics" in text


def test_jinja_render_falls_back_when_template_path_does_not_exist(tmp_path: Path) -> None:
    """Pointing at a non-existent template shouldn't break rendering."""
    bogus_template = tmp_path / "missing.j2"
    lang = _make_lang_with_template(bogus_template)
    kernel, harness = _make_kernel_and_harness(tmp_path)

    # Should silently fall back to the legacy generator, which
    # produces the legacy uppercase section headers.
    text = render_commandment(
        lang,
        CommandmentContext(kernel_path=kernel, harness_path=harness, repo_root=tmp_path),
    )

    assert "## SETUP" in text
    assert "## CORRECTNESS" in text


# ---------------------------------------------------------------------------
# Legacy fallback path
# ---------------------------------------------------------------------------


def test_legacy_fallback_renders_with_harness_path(tmp_path: Path) -> None:
    """Language without a Jinja template -> legacy harness generator."""
    lang = _make_lang_without_template()
    kernel, harness = _make_kernel_and_harness(tmp_path)

    text = render_commandment(
        lang,
        CommandmentContext(kernel_path=kernel, harness_path=harness, repo_root=tmp_path),
    )

    # Legacy generator emits uppercase section headers.
    for section in ("## SETUP", "## CORRECTNESS", "## PROFILE", "## BENCHMARK", "## FULL_BENCHMARK"):
        assert section in text
    # ``run.sh`` is the legacy generator's ``SETUP`` PYTHONPATH wrapper.
    assert "run.sh" in text
    assert str(harness) in text


def test_legacy_fallback_renders_with_eval_commands(tmp_path: Path) -> None:
    """Eval-command flow: explicit perf/correctness commands."""
    lang = _make_lang_without_template()
    kernel, _ = _make_kernel_and_harness(tmp_path)

    text = render_commandment(
        lang,
        CommandmentContext(
            kernel_path=kernel,
            repo_root=tmp_path,
            correctness_command="ls -la",
            performance_command="echo benchmark_result",
        ),
    )

    assert "## SETUP" in text
    assert "echo benchmark_result" in text


def test_legacy_fallback_raises_when_no_input_to_render(tmp_path: Path) -> None:
    """Without a harness or any command, the legacy fallback can't synthesise."""
    lang = _make_lang_without_template()
    kernel = tmp_path / "kernel.py"
    kernel.write_text("# kernel\n", encoding="utf-8")

    with pytest.raises(ValueError, match="legacy fallback needs"):
        render_commandment(
            lang,
            CommandmentContext(kernel_path=kernel, repo_root=tmp_path),
        )


# ---------------------------------------------------------------------------
# out_path writing
# ---------------------------------------------------------------------------


def test_render_commandment_writes_out_path(tmp_path: Path) -> None:
    """When ``out_path`` is given, the rendered text is written there."""
    template_path = tmp_path / "commandment.j2"
    template_path.write_text(_SYNTHETIC_TEMPLATE, encoding="utf-8")
    lang = _make_lang_with_template(template_path)
    kernel, harness = _make_kernel_and_harness(tmp_path)
    out_path = tmp_path / "out" / "COMMANDMENT.md"

    text = render_commandment(
        lang,
        CommandmentContext(kernel_path=kernel, harness_path=harness),
        out_path=out_path,
    )

    assert out_path.is_file()
    assert out_path.read_text(encoding="utf-8") == text


def test_render_commandment_out_path_is_idempotent(tmp_path: Path) -> None:
    """Re-rendering with the same inputs produces identical content."""
    template_path = tmp_path / "commandment.j2"
    template_path.write_text(_SYNTHETIC_TEMPLATE, encoding="utf-8")
    lang = _make_lang_with_template(template_path)
    kernel, harness = _make_kernel_and_harness(tmp_path)
    out_path = tmp_path / "out" / "COMMANDMENT.md"

    text_a = render_commandment(
        lang,
        CommandmentContext(kernel_path=kernel, harness_path=harness),
        out_path=out_path,
    )
    text_b = render_commandment(
        lang,
        CommandmentContext(kernel_path=kernel, harness_path=harness),
        out_path=out_path,
    )

    assert text_a == text_b
    assert out_path.read_text(encoding="utf-8") == text_b


# ---------------------------------------------------------------------------
# Real-language sanity checks (use the registered Triton template if present)
# ---------------------------------------------------------------------------


def test_render_commandment_with_registered_triton_language(tmp_path: Path) -> None:
    """Sanity check against the real Triton template ships in-tree."""
    from minisweagent.kernel_languages import registry

    triton = registry.get("triton")
    if triton is None or triton.commandment_template_path is None:
        pytest.skip("Triton language bundle / template not available")

    kernel, harness = _make_kernel_and_harness(tmp_path)
    text = render_commandment(
        triton,
        CommandmentContext(kernel_path=kernel, harness_path=harness, repo_root=tmp_path),
    )

    assert "## Setup" in text
    assert "## Correctness" in text
    assert str(harness) in text
