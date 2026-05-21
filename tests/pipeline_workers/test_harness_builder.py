"""Tests for Workstream D1 — HarnessBuilder subagent.

Pins:
  - Prompt composition includes the universal contract + Jinja skeleton
    + builder hints + kernel source + optional user tests + retry feedback
  - Success path: validate_harness passes -> ok=True
  - Retry path: first candidate fails, second passes -> attempts=2
  - Failure path: all attempts fail -> HarnessBuildFailed + rejected file
  - Code-fence stripping
  - Strict contract (partial failures trigger retries)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from minisweagent.pipeline_workers.base import SubagentConfig
from minisweagent.pipeline_workers.preprocess.harness_builder import (
    HarnessBuildFailed,
    HarnessBuilder,
)


def _valid_harness_source() -> str:
    """Return a minimal harness that satisfies the universal contract."""
    return """\
import argparse

def main():
    p = argparse.ArgumentParser()
    mutex = p.add_mutually_exclusive_group(required=True)
    mutex.add_argument("--correctness", action="store_true")
    mutex.add_argument("--benchmark", action="store_true")
    mutex.add_argument("--full-benchmark", action="store_true")
    mutex.add_argument("--profile", action="store_true")
    args = p.parse_args()
    if args.correctness:
        print("OK")
    elif args.benchmark:
        print(f"GEAK_RESULT_LATENCY_MS={1.0}")
    elif args.full_benchmark:
        print(f"GEAK_RESULT_LATENCY_MS={1.0}")
        print(f"GEAK_RESULT_SPEEDUP={1.0}")

main()
"""


def _invalid_harness_source() -> str:
    """Return a harness missing BOTH flags and markers (contract fails)."""
    return "def main(): pass\nmain()\n"


def _partial_harness_source() -> str:
    """Missing two flags — triggers retry under strict mode.

    The missing flag substrings must not appear ANYWHERE in the file
    (including comments) or the substring-based validator would false-
    accept.
    """
    return """\
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--correctness", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    print(f"GEAK_RESULT_LATENCY_MS={1.0}")
    print(f"GEAK_RESULT_SPEEDUP={1.0}")

main()
"""


def _fake_language(
    tmp_path: Path,
    *,
    with_template: bool = True,
    with_hints: bool = True,
    with_system_prompt: bool = True,
):
    """Build a MagicMock KernelLanguage-like object.

    Placing content files under ``tmp_path`` so tests are hermetic.
    """
    lang = MagicMock()
    lang.name = "triton"

    if with_template:
        template = tmp_path / "harness.j2"
        template.write_text(
            "# triton harness jinja skeleton\n"
            "# flags: --correctness / --benchmark / --full-benchmark / --profile\n"
        )
        lang.harness_template_path = template
        lang.harness_template = template.read_text()
    else:
        lang.harness_template_path = None
        lang.harness_template = ""

    if with_hints:
        hints = tmp_path / "builder_hints.md"
        hints.write_text("# triton builder hints\nUse @triton.jit entry points.\n")
        lang.builder_hints_path = hints
        lang.builder_hints = hints.read_text()
    else:
        lang.builder_hints = ""

    if with_system_prompt:
        sys_prompt = tmp_path / "system_prompt.md"
        sys_prompt.write_text("You are the triton worker agent.\n")
        lang.system_prompt_path = sys_prompt
        lang.system_prompt = sys_prompt.read_text()
    else:
        lang.system_prompt = ""

    return lang


def _make_builder(
    tmp_path: Path,
    *,
    model_responses: list[str],
    max_retries: int = 1,
    **lang_kwargs,
) -> HarnessBuilder:
    """Build a HarnessBuilder whose .model returns responses in order."""
    lang = _fake_language(tmp_path, **lang_kwargs)
    config = SubagentConfig(
        name="harness_builder",
        model_name="fake-model",
        system_template="sys {language_name}",
        instance_template="inst {language_name}",
        step_limit=1,
        cost_limit=3.0,
        temperature=0.2,
        extra={"max_retries": max_retries},
    )
    builder = HarnessBuilder(language=lang, config=config)
    model = MagicMock()
    responses = iter(model_responses)

    def _query(messages):
        try:
            return next(responses)
        except StopIteration:
            raise RuntimeError("model queried more times than test provided responses")

    model.query = _query
    builder.model = model  # type: ignore[attr-defined]
    return builder


# ──────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────


class TestSuccessOnFirstAttempt:
    def test_valid_harness_lands_as_ok(self, tmp_path: Path) -> None:
        kernel = tmp_path / "kernel.py"
        kernel.write_text("@triton.jit\ndef my_kernel(): pass\n")
        out = tmp_path / "harness.py"

        builder = _make_builder(tmp_path, model_responses=[_valid_harness_source()])
        result = builder.run(kernel_path=kernel, out_path=out)

        assert isinstance(result, dict)
        assert result["ok"] is True
        assert result["attempts_used"] == 1
        assert result["harness_path"] == str(out)
        # File was actually written
        assert out.exists()
        content = out.read_text()
        for flag in ("--correctness", "--benchmark", "--full-benchmark", "--profile"):
            assert flag in content
        for marker in ("GEAK_RESULT_LATENCY_MS", "GEAK_RESULT_SPEEDUP"):
            assert marker in content


class TestStripsCodeFences:
    def test_leading_markdown_fence_removed(self, tmp_path: Path) -> None:
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"

        fenced = "```python\n" + _valid_harness_source() + "```\n"
        builder = _make_builder(tmp_path, model_responses=[fenced])
        result = builder.run(kernel_path=kernel, out_path=out)
        assert result["ok"] is True
        # The written file should not start with ``` anymore
        written = out.read_text()
        assert not written.strip().startswith("```")

    def test_prose_preamble_with_fenced_block_extracted(self, tmp_path: Path) -> None:
        """The exact failure mode observed in production: LLM emits a chatty
        preamble ("I'll start by understanding...") followed by the actual
        code in a ```python ...``` block.  The extractor MUST pick the
        fenced block, NOT the preamble."""
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"

        prose = (
            "I'll start by understanding the existing code, harness, and "
            "profile data to build a proper harness and optimize the kernel.\n"
            "\n"
            "```python\n"
            + _valid_harness_source()
            + "```\n"
            "\n"
            "That should satisfy the universal contract.\n"
        )
        builder = _make_builder(tmp_path, model_responses=[prose])
        result = builder.run(kernel_path=kernel, out_path=out)
        assert result["ok"] is True
        written = out.read_text()
        assert "I'll start by understanding" not in written
        assert "That should satisfy" not in written
        for flag in ("--correctness", "--benchmark", "--full-benchmark", "--profile"):
            assert flag in written

    def test_largest_fenced_block_wins_when_multiple(self, tmp_path: Path) -> None:
        """LLMs sometimes embed small bash snippets ("here's how to run it")
        next to the harness block.  We must pick the largest block, which
        is the harness."""
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"

        response = (
            "First, here's how you'd run it:\n"
            "```bash\n"
            "python harness.py --correctness\n"
            "```\n"
            "\n"
            "And here is the harness:\n"
            "```python\n"
            + _valid_harness_source()
            + "```\n"
        )
        builder = _make_builder(tmp_path, model_responses=[response])
        result = builder.run(kernel_path=kernel, out_path=out)
        assert result["ok"] is True

    def test_pure_prose_response_passes_through_to_validator(
        self, tmp_path: Path
    ) -> None:
        """When the LLM returns nothing but prose (no fence, no code-looking
        line), we hand the prose to the validator unchanged so it can
        report the correct error (missing flags/markers)."""
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"

        prose_only = (
            "I'll start by understanding the existing code, harness, and "
            "profile data to build a proper harness and optimize the kernel.\n"
        )
        builder = _make_builder(
            tmp_path,
            model_responses=[prose_only, prose_only],
            max_retries=1,
        )
        with pytest.raises(HarnessBuildFailed) as exc:
            builder.run(kernel_path=kernel, out_path=out)
        assert "missing" in str(exc.value)


# ──────────────────────────────────────────────────────────────────────
# Wallclock-bounded loop (architectural plan §0.5(b))
# ──────────────────────────────────────────────────────────────────────


class TestSeedHarness:
    """When Layer 2 hands off a user's non-compliant harness as a seed,
    HarnessBuilder must include the seed text in every prompt so the
    LLM iterates on it instead of regenerating from scratch."""

    def test_seed_source_appears_in_prompt(self, tmp_path: Path) -> None:
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"

        seed_src = (
            "# User's partial harness — missing --profile mode\n"
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--correctness', action='store_true')\n"
        )

        captured_prompts: list[str] = []
        builder = _make_builder(
            tmp_path,
            model_responses=[_valid_harness_source()],
        )

        def _q(messages):
            captured_prompts.append(messages[1]["content"])
            return _valid_harness_source()

        builder.model.query = _q  # type: ignore[attr-defined]
        result = builder.run(
            kernel_path=kernel,
            out_path=out,
            seed_harness_source=seed_src,
        )
        assert result["ok"] is True
        assert "STARTING HARNESS (user-provided" in captured_prompts[0]
        # Seed contents are embedded verbatim
        assert "User's partial harness" in captured_prompts[0]
        assert "--correctness" in captured_prompts[0]

    def test_seed_path_is_read_and_injected(self, tmp_path: Path) -> None:
        """Phase layer passes the seed as a Path; builder must read it
        and thread the contents into the prompt."""
        kernel = tmp_path / "k.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"
        seed_file = tmp_path / "user_harness.py"
        seed_file.write_text("# unique marker: user_wrote_this_harness\n")

        captured_prompts: list[str] = []
        builder = _make_builder(
            tmp_path,
            model_responses=[_valid_harness_source()],
        )

        def _q(messages):
            captured_prompts.append(messages[1]["content"])
            return _valid_harness_source()

        builder.model.query = _q  # type: ignore[attr-defined]
        result = builder.run(
            kernel_path=kernel,
            out_path=out,
            seed_harness_path=seed_file,
        )
        assert result["ok"] is True
        assert "user_wrote_this_harness" in captured_prompts[0]


class TestWallclockBoundedLoop:
    def test_loop_terminates_when_budget_exhausted(self, tmp_path: Path) -> None:
        """When validation never passes, the loop must terminate cleanly
        once the wallclock budget is exhausted (no infinite loop)."""
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"

        lang = _fake_language(tmp_path)
        config = SubagentConfig(
            name="harness_builder",
            model_name="fake",
            system_template="sys {language_name}",
            instance_template="inst {language_name}",
            step_limit=1,
            cost_limit=3.0,
            temperature=0.2,
            extra={},
        )
        builder = HarnessBuilder(language=lang, config=config)

        # Always return invalid; loop should rely on wallclock to
        # terminate.  We use a tiny budget (50 ms) so the test is fast,
        # and a model that takes ~30 ms per call so we get exactly 1-2
        # attempts before the deadline.
        import time as _time

        def _slow_q(messages):
            _time.sleep(0.03)
            return _invalid_harness_source()

        builder.model = MagicMock(query=_slow_q)  # type: ignore[attr-defined]

        with pytest.raises(HarnessBuildFailed):
            builder.run(
                kernel_path=kernel,
                out_path=out,
                max_wallclock_seconds=0.05,
            )
        # Rejected file written
        rejected = out.with_suffix(out.suffix + ".rejected")
        assert rejected.exists()

    def test_loop_succeeds_within_budget(self, tmp_path: Path) -> None:
        """When the LLM produces a valid harness within the budget, the
        loop returns ok=True even if max_retries is unbounded."""
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"

        lang = _fake_language(tmp_path)
        config = SubagentConfig(
            name="harness_builder",
            model_name="fake",
            system_template="sys {language_name}",
            instance_template="inst {language_name}",
            step_limit=1,
            cost_limit=3.0,
            temperature=0.2,
            extra={},
        )
        builder = HarnessBuilder(language=lang, config=config)
        responses = iter([_invalid_harness_source(), _valid_harness_source()])
        builder.model = MagicMock(query=lambda m: next(responses))  # type: ignore[attr-defined]

        result = builder.run(
            kernel_path=kernel,
            out_path=out,
            max_wallclock_seconds=30.0,  # 30 s; plenty
        )
        assert result["ok"] is True
        assert result["attempts_used"] == 2

    def test_max_retries_cap_still_honoured(self, tmp_path: Path) -> None:
        """``max_retries`` remains a deterministic safety cap for tests
        that need predictable attempt counts even under a large
        wallclock budget."""
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"

        builder = _make_builder(
            tmp_path,
            model_responses=[_invalid_harness_source(), _invalid_harness_source()],
            max_retries=1,
        )
        with pytest.raises(HarnessBuildFailed):
            builder.run(
                kernel_path=kernel,
                out_path=out,
                max_wallclock_seconds=300.0,
                max_retries=1,
            )
        # ``_make_builder`` only provided 2 responses; if the cap had not
        # fired we would have hit StopIteration instead of HarnessBuildFailed.


# ──────────────────────────────────────────────────────────────────────
# Retry path
# ──────────────────────────────────────────────────────────────────────


class TestRetryOnContractFailure:
    def test_first_attempt_fails_second_passes(self, tmp_path: Path) -> None:
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"

        builder = _make_builder(
            tmp_path,
            model_responses=[_invalid_harness_source(), _valid_harness_source()],
            max_retries=1,
        )
        result = builder.run(kernel_path=kernel, out_path=out)
        assert result["ok"] is True
        assert result["attempts_used"] == 2

    def test_strict_mode_catches_partial_violations(self, tmp_path: Path) -> None:
        """Partial flag coverage (missing --full-benchmark / --profile)
        must NOT pass even though stdout markers are present."""
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"

        # Only the partial candidate; should exhaust retries and fail.
        builder = _make_builder(
            tmp_path,
            model_responses=[_partial_harness_source(), _partial_harness_source()],
            max_retries=1,
        )
        with pytest.raises(HarnessBuildFailed) as exc:
            builder.run(kernel_path=kernel, out_path=out)
        msg = str(exc.value)
        assert "missing required argparse flags" in msg or "missing" in msg

    def test_retry_feedback_injected_into_next_prompt(self, tmp_path: Path) -> None:
        """When first attempt fails, the second attempt's prompt should
        carry the validator's error strings so the LLM can fix them."""
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"

        captured_prompts: list[tuple[str, str]] = []

        lang = _fake_language(tmp_path)
        config = SubagentConfig(
            name="harness_builder",
            model_name="fake",
            system_template="sys {language_name}",
            instance_template="inst {language_name}",
            step_limit=1,
            cost_limit=3.0,
            temperature=0.2,
            extra={"max_retries": 1},
        )
        builder = HarnessBuilder(language=lang, config=config)

        responses = iter([_invalid_harness_source(), _valid_harness_source()])

        def _q(messages):
            sys_msg = next(m["content"] for m in messages if m["role"] == "system")
            inst_msg = next(m["content"] for m in messages if m["role"] == "user")
            captured_prompts.append((sys_msg, inst_msg))
            return next(responses)

        builder.model = MagicMock(query=_q)  # type: ignore[attr-defined]

        result = builder.run(kernel_path=kernel, out_path=out)
        assert result["ok"] is True
        assert len(captured_prompts) == 2

        # Second prompt contains the feedback header + some validator-
        # produced error surface.  The exact string depends on which
        # validator fired (ContractViolation from
        # ``kernel_languages.contract.validate_harness`` when BOTH
        # flags AND markers are missing, vs our strict-mode ``missing
        # required argparse flags`` / ``missing required stdout
        # markers`` when only one category is violated).
        second_inst = captured_prompts[1][1]
        assert "PREVIOUS ATTEMPT" in second_inst
        assert "FAILED THE UNIVERSAL CONTRACT" in second_inst
        assert any(
            phrase in second_inst
            for phrase in (
                "missing required argparse flags",
                "missing required stdout markers",
                "missing required flags",
                "missing required markers",
            )
        )


# ──────────────────────────────────────────────────────────────────────
# Failure path
# ──────────────────────────────────────────────────────────────────────


class TestAllAttemptsFail:
    def test_exhausts_retries_raises_and_writes_rejected_file(
        self, tmp_path: Path
    ) -> None:
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"

        builder = _make_builder(
            tmp_path,
            model_responses=[_invalid_harness_source(), _invalid_harness_source()],
            max_retries=1,
        )
        with pytest.raises(HarnessBuildFailed):
            builder.run(kernel_path=kernel, out_path=out)

        # The rejected candidate is persisted next to the output path
        rejected = out.with_suffix(out.suffix + ".rejected")
        assert rejected.exists()


# ──────────────────────────────────────────────────────────────────────
# Required input validation
# ──────────────────────────────────────────────────────────────────────


class TestInputValidation:
    def test_missing_kernel_path_raises(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path, model_responses=[])
        with pytest.raises(ValueError, match="kernel_path"):
            builder.run(out_path=tmp_path / "harness.py")

    def test_missing_out_path_raises(self, tmp_path: Path) -> None:
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        builder = _make_builder(tmp_path, model_responses=[])
        with pytest.raises(ValueError, match="out_path"):
            builder.run(kernel_path=kernel)

    def test_nonexistent_kernel_raises(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path, model_responses=[])
        with pytest.raises(FileNotFoundError):
            builder.run(
                kernel_path=tmp_path / "does_not_exist.py",
                out_path=tmp_path / "harness.py",
            )


# ──────────────────────────────────────────────────────────────────────
# Prompt composition
# ──────────────────────────────────────────────────────────────────────


class TestPromptComposition:
    def test_universal_contract_always_in_prompt(self, tmp_path: Path) -> None:
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"

        captured: list[str] = []
        lang = _fake_language(tmp_path)
        config = SubagentConfig(
            name="hb",
            model_name="x",
            system_template="",
            instance_template="",
            step_limit=1,
            cost_limit=3.0,
            temperature=0.2,
            extra={"max_retries": 0},
        )
        builder = HarnessBuilder(language=lang, config=config)

        def _q(messages):
            captured.append(messages[1]["content"])
            return _valid_harness_source()

        builder.model = MagicMock(query=_q)  # type: ignore[attr-defined]
        builder.run(kernel_path=kernel, out_path=out)

        assert len(captured) == 1
        inst = captured[0]
        assert "UNIVERSAL HARNESS CONTRACT" in inst
        # All 4 contract flags are namechecked
        for flag in ("--correctness", "--benchmark", "--full-benchmark", "--profile"):
            assert flag in inst
        # The kernel source is embedded verbatim
        assert "KERNEL SOURCE" in inst

    def test_user_test_files_included_when_provided(self, tmp_path: Path) -> None:
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"
        user_test = tmp_path / "user_test.py"
        user_test.write_text("# user-provided reference test\nassert True\n")

        captured: list[str] = []
        builder = _make_builder(tmp_path, model_responses=[_valid_harness_source()])

        def _q(messages):
            captured.append(messages[1]["content"])
            return _valid_harness_source()

        builder.model.query = _q  # type: ignore[attr-defined]
        builder.run(
            kernel_path=kernel,
            out_path=out,
            user_test_files=[user_test],
        )
        assert "USER TEST FILE" in captured[0]
        assert "user_test.py" in captured[0]

    def test_language_system_prompt_flows_into_system_message(
        self, tmp_path: Path
    ) -> None:
        kernel = tmp_path / "kernel.py"
        kernel.write_text("pass\n")
        out = tmp_path / "harness.py"

        captured: list[str] = []
        builder = _make_builder(
            tmp_path,
            model_responses=[_valid_harness_source()],
            with_system_prompt=True,
        )

        def _q(messages):
            captured.append(messages[0]["content"])  # system message
            return _valid_harness_source()

        builder.model.query = _q  # type: ignore[attr-defined]
        builder.run(kernel_path=kernel, out_path=out)

        assert "triton worker agent" in captured[0]


# ──────────────────────────────────────────────────────────────────────
# Config YAML exists
# ──────────────────────────────────────────────────────────────────────


class TestHarnessBuilderConfig:
    def test_config_yaml_exists(self) -> None:
        from minisweagent.pipeline_workers import preprocess

        cfg_path = (
            Path(preprocess.__file__).parent / "configs" / "harness_builder.yaml"
        )
        assert cfg_path.exists(), f"Missing config: {cfg_path}"

    def test_config_yaml_is_loadable(self) -> None:
        import yaml

        from minisweagent.pipeline_workers import preprocess

        cfg_path = (
            Path(preprocess.__file__).parent / "configs" / "harness_builder.yaml"
        )
        data = yaml.safe_load(cfg_path.read_text())
        assert data["name"] == "harness_builder"
        assert data["step_limit"] == 1
        assert "max_retries" in data.get("extra", {})
