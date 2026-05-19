"""Unit tests for ``minisweagent.run.utils.task_parser``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from minisweagent.run.utils import task_parser as tp


class TestResolvePathCase:
    def test_relative_path_returns_none(self, tmp_path: Path) -> None:
        assert tp._resolve_path_case(Path("relative/path")) is None

    def test_resolves_wrong_case_component(self, tmp_path: Path) -> None:
        sub = tmp_path / "MyRepo"
        sub.mkdir()
        (sub / "a.txt").write_text("x")
        wrong = tmp_path / "myrepo" / "a.txt"
        assert not wrong.exists()
        resolved = tp._resolve_path_case(wrong)
        assert resolved is not None
        assert resolved == (tmp_path / "MyRepo" / "a.txt").resolve()

    def test_missing_component_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "does_not_exist" / "file.txt"
        assert tp._resolve_path_case(p.resolve()) is None


class TestNormalizePath:
    def test_empty_returns_none(self) -> None:
        assert tp._normalize_path("") is None

    def test_existing_path_resolves(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("x")
        assert tp._normalize_path(str(f)) == str(f.resolve())

    def test_unknown_path_returns_original_string(self) -> None:
        assert tp._normalize_path("no/such/path/anywhere/xyz") == "no/such/path/anywhere/xyz"


class TestParseTaskInfo:
    class _Model:
        def __init__(self, content: str) -> None:
            self._content = content

        def query(self, messages: list) -> dict:
            return {"content": self._content}

    def test_parses_json_object(self) -> None:
        payload = {
            "kernel_name": "gemm",
            "kernel_url": "https://example.com/k.py",
            "kernel_type": "triton",
            "repo": None,
            "test_command": "pytest",
            "metric": "latency",
            "num_parallel": 2,
            "gpu_ids": "0,1",
            "output_dir": None,
            "model": "m",
            "config": None,
        }
        out = tp.parse_task_info("task", self._Model(json.dumps(payload)))
        assert out["kernel_name"] == "gemm"
        assert out["kernel_type"] == "triton"
        assert out["num_parallel"] == 2

    def test_strips_json_from_markdown_fence(self) -> None:
        inner = json.dumps(
            {
                "kernel_name": "k",
                "kernel_url": None,
                "kernel_type": "hip",
                "repo": None,
                "test_command": None,
                "metric": None,
                "num_parallel": None,
                "gpu_ids": None,
                "output_dir": None,
                "model": None,
                "config": None,
            }
        )
        content = f"Here:\n```json\n{inner}\n```"
        out = tp.parse_task_info("x", self._Model(content))
        assert out["kernel_name"] == "k"
        assert out["kernel_type"] == "hip"

    def test_invalid_kernel_type_becomes_other(self) -> None:
        payload = {
            "kernel_name": None,
            "kernel_url": None,
            "kernel_type": "cuda",
            "repo": None,
            "test_command": None,
            "metric": None,
            "num_parallel": None,
            "gpu_ids": None,
            "output_dir": None,
            "model": None,
            "config": None,
        }
        out = tp.parse_task_info("t", self._Model(json.dumps(payload)))
        assert out["kernel_type"] == "other"

    def test_malformed_json_returns_fallback(self) -> None:
        out = tp.parse_task_info("t", self._Model("not json {{{"))
        assert out["kernel_name"] is None
        assert out["kernel_type"] == "other"

    def test_query_exception_returns_fallback(self) -> None:
        class Bad:
            def query(self, messages):
                raise RuntimeError("boom")

        out = tp.parse_task_info("t", Bad())
        assert out["kernel_name"] is None
        assert out["kernel_type"] == "other"

    def test_repo_resolves_when_path_exists(self, tmp_path: Path) -> None:
        payload = {
            "kernel_name": None,
            "kernel_url": None,
            "kernel_type": "other",
            "repo": str(tmp_path),
            "test_command": None,
            "metric": None,
            "num_parallel": None,
            "gpu_ids": None,
            "output_dir": None,
            "model": None,
            "config": None,
        }
        out = tp.parse_task_info("t", self._Model(json.dumps(payload)))
        assert out["repo"] == str(tmp_path.resolve())


class TestParsePipelineParams:
    class _Model:
        def __init__(self, content: str) -> None:
            self._content = content

        def query(self, messages: list) -> dict:
            return {"content": self._content}

    def test_parses_fields(self) -> None:
        payload = {
            "kernel_url": "/tmp/a.hip",
            "preprocess_dir": None,
            "heterogeneous": True,
            "max_rounds": 3,
            "start_round": 1,
            "pipeline_intent": True,
        }
        out = tp.parse_pipeline_params("t", self._Model(json.dumps(payload)))
        assert out["heterogeneous"] is True
        assert out["max_rounds"] == 3
        assert out["start_round"] == 1
        assert out["pipeline_intent"] is True

    def test_coerces_numeric_strings(self) -> None:
        payload = {
            "kernel_url": None,
            "preprocess_dir": None,
            "heterogeneous": None,
            "max_rounds": "10",
            "start_round": "2",
            "pipeline_intent": False,
        }
        out = tp.parse_pipeline_params("t", self._Model(json.dumps(payload)))
        assert out["max_rounds"] == 10
        assert out["start_round"] == 2

    def test_invalid_int_fields_become_none(self) -> None:
        payload = {
            "kernel_url": None,
            "preprocess_dir": None,
            "heterogeneous": None,
            "max_rounds": "nope",
            "start_round": None,
            "pipeline_intent": False,
        }
        out = tp.parse_pipeline_params("t", self._Model(json.dumps(payload)))
        assert out["max_rounds"] is None

    def test_exception_returns_fallback(self) -> None:
        class Bad:
            def query(self, messages):
                raise RuntimeError("x")

        out = tp.parse_pipeline_params("t", Bad())
        assert out["kernel_url"] is None
        assert out["pipeline_intent"] is False
        assert out["mode"] is None  # fallback always includes mode key

    def test_extracts_quick_mode(self) -> None:
        payload = {
            "kernel_url": None,
            "preprocess_dir": None,
            "heterogeneous": None,
            "max_rounds": None,
            "start_round": None,
            "pipeline_intent": True,
            "mode": "quick",
        }
        out = tp.parse_pipeline_params("t", self._Model(json.dumps(payload)))
        assert out["mode"] == "quick"

    def test_extracts_full_mode(self) -> None:
        payload = {
            "kernel_url": None,
            "preprocess_dir": None,
            "heterogeneous": None,
            "max_rounds": None,
            "start_round": None,
            "pipeline_intent": True,
            "mode": "full",
        }
        out = tp.parse_pipeline_params("t", self._Model(json.dumps(payload)))
        assert out["mode"] == "full"

    def test_normalizes_mode_case_and_whitespace(self) -> None:
        payload = {
            "kernel_url": None,
            "preprocess_dir": None,
            "heterogeneous": None,
            "max_rounds": None,
            "start_round": None,
            "pipeline_intent": True,
            "mode": "  Quick  ",
        }
        out = tp.parse_pipeline_params("t", self._Model(json.dumps(payload)))
        assert out["mode"] == "quick"

    def test_invalid_mode_value_clears_to_none(self) -> None:
        # The LLM might hallucinate a value not in the supported set;
        # we must clear it to None rather than propagate garbage.
        for bad in ("blazing", "fast", "1h", "", 7):
            payload = {
                "kernel_url": None,
                "preprocess_dir": None,
                "heterogeneous": None,
                "max_rounds": None,
                "start_round": None,
                "pipeline_intent": True,
                "mode": bad,
            }
            out = tp.parse_pipeline_params("t", self._Model(json.dumps(payload)))
            assert out["mode"] is None, f"bad value {bad!r} should normalize to None"

    def test_missing_mode_key_returns_none(self) -> None:
        # Backward-compat with old responses that don't include the field.
        payload = {
            "kernel_url": None,
            "preprocess_dir": None,
            "heterogeneous": None,
            "max_rounds": None,
            "start_round": None,
            "pipeline_intent": True,
        }
        out = tp.parse_pipeline_params("t", self._Model(json.dumps(payload)))
        assert out["mode"] is None


class TestInferModeFromText:
    """Regex backstop for mode extraction. Fires when LLM returns no mode."""

    def test_quick_mode_phrase(self) -> None:
        assert tp._infer_mode_from_text("Use quick mode for this run.") == "quick"

    def test_in_quick_mode_phrase(self) -> None:
        assert tp._infer_mode_from_text("Use GPUs 0-3 in quick mode.") == "quick"

    def test_one_hour_phrase(self) -> None:
        assert tp._infer_mode_from_text("Limit the run to 1 hour please.") == "quick"
        assert tp._infer_mode_from_text("Run for one hour total.") == "quick"
        assert tp._infer_mode_from_text("Cap it at 1h.") == "quick"

    def test_full_mode_phrase(self) -> None:
        assert tp._infer_mode_from_text("Use full mode and explore deeply.") == "full"

    def test_two_hours_phrase(self) -> None:
        assert tp._infer_mode_from_text("Run for 2 hours.") == "full"
        assert tp._infer_mode_from_text("Run for two hours.") == "full"

    def test_cli_style_flags(self) -> None:
        assert tp._infer_mode_from_text("Run with --mode quick please.") == "quick"
        assert tp._infer_mode_from_text("Use mode=full.") == "full"

    def test_no_mode_returns_none(self) -> None:
        # Reasonable task text without any mode reference.
        assert tp._infer_mode_from_text("Optimize the silu kernel for me.") is None

    def test_ambiguous_returns_none(self) -> None:
        # If both quick and full are mentioned, abstain rather than guess.
        text = "Try quick mode first, but if needed switch to full mode."
        assert tp._infer_mode_from_text(text) is None

    def test_empty_text(self) -> None:
        assert tp._infer_mode_from_text("") is None
        assert tp._infer_mode_from_text(None) is None  # type: ignore[arg-type]


class TestParsePipelineParamsRegexFallback:
    """When the LLM fails to produce JSON, the regex backstop should still
    populate ``mode`` from obvious natural-language cues, so the budget
    feature isn't silently downgraded to YAML's ``full`` default.
    """

    def _model_returning(self, content: str):
        class _M:
            def query(_self, _messages):
                return {"content": content}

        return _M()

    def test_regex_fills_mode_when_llm_returns_prose(self) -> None:
        # This is the exact failure mode we hit: LLM acknowledges and
        # bails instead of producing JSON.
        prose = "Looking at the task, I need to identify the kernel file. Let me first check the directory."
        out = tp.parse_pipeline_params(
            "Optimize the kernel from /some/dir. Use GPUs 0-3 in quick mode.",
            self._model_returning(prose),
        )
        assert out["mode"] == "quick", "regex fallback must fire after JSON decode failure"

    def test_regex_fills_mode_when_llm_returns_empty(self) -> None:
        out = tp.parse_pipeline_params("Run with full mode for 2 hours.", self._model_returning(""))
        assert out["mode"] == "full"

    def test_regex_does_not_override_explicit_llm_mode(self) -> None:
        # LLM returned "quick" -- the regex must not flip it even if the
        # task text also contains a "full" phrase (it doesn't here, but
        # this asserts the precedence rule).
        payload = {
            "kernel_url": None,
            "preprocess_dir": None,
            "heterogeneous": None,
            "max_rounds": None,
            "start_round": None,
            "pipeline_intent": True,
            "mode": "quick",
        }
        out = tp.parse_pipeline_params(
            "Use full mode for thorough optimization.",
            self._model_returning(json.dumps(payload)),
        )
        # LLM said "quick"; regex would have inferred "full"; LLM wins.
        assert out["mode"] == "quick"

    def test_no_regex_match_keeps_none(self) -> None:
        out = tp.parse_pipeline_params("Optimize the kernel.", self._model_returning(""))
        assert out["mode"] is None  # neither LLM nor regex found a mode


class TestJsonDecodeFailureLogsRawResponse:
    """When the LLM returns non-JSON content, the warning must include a
    truncated view of the raw response so users can debug. Without this,
    parse_pipeline_params errors are indistinguishable across "model
    apologized in plain text" / "transport returned empty" / "markdown
    without a fenced JSON block".
    """

    def _model_returning(self, content: str):
        class _M:
            def query(_self, _messages):
                return {"content": content}

        return _M()

    @staticmethod
    def _capture_warnings():
        import logging

        records: list[logging.LogRecord] = []

        class _H(logging.Handler):
            def emit(self, r: logging.LogRecord) -> None:
                records.append(r)

        h = _H(level=logging.WARNING)
        logger = logging.getLogger("minisweagent.run.utils.task_parser")
        logger.addHandler(h)
        return records, logger, h

    def test_logs_truncated_raw_response_on_json_decode_error(self) -> None:
        records, logger, h = self._capture_warnings()
        try:
            bad = "Sorry, I can't help with that. Here is some prose without any JSON. " + ("x" * 600)
            out = tp.parse_pipeline_params("t", self._model_returning(bad))
            assert out["mode"] is None
            previews = [r.getMessage() for r in records if "model response was not valid JSON" in r.getMessage()]
            assert previews, "expected a warning naming 'model response was not valid JSON'"
            msg = previews[0]
            assert "Sorry, I can't help with that" in msg
            assert "[truncated]" in msg, "long responses must be truncated to keep logs manageable"
        finally:
            logger.removeHandler(h)

    def test_logs_empty_response_distinctly(self) -> None:
        records, logger, h = self._capture_warnings()
        try:
            out = tp.parse_pipeline_params("t", self._model_returning(""))
            assert out["mode"] is None
            msgs = [r.getMessage() for r in records if "not valid JSON" in r.getMessage()]
            assert msgs and "<empty>" in msgs[0]
        finally:
            logger.removeHandler(h)


class TestPromoteKernelUrlDirToFile:
    """The LLM extractor sometimes returns a directory path as ``kernel_url``
    when the user said "the kernel is in <DIR>". Promote to a file inside
    the directory so the run doesn't hard-fail at resolve_kernel_url.
    """

    def test_promotes_kernel_py_in_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "my_kernel_dir"
        d.mkdir()
        (d / "kernel.py").write_text("# kernel\n")
        promoted = tp._promote_kernel_url_dir_to_file(str(d), kernel_name_hint=None, kernel_type="triton")
        assert promoted == str(d / "kernel.py")

    def test_prefers_kernel_name_hint_with_extension(self, tmp_path: Path) -> None:
        # When both ``my_silu.py`` and ``kernel.py`` exist, the name hint wins.
        d = tmp_path / "k"
        d.mkdir()
        (d / "kernel.py").write_text("# generic\n")
        (d / "my_silu.py").write_text("# specific\n")
        promoted = tp._promote_kernel_url_dir_to_file(str(d), kernel_name_hint="my_silu", kernel_type="triton")
        assert promoted == str(d / "my_silu.py")

    def test_promotes_single_hip_file_in_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "hip_kernel"
        d.mkdir()
        # No ``kernel.py`` / ``kernel.hip``; just one .hip file.
        (d / "silu_mul.hip").write_text("// hip\n")
        promoted = tp._promote_kernel_url_dir_to_file(str(d), kernel_name_hint=None, kernel_type="hip")
        assert promoted == str(d / "silu_mul.hip")

    def test_does_not_promote_when_directory_has_no_kernel_file(self, tmp_path: Path) -> None:
        import logging

        d = tmp_path / "empty_dir"
        d.mkdir()
        (d / "README.md").write_text("not a kernel\n")

        records: list[logging.LogRecord] = []

        class _H(logging.Handler):
            def emit(self, r: logging.LogRecord) -> None:
                records.append(r)

        h = _H(level=logging.WARNING)
        logger = logging.getLogger("minisweagent.run.utils.task_parser")
        logger.addHandler(h)
        try:
            promoted = tp._promote_kernel_url_dir_to_file(str(d), kernel_name_hint=None, kernel_type="triton")
            assert promoted == str(d), "must leave path unchanged so caller surfaces a clear error"
            assert any("kernel_url" in r.getMessage() and "directory" in r.getMessage() for r in records)
        finally:
            logger.removeHandler(h)

    def test_passthrough_when_kernel_url_is_already_a_file(self, tmp_path: Path) -> None:
        f = tmp_path / "kernel.py"
        f.write_text("# kernel\n")
        promoted = tp._promote_kernel_url_dir_to_file(str(f), kernel_name_hint=None, kernel_type="triton")
        assert promoted == str(f)

    def test_passthrough_when_path_does_not_exist(self) -> None:
        # A non-existent path is also not a directory; we leave it alone.
        promoted = tp._promote_kernel_url_dir_to_file("/no/such/path", kernel_name_hint=None, kernel_type="triton")
        assert promoted == "/no/such/path"


class TestNormalizeParsedTaskInfoIntegratesPromotion:
    """End-to-end: parse_task_info on JSON containing a directory kernel_url
    must produce a ``kernel_url`` that points at a file inside.
    """

    def _model(self, content: str):
        class _M:
            def query(_self, _messages):
                return {"content": content}

        return _M()

    def test_directory_kernel_url_gets_promoted(self, tmp_path: Path) -> None:
        d = tmp_path / "my_kernel"
        d.mkdir()
        (d / "kernel.py").write_text("# triton kernel\n")
        payload = {
            "kernel_name": "my_kernel",
            "kernel_url": str(d),
            "kernel_type": "triton",
            "repo": str(tmp_path),
            "test_command": None,
            "metric": None,
            "num_parallel": None,
            "gpu_ids": None,
            "output_dir": None,
            "model": None,
            "config": None,
        }
        out = tp.parse_task_info("t", self._model(json.dumps(payload)))
        assert out["kernel_url"] == str(d / "kernel.py")


class TestSanitizeKernelNameForPatchDir:
    """Lock the contract of ``_sanitize_kernel_name_for_patch_dir`` so a
    future cap tweak does not silently change directory layout.

    The cap (``_MAX_KERNEL_DIR_STEM_LEN``) is intentionally a balance
    between log-readability (longer = recognisable kernel names) and
    filesystem path length budget.  Bumping the cap is fine; making it
    smaller again should require an explicit test update.
    """

    def test_short_name_round_trips_unchanged(self) -> None:
        assert tp._sanitize_kernel_name_for_patch_dir("act_and_mul") == "act_and_mul"

    def test_path_separator_becomes_underscore(self) -> None:
        assert tp._sanitize_kernel_name_for_patch_dir("my/kernel") == "my_kernel"

    def test_long_name_gets_hashed_suffix(self) -> None:
        """Long names are truncated to ``cap-1-8`` chars + ``_`` + 8-hex
        SHA-256 prefix.  The two-tier shape (prefix + digest) ensures
        uniqueness without dropping the human-meaningful prefix."""
        long_name = "Cijk_Alik_Bjlk_HBH_MT128x128x32_SE_K1_AS_AmpereTC"
        out = tp._sanitize_kernel_name_for_patch_dir(long_name)
        assert len(out) == tp._MAX_KERNEL_DIR_STEM_LEN
        # The human-meaningful prefix is preserved verbatim.
        assert out.startswith("Cijk_Alik_Bjlk_HBH")
        # And ends with ``_<8-hex>``.
        assert out[-9] == "_"
        assert all(c in "0123456789abcdef" for c in out[-8:])

    def test_long_name_is_deterministic(self) -> None:
        """Identical input yields identical sanitised stem (matters for
        log resumption / patch directory reuse)."""
        long_name = "Cijk_" + "x" * 80
        a = tp._sanitize_kernel_name_for_patch_dir(long_name)
        b = tp._sanitize_kernel_name_for_patch_dir(long_name)
        assert a == b

    def test_cap_value_is_at_least_recognisable(self) -> None:
        """Sanity floor: the cap must leave room for >= 12 characters of
        human-meaningful prefix (after ``_<8-hex>`` overhead).  Below
        that, hipBLASLt-style kernel names become unidentifiable in
        ``logs/`` listings.
        """
        assert tp._MAX_KERNEL_DIR_STEM_LEN >= 12 + 1 + 8, (
            "Stem cap is too small to retain a recognisable kernel prefix; "
            f"got {tp._MAX_KERNEL_DIR_STEM_LEN}, need >= 21."
        )


class TestGeneratePatchOutputDir:
    def test_uses_kernel_name_and_timestamp(self) -> None:
        with patch("minisweagent.run.utils.task_parser.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20250101_120000"
            out = tp.generate_patch_output_dir("my/kernel")
        assert out.replace("\\", "/") == "optimization_logs/my_kernel_20250101_120000"

    def test_none_kernel_name_uses_optimization_prefix(self) -> None:
        with patch("minisweagent.run.utils.task_parser.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20250101_120000"
            out = tp.generate_patch_output_dir(None)
        assert "optimization_20250101_120000" in out.replace("\\", "/")

    def test_respects_base_dir(self) -> None:
        with patch("minisweagent.run.utils.task_parser.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "ts"
            out = tp.generate_patch_output_dir("k", base_dir="custom_logs")
        assert out.replace("\\", "/") == "custom_logs/k_ts"


class TestDisplayParsedConfig:
    def test_includes_patch_output_dir_and_defaults(self) -> None:
        info = {
            "kernel_type": "triton",
            "kernel_name": "n",
            "kernel_url": "u",
            "repo": None,
            "test_command": None,
            "metric": None,
            "num_parallel": None,
            "gpu_ids": None,
            "model": None,
            "config": None,
        }
        text = tp.display_parsed_config(info, "/tmp/out")
        assert "patch_output_dir" in text
        assert "/tmp/out" in text
        assert "triton" in text
        assert "Resolved Configuration" in text


class TestGpuOversubscribeAndMaxConcurrentLlm:
    """Tests for gpu_oversubscribe and max_concurrent_llm extraction."""

    class _Model:
        def __init__(self, content: str) -> None:
            self._content = content

        def query(self, messages: list) -> dict:
            return {"content": self._content}

    def test_extracts_gpu_oversubscribe(self) -> None:
        payload = {
            "kernel_name": "k",
            "kernel_url": None,
            "kernel_type": "hip",
            "repo": None,
            "test_command": None,
            "metric": None,
            "num_parallel": None,
            "gpu_ids": None,
            "output_dir": None,
            "model": None,
            "config": None,
            "gpu_oversubscribe": 2.0,
            "max_concurrent_llm": None,
        }
        out = tp.parse_task_info("task", self._Model(json.dumps(payload)))
        assert out["gpu_oversubscribe"] == 2.0
        assert out["max_concurrent_llm"] is None

    def test_extracts_max_concurrent_llm(self) -> None:
        payload = {
            "kernel_name": "k",
            "kernel_url": None,
            "kernel_type": "hip",
            "repo": None,
            "test_command": None,
            "metric": None,
            "num_parallel": None,
            "gpu_ids": None,
            "output_dir": None,
            "model": None,
            "config": None,
            "gpu_oversubscribe": None,
            "max_concurrent_llm": 8,
        }
        out = tp.parse_task_info("task", self._Model(json.dumps(payload)))
        assert out["max_concurrent_llm"] == 8
        assert out["gpu_oversubscribe"] is None

    def test_null_values_preserved(self) -> None:
        payload = {
            "kernel_name": "k",
            "kernel_url": None,
            "kernel_type": "hip",
            "repo": None,
            "test_command": None,
            "metric": None,
            "num_parallel": None,
            "gpu_ids": None,
            "output_dir": None,
            "model": None,
            "config": None,
            "gpu_oversubscribe": None,
            "max_concurrent_llm": None,
        }
        out = tp.parse_task_info("task", self._Model(json.dumps(payload)))
        assert out["gpu_oversubscribe"] is None
        assert out["max_concurrent_llm"] is None

    def test_invalid_values_cleared(self) -> None:
        payload = {
            "kernel_name": "k",
            "kernel_url": None,
            "kernel_type": "hip",
            "repo": None,
            "test_command": None,
            "metric": None,
            "num_parallel": None,
            "gpu_ids": None,
            "output_dir": None,
            "model": None,
            "config": None,
            "gpu_oversubscribe": "not_a_number",
            "max_concurrent_llm": -5,
        }
        out = tp.parse_task_info("task", self._Model(json.dumps(payload)))
        assert out["gpu_oversubscribe"] is None
        assert out["max_concurrent_llm"] is None

    def test_empty_template_has_new_fields(self) -> None:
        assert "gpu_oversubscribe" in tp._EMPTY_TASK_INFO
        assert "max_concurrent_llm" in tp._EMPTY_TASK_INFO
