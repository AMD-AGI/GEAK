"""Tests for LLM fallback in benchmark_parsing.extract_latency_ms_with_fallback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from minisweagent.run.postprocess.benchmark_parsing import (
    extract_latency_ms,
    extract_latency_ms_with_fallback,
    llm_extract_latency_ms,
)


# ---------------------------------------------------------------------------
# llm_extract_latency_ms — mock LLM responses
# ---------------------------------------------------------------------------


class TestLlmExtractLatencyMs:
    def _mock_model(self, response_content: str) -> MagicMock:
        model = MagicMock()
        model.query.return_value = {"content": response_content}
        return model

    def test_returns_float_from_llm(self):
        model = self._mock_model("6.5767")
        result = llm_extract_latency_ms("some benchmark output", model=model)
        assert result == pytest.approx(6.5767)
        model.query.assert_called_once()

    def test_returns_none_when_llm_says_none(self):
        model = self._mock_model("NONE")
        result = llm_extract_latency_ms("no timing here", model=model)
        assert result is None

    def test_strips_backticks(self):
        model = self._mock_model("`12.345`")
        result = llm_extract_latency_ms("output", model=model)
        assert result == pytest.approx(12.345)

    def test_rejects_out_of_range(self):
        model = self._mock_model("2000000")
        result = llm_extract_latency_ms("output", model=model)
        assert result is None

    def test_handles_non_numeric_response(self):
        model = self._mock_model("I couldn't parse this")
        result = llm_extract_latency_ms("output", model=model)
        assert result is None

    def test_handles_model_exception(self):
        model = MagicMock()
        model.query.side_effect = RuntimeError("API error")
        result = llm_extract_latency_ms("output", model=model)
        assert result is None

    def test_sends_last_80_lines(self):
        lines = [f"line {i}" for i in range(200)]
        text = "\n".join(lines)
        model = self._mock_model("1.0")
        llm_extract_latency_ms(text, model=model)
        call_args = model.query.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "line 120" in user_msg
        assert "line 199" in user_msg

    def test_system_prompt_instructs_sum(self):
        model = self._mock_model("1.0")
        llm_extract_latency_ms("output", model=model)
        call_args = model.query.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "SUM" in system_msg
        assert "microseconds" in system_msg

    def test_auto_creates_model_when_none(self):
        mock_model = self._mock_model("5.0")
        with patch("minisweagent.models.get_model", return_value=mock_model):
            result = llm_extract_latency_ms("output", model=None)
            assert result == pytest.approx(5.0)

    def test_handles_string_response(self):
        model = MagicMock()
        model.query.return_value = "42.5"
        result = llm_extract_latency_ms("output", model=model)
        assert result == pytest.approx(42.5)


# ---------------------------------------------------------------------------
# extract_latency_ms_with_fallback — integration
# ---------------------------------------------------------------------------


class TestExtractLatencyMsWithFallback:
    def test_deterministic_success_skips_llm(self):
        text = "GEAK_RESULT_LATENCY_MS=6.6"
        model = MagicMock()
        result = extract_latency_ms_with_fallback(text, model=model)
        assert result == pytest.approx(6.6)
        model.query.assert_not_called()

    def test_falls_back_to_llm_on_no_match(self):
        text = "Custom format: kernel ran in 0.05 milliseconds per invocation"
        model = MagicMock()
        model.query.return_value = {"content": "0.05"}
        result = extract_latency_ms_with_fallback(text, model=model)
        assert result == pytest.approx(0.05)
        model.query.assert_called_once()

    def test_returns_none_when_both_fail(self):
        text = "no timing at all"
        model = MagicMock()
        model.query.return_value = {"content": "NONE"}
        result = extract_latency_ms_with_fallback(text, model=model)
        assert result is None

    def test_deterministic_parsers_tried_first(self):
        text = "TOTAL_KERNEL_TIME_MS: 6.6\n"
        model = MagicMock()
        result = extract_latency_ms_with_fallback(text, model=model)
        assert result == pytest.approx(6.6)
        model.query.assert_not_called()


# ---------------------------------------------------------------------------
# Verify preprocess copy has the same functions
# ---------------------------------------------------------------------------


class TestPreprocessCopy:
    def test_preprocess_has_fallback(self):
        from minisweagent.run.preprocess.benchmark_parsing import (
            extract_latency_ms_with_fallback as pp_fallback,
            llm_extract_latency_ms as pp_llm,
        )
        assert callable(pp_fallback)
        assert callable(pp_llm)

    def test_preprocess_fallback_works(self):
        from minisweagent.run.preprocess.benchmark_parsing import (
            extract_latency_ms_with_fallback as pp_fallback,
        )
        text = "GEAK_RESULT_LATENCY_MS=10.5"
        result = pp_fallback(text)
        assert result == pytest.approx(10.5)
