"""Regression tests for recipe-env parsing and serving-stack fingerprinting.

Run:  python3 -m pytest interface/test_run_e2e_recipe_env.py -q

Two fixes are pinned here:

  * ``_recipe_env_block`` used to read the ``envs:`` map one line at a time and
    ``partition(":")`` each line, so a MULTI-LINE ``EXTRA_<BE>_ARGS`` (a YAML
    block scalar) was truncated to just the ``|`` / ``>`` indicator and every
    kernel-dispatch flag on the continuation lines was dropped. The launcher then
    served a different stack than the recipe recorded. These tests assert both the
    single-line path (unchanged) and the block-scalar paths (folded + literal),
    plus that a nested map under ``envs:`` is still skipped and parsing recovers
    to the sibling key after a block.

  * ``_serving_stack_signals`` used to ``read_text() + splitlines() + text.lower()``
    (three full O(n) copies of a log that can reach GBs) and appended a pick
    BEFORE checking the cap. It now streams line-by-line and bounds picks before
    the append, so the count is unchanged and picks never exceed the cap.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("run_e2e", _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load()

_REPO_FIXTURES = _HERE / "fixtures" / "recipe_env"


def _write(tmp_path: Path, text: str) -> str:
    p = tmp_path / "baseline_config.with_envs.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


@pytest.mark.parametrize(
    "filename,key,continuation_token",
    [
        (
            "sglang_plain_continuation.yaml",
            "EXTRA_SGLANG_ARGS",
            "--num-continuous-decode-steps",
        ),
        (
            "vllm_json_continuation.yaml",
            "EXTRA_VLLM_ARGS",
            "--speculative-config",
        ),
        (
            "gemma_multiline_json.yaml",
            "EXTRA_VLLM_ARGS",
            "--limit-mm-per-prompt",
        ),
    ],
)
def test_repo_fixture_main_and_fallback_preserve_continuations(
    filename: str, key: str, continuation_token: str
) -> None:
    """Mandatory CI fixtures for the three real continuation shapes."""
    path = _REPO_FIXTURES / filename
    text = path.read_text(encoding="utf-8")
    main = rx._recipe_env_block(str(path))
    fallback = rx._recipe_env_block_scan(text)
    assert continuation_token in main[key]
    assert fallback == main


def test_single_line_and_quoted_values(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "benchmark:\n"
        "  envs:\n"
        "    SIMPLE: value1\n"
        '    QUOTED: "has spaces"\n'
        "    SQUOTED: 'x y'\n",
    )
    envs = rx._recipe_env_block(path)
    assert envs["SIMPLE"] == "value1"
    assert envs["QUOTED"] == "has spaces"
    assert envs["SQUOTED"] == "x y"


def test_folded_block_scalar_joins_with_spaces(tmp_path: Path) -> None:
    # `>` folded scalar: the multi-line EXTRA_VLLM_ARGS must survive as one flag
    # string (the old parser stored just ">").
    path = _write(
        tmp_path,
        "benchmark:\n"
        "  envs:\n"
        "    EXTRA_VLLM_ARGS: >\n"
        "      --kv-cache-dtype fp8\n"
        "      --moe-runner-backend triton\n"
        "      --attention-backend triton\n"
        "    AFTER: recovered\n",
    )
    envs = rx._recipe_env_block(path)
    assert envs["EXTRA_VLLM_ARGS"] == (
        "--kv-cache-dtype fp8 --moe-runner-backend triton --attention-backend triton"
    )
    # Parsing recovers to the sibling key that follows the block scalar.
    assert envs["AFTER"] == "recovered"


def test_literal_block_scalar_keeps_newlines(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "envs:\n"
        "  EXTRA_SGLANG_ARGS: |\n"
        "    --mem-fraction-static 0.8\n"
        "    --disable-radix-cache\n",
    )
    envs = rx._recipe_env_block(path)
    # Literal `|` preserves line structure; whitespace-splitting downstream still
    # yields the individual flags.
    assert envs["EXTRA_SGLANG_ARGS"] == "--mem-fraction-static 0.8\n--disable-radix-cache"
    assert envs["EXTRA_SGLANG_ARGS"].split() == [
        "--mem-fraction-static",
        "0.8",
        "--disable-radix-cache",
    ]


def test_implicit_plain_scalar_continuation_space_folds(tmp_path: Path) -> None:
    """The real blocker: an IMPLICIT plain-scalar continuation (no ``|``/``>``).

    Replay-warm recipes wrap a long ``EXTRA_<BE>_ARGS`` across lines with nothing
    but deeper indentation (this is the exact SGLang TP1 fixture shape from
    Part 5). The old parser stored only the first line and dropped every
    kernel-dispatch flag on the continuation. It must fold with single spaces.
    """
    path = _write(
        tmp_path,
        "benchmark:\n"
        "  envs:\n"
        "    EXTRA_SGLANG_ARGS: --context-length 6144 --watchdog-timeout 1800 --page-size 64\n"
        "      --kv-cache-dtype fp8_e4m3 --num-continuous-decode-steps 2 --quantization fp8\n"
        "    MAGPIE_TRUST_REMOTE_CODE: '1'\n",
    )
    envs = rx._recipe_env_block(path)
    assert envs["EXTRA_SGLANG_ARGS"] == (
        "--context-length 6144 --watchdog-timeout 1800 --page-size 64 "
        "--kv-cache-dtype fp8_e4m3 --num-continuous-decode-steps 2 --quantization fp8"
    )
    # the sibling key after the folded scalar is still parsed.
    assert envs["MAGPIE_TRUST_REMOTE_CODE"] == "1"


def test_implicit_continuation_with_json_colons_survives(tmp_path: Path) -> None:
    """Continuation lines can carry JSON with colons -- a colon-based line scan
    would misread them as new keys. This is the vLLM TP1 speculative-config
    shape from Part 5; the whole JSON value must survive."""
    path = _write(
        tmp_path,
        "benchmark:\n"
        "  envs:\n"
        "    EXTRA_VLLM_ARGS: --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype\n"
        "      fp8_e4m3 --max-model-len 6144 --gpu-memory-utilization\n"
        '      0.87 --speculative-config {"method":"ngram","num_speculative_tokens":7}\n'
        "    VLLM_ROCM_USE_AITER: '1'\n",
    )
    envs = rx._recipe_env_block(path)
    assert envs["EXTRA_VLLM_ARGS"] == (
        "--attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8_e4m3 "
        "--max-model-len 6144 --gpu-memory-utilization 0.87 "
        '--speculative-config {"method":"ngram","num_speculative_tokens":7}'
    )
    assert envs["VLLM_ROCM_USE_AITER"] == "1"


def test_typed_scalars_stringify_for_the_shell(tmp_path: Path) -> None:
    """Ints/quoted-booleans in the recipe become the strings the shell needs."""
    path = _write(
        tmp_path,
        "envs:\n"
        "  MAX_MODEL_LEN: 6144\n"        # int -> "6144"
        "  RANDOM_RANGE_RATIO: 1\n"      # int -> "1"
        "  RUN_EVAL: 'true'\n"           # quoted -> "true"
        "  ROCR_VISIBLE_DEVICES: '0'\n"  # quoted -> "0"
        "  DISABLED: false\n",           # bare YAML bool -> "false"
    )
    envs = rx._recipe_env_block(path)
    assert envs == {
        "MAX_MODEL_LEN": "6144",
        "RANDOM_RANGE_RATIO": "1",
        "RUN_EVAL": "true",
        "ROCR_VISIBLE_DEVICES": "0",
        "DISABLED": "false",
    }
    assert all(isinstance(v, str) for v in envs.values())


def test_nested_map_under_envs_is_skipped(tmp_path: Path) -> None:
    # Values quoted so YAML keeps them as strings (bare ``yes`` is a YAML 1.1
    # boolean, which is exactly how the dumper distinguishes the two).
    path = _write(
        tmp_path,
        "envs:\n"
        '  REAL: "yes"\n'
        "  NESTED:\n"
        "    child: skipme\n"
        '  ALSO_REAL: "yes2"\n',
    )
    envs = rx._recipe_env_block(path)
    assert envs == {"REAL": "yes", "ALSO_REAL": "yes2"}
    assert "NESTED" not in envs and "child" not in envs


def test_nonempty_envs_map_outranks_earlier_empty_map(tmp_path: Path) -> None:
    """An unrelated empty metadata map must not hide the launch environment."""
    text = (
        "metadata:\n"
        "  envs: {}\n"
        "benchmark:\n"
        "  envs:\n"
        "    FOO: bar\n"
    )
    path = _write(tmp_path, text)
    expected = {"FOO": "bar"}
    assert rx._recipe_envs_from_yaml(text) == expected
    assert rx._recipe_env_block_scan(text) == expected
    assert rx._recipe_env_block(path) == expected


def test_block_ends_at_dedent(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "benchmark:\n"
        "  envs:\n"
        "    INSIDE: 1\n"
        "  other: 2\n"          # dedented out of envs: -> not captured
        "top: 3\n",
    )
    envs = rx._recipe_env_block(path)
    assert envs == {"INSIDE": "1"}


def test_block_scalar_under_indented_line_terminates_without_dropping_chars(
    tmp_path: Path, monkeypatch
) -> None:
    """A continuation indented BELOW the block's content indent ends the block.

    The content indentation is fixed by the FIRST content line. A later line that
    is less-indented than that (but still more-indented than the key) must END the
    scalar and be reparsed as a sibling -- NOT be sliced with ``nxt[content_indent:]``,
    which silently drops its leading characters and folds a stray ``KEY: val`` into
    the flag string.

    This guards the ``_recipe_env_block_scan`` fallback, which now runs ONLY when
    PyYAML is unavailable (a malformed recipe with PyYAML present fails closed --
    see ``test_malformed_recipe_fails_closed``). We reach the scanner by forcing
    ``import yaml`` to raise ImportError, then feed input the scanner must handle
    without corrupting the flags that drive kernel dispatch.
    """
    # Force the PyYAML-absent branch so _recipe_envs_from_yaml returns None and
    # _recipe_env_block degrades to the indentation scanner.
    monkeypatch.setitem(sys.modules, "yaml", None)
    path = _write(
        tmp_path,
        "envs:\n"
        "  EXTRA_VLLM_ARGS: |\n"
        "      --kv-cache-dtype fp8\n"   # first content line -> content_indent = 6
        "    STRAY: x\n",                # indent 4: below content, above the key
    )
    envs = rx._recipe_env_block(path)
    # The block keeps its full flag, uncorrupted (no dropped leading chars).
    assert envs["EXTRA_VLLM_ARGS"] == "--kv-cache-dtype fp8"
    # The under-indented line is reparsed as a sibling key, not swallowed.
    assert envs["STRAY"] == "x"


def test_malformed_recipe_fails_closed(tmp_path: Path) -> None:
    """A recipe PyYAML rejects must NOT silently fall to the degraded scanner.

    With PyYAML present, malformed YAML raises ``_RecipeYAMLError`` inside
    ``_recipe_envs_from_yaml``; ``_recipe_env_block`` turns that into an empty
    block (warn) by default, and a hard stop under ``GEAK_STRICT_RECIPE_ENV`` --
    never a truncated flow scalar handed off as a valid environment.
    """
    pytest.importorskip("yaml")
    # A block scalar whose sibling is under-indented is a YAML parse error.
    path = _write(
        tmp_path,
        "envs:\n"
        "  EXTRA_VLLM_ARGS: |\n"
        "      --kv-cache-dtype fp8\n"
        "    STRAY: x\n",
    )
    # Default (non-strict): fail closed to an empty block, not the scanner.
    assert rx._recipe_env_block(path) == {}
    # Strict: refuse to launch.
    import os

    old = os.environ.get("GEAK_STRICT_RECIPE_ENV")
    os.environ["GEAK_STRICT_RECIPE_ENV"] = "1"
    try:
        with pytest.raises(SystemExit):
            rx._recipe_env_block(path)
    finally:
        if old is None:
            os.environ.pop("GEAK_STRICT_RECIPE_ENV", None)
        else:
            os.environ["GEAK_STRICT_RECIPE_ENV"] = old


def test_serving_stack_signals_bounded_and_counts(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    lines = [f"[pid 1] Attention backend selected: triton {i}" for i in range(20)]
    lines.append("aiter aiter AITER")  # 3 case-insensitive mentions
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sig = rx._serving_stack_signals(log)
    assert sig["aiter_mentions"] == 3
    # picks capped at _STACK_SIGNAL_MAX_PICKS even though 20 lines match.
    assert len(sig["kernel_picks"]) == rx._STACK_SIGNAL_MAX_PICKS
    # prefix stripped to the legible tail.
    assert sig["kernel_picks"][0].startswith("Attention backend selected: triton")


def test_serving_stack_signals_missing_log(tmp_path: Path) -> None:
    assert rx._serving_stack_signals(tmp_path / "nope.log") == {}


# ---- fallback scanner (PyYAML-absent / unparseable degraded path) ------------


def test_fallback_scanner_also_folds_implicit_continuation() -> None:
    """The degraded line scan must fold implicit continuation too, so a box
    without PyYAML (or a malformed recipe) still serves the recorded flags."""
    text = (
        "benchmark:\n"
        "  envs:\n"
        "    EXTRA_SGLANG_ARGS: --context-length 6144 --page-size 64\n"
        "      --kv-cache-dtype fp8_e4m3 --quantization fp8\n"
        "    HF_HUB_TRUST_REMOTE_CODE: '1'\n"
    )
    envs = rx._recipe_env_block_scan(text)
    assert envs["EXTRA_SGLANG_ARGS"] == (
        "--context-length 6144 --page-size 64 --kv-cache-dtype fp8_e4m3 --quantization fp8"
    )
    assert envs["HF_HUB_TRUST_REMOTE_CODE"] == "1"


def test_fallback_matches_baseloader_for_supported_empty_and_scalar_values() -> None:
    """Installing PyYAML must not change supported replay bytes."""
    pytest.importorskip("yaml")
    text = (
        "envs:\n"
        "  BARE_EMPTY:\n"
        "  QUOTED_EMPTY: ''\n"
        "  YES_TEXT: yes\n"
        "  HEX_TEXT: 0x10\n"
        "  ON: value\n"
        "  NESTED:\n"
        "    child: skipped\n"
        "  AFTER: kept\n"
    )
    assert rx._recipe_env_block_scan(text) == rx._recipe_envs_from_yaml(text) == {
        "BARE_EMPTY": "",
        "QUOTED_EMPTY": "",
        "YES_TEXT": "yes",
        "HEX_TEXT": "0x10",
        "ON": "value",
        "AFTER": "kept",
    }


def test_strict_mode_refuses_when_pyyaml_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    """Strict alignment cannot treat the non-validating scanner as proof."""
    path = _write(tmp_path, "envs:\n  FOO: bar\n")
    monkeypatch.setitem(sys.modules, "yaml", None)
    monkeypatch.setenv("GEAK_STRICT_RECIPE_ENV", "1")
    with pytest.raises(SystemExit, match="PyYAML is unavailable"):
        rx._recipe_env_block(path)
