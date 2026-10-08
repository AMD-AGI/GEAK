"""Tests for schema-v2 effective launch configuration reconciliation."""
from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Optional

import pytest

from interface import effective_config as ec
from interface.effective_config import EffectiveConfig, resolve_effective_config


def _recipe(tmp_path: Path, framework: str, args: str, **env: str) -> str:
    arg_name = {
        "vllm": "EXTRA_VLLM_ARGS",
        "sglang": "EXTRA_SGLANG_ARGS",
    }[framework]
    env_lines = "\n".join(f"    {key}: {value!r}" for key, value in env.items())
    path = tmp_path / f"{framework}.yaml"
    path.write_text(
        "benchmark:\n"
        f"  framework: {framework}\n"
        "  envs:\n"
        f"    {arg_name}: >\n"
        f"      {args}\n"
        f"{env_lines}\n",
        encoding="utf-8",
    )
    return str(path)


def _handoff(
    recipe: str,
    *,
    framework: str = "vllm",
    launch: str = "",
    extra: str = "",
    accepted: str = "",
    extra_envs: Optional[dict[str, str]] = None,
    accepted_env: str = "",
) -> dict:
    return {
        "schema_version": 2,
        "framework": framework,
        "launch_recipe": recipe,
        "accepted_flags": accepted,
        "accepted_env": accepted_env,
        "baseline_env_spec": {
            "config": {
                "server_launch_flags": launch,
                "extra_server_args": extra,
                "extra_envs": extra_envs or {},
            },
            "overlay_pythonpath": "/tmp/overlay",
            "source_snapshots": [{"id": "snapshot-1", "reproducible": True}],
        },
    }


@pytest.mark.parametrize(
    "framework,recipe_flag",
    [
        ("vllm", "--gpu-memory-utilization 0.9"),
        ("sglang", "--mem-fraction-static 0.9"),
    ],
)
def test_parses_backend_recipe_args(
    tmp_path: Path, framework: str, recipe_flag: str
) -> None:
    recipe = _recipe(tmp_path, framework, recipe_flag)
    result = resolve_effective_config(_handoff(recipe, framework=framework))

    assert shlex.split(result.final_server_args) == recipe_flag.split()
    assert result.base_overlay_pythonpath == "/tmp/overlay"
    assert result.source_snapshots == [{"id": "snapshot-1", "reproducible": True}]


def test_deduplicates_identical_flag_across_all_sources(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path, "vllm", "--block-size=64")
    result = resolve_effective_config(
        _handoff(
            recipe,
            launch="--block-size 64",
            extra="--block-size=64",
            accepted="--block-size 64",
        )
    )

    tokens = shlex.split(result.final_server_args)
    assert tokens == ["--block-size", "64"]
    assert tokens.count("--block-size") == 1
    assert result.conflicts == []


def test_complete_launch_replaces_recipe_and_preserves_unknown_flags(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path, "vllm", "--block-size 8 --recipe-only")
    result = resolve_effective_config(
        _handoff(
            recipe,
            launch="--block-size=16 --unknown-launch value",
            extra="--block-size 32 --delta-only",
            accepted="--block-size=32 --accepted-only yes",
        )
    )

    tokens = shlex.split(result.final_server_args)
    assert tokens.count("--block-size") == 1
    assert tokens[tokens.index("--block-size") + 1] == "32"
    assert "--recipe-only" not in tokens
    assert "--unknown-launch" in tokens
    assert "--delta-only" in tokens
    assert "--accepted-only" in tokens
    assert [entry["higher_source"] for entry in result.conflicts] == [
        "current_best_delta",
    ]


@pytest.mark.parametrize("snapshot", ["", "  "])
@pytest.mark.parametrize("current", ["", "--current 3"])
def test_explicit_replace_suppresses_recipe_fallback(tmp_path, snapshot, current):
    h = _handoff(_recipe(tmp_path, "vllm", "--recipe-only --block-size 128"),
                 launch=snapshot, extra=current, accepted=current)
    h["baseline_env_spec"]["config"]["args_mode"] = "replace"
    assert resolve_effective_config(h).final_server_args == current


def test_replace_preserves_available_observed_script_defaults(tmp_path):
    h = _handoff(_recipe(tmp_path, "vllm", "--recipe-only"),
                 launch="--observed-default 16 --current 3", extra="--current 3")
    h["baseline_env_spec"]["config"]["args_mode"] = "replace"
    assert resolve_effective_config(h).final_server_args == "--observed-default 16 --current 3"


@pytest.mark.parametrize("removal,expected", [
    (["--disable-cuda-graph"], "--block-size 128"),
    (["--block-size=128"], "--disable-cuda-graph"),
    (["--block-size 64"], "--disable-cuda-graph --block-size 128"),
    ("--disable-cuda-graph --block-size", ""),
])
def test_explicit_removal_applies_before_current_flags(tmp_path, removal, expected):
    h = _handoff(_recipe(tmp_path, "vllm", "--disable-cuda-graph --block-size 128"))
    config = h["baseline_env_spec"]["config"]
    config["remove_args"] = removal
    assert resolve_effective_config(h).final_server_args == expected
    config["extra_server_args"] = "--block-size 256"
    assert "--block-size 256" in resolve_effective_config(h).final_server_args


def test_unsets_are_explicit_and_later_assignments_win(tmp_path):
    h = _handoff(_recipe(tmp_path, "sglang", "", SGLANG_AITER_MLA_PERSIST="1", KEEP="2"),
                 framework="sglang")
    original = resolve_effective_config(h)
    h["baseline_env_spec"]["config"]["unset_envs"] = ["SGLANG_AITER_MLA_PERSIST", "AMBIENT_ONLY"]
    removed = resolve_effective_config(h)
    assert removed.final_env == {"KEEP": "2"}
    assert removed.unset_envs == ("AMBIENT_ONLY", "SGLANG_AITER_MLA_PERSIST")
    assert removed.digest != original.digest
    h["accepted_env"] = "SGLANG_AITER_MLA_PERSIST=3"
    added = resolve_effective_config(h)
    assert added.final_env == {"KEEP": "2", "SGLANG_AITER_MLA_PERSIST": "3"}
    assert added.unset_envs == ("AMBIENT_ONLY",)


def test_reenabled_removal_is_auditable_and_not_returned_as_active():
    h = _handoff("", launch="--keep 1 --disabled", extra="--disabled")
    h["baseline_env_spec"]["config"]["remove_args"] = ["--disabled", "--absent"]
    resolved = resolve_effective_config(h)
    assert shlex.split(resolved.final_server_args) == ["--keep", "1", "--disabled"]
    assert resolved.remove_args == ("--absent",)
    assert resolved.manifest["remove_args"] == ["--absent"]
    assert resolved.conflicts == [{
        "kind": "server_flag", "key": "--disabled", "lower_source": "remove_args",
        "lower_value": "--disabled", "higher_source": "current_best_delta", "higher_value": None,
    }]


def test_import_does_not_change_global_module_search_path():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c",
        "import sys; before=list(sys.path); import interface.effective_config; assert sys.path == before"],
        cwd=Path(__file__).resolve().parents[1], check=True)


def test_complete_result_normalizes_removal_controls():
    from interface.run_e2e import _accepted_config_with_env_map
    result = _accepted_config_with_env_map({
        "flags": "--keep 2", "env": "RESET=2", "args_mode": "replace",
        "remove_args": ["--keep", "--absent"], "unset_envs": ["RESET", "ABSENT"],
    })
    assert result["args_mode"] == "replace"
    assert result["remove_args"] == ["--absent"]
    assert result["unset_envs"] == ["ABSENT"]


def test_absent_environment_and_explicit_ambient_removal_have_different_digests():
    h = _handoff("")
    original = resolve_effective_config(h)
    h["baseline_env_spec"]["config"]["unset_envs"] = "AMBIENT_ONLY"
    removed = resolve_effective_config(h)
    assert original.final_env == removed.final_env == {}
    assert original.digest != removed.digest


@pytest.mark.parametrize("key,value", [
    ("args_mode", "unknown"), ("remove_args", 12), ("remove_args", [None]),
    ("unset_envs", 12), ("unset_envs", ["-u"]), ("unset_envs", ["X=1"]),
    ("unset_envs", [None]), ("unset_envs", ["A B"]),
])
def test_malformed_controls_stop_resolution(key, value):
    h = _handoff("")
    h["baseline_env_spec"]["config"][key] = value
    with pytest.raises((TypeError, ValueError)):
        resolve_effective_config(h)


def test_non_conflicting_extra_and_accepted_flags_form_union(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path, "sglang", "--recipe-flag")
    result = resolve_effective_config(
        _handoff(
            recipe,
            framework="sglang",
            extra="--extra-flag one",
            accepted="--accepted-flag two",
        )
    )

    assert shlex.split(result.final_server_args) == [
        "--recipe-flag",
        "--extra-flag",
        "one",
        "--accepted-flag",
        "two",
    ]


def test_conflicting_extra_and_accepted_flag_raises(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path, "vllm", "")
    with pytest.raises(ValueError, match="conflicting server flag.*--block-size"):
        resolve_effective_config(
            _handoff(
                recipe,
                extra="--block-size 16",
                accepted="--block-size=32",
            )
        )


def test_json_values_are_shell_safe_and_semantically_deduplicated(
    tmp_path: Path,
) -> None:
    recipe = _recipe(tmp_path, "vllm", "")
    result = resolve_effective_config(
        _handoff(
            recipe,
            extra="--speculative-config '{\"b\": 2, \"a\": 1}'",
            accepted='--speculative-config={"a":1,"b":2}',
        )
    )

    tokens = shlex.split(result.final_server_args)
    assert tokens == ["--speculative-config", '{"a":1,"b":2}']
    assert json.loads(tokens[1]) == {"a": 1, "b": 2}


def test_env_union_dedupe_override_and_recipe_arg_removal(tmp_path: Path) -> None:
    recipe = _recipe(
        tmp_path,
        "vllm",
        "--dtype auto",
        RECIPE_ONLY="yes",
        SHARED="recipe",
    )
    result = resolve_effective_config(
        _handoff(
            recipe,
            extra_envs={"EXTRA_ONLY": "1", "SHARED": "delta", "SAME": "x"},
            accepted_env="ACCEPTED_ONLY='two words' SAME=x",
        )
    )

    assert result.final_env == {
        "RECIPE_ONLY": "yes",
        "SHARED": "delta",
        "EXTRA_ONLY": "1",
        "SAME": "x",
        "ACCEPTED_ONLY": "two words",
    }
    assert "EXTRA_VLLM_ARGS" not in result.final_env
    assert result.conflicts == [
        {
            "kind": "environment",
            "key": "SHARED",
            "lower_source": "launch_recipe",
            "lower_value": "recipe",
            "higher_source": "current_best_delta",
            "higher_value": "delta",
        }
    ]


def test_conflicting_extra_and_accepted_env_raises(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path, "sglang", "")
    with pytest.raises(ValueError, match="conflicting environment variable.*MODE"):
        resolve_effective_config(
            _handoff(
                recipe,
                framework="sglang",
                extra_envs={"MODE": "fast"},
                accepted_env="MODE=safe",
            )
        )


@pytest.mark.parametrize("quoting", ["bare", "assignment", "value"])
@pytest.mark.parametrize("value", [
    '{"b": 2, "a": 1}',
    '["second", "first"]',
    '{"scale": 1.00}',
    '{ "label": "caf\u00e9" }',
])
def test_json_environment_values_roundtrip_without_normalization(
    tmp_path: Path, value: str, quoting: str
) -> None:
    token = f"CONFIG={value}"
    if quoting == "assignment":
        token = shlex.quote(token)
    elif quoting == "value":
        token = f"CONFIG={shlex.quote(value)}"
    result = resolve_effective_config(_handoff(
        _recipe(tmp_path, "vllm", ""),
        extra_envs={"CONFIG": value},
        accepted_env=token,
    ))

    assert result.final_env["CONFIG"] == value
    assert result.conflicts == []


def test_distinct_json_environment_strings_still_conflict(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="conflicting environment variable.*CONFIG"):
        resolve_effective_config(_handoff(
            _recipe(tmp_path, "vllm", ""),
            extra_envs={"CONFIG": '{"scale":1.0}'},
            accepted_env=shlex.quote('CONFIG={"scale":1.00}'),
        ))


def test_manifest_digest_and_dict_are_deterministic(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path, "vllm", "--dtype auto", A="1")
    first = resolve_effective_config(_handoff(recipe))
    second = resolve_effective_config(_handoff(recipe))

    expected = hashlib.sha256(
        json.dumps(
            first.manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert isinstance(first, EffectiveConfig)
    assert first.digest == second.digest == expected
    assert first.to_dict()["final_server_args"] == "--dtype auto"


def test_schema_v1_is_legacy_accepted_value_passthrough(tmp_path: Path) -> None:
    missing_recipe = str(tmp_path / "does-not-exist.yaml")
    result = resolve_effective_config(
        {
            "schema_version": 1,
            "framework": "vllm",
            "launch_recipe": missing_recipe,
            "accepted_flags": "--legacy=true --boolean",
            "accepted_env": "LEGACY=1",
            "baseline_env_spec": {
                "config": {
                    "server_launch_flags": "--must-not-merge",
                    "extra_server_args": "--must-not-merge",
                    "extra_envs": {"MUST_NOT_MERGE": "1"},
                }
            },
        }
    )

    assert result.final_server_args == "--legacy=true --boolean"
    assert result.final_env == {"LEGACY": "1"}
    assert result.base_overlay_pythonpath == ""
    assert result.source_snapshots == []


# ── parser edges ────────────────────────────────────────────────────────────
# These parsers sit between a recipe written by hand and a server that refuses to start on a
# malformed argument. Ambiguous text is either canonicalised or rejected with the offending token
# named — never silently dropped, which is how a run baselines on a config nobody asked for.


def test_a_bare_json_argument_survives_shell_splitting() -> None:
    """shlex would shred `{"a": 1, "b": 2}` into three tokens on the spaces. It is protected,
    canonicalised (sorted, no spaces) and put back, so two spellings of one value compare equal."""
    tokens = ec._shell_tokens('--override {"b": 2, "a": 1}')
    assert tokens[0] == "--override"
    assert tokens[1] == '{"a":1,"b":2}'


def test_a_quote_escaped_inside_json_does_not_end_the_string() -> None:
    """The scanner tracks its own escapes; a `\\"` that closed the string early would leave the
    braces unbalanced and drop the argument."""
    tokens = ec._shell_tokens('--override {"a": "he said \\"hi\\""}')
    assert json.loads(tokens[1]) == {"a": 'he said "hi"'}


def test_an_unbalanced_brace_is_left_alone() -> None:
    text, protected = ec._protect_bare_json('--override {"a": 1')
    assert protected == {} and text == '--override {"a": 1'


def test_something_that_only_looks_like_json_is_left_alone() -> None:
    text, protected = ec._protect_bare_json("--override {not json}")
    assert protected == {} and text == "--override {not json}"


def test_a_json_shaped_value_that_will_not_parse_is_passed_through() -> None:
    assert ec._canonical_value("[1, 2") == "[1, 2"
    assert ec._canonical_value("[nope]") == "[nope]"
    assert ec._canonical_value("[2, 1]") == "[2,1]"


@pytest.mark.parametrize("token,is_flag", [
    ("--max-model-len", True),
    ("-tp", True),
    ("-", False),          # a lone dash is a value (stdin), not a flag
    ("-1", False),         # a negative number is a value
    ("-1.5", False),
    ("8192", False),
])
def test_what_counts_as_a_flag(token: str, is_flag: bool) -> None:
    assert ec._looks_like_flag(token) is is_flag


def test_a_value_with_no_flag_is_rejected_by_name() -> None:
    """Dropping it would launch a server missing an argument the recipe asked for."""
    with pytest.raises(ValueError, match="no flag"):
        ec._parse_flags("8192 --max-model-len")


def test_an_environment_entry_must_be_a_pair() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        ec._parse_env("JUST_A_NAME")
    with pytest.raises(ValueError, match="KEY=VALUE"):
        ec._parse_env("=novalue")


def test_an_env_mapping_is_taken_as_given() -> None:
    assert ec._parse_env({"A": 1}) == {"A": "1"}
    assert ec._parse_env("") == {}
    assert ec._parse_env(None) == {}


def test_envs_are_found_at_any_depth_and_absence_is_not_an_error() -> None:
    """Recipes nest `envs` under the framework block, but not always at the same depth."""
    assert ec._find_envs({"benchmark": {"envs": {"A": "1"}}}) == {"A": "1"}
    assert ec._find_envs({"a": {"b": {"c": {"envs": {"X": "9"}}}}}) == {"X": "9"}
    assert ec._find_envs({"no": "envs here"}) == {}
    assert ec._find_envs("not a mapping") == {}


def test_a_recipe_that_cannot_be_read_names_itself(tmp_path: Path) -> None:
    """The path is in the message because the usual cause is a recipe that was never mounted
    into the container, and the next question is always "which one"."""
    assert ec._recipe_envs("") == {}
    with pytest.raises(ValueError, match="cannot parse launch recipe"):
        ec._recipe_envs(tmp_path / "absent.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("benchmark: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot parse launch recipe"):
        ec._recipe_envs(bad)


def test_a_handoff_is_read_from_a_path_or_taken_as_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    assert ec._load_handoff(path)["schema_version"] == 2
    assert ec._load_handoff(str(path))["schema_version"] == 2

    given = {"schema_version": 2}
    loaded = ec._load_handoff(given)
    loaded["schema_version"] = 99
    assert given["schema_version"] == 2, "the caller's dict is copied, not aliased"


def test_a_handoff_that_is_not_an_object_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot parse handoff"):
        ec._load_handoff(tmp_path / "absent.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot parse handoff"):
        ec._load_handoff(broken)
    listy = tmp_path / "list.json"
    listy.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        ec._load_handoff(listy)
