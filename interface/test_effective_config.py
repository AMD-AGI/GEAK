"""Tests for schema-v2 effective launch configuration reconciliation."""
from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Optional

import pytest

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


def test_precedence_overrides_and_preserves_unknown_flags(tmp_path: Path) -> None:
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
    assert "--recipe-only" in tokens
    assert "--unknown-launch" in tokens
    assert "--delta-only" in tokens
    assert "--accepted-only" in tokens
    assert [entry["higher_source"] for entry in result.conflicts] == [
        "server_launch_flags",
        "current_best_delta",
    ]


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
