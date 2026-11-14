from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "rebase_commandment.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("rebase_commandment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
SEED_SPEC = importlib.util.spec_from_file_location(
    "seed_fast_test_cache", MODULE_PATH.parent / "seed_fast_test_cache.py"
)
SEED_MODULE = importlib.util.module_from_spec(SEED_SPEC)
assert SEED_SPEC and SEED_SPEC.loader
SEED_SPEC.loader.exec_module(SEED_MODULE)


def test_rebases_only_eval_dir(tmp_path: Path) -> None:
    old = "/tmp/geak_runs/team_old/ws_kernel"
    target_eval = tmp_path / "geak_runs" / "team_new" / "ws_kernel"
    target_eval.mkdir(parents=True)
    source = tmp_path / "cached.md"
    source.write_text(
        f"# contract\nEVAL_DIR={old}\ncd {old}/workspace\n"
        "candidate=<CANDIDATE_TREE>\n",
        encoding="utf-8",
    )

    result = MODULE.rebase_commandment(
        source, target_eval / "COMMANDMENT.md", target_eval
    )
    output = (target_eval / "COMMANDMENT.md").read_text(encoding="utf-8")

    assert result["materialized"] is True
    assert old not in output
    assert str(target_eval) in output
    assert "candidate=<CANDIDATE_TREE>" in output


def test_rejects_fixed_candidate_path(tmp_path: Path) -> None:
    old = "/tmp/geak_runs/team_old/ws_kernel"
    target_eval = tmp_path / "geak_runs" / "team_new" / "ws_kernel"
    target_eval.mkdir(parents=True)
    source = tmp_path / "cached.md"
    source.write_text(
        f"EVAL_DIR={old}\n"
        "candidate=/tmp/geak_state/state/candidates/fixed/tree\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fixed candidate"):
        MODULE.rebase_commandment(
            source, target_eval / "COMMANDMENT.md", target_eval
        )


def test_rejects_multiple_eval_dirs(tmp_path: Path) -> None:
    target_eval = tmp_path / "geak_runs" / "team_new" / "ws_kernel"
    target_eval.mkdir(parents=True)
    source = tmp_path / "cached.md"
    source.write_text(
        "a=/tmp/geak_runs/team_a/ws_kernel/workspace\n"
        "b=/tmp/geak_runs/team_b/ws_kernel/workspace\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one"):
        MODULE.rebase_commandment(
            source, target_eval / "COMMANDMENT.md", target_eval
        )


def test_seeds_self_contained_keyed_cache(tmp_path: Path) -> None:
    source_eval = tmp_path / "geak_runs" / "team_old" / "ws_kernel"
    source_eval.mkdir(parents=True)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    bench = tmp_path / "bench.py"
    bench.write_text("print('bench')\n", encoding="utf-8")
    evidence = source_eval / "setup_control" / "evidence_manifest.json"
    evidence.parent.mkdir()
    evidence.write_text('{"passed": true}\n', encoding="utf-8")
    (source_eval / "baseline_timing.json").write_text(
        '{"reliable": true}\n', encoding="utf-8"
    )
    (source_eval / "COMMANDMENT.md").write_text(
        f"EVAL_DIR={source_eval}\ncd {source_eval}/workspace\n",
        encoding="utf-8",
    )
    (source_eval / "setup_ab_control.json").write_text(
        json.dumps(
            {
                "claim_complete": True,
                "ran": True,
                "passed": True,
                "evidence_manifest": str(evidence),
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "state" / "fast_test_cache"

    result = SEED_MODULE.seed_cache(
        source_eval,
        cache,
        frozen,
        bench,
        '{"magnitude":"constructed"}',
        '["8192_uniform"]',
    )
    cached_control = json.loads(
        (cache / "setup_ab_control.json").read_text(encoding="utf-8")
    )

    assert result["published"] is True
    assert (cache / "fast_test_key.json").is_file()
    assert (cache / "setup_ab_control_evidence_manifest.json").is_file()
    assert cached_control["evidence_manifest"] == str(
        cache / "setup_ab_control_evidence_manifest.json"
    )
