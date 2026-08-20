#!/usr/bin/env python3
"""Behavior test for run_model.sh's advisory material audit."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUN_MODEL = REPO / "ci" / "node" / "run_model.sh"


def test_missing_optional_material_is_advisory(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    model_dir = runtime / "fixture-model"
    model_dir.mkdir(parents=True)
    (model_dir / "handoff.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model_path": "/nonexistent/dry-run-model",
                "framework": "vllm",
                "tp": 1,
                "exp_root": "/nonexistent/source-exp",
            }
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env.update(
        HF_LOGS=str(runtime),
        GEAK_ROOT=str(REPO),
        INFERENCEX_PATH="",
        EXP_ROOT=str(tmp_path / "exp"),
        OUT_DIR=str(tmp_path / "out"),
        RUN_TS="material-audit-test",
    )
    proc = subprocess.run(
        ["bash", str(RUN_MODEL), "fixture-model", "--dry-run"],
        env=env,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )

    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert "material audit: fixture-model" in output
    assert "MISSING (advisory)" in output
    assert "optional TraceLens/trace prior(s) missing" in output
    assert "DRY-RUN mapping OK" in output
