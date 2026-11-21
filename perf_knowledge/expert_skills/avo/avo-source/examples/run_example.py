"""Run one portable source-level AVO example."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parents[1]
RUNNER = REPO_ROOT / "avo-workflow" / "run_geak_avo.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT.parent / f"{REPO_ROOT.name}-runs" / "avo-examples"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", help="case directory name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--budget-s", type=int)
    parser.add_argument("--final-phase-reserve-s", type=int)
    parser.add_argument("--gpu-ids")
    parser.add_argument("--operator-directions", type=int)
    parser.add_argument("--start-policy", choices=("best", "topk", "diverse", "elites"))
    parser.add_argument("--start-k", type=int)
    parser.add_argument("--escalate-workers", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--verify-repeats", type=int)
    parser.add_argument("--use-profiling", choices=("true", "false"))
    ns = parser.parse_args()
    for name in (
        "budget_s", "final_phase_reserve_s", "operator_directions", "start_k", "escalate_workers",
        "max_steps", "verify_repeats",
    ):
        value = getattr(ns, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    case_dir = (EXAMPLES_DIR / ns.case).resolve()
    if case_dir.parent != EXAMPLES_DIR or not case_dir.is_dir():
        parser.error(f"unknown example: {ns.case}")

    config_path = case_dir / "handoff.json"
    handoff = json.loads(config_path.read_text())
    handoff["kernel_path"] = str((case_dir / handoff["kernel_path"]).resolve())
    output_root = Path(os.environ.get(
        "AVO_EXAMPLE_RUNS_ROOT", DEFAULT_OUTPUT_ROOT)).expanduser().resolve()
    case_output = output_root / ns.case
    configured_exp = Path(str(handoff.get("exp_root") or "runs"))
    handoff["exp_root"] = str(
        configured_exp.resolve() if configured_exp.is_absolute()
        else (case_output / configured_exp).resolve())
    if str(handoff.get("target_language", "")).lower() == "flydsl":
        env_script = os.environ.get("AVO_FLYDSL_ENV", "")
        if not env_script and not ns.dry_run:
            parser.error("AVO_FLYDSL_ENV must name the FlyDSL environment script")
        if env_script:
            env_path = Path(env_script).expanduser().resolve()
            if not env_path.is_file():
                parser.error(f"AVO_FLYDSL_ENV does not exist: {env_path}")
            handoff["kernel_env_script"] = str(env_path)
    overrides = {
        "budget_s": ns.budget_s,
        "final_phase_reserve_s": ns.final_phase_reserve_s,
        "gpu_ids": ns.gpu_ids,
        "operator_directions": ns.operator_directions,
        "start_policy": ns.start_policy,
        "start_k": ns.start_k,
        "escalate_workers": ns.escalate_workers,
        "max_steps": ns.max_steps,
        "verify_repeats": ns.verify_repeats,
        "use_profiling": ns.use_profiling,
    }
    handoff.update({key: value for key, value in overrides.items()
                    if value is not None})

    with tempfile.TemporaryDirectory(prefix="avo-example-") as tmp:
        resolved = Path(tmp) / "handoff.json"
        resolved.write_text(json.dumps(handoff, indent=2) + "\n")
        if ns.dry_run:
            result = Path(tmp) / "result.json"
        else:
            state_dir = case_output / ".avo"
            state_dir.mkdir(parents=True, exist_ok=True)
            result = state_dir / "result.json"
        command = [sys.executable, str(RUNNER), str(resolved), str(result)]
        if ns.dry_run:
            command.append("--dry-run")
        return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
