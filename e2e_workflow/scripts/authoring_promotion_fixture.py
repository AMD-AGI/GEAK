#!/usr/bin/env python3
"""Deterministic patch-promotion fixture for the authoring E2E plumbing.

This is deliberately not a performance benchmark.  It starts at the Integrator
handoff and proves that a patch, overlay, required accepted_env binding,
provenance, engagement marker, banked record, final bundle, Director-style
numeric check, and canonical workflow return survive as one reproducible chain.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
OVERLAY_SETUP = SCRIPT_DIR / "overlay_setup.py"
ENV_BINDING = "GEAK_AUTHORING_FIXTURE=enabled"
TARGET_SEAM = "fixture_target:kernel"
CANDIDATE_BINDING = {
    "kind": "rebind",
    "target": TARGET_SEAM,
    "impl_module": "fixture_impl",
    "impl_attr": "kernel",
    "file": "kernel_src/fixture_impl.py",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(argv: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, env=env, check=True, capture_output=True, text=True)


def build_fixture(eval_dir: Path) -> dict[str, Any]:
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    source = eval_dir / "source"
    task = eval_dir / "task"
    overlay = eval_dir / "overlay" / "candidate"
    source.mkdir(parents=True)
    task.mkdir(parents=True)

    baseline_text = "def kernel(x):\n    return x + 1\n"
    authored_text = """import json
import os
from pathlib import Path

def kernel(x):
    if os.environ.get("GEAK_AUTHORING_FIXTURE") != "enabled":
        return -999
    marker = Path(os.environ["GEAK_AUTHORING_MARKER"])
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"engaged": True, "input": x}) + "\\n", encoding="utf-8")
    return x + 1
"""
    (source / "fixture_target.py").write_text(baseline_text, encoding="utf-8")
    impl = task / "fixture_impl.py"
    impl.write_text(authored_text, encoding="utf-8")

    patch = task / "final_patch.diff"
    patch.write_text("".join(difflib.unified_diff(
        baseline_text.splitlines(keepends=True),
        authored_text.splitlines(keepends=True),
        fromfile="a/fixture_target.py",
        tofile="b/fixture_target.py",
    )), encoding="utf-8")
    oracle = task / "reference_io.json"
    _write_json(oracle, {"cases": [{"input": i, "output": i + 1} for i in (-7, 0, 4, 19)]})

    _run([
        "python3", str(OVERLAY_SETUP), "add-rebind",
        "--overlay", str(overlay),
        "--target", TARGET_SEAM,
        "--impl-module", "fixture_impl",
        "--impl-attr", "kernel",
        "--impl-file", str(impl),
    ])

    marker = eval_dir / "engagement" / "fixture.json"
    integration = {
        "short_name": "authoring_plumbing_fixture",
        "winner_kind": "patch",
        "gate": "accepted",
        "ab_complete": True,
        "provenance_ok": True,
        "provenance": {
            "authored_patch_sha256": _sha256(patch),
            "oracle_sha256": _sha256(oracle),
            "target_seam": TARGET_SEAM,
            "candidate_binding": CANDIDATE_BINDING,
        },
        "engagement_evidence": str(marker),
        "accepted_overlay": str(overlay),
        "accepted_env": ENV_BINDING,
        "accepted_flags": "",
        "output_parity": "pass",
        "parity_kind": "op_tolerance",
        "reason": "plumbing fixture only; no performance claim",
    }
    _write_json(eval_dir / "integrate_result.json", integration)

    # Deterministic bank: preserve replay identity and the exact accepted binding.
    banked = {
        "name": integration["short_name"],
        "winner_kind": "patch",
        "patch": str(patch),
        "accepted_overlay": integration["accepted_overlay"],
        "apply_env": integration["accepted_env"],
        "apply_flags": integration["accepted_flags"],
        "provenance_ok": integration["provenance_ok"],
        "provenance": integration["provenance"],
        "engagement_evidence": integration["engagement_evidence"],
        "target_callable": TARGET_SEAM,
        "candidate_bind": CANDIDATE_BINDING,
    }
    _write_json(eval_dir / "accepted_kernels.json", [banked])

    final = eval_dir / "final"
    final_overlay = final / "overlay"
    final.mkdir(parents=True)
    shutil.copytree(overlay, final_overlay)
    shutil.copy2(patch, final / "final_patch.diff")
    shutil.copy2(source / "fixture_target.py", final / "fixture_target.py")
    driver = final / "fixture_driver.py"
    driver.write_text(
        "import json\n"
        "from fixture_target import kernel\n"
        "print(json.dumps([kernel(x) for x in (-7, 0, 4, 19)]))\n",
        encoding="utf-8",
    )
    launch = final / "final_launch.sh"
    launch.write_text(
        "#!/bin/sh\nset -eu\n"
        'E=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)\n'
        'export PYTHONPATH="$E/final/overlay:$E/final${PYTHONPATH:+:$PYTHONPATH}"\n'
        f"export {ENV_BINDING}\n"
        'export GEAK_AUTHORING_MARKER="$E/engagement/fixture.json"\n'
        'exec python3 "$E/final/fixture_driver.py"\n',
        encoding="utf-8",
    )
    launch.chmod(0o755)

    # Director-style independent check: baseline and final launcher must agree
    # numerically, while only the candidate may create the engagement marker.
    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = str(source)
    base_env.pop("GEAK_AUTHORING_FIXTURE", None)
    baseline = json.loads(_run(["python3", str(driver)], env=base_env).stdout)
    candidate = json.loads(_run([str(launch)]).stdout)
    oracle_outputs = [case["output"] for case in json.loads(oracle.read_text())["cases"]]
    marker_payload = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}

    checks = {
        "patch_nonempty": (final / "final_patch.diff").stat().st_size > 0,
        "overlay_nonempty": any(final_overlay.iterdir()),
        "accepted_env_in_launcher": ENV_BINDING in launch.read_text(encoding="utf-8"),
        "provenance_ok": integration["provenance_ok"],
        "patch_digest_match": _sha256(final / "final_patch.diff")
        == integration["provenance"]["authored_patch_sha256"],
        "oracle_digest_match": _sha256(oracle) == integration["provenance"]["oracle_sha256"],
        "target_and_binding_match": (
            integration["provenance"]["target_seam"] == TARGET_SEAM
            and integration["provenance"]["candidate_binding"] == CANDIDATE_BINDING
        ),
        "numeric_gate": baseline == candidate == oracle_outputs,
        "engagement_marker": marker_payload.get("engaged") is True,
    }
    status = "pass" if all(checks.values()) else "fail"
    director = {
        "validation_status": status,
        "checks": checks,
        "baseline_outputs": baseline,
        "candidate_outputs": candidate,
        "performance_claim": False,
        "note": "Fixture validates plumbing only.",
    }
    _write_json(eval_dir / "director_e2e_validation.json", director)

    workflow_return = {
        "schema_version": 1,
        "mode": "authoring_promotion_fixture",
        "eval_dir": str(eval_dir),
        "validation_status": status,
        "final_overlay": str(final_overlay),
        "final_patch": str(final / "final_patch.diff"),
        "final_launch_script": str(launch),
        "accepted_kernels": [banked],
        "accepted_config": {"env": ENV_BINDING, "flags": ""},
        "output_parity": "pass" if checks["numeric_gate"] else "fail",
        "engagement_evidence": str(marker),
        "handoff_error_class": "",
        "recovered_from_disk": False,
        "performance_claim": False,
    }
    _write_json(eval_dir / "workflow_return.json", workflow_return)
    if status != "pass":
        raise RuntimeError(f"authoring promotion fixture failed: {checks}")
    return workflow_return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_fixture(args.eval_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
