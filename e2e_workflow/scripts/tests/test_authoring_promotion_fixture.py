from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "authoring_promotion_fixture.py"
SPEC = importlib.util.spec_from_file_location("authoring_promotion_fixture", SCRIPT)
fixture = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(fixture)


def test_patch_overlay_env_promotion_reaches_canonical_return(tmp_path):
    eval_dir = tmp_path / "authoring_fixture"
    result = fixture.build_fixture(eval_dir)

    assert result["validation_status"] == "pass"
    assert result["handoff_error_class"] == ""
    assert result["recovered_from_disk"] is False
    assert result["performance_claim"] is False

    integration = json.loads((eval_dir / "integrate_result.json").read_text())
    assert integration["winner_kind"] == "patch"
    assert integration["provenance_ok"] is True
    assert integration["accepted_env"] == fixture.ENV_BINDING
    assert integration["accepted_overlay"]
    assert integration["engagement_evidence"]

    banked = json.loads((eval_dir / "accepted_kernels.json").read_text())[0]
    assert banked["patch"] and banked["accepted_overlay"]
    assert banked["apply_env"] == fixture.ENV_BINDING
    assert banked["provenance"] == integration["provenance"]

    director = json.loads((eval_dir / "director_e2e_validation.json").read_text())
    assert all(director["checks"].values())
    assert director["baseline_outputs"] == director["candidate_outputs"]

    canonical = json.loads((eval_dir / "workflow_return.json").read_text())
    assert canonical == result
    assert Path(canonical["final_overlay"]).is_dir()
    assert Path(canonical["final_patch"]).stat().st_size > 0
    assert fixture.ENV_BINDING in Path(canonical["final_launch_script"]).read_text()
