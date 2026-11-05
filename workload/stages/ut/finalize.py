#!/usr/bin/env python3
"""UT acceptance: immutable profile inputs, full GPU gates and source delivery."""
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import verify_input


def main():
    scripts = HERE.parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from emit_result import emit, read_json
    out = os.environ.get("OUT")
    if not out:
        return 2
    try:
        doc = verify_input(out)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return emit("blocked", artifacts={}, notes="UT profile input is invalid: " + str(exc))
    # Reuse the established task/source acceptance checks, not a second definition
    # of a passing UT. Whole-serving coverage is a separate profile/model result.
    path = HERE.parent / "profile/finalize.py"
    spec = importlib.util.spec_from_file_location("profile_delivery_finalizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rc = module.main(params_override=doc["selection_params"], require_serving_coverage=False)
    result = read_json(str(Path(out) / "result.json"))
    if not result:
        return rc
    receipt = read_json(str(Path(out) / "profile_input.json"))
    result.setdefault("artifacts", {})["profile_input"] = "profile_input.json"
    result.setdefault("metrics", {})["profile_handoff_sha256"] = receipt["handoff_sha256"]
    return emit(result["status"], artifacts=result["artifacts"], metrics=result["metrics"],
                notes=result.get("notes", ""))


if __name__ == "__main__":
    raise SystemExit(main())
