#!/usr/bin/env python3
"""UT acceptance: immutable profile inputs, GPU correctness gates and source delivery."""
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import verify_input


def main():
    sys.path.insert(0, str(HERE.parent / "_shared"))
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
    result["metrics"]["required_ut_validation"] = "correctness"
    tasks = Path(out) / "tasks"
    results = [read_json(str(p / "run/ut_result.json")) or {} for p in tasks.iterdir()
               if p.is_dir()] if tasks.is_dir() else []
    result["metrics"]["tasks_with_performance_measurements"] = sum(
        r.get("performance_measured") is True or
        (r.get("gates") or {}).get("timing", {}).get("status") == "pass" for r in results)
    return emit(result["status"], artifacts=result["artifacts"], metrics=result["metrics"],
                notes=result.get("notes", ""))


if __name__ == "__main__":
    raise SystemExit(main())
