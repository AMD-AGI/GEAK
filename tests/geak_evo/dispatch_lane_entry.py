"""Optimizer command for the lane composition check in
kernel_workflow/scripts/test_optimizer_dispatch.js: the real dispatch adapter over the
CPU owner fixture of test_command.py. A test helper, not a production entry.

  dispatch_lane_entry.py --probe            print the admitted task (ref, files, cases)
  dispatch_lane_entry.py --task T --out O   run geak_evo.dispatch.main on the lane's task

EVO_LANE_FIXTURE_ROOT is the fixture directory, so the probe and the run build the same
admitted task (same paths and bytes, so the same task ref). EVO_LANE_DISPATCH_ROOT is the
lane's <eval_dir>/optimizer, which the owner binds in advance.
"""

import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest import mock

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]

from geak_evo import command, dispatch, result  # noqa: E402

CASES = {"lane-a": "case-a", "lane-b": "case-b"}
DEVICE = {"arch": "gfx950", "gpus": ["0"]}


def fixture():
    root, real = Path(os.environ["EVO_LANE_FIXTURE_ROOT"]), tempfile.mkdtemp
    def mkdtemp(*args, **kwargs):
        if root.exists():
            return real(*args, **kwargs)
        root.mkdir(mode=0o700)
        return str(root)
    with mock.patch.object(tempfile, "mkdtemp", mkdtemp):
        import test_command
        owner = test_command.CommandTests(methodName="runTest")
        owner.setUp()
    clock = command._document(owner.config["search"]["clock_ref"])
    clock["started_unix"] = time.time()
    owner.config["search"]["clock_ref"] = owner.save("clock.json", clock)
    return owner


def main(argv):
    owner = fixture()
    try:
        contract = owner.f.contract
        native = command._document(owner.config["search"]["templates"]["config"])
        if argv == ["--probe"]:
            files = {name: base64.b64encode((owner.f.task / name).read_bytes()).decode()
                     for name in contract["artifacts"]}
            print(json.dumps({"task_ref": native["contract_ref"], "files": files,
                              "editable": contract["source_paths"], "cases": CASES, "device": DEVICE}))
            return 0
        binding = {"schema": "geak-evo-dispatch-owner-v1", "dispatch_root": os.environ["EVO_LANE_DISPATCH_ROOT"],
                   "owner_config_ref": owner.save("owner.json", owner.config), "task_ref": native["contract_ref"],
                   "max_steps": 1, "export_seconds": 30, "case_map": CASES, "device": DEVICE}
        with mock.patch.object(command, "prepare_native_search", return_value=owner.search), \
             mock.patch.object(command, "validate_final_recommendation", return_value=owner.boundary.selection), \
             mock.patch.object(result, "validate_final_recommendation", return_value=owner.boundary.selection):
            return dispatch.main(argv, owner_binding_ref=owner.save("dispatch-owner.json", binding),
                                 store=owner.f.store, services=owner.services, evaluator=owner.f.engine,
                                 budget=owner.budget, episode=lambda *args: None)
    finally:
        owner.doCleanups()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
