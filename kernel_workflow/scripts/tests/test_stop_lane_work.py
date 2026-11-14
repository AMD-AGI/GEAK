import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
STOP = SCRIPTS / "stop_lane_work.py"
GPU_LOCK = SCRIPTS / "gpu_lock.sh"


def load_stop():
    spec = importlib.util.spec_from_file_location("stop_lane_work", STOP)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _lane(tmp_path):
    lane = tmp_path / "exp" / "team_K_1" / "K"
    (lane / "workspace").mkdir(parents=True)
    return lane


def _spawn(cwd, *args):
    # start_new_session: like a detached agent job, not a child the test would reap on its own.
    return subprocess.Popen(list(args) or ["sleep", "300"], cwd=cwd, start_new_session=True)


def _run_stop(lane, *extra):
    out = subprocess.run([sys.executable, str(STOP), "--eval-dir", str(lane), "--grace", "2", *extra],
                         capture_output=True, text=True, timeout=60)
    return out.returncode, json.loads(out.stdout)


def _gone(proc):
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        return False
    return True


def test_stops_work_inside_the_lane_and_its_descendants(tmp_path):
    lane = _lane(tmp_path)
    in_ws = _spawn(lane / "workspace")
    # A shell running inside the lane: its backgrounded child goes with it.
    tree = _spawn(lane / "workspace", "bash", "-c", "sleep 300 & wait")
    # Started outside the lane but naming a path under it, like `python3 <EVAL_DIR>/tools/x.py`.
    named = _spawn(tmp_path, sys.executable, "-c", "import time; time.sleep(300)", f"{lane}/tools/x.py")
    outside = _spawn(tmp_path)
    sibling = tmp_path / "exp" / "team_K_10" / "K"
    sibling.mkdir(parents=True)
    neighbour = _spawn(sibling)
    try:
        time.sleep(0.3)
        rc, result = _run_stop(lane)
        assert rc == 0 and result["ok"]
        assert _gone(in_ws) and _gone(tree) and _gone(named)
        stopped = {p["pid"] for p in result["stopped"]}
        assert {in_ws.pid, tree.pid, named.pid} <= stopped
        assert outside.poll() is None and neighbour.poll() is None
        assert not (lane / ".geak_stop").exists()
    finally:
        for p in (in_ws, tree, named, outside, neighbour):
            if p.poll() is None:
                p.kill()


def test_block_writes_stop_file_and_gpu_lock_refuses_under_it(tmp_path):
    lane = _lane(tmp_path)
    rc, result = _run_stop(lane, "--block", "--reason", "benchmark_engineer returned no baseline")
    assert rc == 0 and result["blocked"]
    assert "benchmark_engineer" in (lane / ".geak_stop").read_text()

    ran = lane / "workspace" / "ran"
    out = subprocess.run(["bash", str(GPU_LOCK), "0", "touch", str(ran)], cwd=lane / "workspace",
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 87
    assert "GEAK_LANE_STOPPED" in out.stderr
    assert not ran.exists()


def test_path_matching_respects_component_boundaries():
    stop = load_stop()
    roots = ("/r/exp/team_a",)
    assert stop.mentions("python3 /r/exp/team_a/K/unittest.py", roots)
    assert stop.mentions("cd '/r/exp/team_a' && x", roots)
    assert not stop.mentions("python3 /r/exp/team_ab/K/unittest.py", roots)
    assert not stop.mentions("cp /backup/r/exp/team_a/x .", roots)


def test_never_selects_its_own_ancestors(tmp_path):
    stop = load_stop()
    lane = _lane(tmp_path)
    here = os.getcwd()
    os.chdir(lane)
    try:
        chosen = stop.select(stop.roots_of(str(lane)))
    finally:
        os.chdir(here)
    assert os.getpid() not in chosen


def test_refuses_a_shallow_or_missing_dir(tmp_path):
    out = subprocess.run([sys.executable, str(STOP), "--eval-dir", "/tmp"], capture_output=True, text=True)
    assert out.returncode == 2
    out = subprocess.run([sys.executable, str(STOP), "--eval-dir", str(tmp_path / "missing")],
                         capture_output=True, text=True)
    assert out.returncode == 2
