"""Full-worker CPU regressions for evidence lost on overwrite, revert or restart."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def worker(tmp_path):
    task, work, skill = tmp_path / "task", tmp_path / "work", tmp_path / "skill"
    write(task / "kernel_src/kernel.py", "VALUE = 1\n")
    write(task / "meta.json", '{"candidate_bind":{"file":"kernel_src/kernel.py"}}')
    write(task / "unittest.py", "# CPU fixture\n")
    write(task / "run/extraction.json", '{"source":"original extraction receipt"}\n')
    write(tmp_path / "clock", "1000\n")
    write(tmp_path / "bin/date", '#!/bin/sh\nif [ "$1" = +%s ]; then cat "$TEST_CLOCK"; else exec /bin/date "$@"; fi\n')
    (tmp_path / "bin/date").chmod(0o755)
    write(skill / "prompt.sh", 'echo optimize > "$PROMPT_OUT"\n')
    write(tmp_path / "runner.sh", 'echo runner >> "$WORK_ROOT/runner_calls"\n'
          'echo "VALUE = 2" > "$TASK_DIR/kernel_src/kernel.py"\n'
          'echo changed-by-coding > "$TASK_DIR/run/extraction.json"\n'
          'echo 1002 > "$TEST_CLOCK"\n')
    env = dict(os.environ, WORK_ROOT=str(work), TASK_DIR=str(task), SKILL_ROOT=str(skill),
               RUNNER=str(tmp_path / "runner.sh"), TARGET="2", ATTEMPT_CAP="2", ATTEMPT_MIN="1",
               MEASURE_TIMEOUT="5", TEST_CLOCK=str(tmp_path / "clock"),
               PATH=str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])

    def execute(**overrides):
        with (tmp_path / "worker.log").open("a") as log:
            result = subprocess.run(["bash", str(SCRIPTS / "worker.sh"), "fixture", "cpu"],
                env=dict(env, **overrides), stdout=log, stderr=subprocess.STDOUT, timeout=15)
        return result.returncode

    return task, work, skill, execute


def custom_measure(skill, outcomes, *, make_fifo=False):
    rows = "\n".join(f'{i}) printf "%s\\n" {json.dumps(output)}; exit {rc} ;;'
                     for i, (output, rc) in enumerate(outcomes, 1))
    body = '''i=$(cat "$RUN_DIR/measure_count" 2>/dev/null || echo 0)
i=$(( i + 1 )); echo "$i" > "$RUN_DIR/measure_count"
mkdir -p "$TASK_DIR/run"
printf '{"sweep":%s,"fixture":"CPU only"}\\n' "$i" > "$TASK_DIR/run/ut_result.json"
printf 'native raw sweep %s\\n' "$i" > "$RUN_DIR/.measure.log"
printf 'driver stderr sweep %s\\n' "$i" >&2
'''
    if make_fifo:
        body += 'mkfifo "$TASK_DIR/run/z_unretained_fifo"\n'
    body += 'case "$i" in\n' + rows + '\n*) exit 99 ;;\nesac\n'
    write(skill / "measure.sh", body)


def archives(work):
    return sorted((work / "runs/fixture/attempt_1/worker_measurements").glob("sweep-*"))


def test_presearch_receipts_survive_first_coding_and_rejected_measurement(worker):
    task, work, skill, execute = worker
    custom_measure(skill, [("STATUS=0 SPEEDUP=1.8 CLEAN= SINGLE=0", 7)])
    assert execute() == 0
    run = work / "runs/fixture"
    before = Path((run / "pre_search_evidence.path").read_text().strip())
    assert json.loads((before / "run/extraction.json").read_text())["source"] == "original extraction receipt"
    assert (before / "kernel_src/kernel.py").read_text() == "VALUE = 1\n"
    saved, = archives(work)
    record = json.loads((saved / "retention.json").read_text())
    assert record["complete"] and record["measure_exit_code"] == 7
    assert record["native_exit_code"] is None
    assert "STATUS=0" in (saved / "measure.stdout.log").read_text()
    assert "driver stderr sweep 1" in (saved / "measure.stderr.log").read_text()
    assert "native raw sweep 1" in (saved / "native.log").read_text()
    assert json.loads((saved / "run/ut_result.json").read_text())["sweep"] == 1
    assert (saved / "kernel_src/kernel.py").read_text() == "VALUE = 2\n"
    assert (task / "kernel_src/kernel.py").read_text() == "VALUE = 1\n"


@pytest.mark.parametrize("second", [("STATUS=1 SPEEDUP=1.2 CLEAN=1.2 SINGLE=0", 0),
                                    ("STATUS=0 SPEEDUP= CLEAN= SINGLE=0", 9)])
def test_both_straddle_sweeps_have_independent_raw_evidence(worker, second):
    task, work, skill, execute = worker
    custom_measure(skill, [("STATUS=1 SPEEDUP=1.04 CLEAN=1.08 SINGLE=1", 0), second])
    assert execute() == 0
    first, last = archives(work)
    assert first.name.startswith("sweep-1-") and last.name.startswith("sweep-2-")
    for sequence, saved in enumerate((first, last), 1):
        assert json.loads((saved / "run/ut_result.json").read_text())["sweep"] == sequence
        assert (saved / "native.log").read_text() == f"native raw sweep {sequence}\n"
        assert (saved / "kernel_src/kernel.py").read_text() == "VALUE = 2\n"
        assert (saved / "kernel_src/kernel.py").stat().st_ino != (task / "kernel_src/kernel.py").stat().st_ino
    assert json.loads((last / "retention.json").read_text())["measure_exit_code"] == second[1]
    # The moved compatibility log and working task are mutable; archived bytes must not be.
    (last / "previous_measure.log").write_text("mutated later")
    (task / "run/ut_result.json").write_text("overwritten later")
    (task / "kernel_src/kernel.py").write_text("VALUE = 99\n")
    assert (first / "native.log").read_text() == "native raw sweep 1\n"
    assert json.loads((last / "run/ut_result.json").read_text())["sweep"] == 2
    assert (last / "kernel_src/kernel.py").read_text() == "VALUE = 2\n"


def test_default_measure_retains_actual_failed_native_exit(worker):
    task, work, _, execute = worker
    write(task / "unittest.py", 'import json,pathlib,time\n'
          'd={"started":time.time(),"ok":False,"exit_code":13,"gates":{"correctness":{"ok":False}}}\n'
          'pathlib.Path("run/ut_result.json").write_text(json.dumps(d))\n'
          'print("native partial evidence before exit 13")\nraise SystemExit(13)\n')
    assert execute() == 0
    saved, = archives(work)
    record = json.loads((saved / "retention.json").read_text())
    assert record["native_exit_code"] == 13 and record["measure_exit_code"] == 0
    assert "exit 13" in (saved / "native.log").read_text()
    assert json.loads((saved / "run/ut_result.json").read_text())["exit_code"] == 13
    assert (saved / "kernel_src/kernel.py").read_text() == "VALUE = 2\n"
    assert (task / "kernel_src/kernel.py").read_text() == "VALUE = 1\n"


def test_copy_failure_stops_before_revert_and_recovers_without_remeasurement(worker):
    task, work, skill, execute = worker
    custom_measure(skill, [("STATUS=0 SPEEDUP= CLEAN= SINGLE=0", 0)], make_fifo=True)
    assert execute() == 2
    run = work / "runs/fixture"
    pending = run / ".evidence_pending"
    first = Path(pending.read_text().strip())
    record = json.loads((first / "retention.json").read_text())
    assert not record["complete"] and "non-regular evidence" in record["error"]
    assert (first / "run/ut_result.json").exists()  # Partial archive survives.
    assert (task / "kernel_src/kernel.py").read_text() == "VALUE = 2\n"
    assert execute(TARGET="0") == 78  # Resource problem unresolved: no next attempt.
    assert (work / "runner_calls").read_text().splitlines() == ["runner"]
    (task / "run/z_unretained_fifo").unlink()  # Resolve only the synthetic fixture obstacle.
    assert execute(TARGET="0") == 0
    assert not pending.exists()
    assert (run / "measure_count").read_text().strip() == "1"
    assert (work / "runner_calls").read_text().splitlines() == ["runner"]
    complete = [p for p in archives(work) if json.loads((p / "retention.json").read_text())["complete"]]
    assert len(complete) == 1 and complete[0] != first
    assert json.loads((complete[0] / "run/ut_result.json").read_text())["sweep"] == 1
    assert (complete[0] / "kernel_src/kernel.py").read_text() == "VALUE = 2\n"
    assert not json.loads((first / "retention.json").read_text())["complete"]


def test_completed_copy_with_uncleared_marker_resumes_safely(worker):
    _, work, skill, execute = worker
    custom_measure(skill, [("STATUS=1 SPEEDUP=1.01 CLEAN= SINGLE=0", 0)])
    assert execute() == 0
    saved, = archives(work)
    write(work / "runs/fixture/.evidence_pending", str(saved) + "\n")
    assert execute(TARGET="0") == 0
    assert not (work / "runs/fixture/.evidence_pending").exists()
    assert len(archives(work)) == 1
    assert (work / "runner_calls").read_text().splitlines() == ["runner"]


def test_archive_destination_cannot_enter_task_tree(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    destination = task / "run/recursive_archive"
    result = subprocess.run([sys.executable, str(SCRIPTS / "retain_measurement.py"), "begin",
                             str(task), str(destination)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "outside the entire task" in result.stderr
    assert not destination.exists()


@pytest.mark.parametrize("failure", ["error", "kill"])
def test_begin_never_returns_before_atomic_pending_publication(tmp_path, failure):
    task = tmp_path / "task"
    task.mkdir()
    pending = tmp_path / ".evidence_pending"
    previous = str(tmp_path / "previous-intact-archive") + "\n"
    pending.write_text(previous)
    code = '''import importlib.util,os,pathlib,signal,sys
spec=importlib.util.spec_from_file_location('retention',sys.argv[1])
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
pending=pathlib.Path(sys.argv[4]);original=os.replace
def interrupted(source,destination):
    if pathlib.Path(destination)==pending:
        if sys.argv[5]=='kill':os.kill(os.getpid(),signal.SIGKILL)
        raise OSError('injected pointer publication failure')
    return original(source,destination)
os.replace=interrupted
print(r.begin(pathlib.Path(sys.argv[2]),pathlib.Path(sys.argv[3]),'sweep-1',pending))
'''
    result = subprocess.run([sys.executable, "-c", code, str(SCRIPTS / "retain_measurement.py"),
                             str(task), str(tmp_path / "archives"), str(pending), failure],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert result.stdout == ""  # The worker cannot start a sweep using an unpublished path.
    assert pending.read_text() == previous
    partial, = (tmp_path / "archives").iterdir()
    assert json.loads((partial / "retention.json").read_text())["complete"] is False
    temporary, = tmp_path.glob(".evidence_pending.*.tmp")
    assert temporary.read_text() == str(partial) + "\n"


def retention_module():
    spec = importlib.util.spec_from_file_location("retention_pointer_fixture", SCRIPTS / "retain_measurement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_presearch_pointer_preserves_previous_and_recovers_completed_copy(tmp_path, monkeypatch):
    r = retention_module()
    task = tmp_path / "task"
    write(task / "run/original.json", '{"CPU fixture":"original"}')
    write(task / "kernel_src/kernel.py", "VALUE = 1\n")
    pending = tmp_path / ".evidence_pending"
    directory = r.begin(task, tmp_path / "archives", "pre-search", pending)
    complete = tmp_path / "pre_search_evidence.path"
    previous = str(tmp_path / "previous-intact-archive") + "\n"
    complete.write_text(previous)
    original = os.replace

    def fail_publication(source, destination):
        if Path(destination) == complete:
            raise OSError("injected completed pointer publication failure")
        return original(source, destination)

    with monkeypatch.context() as patcher:
        patcher.setattr(r.os, "replace", fail_publication)
        with pytest.raises(OSError, match="completed pointer"):
            r.retain(task, directory, complete_pointer=complete)
    assert complete.read_text() == previous
    assert pending.read_text() == str(directory) + "\n"
    assert json.loads((directory / "retention.json").read_text())["complete"] is True
    assert r.recover(task, pending, None) == directory
    assert complete.read_text() == str(directory) + "\n"
    assert not pending.exists()
    assert len(list((tmp_path / "archives").iterdir())) == 1


def test_failed_recovery_pointer_keeps_old_archive_address(tmp_path, monkeypatch):
    r = retention_module()
    task = tmp_path / "task"
    write(task / "run/partial.json", '{"CPU fixture":"partial"}')
    write(task / "kernel_src/kernel.py", "VALUE = 2\n")
    pending = tmp_path / ".evidence_pending"
    previous = r.begin(task, tmp_path / "archives", "sweep-1", pending)
    write(previous / "measure.stdout.log", "STATUS=0 SPEEDUP=\n")
    original = os.replace

    def fail_publication(source, destination):
        if Path(destination) == pending:
            raise OSError("injected recovery pointer publication failure")
        return original(source, destination)

    with monkeypatch.context() as patcher:
        patcher.setattr(r.os, "replace", fail_publication)
        with pytest.raises(OSError, match="recovery pointer"):
            r.recover(task, pending, None)
    assert pending.read_text() == str(previous) + "\n"
    assert (previous / "measure.stdout.log").read_text() == "STATUS=0 SPEEDUP=\n"
    recovered = r.recover(task, pending, None)
    assert recovered != previous
    assert json.loads((recovered / "retention.json").read_text())["complete"] is True
    assert (recovered / "measure.stdout.log").read_text() == "STATUS=0 SPEEDUP=\n"
    assert not pending.exists()


def test_root_baseline_random_is_retained_as_independent_opaque_bytes(tmp_path):
    r = retention_module()
    task = tmp_path / "task"
    write(task / "kernel_src/kernel.py", "VALUE = 1\n")
    oracle = task / "_baseline_random.pt"
    payload = b"\x80\x02CPU fixture oracle bytes\x00\xff"
    oracle.write_bytes(payload)
    directory = r.begin(task, tmp_path / "archives", "sweep-1")
    r.retain(task, directory)
    saved = directory / oracle.name
    assert saved.read_bytes() == payload
    assert saved.stat().st_ino != oracle.stat().st_ino
    oracle.write_bytes(b"overwritten by another synthetic sweep")
    assert saved.read_bytes() == payload
    record = json.loads((directory / "retention.json").read_text())
    assert record["complete"] and oracle.name in record["artifact_roots"]
    assert oracle.name not in record["missing"]
    assert record["copied_bytes"] == len(payload) + (task / "kernel_src/kernel.py").stat().st_size


def test_missing_root_baseline_random_is_explicit_and_optional(tmp_path):
    r = retention_module()
    task = tmp_path / "task"
    write(task / "kernel_src/kernel.py", "VALUE = 1\n")
    directory = r.begin(task, tmp_path / "archives", "sweep-1")
    r.retain(task, directory)
    record = json.loads((directory / "retention.json").read_text())
    assert record["complete"]
    assert "_baseline_random.pt" in record["artifact_roots"]
    assert "_baseline_random.pt" in record["missing"]
    assert not (directory / "_baseline_random.pt").exists()


@pytest.mark.parametrize("broken", [False, True])
def test_root_baseline_random_link_is_rejected_without_losing_pending_evidence(tmp_path, broken):
    r = retention_module()
    task = tmp_path / "task"
    write(task / "kernel_src/kernel.py", "VALUE = 1\n")
    external = tmp_path / "external-oracle.pt"
    if not broken:
        external.write_bytes(b"CPU fixture external bytes")
    (task / "_baseline_random.pt").symlink_to(external)
    pending = tmp_path / ".evidence_pending"
    directory = r.begin(task, tmp_path / "archives", "sweep-1", pending)
    with pytest.raises(ValueError, match="non-regular evidence"):
        r.retain(task, directory)
    record = json.loads((directory / "retention.json").read_text())
    assert not record["complete"]
    assert "_baseline_random.pt" in record["artifact_roots"]
    assert pending.read_text() == str(directory) + "\n"
    assert (directory / "kernel_src/kernel.py").is_file()
    assert not (directory / "_baseline_random.pt").exists()
    if not broken:
        assert external.read_bytes() == b"CPU fixture external bytes"


def test_old_completed_archive_is_not_relabelled_for_new_oracle_policy(tmp_path):
    r = retention_module()
    task = tmp_path / "task"
    write(task / "kernel_src/kernel.py", "VALUE = 1\n")
    (task / "_baseline_random.pt").write_bytes(b"new unassociated CPU fixture bytes")
    directory = tmp_path / "legacy_archive"
    directory.mkdir()
    legacy = json.dumps({"complete": True, "kind": "sweep-1", "task_dir": str(task.resolve())})
    (directory / "retention.json").write_text(legacy)
    pending = tmp_path / ".evidence_pending"
    r.write_pointer(pending, directory)
    assert r.recover(task, pending, None) == directory
    assert (directory / "retention.json").read_text() == legacy
    assert not (directory / "_baseline_random.pt").exists()
