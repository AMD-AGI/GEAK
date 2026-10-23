"""Regression tests for source identity across real worker KEEP/revert and cold runs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import source_tree

LEAF = 'aiter_flydsl/muse_moe_gemm.py'
TOP = 'dispatcher.py'


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def task_fixture(tmp_path, form='object'):
    task = tmp_path / 'tasks/nested'
    for directory in ('kernel_src', 'baseline_ref'):
        write(task / directory / LEAF, 'VALUE = 1\n')
        write(task / directory / TOP, '# immutable dispatcher\n')
    if form in ('object', 'file_object'):
        leaf = {('source_file' if form == 'object' else 'file'): 'kernel_src/' + LEAF}
    else:
        leaf = 'kernel_src/' + LEAF
    write(task / 'meta.json', json.dumps({'kernel_extraction': {'editable_leaves': [leaf]},
                                        'candidate_bind': {'file': 'kernel_src/' + TOP}}))
    write(task / 'unittest.py', '# frozen test entry\n')
    return task

def run_source_tree(*argv):
    return subprocess.run([sys.executable, str(SCRIPTS / 'source_tree.py')]
                          + [str(a) for a in argv], capture_output=True, text=True)


@pytest.mark.parametrize('form', ['object', 'file_object', 'string'])
@pytest.mark.parametrize('outcome', ['keep', 'reject', 'forbidden', 'frozen', 'measurement_mutates'])
def test_worker_tracks_nested_computation(tmp_path, form, outcome):
    task = task_fixture(tmp_path, form)
    work = tmp_path / 'work'
    skill = tmp_path / 'skill'
    clock_file = tmp_path / 'clock'
    clock_file.write_text('1000\n')
    bins = tmp_path / 'bin'
    write(bins / 'date', '#!/bin/sh\nif [ "$1" = +%s ]; then cat "$TEST_CLOCK"; else exec /bin/date "$@"; fi\n')
    (bins / 'date').chmod(0o755)
    write(skill / 'prompt.sh', 'echo "$KERNEL_SRC_ALL" > "$PROMPT_OUT"\n')
    write(skill / 'measure.sh', 'echo measured > "$RUN_DIR/.measure.log"\n'
          'if [ "$TEST_OUTCOME" = measurement_mutates ]; then echo changed > "$TASK_DIR/unittest.py"; fi\n'
          'echo "STATUS=$TEST_OK SPEEDUP=1.8"\n')
    runner = tmp_path / 'runner.sh'
    write(runner, 'echo "VALUE = 2" > "$TASK_DIR/kernel_src/' + LEAF + '"\n'
          'if [ "$TEST_OUTCOME" = forbidden ]; then echo changed > "$TASK_DIR/kernel_src/dispatcher.py"; fi\n'
          'if [ "$TEST_OUTCOME" = frozen ]; then echo changed > "$TASK_DIR/unittest.py"; fi\n'
          'echo 1002 > "$TEST_CLOCK"\n')
    frozen = (task / 'meta.json').read_bytes()
    env = dict(os.environ, WORK_ROOT=str(work), TASK_DIR=str(task), SKILL_ROOT=str(skill),
               RUNNER=str(runner), TARGET='2', ATTEMPT_CAP='2', ATTEMPT_MIN='1',
               MEASURE_TIMEOUT='1', TEST_CLOCK=str(clock_file), TEST_OUTCOME=outcome,
               TEST_OK='0' if outcome == 'reject' else '1',
               PATH=str(bins) + ':' + os.environ['PATH'])
    stdout = tmp_path / 'worker.log'
    # The worker's background heartbeat inherits stdout; use a file, not a PIPE.
    with stdout.open('w') as log:
        result = subprocess.run(['bash', str(SCRIPTS / 'worker.sh'), 'nested', 'cpu'],
                                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=15)
    assert result.returncode == 0, stdout.read_text()
    run = work / 'runs/nested'
    assert (run / '.prompt_1.txt').read_text().strip() == LEAF
    assert (run / 'snapshots/before_attempt_1' / LEAF).read_text() == 'VALUE = 1\n'
    assert (run / 'snapshots/before_attempt_1' / TOP).read_bytes() == (task / 'baseline_ref' / TOP).read_bytes()
    assert (task / 'meta.json').read_bytes() == frozen
    assert (task / 'unittest.py').read_text() == '# frozen test entry\n'
    assert (task / 'kernel_src' / TOP).read_bytes() == (task / 'baseline_ref' / TOP).read_bytes()
    if outcome == 'keep':
        assert (run / 'best_kernel_src' / LEAF).read_text() == 'VALUE = 2\n'
        assert (run / 'best_kernel_src' / TOP).is_file()
        assert (task / 'kernel_src' / LEAF).read_text() == 'VALUE = 2\n'
    else:
        assert (task / 'kernel_src' / LEAF).read_text() == 'VALUE = 1\n'
        assert not (run / 'best_kernel_src').exists()
    assert (run / '.measure.log').exists() is (outcome not in ('forbidden', 'frozen'))


@pytest.mark.parametrize('name', ['../escape.py', '/tmp/escape.py', 'kernel_src/missing.py', 'kernel_src/a/../../b.py'])
def test_declared_invalid_source_does_not_fall_back(tmp_path, name):
    task = task_fixture(tmp_path)
    write(task / 'meta.json', json.dumps({'kernel_extraction': {'editable_leaves': [{'source_file': name}]}}))
    with pytest.raises(ValueError):
        source_tree.editable_sources(task)


def test_restore_removes_added_files_and_bytecode(tmp_path):
    task = task_fixture(tmp_path)
    snapshot = tmp_path / 'snapshot'
    source_tree.copy_source_tree(task / 'kernel_src', snapshot)
    write(task / 'kernel_src/added.py', 'x = 1')
    write(task / 'kernel_src/__pycache__/dispatcher.pyc', 'stale')
    source_tree.copy_source_tree(snapshot, task / 'kernel_src')
    assert sorted(source_tree.source_files(task / 'kernel_src')) == [LEAF, TOP]
    assert not (task / 'kernel_src/__pycache__').exists()


def test_check_rejects_edits_outside_declared_leaves(tmp_path):
    """The gate the optimize README asks for before any number is trusted."""
    task = task_fixture(tmp_path)
    before = tmp_path / 'before'
    source_tree.copy_source_tree(task / 'kernel_src', before)
    write(task / 'kernel_src' / LEAF, 'VALUE = 2\n')
    write(task / 'kernel_src' / TOP, '# changed immutable dispatcher\n')
    allowed = ' '.join(source_tree.editable_sources(task))
    r = run_source_tree('check', before, task / 'kernel_src', '--allowed', allowed)
    assert r.returncode == 2, r.stdout
    assert TOP in r.stdout


def test_check_frozen_refuses_a_modified_harness(tmp_path):
    task = task_fixture(tmp_path)
    snapshot = tmp_path / 'frozen_harness'
    source_tree.freeze_harness(task, snapshot)
    write(task / 'unittest.py', "print('CORRECTNESS: PASS')\nprint('GEAK_GEOMEAN_SPEEDUP 10')\n")
    r = run_source_tree('check-frozen', task, snapshot)
    assert r.returncode == 2, r.stdout
    assert 'frozen harness changed: unittest.py' in r.stdout
    source_tree.restore_harness(task, snapshot)
    assert (task / 'unittest.py').read_text() == '# frozen test entry\n'
    assert run_source_tree('check-frozen', task, snapshot).returncode == 0


@pytest.mark.parametrize('operation', ['add', 'delete'])
def test_check_rejects_source_tree_membership_changes(tmp_path, operation):
    task = task_fixture(tmp_path)
    before = tmp_path / 'before'
    source_tree.copy_source_tree(task / 'kernel_src', before)
    write(task / 'kernel_src' / LEAF, 'VALUE = 2\n')
    allowed = ' '.join(source_tree.editable_sources(task))
    if operation == 'add':
        write(task / 'kernel_src/added.py', '# undeclared source\n')
    else:
        (task / 'kernel_src' / TOP).unlink()
    r = run_source_tree('check', before, task / 'kernel_src', '--allowed', allowed)
    assert r.returncode == 2, r.stdout
    assert ('added.py' if operation == 'add' else TOP) in r.stdout


def test_frozen_check_reads_bytes_when_mtime_and_size_match(tmp_path):
    task = task_fixture(tmp_path)
    snapshot = tmp_path / 'frozen'
    source_tree.freeze_harness(task, snapshot)
    assert source_tree.changed_harness(task, snapshot) == []
    path = task / 'unittest.py'
    previous = path.stat()
    path.write_text(path.read_text().replace('frozen', 'broken'))
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert source_tree.changed_harness(task, snapshot) == ['unittest.py']


def test_worker_default_measure_accepts_fresh_file_only_receipt(tmp_path):
    task = task_fixture(tmp_path)
    write(task / 'unittest.py',
          'import json, pathlib, time\n'
          'started = time.time()\nns = {}\n'
          'exec(pathlib.Path("kernel_src/' + LEAF + '").read_text(), ns)\n'
          'pathlib.Path("run").mkdir(exist_ok=True)\n'
          'd = {"started": started, "ok": True, "exit_code": 0, "gates": '
          '{"correctness": {"ok": True}, "timing": {"ok": True, '
          '"aggregate": {"primary": ns["VALUE"], "weighted": ns["VALUE"]}}}}\n'
          'pathlib.Path("run/ut_result.json").write_text(json.dumps(d))\n'
          'print("== gates ==")\n')
    clock_file = tmp_path / 'clock'
    clock_file.write_text('1000\n')
    bins = tmp_path / 'bin'
    write(bins / 'date', '#!/bin/sh\nif [ "$1" = +%s ]; then cat "$TEST_CLOCK"; else exec /bin/date "$@"; fi\n')
    (bins / 'date').chmod(0o755)
    skill = tmp_path / 'skill'
    write(skill / 'prompt.sh', 'echo optimize > "$PROMPT_OUT"\n')
    runner = tmp_path / 'runner.sh'
    write(runner, 'echo "VALUE = 2" > "$TASK_DIR/kernel_src/' + LEAF + '"\n'
          'echo 1002 > "$TEST_CLOCK"\n')
    work = tmp_path / 'work'
    env = dict(os.environ, WORK_ROOT=str(work), TASK_DIR=str(task), SKILL_ROOT=str(skill),
               RUNNER=str(runner), TARGET='2', ATTEMPT_CAP='2', ATTEMPT_MIN='1',
               MEASURE_TIMEOUT='10', TEST_CLOCK=str(clock_file),
               PATH=str(bins) + ':' + os.environ['PATH'])
    with (tmp_path / 'worker.log').open('w') as log:
        result = subprocess.run(['bash', str(SCRIPTS / 'worker.sh'), 'nested', 'cpu'],
                                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=15)
    assert result.returncode == 0
    assert (work / 'runs/nested/best_kernel_src' / LEAF).read_text() == 'VALUE = 2\n'
    status = json.loads((work / 'state/nested.status.json').read_text())
    assert status['best_speedup'] == 2


def test_full_budget_runs_a_last_scored_attempt_instead_of_early_done(tmp_path):
    task = task_fixture(tmp_path)
    work, skill, bins = tmp_path / 'work', tmp_path / 'skill', tmp_path / 'bin'
    clock = tmp_path / 'clock'
    clock.write_text('1000')
    write(bins / 'date', '#!/bin/sh\nif [ "$1" = +%s ]; then cat "$TEST_CLOCK"; else exec /bin/date "$@"; fi\n')
    (bins / 'date').chmod(0o755)
    write(skill / 'prompt.sh', 'echo optimize > "$PROMPT_OUT"\n')
    write(skill / 'measure.sh', 'echo measured >> "$RUN_DIR/measured_attempts"\necho "STATUS=1 SPEEDUP=1.8"\n')
    runner = tmp_path / 'runner.sh'
    write(runner, 'echo "VALUE = $(cat "$TEST_CLOCK")" > "$TASK_DIR/kernel_src/' + LEAF + '"\n'
          'echo $(( $(cat "$TEST_CLOCK") + 6 )) > "$TEST_CLOCK"\n')
    env = dict(os.environ, WORK_ROOT=str(work), TASK_DIR=str(task), SKILL_ROOT=str(skill),
               RUNNER=str(runner), TARGET='10', ATTEMPT_CAP='10', ATTEMPT_MIN='2',
               MEASURE_TIMEOUT='4', REQUIRE_FULL_BUDGET='1', TEST_CLOCK=str(clock),
               PATH=str(bins) + ':' + os.environ['PATH'])
    with (tmp_path / 'worker.log').open('w') as log:
        result = subprocess.run(['bash', str(SCRIPTS / 'worker.sh'), 'nested', 'cpu'],
                                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=15)
    assert result.returncode == 0
    status = json.loads((work / 'state/nested.status.json').read_text())
    assert status['credited_seconds'] == 12 and status['attempts'] == 2
    assert (work / 'runs/nested/measured_attempts').read_text().splitlines() == ['measured', 'measured']


def test_worker_resumes_from_newer_committed_credit(tmp_path):
    task = task_fixture(tmp_path)
    work, skill = tmp_path / 'work', tmp_path / 'skill'
    write(work / 'state/nested.credit', '10')
    write(work / 'state/nested.status.json', '{"status":"done","credited_seconds":20}')
    write(skill / 'prompt.sh', 'exit 99\n')
    runner = tmp_path / 'runner.sh'
    write(runner, 'exit 99\n')
    env = dict(os.environ, WORK_ROOT=str(work), TASK_DIR=str(task), SKILL_ROOT=str(skill),
               RUNNER=str(runner), TARGET='20', REQUIRE_FULL_BUDGET='1')
    with (tmp_path / 'worker.log').open('w') as log:
        result = subprocess.run(['bash', str(SCRIPTS / 'worker.sh'), 'nested', 'cpu'],
                                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=15)
    assert result.returncode == 0
    status = json.loads((work / 'state/nested.status.json').read_text())
    assert status['credited_seconds'] >= 20
    assert not list((work / 'runs/nested').glob('attempt_*'))
