"""CPU checks for the reference runner's literal overlay-build command prefix."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'workload/stages/applyback/reference-replay-ab/run.sh'


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def fixture(tmp_path, prefix=None):
    skill = tmp_path/'slot/reference-replay-ab'
    skill.mkdir(parents=True)
    shutil.copy2(RUNNER, skill/'run.sh')
    write(skill/'summarize.py', 'print("CPU fixture preflight")\n')
    write(skill.parent/'build_overlay.py',
          'import json,os,pathlib,sys\n'
          'pathlib.Path(os.environ["TEST_BUILD_ARGV"]).write_text(json.dumps(sys.argv[1:]))\n'
          'print("fixture builder stdout", flush=True)\n'
          'print("fixture builder stderr", file=sys.stderr, flush=True)\n'
          'raise SystemExit(int(os.environ["TEST_BUILD_RC"]))\n')
    harness = tmp_path/'harness'
    write(harness/'entry.sh', 'echo "CPU test must stop before native replay" >&2\nexit 99\n')
    write(harness/'cases.json', '{}\n')
    write(tmp_path/'reference.json', '{}\n')
    write(tmp_path/'optimized.json', '{}\n')
    params = {'harness_dir': str(harness), 'harness_cmd': 'entry.sh', 'case_file': 'cases.json',
              'reference_run': str(tmp_path/'reference.json'), 'cache_root': str(tmp_path/'cache')}
    if prefix is not None:
        params['overlay_build_command'] = prefix
    write(tmp_path/'params.json', json.dumps(params))
    bins = tmp_path/'bin'
    write(bins/'curl', '#!/bin/sh\nexit 22\n')
    (bins/'curl').chmod(0o755)
    out = tmp_path/'out'
    out.mkdir()
    env = dict(os.environ, OUT=str(out), RUN_ROOT=str(tmp_path/'run'), SKILL_ROOT=str(skill),
               GEAK_ROOT=str(tmp_path/'pinned repo with spaces'), WORKLOAD_ROOT=str(ROOT/'workload'),
               PARAMS=str(tmp_path/'params.json'), OPTIMIZED=str(tmp_path/'optimized.json'),
               TASKS_DIR=str(tmp_path/'tasks'), STAGE='applyback', STAGE_SKILL='reference-replay-ab',
               TEST_BUILD_ARGV=str(tmp_path/'builder_argv.json'), TEST_BUILD_RC='17',
               PATH=str(bins)+os.pathsep+os.environ['PATH'])
    return skill/'run.sh', env


@pytest.mark.parametrize('custom_prefix', [False, True])
def test_builder_argv_and_failure_exit_survive_tee(tmp_path, custom_prefix):
    prefix = None
    literal = 'space ; $(touch forbidden-marker) "quotes"'
    if custom_prefix:
        launcher = tmp_path/'prefix with spaces.py'
        write(launcher, 'import json,os,pathlib,subprocess,sys\n'
              'pathlib.Path(os.environ["TEST_PREFIX_ARGV"]).write_text(json.dumps(sys.argv[1:]))\n'
              'raise SystemExit(subprocess.run([sys.executable]+sys.argv[3:]).returncode)\n')
        prefix = [sys.executable, str(launcher), literal, '']
    runner, env = fixture(tmp_path, prefix)
    env['TEST_PREFIX_ARGV'] = str(tmp_path/'prefix_argv.json')
    process = subprocess.run(['bash', str(runner)], env=env, cwd=tmp_path,
                             capture_output=True, text=True, timeout=20)
    assert process.returncode == 17, process.stdout + process.stderr
    result = json.loads((tmp_path/'out/result.json').read_text())
    assert result['status'] == 'fail' and 'exit 17' in result['notes']
    arguments = json.loads((tmp_path/'builder_argv.json').read_text())
    assert arguments[arguments.index('--geak-root') + 1] == env['GEAK_ROOT']
    assert arguments[arguments.index('--settings') + 1] == env['PARAMS']
    assert arguments[arguments.index('--tasks-dir') + 1] == env['TASKS_DIR']
    log = (tmp_path/'out/overlay.log').read_text()
    assert 'fixture builder stdout' in log and 'fixture builder stderr' in log
    assert not (tmp_path/'forbidden-marker').exists()
    if custom_prefix:
        received = json.loads((tmp_path/'prefix_argv.json').read_text())
        assert received[:2] == [literal, '']


@pytest.mark.parametrize('prefix', [['/missing/overlay-builder'], [], 'python3'])
def test_unavailable_or_invalid_prefix_is_blocked_78(tmp_path, prefix):
    runner, env = fixture(tmp_path, prefix)
    process = subprocess.run(['bash', str(runner)], env=env, cwd=tmp_path,
                             capture_output=True, text=True, timeout=20)
    assert process.returncode == 78, process.stdout + process.stderr
    result = json.loads((tmp_path/'out/result.json').read_text())
    assert result['status'] == 'blocked'
    assert (tmp_path/'out/overlay.log').read_text().strip()
    assert not (tmp_path/'builder_argv.json').exists()


@pytest.mark.parametrize('builder_exit,stage_exit,status', [(3, 0, 'blocked'), (78, 78, 'blocked')])
def test_existing_no_keep_and_explicit_precondition_exits(tmp_path, builder_exit, stage_exit, status):
    runner, env = fixture(tmp_path)
    env['TEST_BUILD_RC'] = str(builder_exit)
    process = subprocess.run(['bash', str(runner)], env=env, cwd=tmp_path,
                             capture_output=True, text=True, timeout=20)
    assert process.returncode == stage_exit, process.stdout + process.stderr
    assert json.loads((tmp_path/'out/result.json').read_text())['status'] == status
