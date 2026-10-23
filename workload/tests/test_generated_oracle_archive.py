"""Real CPU subprocesses reproduce parent -> correctness -> oracle/timing legs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


@pytest.fixture
def task(tmp_path):
    root = tmp_path / "task"
    run = root / "run"
    run.mkdir(parents=True)
    overlay = root / "_cand_overlay"
    write(overlay / "sitecustomize.py", '''import os,sys
_is_enumerator = sys.argv and os.path.basename(sys.argv[0]) == "rocm_agent_enumerator"
if not _is_enumerator:
    with open(os.environ['OVERLAY_MARKER'],'a') as f:f.write('original-overlay-active\\n')
''')
    write(root / "leg_runner.py", '''import json,os,pathlib,sys
root=pathlib.Path(__file__).parent
with open(os.environ['LEG_EVENTS'],'a') as f:
    f.write(json.dumps({'argv':sys.argv,'profile':sys.getprofile() is not None})+'\\n')
if sys.argv[1]=='oracle':
    counter=root/'run/count';n=int(counter.read_text())+1 if counter.exists() else 1
    counter.write_text(str(n))
    (root/'_baseline_random.pt').write_bytes(('opaque-v%d-seed%s-draw%s'%(n,sys.argv[2],sys.argv[3])).encode())
''')
    write(root / "harness_lib.py", '''import os,pathlib,subprocess,sys
def baseline_random_outputs(task_dir, meta, *, seed=0, draws=0, timeout=10):
    task=pathlib.Path(task_dir)
    p=subprocess.Popen([sys.executable,str(task/'leg_runner.py'),'oracle',str(seed),str(draws)])
    if p.wait():raise RuntimeError('native leg failed')
    out=task/'_baseline_random.pt'
    if meta.get('bad_link'):
        out.unlink();out.symlink_to(os.environ['EXTERNAL_ORACLE'])
    if meta.get('remove_file'):out.unlink()
    if meta.get('native_error'):raise RuntimeError('original native failure')
    if 'native_exit' in meta:raise SystemExit(meta['native_exit'])
    if meta.get('return_none'):return None
    return {'seed':seed,'draws':draws}
''')
    write(root / "unittest.py", "# replaced by each CPU fixture\n")
    (tmp_path / "external.pt").write_bytes(b"external fixture unchanged")
    return root


def run_entry(tmp_path, task, *, archive=True, env_extra=None):
    env = dict(os.environ, OVERLAY_MARKER=str(tmp_path / "overlay.log"),
               LEG_EVENTS=str(tmp_path / "legs.jsonl"), EXTERNAL_ORACLE=str(tmp_path / "external.pt"))
    env.pop("WORK_ROOT", None)
    env.pop("WORKLOAD_ORACLE_ARCHIVE_DIR", None)
    if archive:
        env["WORKLOAD_ORACLE_ARCHIVE_DIR"] = str(tmp_path / "archives")
    env.update(env_extra or {})
    # Invoke the actual entry API and check restoration even on its native failures.
    driver = '''import importlib.util,json,subprocess,sys
sys.path.insert(0,sys.argv[1])
import run_task_entry
prior_run=subprocess.run
try:
    code=run_task_entry.main([sys.argv[2]])
except BaseException:
    print('RESTORED',sys.getprofile() is None,subprocess.run is prior_run)
    raise
print('RESTORED',sys.getprofile() is None,subprocess.run is prior_run)
raise SystemExit(code)
'''
    return subprocess.run([sys.executable, "-c", driver, str(SCRIPTS), str(task)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)


def nested_entry(task, meta=None, *, check=False):
    body = '''import json,os,pathlib,subprocess,sys
TASK=pathlib.Path(%r);RUN=TASK/'run'
assert pathlib.Path.cwd()==RUN and sys.path[0]==str(RUN)
sys.path.insert(0,str(TASK))
import harness_lib as h
assert sys.argv == [__file__]
assert pathlib.Path.cwd()==RUN and sys.path[0]==str(TASK)
for seed,draws in [(0,2),(3,4)]:
    value=h.baseline_random_outputs(TASK,%r,seed=seed,draws=draws)
    assert value=={'seed':seed,'draws':draws}
subprocess.run([sys.executable,str(TASK/'leg_runner.py'),'time'],check=True)
''' % (str(task), meta or {})
    write(task / "unittest.py", '''import json,os,pathlib,subprocess,sys,unittest.mock
TASK=pathlib.Path(__file__).parent;RUN=TASK/'run'
env=dict(os.environ,PYTHONPATH=str(TASK/'_cand_overlay'))
for label in ['primary','after_restore']:
    script=RUN/('correctness_'+label+'.py');script.write_text(%r)
    requested=[sys.executable,str(script)]
    result=subprocess.run(requested,env=env,cwd=RUN,capture_output=True,text=True,check=%r)
    assert result.args==requested
    print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr)
    if result.returncode:raise SystemExit(result.returncode)
print('NATIVE SUCCESS')
''' % (body, check))


def observations(tmp_path):
    return [json.loads(p.read_text()) for p in (tmp_path / "archives").rglob("observation.json")]


def test_nested_four_overwrites_preserve_overlay_argv_and_unwrapped_timing(tmp_path, task):
    nested_entry(task)
    original_harness = (task / "harness_lib.py").read_bytes()
    original_overlay = (task / "_cand_overlay/sitecustomize.py").read_bytes()
    result = run_entry(tmp_path, task)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESTORED True True" in result.stdout
    copies = sorted(p.read_bytes() for p in (tmp_path / "archives").rglob("_baseline_random.pt"))
    assert copies == [b"opaque-v1-seed0-draw2", b"opaque-v2-seed3-draw4",
                      b"opaque-v3-seed0-draw2", b"opaque-v4-seed3-draw4"]
    assert (task / "_baseline_random.pt").read_bytes() == copies[-1]
    assert (task / "harness_lib.py").read_bytes() == original_harness
    assert (task / "_cand_overlay/sitecustomize.py").read_bytes() == original_overlay
    children = [d for d in observations(tmp_path) if d["provenance"]]
    assert len(children) == 2
    for child in children:
        p = child["provenance"]
        assert Path(p["bootstrap_sys_argv"][0]).parent == task / "run"
        assert p["original_argv"][0].endswith(("correctness_primary.py", "correctness_after_restore.py"))
        assert p["pythonpath"] == str(task / "_cand_overlay")
        assert "non-rocm_agent_enumerator" in p["checked_argv_branch"]
        assert any("correctness_oracle_archive_" in item for item in p["os_argv"])
    legs = [json.loads(line) for line in (tmp_path / "legs.jsonl").read_text().splitlines()]
    assert len(legs) == 6 and all(not row["profile"] for row in legs)
    assert all(row["argv"][0] == str(task / "leg_runner.py") for row in legs)
    assert [row["argv"][1] for row in legs].count("time") == 2
    assert (tmp_path / "overlay.log").read_text().splitlines() == ["original-overlay-active"] * 8
    assert len(list((task / "run").glob("correctness_oracle_archive_*.py"))) == 2


@pytest.mark.parametrize("code", [None, 0, 17])
def test_direct_parent_calls_and_original_system_exit_are_preserved(tmp_path, task, code):
    write(task / "unittest.py", '''import pathlib,sys,unittest.mock
import harness_lib as h
task=pathlib.Path(__file__).parent
result=h.baseline_random_outputs(task,{},seed=7,draws=3)
assert result=={'seed':7,'draws':3}
raise SystemExit(%r)
''' % code)
    result = run_entry(tmp_path, task)
    assert result.returncode == (code or 0), result.stderr
    assert "RESTORED True True" in result.stdout
    assert [p.read_bytes() for p in (tmp_path / "archives").rglob("_baseline_random.pt")] == [b"opaque-v1-seed7-draw3"]


@pytest.mark.parametrize("nested", [False, True])
def test_archive_failure_aborts_further_generations_despite_exception_gate(tmp_path, task, nested):
    if nested:
        nested_entry(task, {"bad_link": True})
    else:
        write(task / "unittest.py", '''import pathlib
import harness_lib as h
task=pathlib.Path(__file__).parent
try:h.baseline_random_outputs(task,{'bad_link':True},seed=1,draws=1)
except Exception:pass
h.baseline_random_outputs(task,{},seed=2,draws=2)
''')
    result = run_entry(tmp_path, task)
    assert result.returncode == 78, result.stdout + result.stderr
    assert (task / "run/count").read_text() == "1"
    assert "RESTORED True True" in result.stdout
    assert any(d["status"] == "archive_failed" for d in observations(tmp_path))
    assert (tmp_path / "external.pt").read_bytes() == b"external fixture unchanged"


@pytest.mark.parametrize("meta, expected", [({"native_error": True}, 1),
                                          ({"bad_link": True, "native_exit": 17}, 17)])
def test_child_native_failure_survives_even_when_archival_also_fails(tmp_path, task, meta, expected):
    nested_entry(task, meta)
    result = run_entry(tmp_path, task)
    assert result.returncode == expected, result.stdout + result.stderr
    assert (task / "run/count").read_text() == "1"
    assert "RESTORED True True" in result.stdout
    if meta.get("native_error"):
        assert "original native failure" in result.stderr


def test_called_process_error_retains_original_requested_command(tmp_path, task):
    nested_entry(task, {"native_exit": 17}, check=True)
    result = run_entry(tmp_path, task)
    assert result.returncode == 1
    final = result.stderr.splitlines()[-1]
    assert "CalledProcessError" in final and "correctness_primary.py" in final
    assert "correctness_oracle_archive_" not in final


def test_no_archive_is_a_noop_and_zero_calls_do_not_claim_capture(tmp_path, task):
    write(task / "unittest.py", "import unittest.mock\nprint('stdlib-ok')\n")
    result = run_entry(tmp_path, task, archive=False)
    assert result.returncode == 0 and "stdlib-ok" in result.stdout
    assert not (tmp_path / "archives").exists()
    result = run_entry(tmp_path, task)
    assert result.returncode == 0
    assert observations(tmp_path)[0]["status"] == "not_observed"


@pytest.mark.parametrize("kind", ["inside_task", "bad_pending", "oracle_link"])
def test_invalid_archive_handoff_or_oracle_link_prevents_generation(tmp_path, task, kind):
    write(task / "unittest.py", "import pathlib,harness_lib\nharness_lib.baseline_random_outputs(pathlib.Path(__file__).parent, {})\n")
    extra = {}
    archive = True
    if kind == "inside_task":
        extra["WORKLOAD_ORACLE_ARCHIVE_DIR"] = str(task / "bad_archive")
    elif kind == "bad_pending":
        archive = False
        work = tmp_path / "work"; pending = work / "runs" / task.name / ".evidence_pending"
        target = work / "runs" / task.name / "attempt_1/evidence"
        target.mkdir(parents=True)
        write(target / "retention.json", json.dumps({"task_dir": str(tmp_path / "wrong"), "complete": False}))
        write(pending, str(target)); extra["WORK_ROOT"] = str(work)
    else:
        (task / "_baseline_random.pt").symlink_to(tmp_path / "external.pt")
    result = run_entry(tmp_path, task, archive=archive, env_extra=extra)
    assert result.returncode == 78, result.stderr
    assert not (task / "run/count").exists()


def test_worker_pending_archive_is_used_without_explicit_cold_env(tmp_path, task):
    work = tmp_path / "work"; target = work / "runs" / task.name / "attempt_1/evidence"
    target.mkdir(parents=True)
    write(target / "retention.json", json.dumps({"task_dir": str(task), "complete": False}))
    write(work / "runs" / task.name / ".evidence_pending", str(target))
    write(task / "unittest.py", "import pathlib,harness_lib\nharness_lib.baseline_random_outputs(pathlib.Path(__file__).parent, {},seed=1,draws=2)\n")
    result = run_entry(tmp_path, task, archive=False, env_extra={"WORK_ROOT": str(work)})
    assert result.returncode == 0, result.stderr
    assert len(list(target.rglob("_baseline_random.pt"))) == 1


def test_existing_profiler_is_refused_and_preserved(tmp_path, task, monkeypatch):
    spec = importlib.util.spec_from_file_location("oracle_profile_fixture", SCRIPTS / "generated_oracle_archive.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    prior = lambda frame, event, arg: None
    sys.setprofile(prior)
    try:
        with pytest.raises(module.OracleArchiveFailure, match="existing sys profiler"):
            with module.Observation(task, tmp_path / "archives"):
                raise AssertionError("entry must not run")
        assert sys.getprofile() is prior
    finally:
        sys.setprofile(None)


def test_unsupported_argv_sensitive_startup_is_not_silently_wrapped(tmp_path, task):
    nested_entry(task)
    with (task / "_cand_overlay/sitecustomize.py").open("a") as stream:
        stream.write("if sys.argv[0].endswith('correctness_primary.py'): pass\n")
    result = run_entry(tmp_path, task)
    assert result.returncode == 78 and "argv-sensitive" in result.stderr
    assert not (task / "run/count").exists()


def test_none_return_is_unchanged_and_copy_failure_blocks_next_call(tmp_path, task):
    write(task / "unittest.py", '''import pathlib,harness_lib as h
task=pathlib.Path(__file__).parent
assert h.baseline_random_outputs(task,{'return_none':True},seed=1,draws=2) is None
assert h.baseline_random_outputs(task,{'return_none':True,'bad_link':True},seed=2,draws=2) is None
h.baseline_random_outputs(task,{},seed=3,draws=2)
''')
    result = run_entry(tmp_path, task)
    assert result.returncode == 78, result.stdout + result.stderr
    assert (task / "run/count").read_text() == "2"
    assert "RESTORED True True" in result.stdout
    assert len(list((tmp_path / "archives").rglob("_baseline_random.pt"))) == 1


def test_successful_none_return_keeps_native_success(tmp_path, task):
    write(task / "unittest.py", '''import pathlib,harness_lib as h
task=pathlib.Path(__file__).parent
assert h.baseline_random_outputs(task,{'return_none':True},seed=5,draws=3) is None
print('native-none-success')
''')
    result = run_entry(tmp_path, task)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native-none-success" in result.stdout and "RESTORED True True" in result.stdout
    record, = observations(tmp_path)
    assert record["status"] == "captured" and not record["native_failure"]
    assert record["calls"][0]["returned_none"] is True
    assert [p.read_bytes() for p in (tmp_path / "archives").rglob("_baseline_random.pt")] == [b"opaque-v1-seed5-draw3"]


@pytest.mark.parametrize("copy_failure", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_caught_native_error_preserved_and_archive_failure_latched(tmp_path, task, copy_failure, nested):
    body = '''import pathlib,sys
task=pathlib.Path(%r)
sys.path.insert(0,str(task));import harness_lib as h
try:
    h.baseline_random_outputs(task,{'native_error':True,'bad_link':%r},seed=1,draws=2)
except RuntimeError as exc:
    assert str(exc)=='original native failure'
    print('caught-original-native')
else:
    raise AssertionError('native error was lost')
assert h.baseline_random_outputs(task,{},seed=2,draws=2)=={'seed':2,'draws':2}
print('native-continued-success')
''' % (str(task), copy_failure)
    if nested:
        write(task / "unittest.py", '''import os,pathlib,subprocess,sys
task=pathlib.Path(__file__).parent;script=task/'run/correctness_primary.py'
script.write_text(%r)
result=subprocess.run([sys.executable,str(script)],cwd=script.parent,
                      env=dict(os.environ,PYTHONPATH=str(task/'_cand_overlay')))
raise SystemExit(result.returncode)
''' % body)
    else:
        write(task / "unittest.py", body)
    result = run_entry(tmp_path, task)
    assert result.returncode == (78 if copy_failure else 0), result.stdout + result.stderr
    assert "caught-original-native" in result.stdout and "RESTORED True True" in result.stdout
    assert ("native-continued-success" in result.stdout) is not copy_failure
    assert (task / "run/count").read_text() == ("1" if copy_failure else "2")
    records = observations(tmp_path)
    calls = [call for record in records for call in record["calls"]]
    assert calls[0]["returned_none"] is True
    assert all(record["status"] == ("archive_failed" if copy_failure else "captured") for record in records)
    copies = sorted(p.read_bytes() for p in (tmp_path / "archives").rglob("_baseline_random.pt"))
    assert copies == ([] if copy_failure else [b"opaque-v1-seed1-draw2", b"opaque-v2-seed2-draw2"])


def test_timeout_preserves_original_command_and_failure(tmp_path, task):
    body = '''import pathlib,time
import sys
TASK=pathlib.Path(%r)
sys.path.insert(0,str(TASK));import harness_lib as h
time.sleep(5)
h.baseline_random_outputs(TASK,{},seed=0,draws=2)
''' % str(task)
    write(task / "unittest.py", '''import os,pathlib,subprocess,sys
task=pathlib.Path(__file__).parent;script=task/'run/correctness_primary.py'
script.write_text(%r)
subprocess.run([sys.executable,str(script)],cwd=script.parent,
               env=dict(os.environ,PYTHONPATH=str(task/'_cand_overlay')),timeout=0.1)
''' % body)
    result = run_entry(tmp_path, task)
    assert result.returncode == 1
    final = result.stderr.splitlines()[-1]
    assert "TimeoutExpired" in final and "correctness_primary.py" in final
    assert "correctness_oracle_archive_" not in final
    assert not (task / "run/count").exists()


def test_removed_observer_is_reported_as_failure_not_zero_capture(tmp_path, task):
    write(task / "unittest.py", "import sys\nsys.setprofile(None)\n")
    result = run_entry(tmp_path, task)
    assert result.returncode == 78
    assert observations(tmp_path)[0]["status"] == "archive_failed"
    assert "RESTORED True True" in result.stdout
