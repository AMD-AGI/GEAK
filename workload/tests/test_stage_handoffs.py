"""Stage continuation binds receipts and cannot revive a failed script's old success."""
import argparse
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import stage


def fixture_pipeline(tmp_path, monkeypatch):
    root = tmp_path/'workload'
    monkeypatch.setattr(stage, 'WORKLOAD_ROOT', str(root))
    cfg = {'run_root': str(tmp_path/'run'), 'runner': 'stub',
           'target': {'enforce': False}, 'order': ['baseline', 'optimize'],
           'stages': {name: {'skill': 'fixture', 'mode': 'script', 'params': {}}
                      for name in ['baseline', 'optimize']}}
    for name in cfg['order']:
        skill = root/'stages'/name/'fixture'
        skill.mkdir(parents=True)
        (skill/'SKILL.md').write_text('Fixture stage.\n')
        result = {'stage': name, 'skill': 'fixture', 'status': 'ok',
                  'artifacts': {'evidence': 'evidence.txt'}, 'metrics': {}}
        (skill/'run.sh').write_text(
            'echo executed >> "$RUN_ROOT/invocations"\n'
            'echo measured > "$OUT/evidence.txt"\n'
            'echo retained-log\n'
            'cat > "$OUT/result.json" <<\'RECEIPT\'\n'
            + json.dumps(result) + '\nRECEIPT\n')
    return cfg, root


def run_stage(cfg, name):
    return stage.cmd_run(cfg, argparse.Namespace(stage=name))


def test_failed_script_preserves_previous_receipt_and_log_without_reusing_success(tmp_path, monkeypatch):
    cfg, root = fixture_pipeline(tmp_path, monkeypatch)
    assert run_stage(cfg, 'baseline') == 0
    out = tmp_path/'run/baseline/out'
    previous = (out/'result.json').read_bytes()
    (root/'stages/baseline/fixture/run.sh').write_text('echo failed-new-invocation\nexit 9\n')
    assert run_stage(cfg, 'baseline') == 1
    assert not (out/'result.json').exists()
    assert next(out.glob('result.json.before-run.*')).read_bytes() == previous
    assert 'retained-log' in next(out.glob('stage.log.before-run.*')).read_text()
    assert 'failed-new-invocation' in (out/'stage.log').read_text()
    assert stage.load_state(cfg)['stages']['baseline']['status'] == 'fail'


def test_nonzero_script_cannot_publish_an_ok_receipt(tmp_path, monkeypatch):
    cfg, root = fixture_pipeline(tmp_path, monkeypatch)
    script = root/'stages/baseline/fixture/run.sh'
    script.write_text(script.read_text() + '\nexit 9\n')
    assert run_stage(cfg, 'baseline') == 1
    errors = stage.load_state(cfg)['stages']['baseline']['errors']
    assert any('exited 9 despite status=ok' in e for e in errors)


def test_rerun_preserves_declared_payload_bytes_as_well_as_result_and_log(tmp_path, monkeypatch):
    cfg, root = fixture_pipeline(tmp_path, monkeypatch)
    assert run_stage(cfg, 'baseline') == 0
    out = tmp_path/'run/baseline/out'
    previous = (out/'evidence.txt').read_bytes()
    script = root/'stages/baseline/fixture/run.sh'
    script.write_text(script.read_text().replace('echo measured', 'echo new-measurement'))
    assert run_stage(cfg, 'baseline') == 0
    assert (out/'evidence.txt').read_bytes() != previous
    saved = list((out/'.previous_runs').glob('*/evidence.txt'))
    assert len(saved) == 1 and saved[0].read_bytes() == previous
    assert (saved[0].parent/'result.json').is_file()


def test_payload_preservation_failure_blocks_before_new_execution(tmp_path, monkeypatch):
    cfg, _ = fixture_pipeline(tmp_path, monkeypatch)
    assert run_stage(cfg, 'baseline') == 0
    before = (tmp_path/'run/invocations').read_bytes()
    def unavailable(*args, **kwargs):
        raise OSError('quota unavailable')
    monkeypatch.setattr(stage.shutil, 'copy2', unavailable)
    assert run_stage(cfg, 'baseline') == 1
    assert (tmp_path/'run/invocations').read_bytes() == before
    assert stage.load_state(cfg)['stages']['baseline']['status'] == 'blocked'


@pytest.mark.parametrize('change', ['upstream', 'params', 'artifact', 'input_receipt'])
def test_resume_rejects_stale_or_incomplete_handoffs_before_dispatch(tmp_path, monkeypatch, capsys, change):
    cfg, _ = fixture_pipeline(tmp_path, monkeypatch)
    assert run_stage(cfg, 'baseline') == 0
    assert run_stage(cfg, 'optimize') == 0
    out = tmp_path/'run/optimize/out'
    if change == 'upstream':
        p = tmp_path/'run/baseline/out/result.json'
        doc = json.loads(p.read_text()); doc['metrics']['throughput'] = 42
        p.write_text(json.dumps(doc))
    elif change == 'params':
        cfg['stages']['optimize']['params']['time_budget_sec'] = 172800
    elif change == 'artifact':
        (out/'evidence.txt').rename(out/'retained-evidence.txt')
    else:
        (out/'inputs.json').write_text('{interrupted write')
    before = (tmp_path/'run/invocations').read_bytes()
    capsys.readouterr()
    assert stage.cmd_next(cfg, argparse.Namespace()) == 0
    assert capsys.readouterr().out.strip() == 'optimize'
    assert stage.run_order(cfg, argparse.Namespace(force=False), ['optimize'], stage.load_state(cfg)) == 1
    assert (tmp_path/'run/invocations').read_bytes() == before
    assert stage.load_state(cfg)['stages']['optimize']['status'] == 'fail'


def test_unchanged_handoff_resumes_without_executing_either_stage(tmp_path, monkeypatch):
    cfg, _ = fixture_pipeline(tmp_path, monkeypatch)
    assert run_stage(cfg, 'baseline') == 0
    assert run_stage(cfg, 'optimize') == 0
    before = (tmp_path/'run/invocations').read_bytes()
    assert stage.run_order(cfg, argparse.Namespace(force=False), cfg['order'], stage.load_state(cfg)) == 0
    assert (tmp_path/'run/invocations').read_bytes() == before
