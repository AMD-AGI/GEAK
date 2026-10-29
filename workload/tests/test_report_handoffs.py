"""A retained result is evidence, not permission to hide a later validation failure."""
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1]/'scripts/report.py'
SPEC = importlib.util.spec_from_file_location('handoff_report', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_driver_rejection_wins_over_retained_success(tmp_path):
    write(tmp_path/'optimize/out/result.json', {'skill':'fixture', 'status':'ok',
        'metrics': {'best_speedup': 99}, 'notes':'old success'})
    write(tmp_path/'STATE.json', {'stages': {'optimize': {'status':'fail',
        'errors': ['stage inputs changed (upstream_results)']}}})
    row = MODULE.stage_rows(str(tmp_path))[0]
    assert row['status'] == 'fail' and row['metrics'] == {}
    assert 'upstream_results' in row['notes']


@pytest.mark.parametrize('failure', ['stage', 'summary'])
def test_failed_current_applyback_does_not_render_a_prior_gain(tmp_path, failure):
    write(tmp_path/'applyback/out/result.json', {'skill':'fixture',
        'status':'fail' if failure == 'stage' else 'ok',
        'metrics': {'e2e_delta_pct':10}, 'notes':'old gain claim'})
    write(tmp_path/'applyback/out/ab.json', {'baseline':{'throughput':100},
        'candidate':{'throughput':110,'repeats':2},'e2e_delta_pct':10,'verdict':'gain'})
    write(tmp_path/'applyback/out/e2e/bench_summary.json',
        {'overlay_loaded_by_every_leg': failure != 'summary',
         'error': 'current proof missing' if failure == 'summary' else None})
    assert MODULE.main(['--run-root', str(tmp_path)]) == 0
    report = (tmp_path/'REPORT.md').read_text()
    assert '**whole machine**' not in report and '**gain**' not in report
    assert MODULE.stage_rows(str(tmp_path))[0]['status'] == 'fail'
    assert MODULE.stage_rows(str(tmp_path))[0]['metrics'] == {}
    if failure == 'summary':
        assert 'old gain claim' not in report
    assert 'no end-to-end gain is reported' in report
