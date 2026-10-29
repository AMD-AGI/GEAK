"""Native P90 must not silently become reciprocal tail latency."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

from test_applyback_contracts import summary_fixture, run_summary, json_write

PATH = Path(__file__).resolve().parents[1] / "stages" / "applyback" / "reference-replay-ab" / "summarize.py"
SPEC = importlib.util.spec_from_file_location("native_interactivity_reader", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_native_percentile_is_not_inverse_itl():
    stock = {"admissible": True, "native_interactivity_p90": 200, "itl_p90": 0.01}
    legs = [{"admissible": True, "native_interactivity_p90": value, "itl_p90": 0.02}
            for value in [215, 205, 210]]
    result = MODULE.native_interactivity_summary(stock, legs)
    assert result["available"]
    assert result["baseline"] == 200
    assert result["candidate"] == 210
    assert abs(result["delta_pct"] - 5) < 1e-10
    assert result["metric_path"] == "request_metrics.latency.intvty.p90"


def test_missing_native_rate_does_not_fall_back():
    stock = {"admissible": True, "itl_p90": 0.01}
    legs = [{"admissible": True, "native_interactivity_p90": 210}]
    result = MODULE.native_interactivity_summary(stock, legs)
    assert not result["available"]
    assert "baseline" not in result


def test_no_partial_native_median():
    stock = {"admissible": True, "native_interactivity_p90": 200}
    legs = [{"admissible": True, "native_interactivity_p90": 210},
            {"admissible": False, "native_interactivity_p90": 220}]
    assert not MODULE.native_interactivity_summary(stock, legs)["available"]


def test_invalid_native_metric_rejected():
    stock = {"admissible": True, "native_interactivity_p90": 200}
    for value in [True, 0, -1, float("nan"), float("inf")]:
        assert not MODULE.native_interactivity_summary(
            stock, [{"admissible": True, "native_interactivity_p90": value}])["available"]


def test_required_metric_setting_cannot_silently_disable_itself():
    with pytest.raises(ValueError, match="must be a boolean"):
        MODULE.validate_settings({"require_native_interactivity_p90": "true"})


@pytest.mark.parametrize('missing', ['stock', 'two'])
def test_required_p90_refuses_a_complete_arm_that_lacks_the_native_metric(tmp_path, missing):
    _, _, _, settings = summary_fixture(tmp_path)
    settings['require_native_interactivity_p90'] = True
    json_write(tmp_path/'settings.json', settings)
    for tag in ['stock', 'one', 'two', 'three']:
        if tag == missing:
            continue
        p = tmp_path/'runs'/tag/(tag+'.json')
        data = json.loads(p.read_text())
        data['request_metrics']['latency'] = {'intvty': {'p90': 200}}
        json_write(p, data)
    result = run_summary(tmp_path)
    assert result.returncode == 3
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert 'output_token_throughput' not in summary
    assert 'required native P90' in result.stdout


def test_native_p90_survives_summary_finalizer_and_report(tmp_path):
    _, _, applied, settings = summary_fixture(tmp_path)
    settings['require_native_interactivity_p90'] = True
    json_write(tmp_path/'settings.json', settings)
    for tag, rate in [('stock', 200), ('one', 215), ('two', 180), ('three', 210)]:
        p = tmp_path/'runs'/tag/(tag+'.json')
        data = json.loads(p.read_text())
        data['request_metrics']['latency'] = {'intvty': {'p90': rate}, 'itl': {'p90': 0.02}}
        json_write(p, data)
    assert run_summary(tmp_path).returncode == 0
    root = tmp_path/'campaign'
    out = root/'applyback/out'
    (out/'e2e').mkdir(parents=True)
    (out/'overlay').mkdir()
    applied['applied'][0].update(kernel='task', target='fixture:operation', verified_speedup=1.4)
    json_write(out/'applied.json', applied)
    shutil.copy2(tmp_path/'summary.json', out/'e2e/bench_summary.json')
    workload = PATH.parents[3]
    result = subprocess.run([sys.executable, str(workload/'stages/applyback/finalize.py'),
        '--e2e-dir', str(out/'e2e'), '--baseline', str(tmp_path/'reference.json')],
        env=dict(os.environ, OUT=str(out), STAGE='applyback', STAGE_SKILL='reference-replay-ab'),
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    ab = json.loads((out/'ab.json').read_text())
    receipt = json.loads((out/'result.json').read_text())
    assert ab['native_interactivity_p90']['candidate_runs'] == [215, 180, 210]
    assert ab['native_interactivity_p90']['candidate'] == 210
    assert receipt['metrics']['native_p90_interactivity_candidate'] == 210
    assert receipt['metrics']['candidate_tok_s'] == 110
    report = subprocess.run([sys.executable, str(workload/'scripts/report.py'), '--run-root', str(root)],
                            capture_output=True, text=True)
    assert report.returncode == 0, report.stderr
    assert 'request_metrics.latency.intvty.p90' in (root/'REPORT.md').read_text()
