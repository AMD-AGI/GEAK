"""Gluon compiler capability and optional hardware-source gate regressions."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
BRIDGE = ROOT / 'perf_knowledge/expert_skills/skills/gluon_authoring/scripts/ttgir_bridge.py'


def run(*args, **env):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True,
                          env=dict(os.environ, **env), timeout=30)


def test_capability_inventory_does_not_hide_missing_triton_behind_override(tmp_path):
    (tmp_path/'triton.py').write_text("raise ModuleNotFoundError('triton deliberately unavailable')")
    r = run(sys.executable, BRIDGE, 'capabilities', '--json', PYTHONPATH=str(tmp_path),
            TTGIR_BRIDGE_CAPS='distributed_norm,shared_norm,layout_view')
    assert r.returncode == 3, r.stdout + r.stderr
    data = json.loads(r.stdout)
    assert data['layout_view'] is False and not data.get('native_apis')
    assert 'triton deliberately unavailable' in data['probe_diagnostic']


def test_child_abort_preserves_completed_capabilities(monkeypatch):
    spec = importlib.util.spec_from_file_location('minimal_bridge_test', BRIDGE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    api = {'triton_version': 'vendor-build', 'native_apis': {
        'scale_layout': {'available': True, 'parameters': ['dot_operand_layout', 'shape']}}}
    completed = subprocess.CompletedProcess([], -6, 'API:'+json.dumps(api)+'\nCAP:distributed_norm\n', 'shared normalization aborted')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: completed)
    monkeypatch.setenv('TTGIR_BRIDGE_CAPS', 'shared_norm')
    data = module.capabilities(force_probe=True)
    assert data['distributed_norm'] and not data['shared_norm']
    assert data['native_apis'] == api['native_apis']
    assert 'aborted' in data['probe_diagnostic']


def test_optional_calculator_gap_does_not_block_compiler_layouts(tmp_path):
    # Exercise the real gate's control flow with a CPU-only locus and no profiler.
    script = tmp_path/'env_gate.sh'
    script.write_text((BRIDGE.parent/'env_gate.sh').read_text())
    (tmp_path/'locus.sh').write_text('locus_run(){ return 1; }; locus_have(){ return 1; }\n')
    (tmp_path/'hw_sources.sh').write_text('echo "calculator unavailable"; exit 1\n')
    r = run('bash', script, '--workdir', tmp_path, TILE_DISK_FLOOR_GB='0',
            TILE_TMPDIR=str(tmp_path), TILE_KERNEL_CONTAINER='')
    assert r.returncode == 3, r.stdout + r.stderr  # profiling degraded, not hard-blocked
    assert 'native Triton layout APIs' in r.stdout
    assert 'blocked_environment' not in r.stdout and 'only source' not in r.stdout
