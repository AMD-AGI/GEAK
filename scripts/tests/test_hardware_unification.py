# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-entry regression checks for the PR1 shared hardware contract."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
KT = ROOT / 'kernel_workflow/scripts/kernel_tools'
PACK = ROOT / 'perf_knowledge/expert_skills/skills/gluon_authoring'
sys.path.insert(0, str(ROOT / 'scripts'))
import hardware_context as hc  # noqa: E402 (shared repository-local module)


def run(*args, env=None):
    clean = {k: v for k, v in os.environ.items() if k not in ('GEAK_HARDWARE_CONTEXT', 'GEAK_HW_DATA_DIR')}
    return subprocess.run([sys.executable, *map(str, args)], env={**clean, **(env or {})},
                          cwd='/tmp', capture_output=True, text=True, timeout=30)


def context_file(tmp_path, name='MI308X', cu=80):
    p = tmp_path / 'context.json'
    p.write_text(json.dumps(hc.build({'gfx': 'gfx942', 'marketing_name': 'AMD Instinct ' + name,
                                    'physical_cu_count': cu})))
    return p


def test_legacy_consumers_use_shared_occupancy_and_data():
    script = f'''
import sys
sys.path.insert(0, {str(PACK / 'scripts')!r})
import amd_occupancy, probe, asm_loop_audit
from pathlib import Path
assert Path(amd_occupancy.__file__).resolve() == Path({str(KT / 'amd_occupancy.py')!r})
assert probe._OCC is amd_occupancy and asm_loop_audit._OCC is amd_occupancy
assert amd_occupancy._arch_facts('gfx1151').get('lds_per_wgp_kib') is None
assert amd_occupancy.lds_per_cu('gfx950') == 160 * 1024
assert amd_occupancy.waves_by_vgpr(249, 'gfx1201')[0] == 5
'''
    result = run('-c', script)
    assert result.returncode == 0, result.stderr
    assert not (PACK / 'references/hardware/hw_constants.json').exists()


@pytest.mark.parametrize('tool,args', [
    ('hw_budget.py', ['--workload', 'gemm', '--shapes', 'M=1024,N=1024,K=1024', '--json']),
    ('mem_bw_probe.py', []),
])
@pytest.mark.parametrize('sku', ['MI300X', 'mi300x', 'Mi300X'])
def test_tools_refuse_wrong_sku_before_measurement(tmp_path, tool, args, sku):
    p = context_file(tmp_path)
    result = run(KT / tool, '--sku', sku, *args, env={'GEAK_HARDWARE_CONTEXT': str(p)})
    assert result.returncode != 0 and 'conflicts with the run hardware context' in result.stderr


@pytest.mark.parametrize('sku', ['mi355x', 'Mi350X', 'r9700', 'ai_max_395'])
def test_standalone_budget_sku_case_preserves_result(sku):
    args = ['--workload', 'gemm', '--shapes', 'M=4096,N=4096,K=4096', '--dtype', 'bf16', '--json']
    reference = run(KT / 'hw_budget.py', '--sku', sku.upper(), *args)
    result = run(KT / 'hw_budget.py', '--sku', sku, *args)
    assert reference.returncode == 0, reference.stderr
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data == json.loads(reference.stdout)
    assert data['sku'] == sku.upper()
    assert data['workload']['may_gate'] is False


@pytest.mark.parametrize('args', [
    ['roofline', '--M', '4096', '--N', '4096', '--K', '4096', '--dtype', 'bf16', '--json'],
    ['budget', '--bm', '64', '--bn', '64', '--bk', '32', '--num-warps', '4'],
    ['occ', '--vgpr', '128'],
])
@pytest.mark.parametrize('with_context', [False, True])
def test_calculator_cli_accepts_identity_sku_case(tmp_path, args, with_context):
    extra = ['--hardware-context', context_file(tmp_path)] if with_context else []
    reference = run(KT / 'calc_perf.py', *args, '--sku', 'MI308X', *extra)
    assert reference.returncode == 0, reference.stderr
    for sku in ('mi308x', 'Mi308X'):
        result = run(KT / 'calc_perf.py', *args, '--sku', sku, *extra)
        assert result.returncode == 0, result.stderr
        assert result.stdout == reference.stdout


def test_bandwidth_probe_accepts_mixed_case_sku_before_measurement(monkeypatch):
    # Exercise CLI parsing and planning, stopping at the GPU measurement boundary.
    monkeypatch.syspath_prepend(str(KT))
    monkeypatch.delenv('GEAK_HARDWARE_CONTEXT', raising=False)
    monkeypatch.delenv('GEAK_HW_DATA_DIR', raising=False)
    import mem_bw_probe

    class MeasurementReached(Exception):
        pass

    def stop_before_gpu(*args):
        raise MeasurementReached

    monkeypatch.setattr(mem_bw_probe, 'measure', stop_before_gpu)
    monkeypatch.setattr(sys, 'argv', ['mem_bw_probe.py', '--sku', 'mI350x'])
    with pytest.raises(MeasurementReached):
        mem_bw_probe.main()


@pytest.mark.parametrize('tool', ['hw_budget.py', 'mem_bw_probe.py'])
def test_unknown_explicit_sku_is_still_refused(tool):
    result = run(KT / tool, '--sku', 'mi_unknown')
    assert result.returncode != 0 and 'unknown SKU' in result.stderr


@pytest.mark.parametrize('sku', ['mi308x', 'Mi308X'])
def test_budget_explicit_sku_case_preserves_context(tmp_path, sku):
    p = context_file(tmp_path)
    result = run(KT / 'hw_budget.py', '--hardware-context', p, '--sku', sku, '--json')
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data['sku'] == 'MI308X'
    assert data['hardware_context_sha256'] == hc.catalog()[1]


@pytest.mark.parametrize('tool', ['hw_budget.py', 'mem_bw_probe.py'])
def test_unknown_scope_cannot_supply_product_peaks(tmp_path, tool):
    p = context_file(tmp_path, 'MI300X', 80)
    result = run(KT / tool, '--hardware-context', p, '--sku', 'unknown')
    assert result.returncode != 0 and 'no supported product peaks' in result.stderr


def test_budget_adopts_context_without_explicit_sku(tmp_path):
    p = context_file(tmp_path)
    result = run(KT / 'hw_budget.py', '--hardware-context', p, '--workload', 'gemm',
                 '--shapes', 'M=1024,N=1024,K=1024', '--json')
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data['workload']['compute_ceiling_tflops'] == [344, 344]
    assert data['workload']['may_gate'] is False
    assert data['hardware_context_sha256'] == hc.catalog()[1]
    unknown = context_file(tmp_path, 'MI300X', 80)
    result = run(KT / 'hw_budget.py', '--hardware-context', unknown)
    assert result.returncode != 0 and 'no supported product peaks' in result.stderr


def test_occupancy_refuses_cross_arch_context(tmp_path):
    p = context_file(tmp_path)
    result = run(KT / 'amd_occupancy.py', '--vgpr', '64', '--arch', 'gfx950', '--hardware-context', p)
    assert result.returncode != 0 and 'architecture conflicts' in result.stderr


def test_catalog_override_is_shared_by_identity_and_calculators(tmp_path):
    catalog = json.loads((ROOT / 'perf_knowledge/hardware/data/sku.json').read_text())
    catalog['skus']['MI308X']['peak_tflops']['bf16'] = 345.0
    (tmp_path / 'sku.json').write_text(json.dumps(catalog))
    result = run('-c', f"import sys; sys.path.insert(0, {str(ROOT / 'scripts')!r}); "
                 "import hardware_context as h; print(h.catalog()[0]['MI308X']['peak_tflops']['bf16'])",
                 env={'GEAK_HW_DATA_DIR': str(tmp_path)})
    assert result.returncode == 0 and float(result.stdout) == 345.0


def test_combined_check_rejects_stale_generated_docs(tmp_path):
    # A stale generated table must fail even when the JSON itself is valid.
    page = tmp_path / 'table.md'
    page.write_text('<!-- BEGIN GENERATED by extract_sku.py: table gfx942 -->\nstale\n'
                    '<!-- END GENERATED by extract_sku.py -->\n')
    result = run('-c', f"import sys; sys.path.insert(0, {str(KT)!r}); import extract_sku as e; "
                 f"e.doc_files=lambda: iter([{str(page)!r}]); "
                 "sys.argv=['extract_sku.py','--check','--check-docs']; e.main()")
    assert result.returncode == 1 and 'stale' in result.stdout


def test_legacy_container_entry_maps_hardware_context(tmp_path):
    host = tmp_path / 'host root'
    host.mkdir()
    env = {**os.environ, 'TILE_KERNEL_HOST_WORKDIR': str(host),
           'TILE_KERNEL_CONTAINER_WORKDIR': '/work/run',
           'GEAK_HARDWARE_CONTEXT': str(host / 'hardware context.json'),
           'LOCUS_ENV': 'GEAK_HARDWARE_CONTEXT'}
    result = subprocess.run(['bash', '-c', 'source "$1"; _locus_env_args', 'bash',
                             str(PACK / 'scripts/locus.sh')], env=env, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split(b'\0') == [b'-e', b'GEAK_HARDWARE_CONTEXT=/work/run/hardware context.json', b'']
