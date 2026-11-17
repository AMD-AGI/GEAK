# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gluon_profile.py (the Gluon pack's profiling entry) end to end on the fake profiler (CPU-only)."""
import json
import subprocess
import sys
from pathlib import Path

from _fake_rocprofv3 import SCRIPTS, _decoder, _env

GP = SCRIPTS.parents[1] / 'perf_knowledge/expert_skills/skills/gluon_authoring/scripts/gluon_profile.py'

# Writes a Triton-cache-shaped dump (for capture.sh's IR layer) and checks its argv survived.
APP = '''import json, os, sys
from pathlib import Path
assert sys.argv[1:] == ['--case', 'c 2'], sys.argv
cache = os.environ.get('TRITON_CACHE_DIR')
if cache:
    p = Path(cache) / 'hash'
    p.mkdir(parents=True, exist_ok=True)
    asm = """.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
_gemm_fwd_kernel:
.LBB0_1:
\tbuffer_load_dwordx4 v[0:3], v4, s[0:3], 0 offen
\ts_waitcnt vmcnt(0)
\tds_read_b128 v[8:11], v5
\tv_mfma_f32_16x16x32_bf16 a[0:3], v[8:11], v[12:15], a[0:3]
\tv_mfma_f32_16x16x32_bf16 a[0:3], v[8:11], v[12:15], a[0:3]
\ts_cbranch_scc1 .LBB0_1
\ts_endpgm
.amdhsa_kernel _gemm_fwd_kernel
.amdhsa_next_free_vgpr 128
.amdhsa_accum_offset 128
.end_amdhsa_kernel
"""
    for ext, text in {'ttgir': 'module {}', 'llir': 'define void @k()', 'amdgcn': asm,
                      'json': json.dumps({'name': '_gemm_fwd_kernel', 'shared': 65536, 'num_warps': 4})}.items():
        (p / ('_gemm_fwd_kernel.' + ext)).write_text(text)
'''


def _profile(tmp_path, out, *extra):
    _decoder(tmp_path)
    src = tmp_path / 'app.py'
    src.write_text(APP)
    env = _env(tmp_path, arch='gfx942', ROCPROF_ATT_LIBRARY_PATH=str(tmp_path), GEAK_GLUON_PACK_DIR=str(tmp_path / 'none'))
    env.pop('TILE_KERNEL_CONTAINER', None)
    return subprocess.run([sys.executable, str(GP), str(out), '--kernel', '_gemm_fwd_kernel', '--source', str(src),
                           '--gpu', '0', '--arch', 'gfx942', '--timeout', '60', *extra, '--',
                           sys.executable, str(src), '--case', 'c 2'],
                          env=env, capture_output=True, text=True, timeout=600, cwd=tmp_path)


def test_one_call_fills_every_dial_with_layer_states(tmp_path):
    out = tmp_path / 'prof'
    r = _profile(tmp_path, out)
    prof = json.loads((out / 'profile.json').read_text())
    layers = {k: v['state'] for k, v in prof['layers'].items()}
    assert layers['trace'] == 'ok' and layers['pmc'] == 'ok' and layers['att'] == 'ok', (layers, r.stdout, r.stderr)
    assert layers['ir'] == 'ok', (layers, (out / 'logs/capture.log').read_text()[-2000:])
    assert r.returncode == 0, r.stdout + r.stderr
    d = prof['dials']
    assert d['D']['dispatches'] >= 1
    assert d['C']['MfmaUtil'] == 50 and not d['C']['dropped_counters']
    assert d['B']['mfma']['cycles_per_mfma'] == 10.0          # fixture: 2 MFMA, 20 cycles apart
    assert d['B']['hotspots']['top_instructions'][0]['asm'].startswith('s_waitcnt vmcnt(0)')
    assert d['A']['occupancy']['lds_bytes_per_wg'] == 65536 and d['A']['occupancy']['lds_wg_per_cu'] == 1
    assert prof['source_sha256'] and prof['complete'] is True
    # every profiler ran the same generated script, so a quoted argument kept its boundary
    assert "'c 2'" in (out / 'cmd.sh').read_text()
    report = (out / 'report.md').read_text()
    assert '## B - instruction stream (ATT)' in report and 's_waitcnt vmcnt(0)' in report


def test_compare_writes_the_profile_delta(tmp_path):
    first = tmp_path / 'r1'
    assert _profile(tmp_path, first).returncode == 0
    second = tmp_path / 'r2'
    r = _profile(tmp_path, second, '--compare', str(first), '--layers', 'trace,att')
    assert r.returncode == 0, r.stdout + r.stderr
    prof = json.loads((second / 'profile.json').read_text())
    assert 'pmc' not in prof['layers'] and prof['dials'].get('C') is None
    text = (second / 'compare.md').read_text()
    assert 'Profile delta' in text and 'C.MfmaUtil' in text.split('one side only')[-1]


def test_a_failed_layer_is_recorded_not_zeroed(tmp_path):
    out = tmp_path / 'prof'
    r = subprocess.run([sys.executable, str(GP), str(out), '--kernel', 'no_such_kernel', '--source', str(GP),
                        '--gpu', '0', '--layers', 'trace', '--timeout', '30', '--', 'true'],
                       env=_env(tmp_path), capture_output=True, text=True, timeout=300)
    prof = json.loads((out / 'profile.json').read_text())
    assert r.returncode in (4, 5), r.stdout + r.stderr
    assert prof['layers']['trace']['state'] != 'ok' and prof['dials']['D'] is None
    assert prof['complete'] is False


def test_refuses_without_a_gpu_or_a_held_lock(tmp_path):
    r = subprocess.run([sys.executable, str(GP), str(tmp_path / 'o'), '--kernel', 'k', '--source', str(GP), '--', 'true'],
                       env=_env(tmp_path), capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and 'gpu_lock' in r.stderr
    assert not (tmp_path / 'o').exists()


def test_selftest():
    assert subprocess.run([sys.executable, str(GP), '--selftest'], capture_output=True, text=True).returncode == 0


def test_a_busy_gpu_is_waited_out_never_measured(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('gluon_profile_mod', GP)
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    calls = []

    def fake_run(cmd, log_path, timeout=None, env=None):
        calls.append(cmd)
        Path(log_path).write_text('ERROR: GPU 5 ' + gp.BUSY_MARK + ' after 30s' if len(calls) < 3 else 'ok')
        return 1 if len(calls) < 3 else 0

    monkeypatch.setattr(gp, '_run', fake_run)
    monkeypatch.setattr(gp.time, 'sleep', lambda s: None)
    a = type('A', (), {'busy_wait': '600'})()
    assert gp._run_gpu(a, ['x'], tmp_path / 'l.log') == 0 and len(calls) == 3
    assert [w['waited_s'] for w in gp.BUSY_LOG] == [60, 120]
    gp.BUSY_LOG.clear(); calls.clear()
    a.busy_wait = '60'
    monkeypatch.setattr(gp, '_run', lambda cmd, log_path, timeout=None, env=None:
                        (Path(log_path).write_text(gp.BUSY_MARK), 1)[1])
    assert gp._run_gpu(a, ['x'], tmp_path / 'l.log') == 1
    assert gp.BUSY_LOG[-1].get('gave_up') is True
