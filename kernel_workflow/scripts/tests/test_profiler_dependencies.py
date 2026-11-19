"""CLI capability checks, actual ELF loading, and SDK-managed ATT without GPU collection."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

KT = Path(__file__).resolve().parents[1] / 'kernel_tools'
sys.path.insert(0, str(KT))
import profiler_deps as deps


def sdk(legacy=True):
    return {'status': 'cli_available', 'help': '--att --kernel-trace --pmc ' +
            ('--att-library-path' if legacy else '')}


@pytest.fixture
def decoder_root(tmp_path, monkeypatch):
    for key in ('KT_ATT_SYSTEM_DIRS', 'ROCPROF_ATT_LIBRARY_PATH', 'LD_LIBRARY_PATH'):
        monkeypatch.setenv(key, str(tmp_path))
    return tmp_path


@pytest.mark.parametrize('kind,status', [('v1', 'external_candidate'),
                                        ('v2', 'decoder_incompatible'),
                                        ('broken', 'decoder_incompatible')])
def test_decoder_load_and_api_are_checked(decoder_root, kind, status):
    library = decoder_root / 'librocprof-trace-decoder.so'
    if kind == 'broken':
        library.write_text('not an ELF shared library')
    else:
        cc = shutil.which('cc')
        if not cc:
            pytest.skip('C compiler needed for a synthetic decoder library')
        symbol = 'parse_data' if kind == 'v1' else 'create_handle'
        subprocess.run([cc, '-shared', '-fPIC', '-x', 'c', '-', '-o', str(library)],
                       input=f'void rocprof_trace_decoder_{symbol}(void) {{}}', text=True, check=True)
    result = deps.att_probe(sdk())
    assert result['status'] == status and result['gpu_validation'] == 'not_attempted'


def test_sdk_managed_decoder_does_not_require_a_separate_library(decoder_root):
    assert deps.att_probe(sdk(False))['status'] == 'sdk_managed_unverified'
    assert deps.att_probe(sdk())['status'] == 'decoder_absent'


def test_missing_and_broken_profiler_are_distinct(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', str(tmp_path))
    assert deps.tool('rocprof-compute')['status'] == 'absent'
    executable = tmp_path / 'rocprof-compute'
    executable.write_text('#!/bin/sh\necho missing_python_package >&2\nexit 1\n')
    executable.chmod(0o755)
    result = deps.tool('rocprof-compute')
    assert result['status'] == 'broken' and 'missing_python_package' in result['version']


def test_att_arguments_preserve_custom_path_with_spaces(tmp_path):
    # Source the actual common shell helper; synthetic SDK-managed response runs
    # through a fake locus, proving the host detector is not the source of truth.
    record = tmp_path / 'report.json'
    document = {'att': {'status': 'external_candidate', 'directory': '/container/decoder with spaces'}}
    script = '''source "$1/_kt_common.sh"
locus_run() { printf '%s' "$PROBE_RESPONSE"; }
kt_att_prepare locus "$2" || exit
printf '%s\\n' "${KT_ATT_ARGS[@]}"
'''
    result = subprocess.run(['bash', '-c', script, '_', str(KT), str(record)],
                            env=dict(os.environ, PROBE_RESPONSE=json.dumps(document)),
                            capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == ['--att-library-path', '/container/decoder with spaces']
    assert json.loads(record.read_text()) == document
