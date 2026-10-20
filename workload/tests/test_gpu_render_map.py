"""CPU checks for logical GPU ids in containers with host sysfs node names."""
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

GEAK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GEAK / 'workload/scripts'))
import scheduler


@pytest.mark.parametrize('gpus, renders', [([1], 'renderD136'), ([0, 2, 7], 'renderD128,renderD144,renderD184')])
def test_docker_launch_records_logical_to_physical_mapping(tmp_path, monkeypatch, gpus, renders):
    commands = []
    def fake_sh(argv, **kw):
        commands.append(argv)
        return 0, 'cpu-stub'
    original_exists = scheduler.os.path.exists
    monkeypatch.setattr(scheduler, 'sh', fake_sh)
    monkeypatch.setattr(scheduler.os.path, 'exists', lambda p: True if str(p).startswith('/dev/dri/') else original_exists(p))
    backend = scheduler.DockerBackend(str(tmp_path), 'pinned:test')
    assert backend.start('fixture', gpus, {})
    launch = commands[-1]
    logical = ','.join(map(str, range(len(gpus))))
    assert 'GPU_IDS=' + logical in launch
    assert 'GEAK_GPU_ALLOWED=' + logical in launch
    assert 'GEAK_GPU_RENDER_DEVICES=' + renders in launch


@pytest.mark.parametrize('gpu, mapping, expected', [
    ('0', '', 1),                 # physical GPU 0 is busy
    ('1', '', 0),                 # legacy physical GPU 1 is idle
    ('0', 'renderD136', 0),       # logical GPU 0 is physical GPU 1
    ('1', 'renderD136', 1),       # undeclared logical GPU
    ('0', '../renderD136', 1),    # invalid mapping must not escape sysfs lookup
])
def test_idle_check_uses_the_mapped_device(tmp_path, gpu, mapping, expected):
    lock = (GEAK / 'kernel_workflow/scripts/gpu_lock.sh').read_text()
    function = re.search(r'^_gpu_is_idle\(\) \{.*?^\}', lock, re.M | re.S).group()
    for render, busy in [('renderD128', 100), ('renderD136', 0)]:
        folder = tmp_path / render
        folder.mkdir()
        (folder / 'gpu_busy_percent').write_text(str(busy))
        (folder / 'mem_info_vram_used').write_text(str(300 * 1024 * 1024))
    bins = tmp_path / 'bin'
    bins.mkdir()
    shim = bins / 'readlink'
    shim.write_text('#!/usr/bin/env python3\nimport os,sys\nfrom pathlib import Path\n'
                    'print(Path(os.environ["MOCK_SYSFS"]) / sys.argv[-1].split("/")[-2])\n')
    shim.chmod(0o755)
    env = dict(os.environ, MOCK_SYSFS=str(tmp_path), GEAK_GPU_RENDER_DEVICES=mapping,
               PATH=str(bins) + ':' + os.environ['PATH'])
    result = subprocess.run(['bash', '-c', function + '\n_gpu_is_idle "$1"', 'cpu-test', gpu],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == expected, result.stdout + result.stderr
