"""parse_pmc.derived keeps raw counts raw and real zeros zero (no GPU)."""
import importlib.util
import sys
from pathlib import Path

KT = Path(__file__).resolve().parents[1] / 'kernel_tools'


def load(name):
    spec = importlib.util.spec_from_file_location(name, KT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(KT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_raw_counts_do_not_become_absolute_utilization():
    d = load('parse_pmc').derived({'vals': {'SQ_VALU_MFMA_BUSY_CYCLES': [10.0], 'SQ_ACTIVE_INST_VALU': [1.0]}})
    for field in ('MfmaUtil', 'VALUBusy', 'VALUUtilization', 'LDSBankConflict', 'bubble_mfma_idle'):
        assert d[field] is None
    assert d['raw_activity']['values']['SQ_VALU_MFMA_BUSY_CYCLES'] == 10


def test_real_zero_utilization_and_occupancy_are_preserved():
    d = load('parse_pmc').derived({'vals': {'MfmaUtil': [0], 'VALUBusy': [0],
        'MeanOccupancyPerActiveCU': [0], 'OccupancyPercent': [50], 'SQ_ACTIVE_INST_VALU': [10]}})
    assert (d['MfmaUtil'], d['VALUBusy'], d['occupancy'], d['bubble_mfma_idle']) == (0, 0, 0, 100)


def test_fetch_write_route_needs_both_counters():
    md = load('parse_pmc').memory_derived
    meta = {'duration_ns': 1000.0, 'n_dispatches': 1}
    only_fetch = md({'vals': {'FETCH_SIZE': [1000.0]}, 'meta': meta})
    assert 'fetch_write' not in only_fetch['dram_bytes_by_route']
    assert 'achieved_dram_tb_s' not in only_fetch and 'achieved_tb_s_fetch_write' not in only_fetch
    assert 'WRITE_SIZE was not' in only_fetch['_fetch_write_note']
    only_write = md({'vals': {'WRITE_SIZE': [1000.0], 'TCC_MISS_sum': [10.0]}, 'meta': meta})
    assert 'fetch_write' not in only_write['dram_bytes_by_route'] and only_write['_bytes_route'] == 'tcc_miss'
    assert 'FETCH_SIZE was not' in only_write['_fetch_write_note']
    both = md({'vals': {'FETCH_SIZE': [1000.0], 'WRITE_SIZE': [0.0]}, 'meta': meta})
    assert both['dram_bytes_by_route']['fetch_write'] == 1000 * 1024 and '_fetch_write_note' not in both
    assert 'achieved_dram_tb_s' in both
