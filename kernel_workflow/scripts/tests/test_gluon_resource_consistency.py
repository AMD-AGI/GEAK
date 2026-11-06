"""Gluon decisions consume the same resource record as the shared tools."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
KT = ROOT / 'kernel_workflow/scripts/kernel_tools'
sys.path.insert(0, str(KT))
from amd_resources import analyze
import amd_occupancy

spec = importlib.util.spec_from_file_location('gluon_parity', ROOT /
    'perf_knowledge/expert_skills/skills/gluon_authoring/scripts/parity_gate.py')
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


@pytest.mark.parametrize('arch,body', [
    ('gfx942', '.vgpr_count: 168\n.agpr_count: 8\n; NumVgprs: 160\n'),
    ('gfx950', '.amdhsa_next_free_vgpr 64\n; Occupancy: 0\n'),
    ('gfx1201', '.amdhsa_next_free_vgpr 64\n.amdhsa_wavefront_size32 0\n'),
    ('gfx942', ''),
])
def test_register_and_occupancy_contracts_match(arch, body):
    asm = f'.amdgcn_target "amdgcn-amd-amdhsa--{arch}"\n' + body
    shared = analyze(asm)
    facts = parity.asm_facts(asm)
    assert facts['occupancy_waves_per_simd'] == shared['waves_per_simd']
    assert facts['occupancy_waves_per_simd'] == amd_occupancy.occupancy_from_asm(asm)[0]
    assert facts['arch_vgpr'] == shared['vgpr_arch']
    assert facts['next_free_vgpr'] == shared['vgpr_combined']
    assert facts['agpr'] == shared['agpr']


def test_parity_uses_actual_launch_and_assembly_target():
    asm = '.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\n.amdhsa_next_free_vgpr 64\n.max_flat_workgroup_size: 1024\n'
    unknown = parity.occupancy_terms(asm, lds_bytes=32768, arch='gfx950')
    assert unknown['arch'] == 'gfx942' and unknown['lds_term'] is None
    actual = parity.occupancy_terms(asm, lds_bytes=32768, workgroup_size=256)
    assert actual['waves_per_simd'] == actual['lds_term'] == 2


def test_spill_register_count_is_not_scratch_bytes():
    facts = parity.asm_facts('.vgpr_spill_count: 3\n')
    assert facts['spill_bytes'] is None
