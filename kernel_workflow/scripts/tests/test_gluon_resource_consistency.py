"""The shared resource record and the occupancy model agree (the Gluon pack's tools consume both)."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
KT = ROOT / 'kernel_workflow/scripts/kernel_tools'
sys.path.insert(0, str(KT))
from amd_resources import analyze
import amd_occupancy


@pytest.mark.parametrize('arch,body', [
    ('gfx942', '.vgpr_count: 168\n.agpr_count: 8\n; NumVgprs: 160\n'),
    ('gfx950', '.amdhsa_next_free_vgpr 64\n; Occupancy: 0\n'),
    ('gfx1201', '.amdhsa_next_free_vgpr 64\n.amdhsa_wavefront_size32 0\n'),
    ('gfx942', ''),
])
def test_register_and_occupancy_contracts_match(arch, body):
    asm = f'.amdgcn_target "amdgcn-amd-amdhsa--{arch}"\n' + body
    shared = analyze(asm)
    assert shared['waves_per_simd'] == amd_occupancy.occupancy_from_asm(asm)[0]
