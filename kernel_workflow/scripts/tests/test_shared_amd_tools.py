"""Cross-entry-point contracts for the shared AMD tools (no GPU required)."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
KT = ROOT / 'kernel_workflow/scripts/kernel_tools'
PACK = ROOT / 'perf_knowledge/expert_skills/skills/gluon_authoring/scripts'
sys.path.insert(0, str(KT))
from amd_resources import analyze
import amd_occupancy


def call(path, *args):
    return subprocess.run([sys.executable, str(path), *args], text=True,
                          capture_output=True, timeout=30)


@pytest.mark.parametrize('arch,instruction', [
    ('gfx942', 'v_mfma_f32_16x16x16_f16'),
    ('gfx950', 'v_mfma_f32_16x16x32_f16'),
    ('gfx1201', 'v_wmma_f32_16x16x16_f16')])
def test_isa_legacy_shared_and_pack_paths_agree(arch, instruction):
    outputs = [call(path, '--arch', arch, 'encoding', instruction) for path in
               (KT/'amd_isa.py', KT/'gfx950_isa.py', PACK/'gfx950_isa.py', PACK/'amd_isa.py')]
    assert all(p.returncode == 0 for p in outputs), [p.stderr for p in outputs]
    assert len({p.stdout for p in outputs}) == 1


def test_generic_entry_requires_arch_and_preserves_capability_scope(monkeypatch):
    assert call(KT/'amd_isa.py', 'capabilities').returncode == 2
    for name in ('TILE_HW_SEED', 'TILE_HW_TOOLS', 'TILE_HW_CACHE'):
        monkeypatch.setenv(name, '')
    r = call(KT/'amd_isa.py', '--arch', 'gfx1201', 'capabilities', '--json')
    data = json.loads(r.stdout)
    assert data['requested_arch'] == 'gfx1201' and data['isa_family'] == 'rdna4'
    assert data['queries']['encoding']['status'] == 'available'
    assert data['queries']['layout']['status'] == 'requires_calculator'
    assert data['queries']['layout']['asset_present'] is False
    assert data['queries']['waitcnt']['status'] == 'not_extracted'
    assert call(KT/'amd_isa.py', '--arch', 'gfx9999', 'capabilities').returncode == 3


@pytest.mark.parametrize('asm,expected', [
    ('.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\n', None),
    ('.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"\n.amdhsa_wavefront_size32 0\n.amdhsa_next_free_vgpr 128\n', None),
    ('.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\n.amdhsa_next_free_vgpr 64\n; Occupancy: 0\n', 0),
])
def test_shared_parser_and_occupancy_agree_on_unknown_and_zero(asm, expected):
    assert analyze(asm)['waves_per_simd'] == expected
    assert amd_occupancy.occupancy_from_asm(asm)[0] == expected


def test_actual_launch_cannot_come_from_compile_bound():
    asm = '.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\n.amdhsa_next_free_vgpr 64\n.max_flat_workgroup_size: 1024\n'
    unknown = analyze(asm, lds_bytes=32768)
    actual = analyze(asm, lds_bytes=32768, num_warps=4)
    assert unknown['wg_size'] is None and unknown['waves_by_lds_term'] is None
    assert actual['wg_size'] == 256 and actual['wg_size_compile_bound'] == 1024
    assert actual['waves_per_simd'] == 2
    assert amd_occupancy.occupancy_from_asm(asm, lds_bytes_per_wg=32768, workgroup_size=256)[0] == 2


def test_agpr_total_is_not_readded_and_arch_count_stays_distinct():
    asm = ('.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\n'
           '; NumVgprs: 160\namdhsa.kernels:\n - .agpr_count: 8\n   .vgpr_count: 168\n')
    r = analyze(asm)
    assert (r['vgpr_arch'], r['agpr'], r['vgpr_combined'], r['waves_per_simd']) == (160, 8, 168, 3)


def test_partial_ir_is_not_a_successful_capture(tmp_path):
    cmd = "import os; from pathlib import Path; p=Path(os.environ['TRITON_CACHE_DIR'])/'h'; p.mkdir(); (p/'k.ttgir').write_text('module {}')"
    r = subprocess.run(['bash', str(KT/'dump_ir.sh'), '--arch', 'gfx942', '--out', str(tmp_path),
                        '--', sys.executable, '-c', cmd], capture_output=True, text=True)
    assert r.returncode == 3 and 'incomplete compilation' in r.stderr
    assert not (tmp_path/'variant/ir_manifest.json').exists()


def test_multi_kernel_assembly_cannot_mix_resources():
    with pytest.raises(ValueError, match='one kernel'):
        analyze('.amdhsa_kernel a\n.amdhsa_next_free_vgpr 64\n.amdhsa_kernel b\n.amdhsa_next_free_vgpr 128\n')


def test_assembly_audit_uses_shared_register_record():
    import asm_loop_audit
    asm = '.amdhsa_next_free_vgpr 168\n.amdhsa_accum_offset 160\n.amdhsa_next_free_sgpr 32\n'
    r = analyze(asm)
    assert asm_loop_audit.kd_regs(asm.splitlines()) == {
        'vgpr': r['vgpr_combined'], 'sgpr': r['sgpr'], 'accum_vgpr': r['agpr']}
