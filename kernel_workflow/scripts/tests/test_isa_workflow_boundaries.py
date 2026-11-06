"""Run real CLI boundaries with compiler fixtures; no GPU or network required."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

KT = Path(__file__).resolve().parents[1] / 'kernel_tools'


def run(*args, cwd=None, **env):
    return subprocess.run([str(a) for a in args], cwd=cwd, env=dict(os.environ, **env),
                          capture_output=True, text=True, timeout=30)


def compiler(tmp_path, groups, chdir=False):
    script = tmp_path / 'compile.py'
    script.write_text('import os,json\nfrom pathlib import Path\n' +
        ("os.chdir('child')\n" if chdir else '') +
        'groups=' + repr(groups) + '\n' +
        '''for group,name,shared in groups:
 p=Path(os.environ['TRITON_CACHE_DIR'])/group
 p.mkdir(parents=True)
 for ext in ['ttgir','llir','amdgcn']:
  (p/(name+'.'+ext)).write_text('.amdgcn_target "amdgcn-amd-amdhsa--gfx950"\\n.amdhsa_kernel '+name+'\\n.vgpr_count: 64\\n.max_flat_workgroup_size: 1024\\n')
 (p/(name+'.json')).write_text(json.dumps({'name':name,'shared':shared,'num_warps':4}))
''')
    return script


@pytest.mark.parametrize('arch', ['gfx9999', 'unknown'])
def test_unsupported_arch_does_not_invent_cdna_occupancy(arch, tmp_path):
    r = run(sys.executable, KT/'probe.py', 'plan', '--arch', arch, '--warps','4','--acc','a=128x128')
    assert r.returncode == 2 and 'no register model' in r.stderr
    (tmp_path/'k.s').write_text(f'.amdgcn_target "amdgcn-amd-amdhsa--{arch}"\n.vgpr_count: 128\n')
    r = run(sys.executable, KT/'probe.py','measure','--dir',tmp_path,'--arch',arch,'--json',tmp_path/'out.json')
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path/'out.json').read_text())['kernels'][0]['waves_per_simd'] is None


def test_gfx1201_uses_rdna4_encoding_offline():
    for command in ([sys.executable,KT/'gfx950_isa.py','--arch','gfx1201'],
                    [sys.executable,KT/'layout_facts.py','gfx1201']):
        r = run(*command, 'encoding', 'v_wmma_f32_16x16x16_f16')
        assert r.returncode == 0 and 'RDNA4' in r.stdout, r.stderr


@pytest.mark.parametrize('groups', [[('a','body',4096),('b','reduce',8192)],
                                   [('a','body',4096),('b','body',8192)]])
def test_dump_refuses_ambiguous_compilations(tmp_path, groups):
    script = compiler(tmp_path, groups)
    r = run('bash',KT/'dump_ir.sh',sys.executable,script,'--out',tmp_path/'ir','--variant','v',
            DUMP_IR_NO_ROCMINFO='1')
    assert r.returncode == 3 and 'E_IR_SELECTION' in r.stderr
    assert not (tmp_path/'ir/v/v.amdgcn').exists()


def test_dump_retires_previous_artifacts_when_new_compile_is_empty(tmp_path):
    dest=tmp_path/'ir/v';dest.mkdir(parents=True)
    (dest/'v.amdgcn').write_text('old kernel')
    r = run('bash',KT/'dump_ir.sh','true','--out',tmp_path/'ir','--variant','v',DUMP_IR_NO_ROCMINFO='1')
    assert r.returncode == 3 and not (dest/'v.amdgcn').exists()
    assert any(p.read_text() == 'old kernel' for p in (tmp_path/'ir').glob('v.previous.*/v.amdgcn'))


def test_dump_to_probe_uses_selected_metadata_and_actual_workgroup(tmp_path):
    name='body_'+'x'*70
    script=compiler(tmp_path,[('h1',name,81920),('h2','reduce',1024)],chdir=True)
    (tmp_path/'child').mkdir()
    r=run('bash',KT/'dump_ir.sh',sys.executable,script,'--out','ir','--variant','v',
          '--kernel-name',name,cwd=tmp_path,DUMP_IR_NO_ROCMINFO='1')
    assert r.returncode == 0,r.stderr
    dest=tmp_path/'ir/v'
    assert len(list(dest.glob('meta_*.json'))) == 1
    r=run(sys.executable,KT/'probe.py','measure','--dir',dest,'--json',dest/'probe.json')
    assert r.returncode == 0,r.stderr
    for k in json.loads((dest/'probe.json').read_text())['kernels']:
        assert k['kernel_name'] == name
        assert k['wg_size'] == 256 and k['wg_size_compile_bound'] == 1024
        assert k['waves_by_lds_term'] == 2 and k['waves_per_simd'] == 2


def test_dump_preserves_caller_arguments(tmp_path):
    script=compiler(tmp_path,[('h','body',1024)])
    script.write_text(script.read_text()+"\nimport sys\nassert sys.argv[1:]==['--check','--backends','gluon']\n")
    r=run('bash',KT/'dump_ir.sh',sys.executable,script,'--check','--backends','gluon',
          '--out',tmp_path/'ir',DUMP_IR_NO_ROCMINFO='1')
    assert r.returncode == 0,r.stderr


def test_dump_separator_preserves_overlapping_caller_options(tmp_path):
    script = compiler(tmp_path, [('h', 'body', 1024)])
    args = ['--arch', 'gfx942', '--out', 'caller output', '--variant', 'candidate', '--help']
    script.write_text(script.read_text() + '\nimport sys\nassert sys.argv[1:] == ' + repr(args) + '\n')
    r = run('bash', KT/'dump_ir.sh', '--out', tmp_path/'ir', '--variant', 'v',
            '--', sys.executable, script, *args, DUMP_IR_NO_ROCMINFO='1')
    assert r.returncode == 0, r.stderr
    assert (tmp_path/'ir/v/v.amdgcn').is_file()


def test_conflicting_same_name_metadata_is_not_silently_overwritten(tmp_path):
    for i in range(2):
        p=tmp_path/str(i);p.mkdir()
        (p/'k.json').write_text(json.dumps({'name':'k','shared':32768*(i+1),'num_warps':4}))
    (tmp_path/'k.s').write_text('.amdgcn_target "amdgcn-amd-amdhsa--gfx950"\n.vgpr_count: 64\n.max_flat_workgroup_size: 256\n')
    r=run(sys.executable,KT/'probe.py','measure','--dir',tmp_path,'--json',tmp_path/'out.json')
    assert r.returncode == 0,r.stderr
    assert json.loads((tmp_path/'out.json').read_text())['kernels'][0]['waves_by_lds_term'] is None


def test_compiler_workgroup_bound_does_not_supply_launch_geometry(tmp_path):
    (tmp_path/'k.s').write_text('.amdgcn_target "amdgcn-amd-amdhsa--gfx950"\n.vgpr_count: 64\n.max_flat_workgroup_size: 1024\n')
    (tmp_path/'k.json').write_text(json.dumps({'name':'k','shared':81920}))
    r=run(sys.executable,KT/'probe.py','measure','--dir',tmp_path,'--json',tmp_path/'out.json')
    assert r.returncode == 0,r.stderr
    k=json.loads((tmp_path/'out.json').read_text())['kernels'][0]
    assert k['wg_size'] is None and k['waves_by_lds_term'] is None
    assert 'upper bound' in k['waves_warning']


@pytest.mark.parametrize('descriptor', [True, False])
def test_llvm_agpr_metadata_is_already_combined(tmp_path, descriptor):
    # ROCm LLVM 22: clobber v0..v158 and a0..a7. The resulting 160 ArchVGPRs
    # plus 8 AGPRs are already included in .vgpr_count; double counting drops
    # the model from 3 to 2 waves/SIMD when LLVM's occupancy comment is stripped.
    asm = '.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\n'
    if descriptor:
        asm += '.amdhsa_next_free_vgpr 168\n; NumVgprs: 160\n; NumAgprs: 8\n'
    asm += 'amdhsa.kernels:\n  - .agpr_count: 8\n    .vgpr_count: 168\n'
    (tmp_path/'k.s').write_text(asm)
    r=run(sys.executable,KT/'probe.py','measure','--dir',tmp_path,'--json',tmp_path/'out.json')
    assert r.returncode == 0,r.stderr
    k=json.loads((tmp_path/'out.json').read_text())['kernels'][0]
    assert k['vgpr_combined'] == 168 and k['waves_by_vgpr_term'] == 3
    assert k['agpr'] == 8
    assert k['vgpr_arch'] == (160 if descriptor else None)


def test_rdna_wave64_does_not_use_wave32_model(tmp_path):
    (tmp_path/'k.s').write_text('.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"\n.amdhsa_wavefront_size32 0\n.vgpr_count: 128\n')
    r=run(sys.executable,KT/'probe.py','measure','--dir',tmp_path,'--json',tmp_path/'out.json')
    assert r.returncode == 0,r.stderr
    assert json.loads((tmp_path/'out.json').read_text())['kernels'][0]['waves_per_simd'] is None


def test_layout_queries_use_seed_without_cloning(tmp_path):
    seed=tmp_path/'seed/amd_matrix_instruction_calculator';seed.mkdir(parents=True)
    (seed/'matrix_calculator.py').write_text("import sys\nassert sys.argv[1:]==['-a','rdna4','-L']\nprint('seed calculator used')\n")
    r=run(sys.executable,KT/'layout_facts.py','gfx1201','--list',TILE_HW_SEED=str(seed.parent),
          TILE_HW_TOOLS='',TILE_HW_CACHE=str(tmp_path/'missing-cache'))
    assert r.returncode == 0 and 'seed calculator used' in r.stdout,r.stderr
    r=run(sys.executable,KT/'layout_facts.py','cdna3','--list',TILE_HW_SEED='',TILE_HW_TOOLS='',
          TILE_HW_CACHE=str(tmp_path/'missing-cache'))
    assert r.returncode == 3 and not (tmp_path/'missing-cache').exists()
