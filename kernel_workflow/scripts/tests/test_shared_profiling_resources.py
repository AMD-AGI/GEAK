"""ATT diagnostics preserve missing metadata and reuse the shared AMD model."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kernel_tools'))
from amd_resources import analyze
from hotspot_analyzer import detect_arch_and_reg_pressure, print_reg_pressure


def test_att_without_resources_never_invents_occupancy():
    result = detect_arch_and_reg_pressure([])
    assert result['occupancy'] is None and result['combined_vgpr_alloc'] is None
    print_reg_pressure(result)


def test_att_and_static_analysis_agree_on_lds_and_zero():
    for arch in ('gfx942', 'gfx950', 'gfx1201'):
        meta = dict(arch=arch, csv_vgpr=128, csv_accum_vgpr=0, csv_lds=200000, csv_wg=256)
        result = detect_arch_and_reg_pressure([], meta)
        shared = analyze('', arch=arch, vgpr=128, lds_bytes=200000, workgroup_size=256)
        assert result['occupancy'] == shared['waves_per_simd']
        assert result['lds_limit'] == shared['waves_by_lds_term']
        if arch != 'gfx1201':
            assert result['occupancy'] == 0


def test_att_wave64_rdna_model_is_withheld():
    result = detect_arch_and_reg_pressure([], dict(arch='gfx1201', wave_size=64,
        csv_vgpr=64, csv_accum_vgpr=0))
    assert result['occupancy'] is None


def test_att_metadata_requires_an_unambiguous_dispatch(tmp_path):
    from hotspot_analyzer import read_kernel_metadata
    import csv
    fields = ['Kernel_Name', 'Dispatch_Id', 'VGPR_Count', 'Accum_VGPR_Count',
              'SGPR_Count', 'LDS_Block_Size', 'Workgroup_Size_X', 'Workgroup_Size_Y', 'Workgroup_Size_Z']
    with (tmp_path / 'kernel_trace.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows([['kernel', 5, 64, 0, 32, 0, 256, 1, 1],
                         ['kernel', 6, 128, 0, 32, 32768, 256, 1, 1]])
    assert read_kernel_metadata(str(tmp_path), 'kernel') == {}
    dispatch = tmp_path / 'ui_output_agent_0_dispatch_6'
    dispatch.mkdir()
    matched = read_kernel_metadata(str(dispatch), 'kernel')
    assert matched['csv_vgpr'] == 128 and matched['csv_lds'] == 32768
