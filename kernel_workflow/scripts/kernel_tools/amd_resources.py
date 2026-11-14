"""One AMD compilation resource reader for shared tools and language-specific callers.

All values are static compiler facts or model bounds, not achieved occupancy.
Launch geometry is never inferred from max_flat_workgroup_size. Missing fields
stay None; a numeric zero is retained. Sources accompany the parsed values.
"""
from __future__ import annotations

import re

import amd_occupancy as occupancy
from amd_artifacts import assembly_kernel_name


def _integer(text, pattern):
    match = re.search(pattern, text, re.MULTILINE)
    return int(match.group(1)) if match else None


def _metadata(text, name):
    return _integer(text, rf"^\s*(?:-\s*)?\.{name}:\s*(\d+)")


def analyze(text, *, arch=None, vgpr=None, lds_bytes=None, workgroup_size=None,
            num_warps=None, lds_per_cu=None):
    """Parse ONE kernel's assembly and compose register/LDS bounds.

    An explicit vgpr is already a combined allocation. Explicit LDS/launch inputs
    must come from this compilation's metadata or its caller's launch record.
    Multiple kernels in one assembly are rejected rather than mixed by regex.
    """
    kernel_name = assembly_kernel_name(text)
    observed_arch = occupancy.arch_from_asm(text)
    arch = observed_arch or arch
    model = occupancy.model_for(arch)
    sources = {'arch': 'assembly' if observed_arch else 'caller' if arch else 'missing'}
    counts = [
        (vgpr, 'caller combined allocation'),
        (_integer(text, r'^\s*\.amdhsa_next_free_vgpr\s+(\d+)'), 'kernel descriptor'),
        (_integer(text, r'^;\s*TotalNumVgprs:\s*(\d+)'), 'LLVM TotalNumVgprs'),
        (_metadata(text, 'vgpr_count'), 'AMDHSA combined vgpr_count'),
    ]
    combined, sources['vgpr_combined'] = next(((n, src) for n, src in counts if n is not None),
                                             (None, 'missing'))
    arch_vgpr = _integer(text, r'^;\s*NumVgprs:\s*(\d+)')
    if arch_vgpr is None:
        arch_vgpr = _integer(text, r'^\s*\.set\s+\S*num_vgpr,\s*(\d+)')
    agpr = _metadata(text, 'agpr_count')
    if agpr is None:
        agpr = _integer(text, r'^;\s*NumAgprs:\s*(\d+)')
    if arch_vgpr is None and agpr == 0:
        arch_vgpr = combined
    sources['vgpr_arch'] = 'LLVM architectural count' if arch_vgpr is not None else 'missing'
    wave_flag = _integer(text, r'^\s*\.amdhsa_wavefront_size32\s+([01])\s*$')
    wave = 32 if wave_flag == 1 else 64 if wave_flag == 0 else _metadata(text, 'wavefront_size')
    sources['wave_size'] = 'assembly' if wave is not None else 'architecture default'
    if wave is None and model:
        wave = model['wave_size']
    supported_wave = bool(model and wave == model['wave_size'])
    bound = _metadata(text, 'max_flat_workgroup_size')
    sources['workgroup_size'] = 'caller launch' if workgroup_size is not None else 'missing'
    if workgroup_size is None and num_warps is not None and wave:
        workgroup_size = num_warps * wave
        sources['workgroup_size'] = 'matched num_warps x wave size'
    if workgroup_size is not None and workgroup_size <= 0:
        raise ValueError('actual workgroup size must be positive')
    if lds_bytes is None:
        static_lds, field = occupancy.lds_bytes_per_wg_from_asm(text)
        # Zero in the KD does not account for Triton's launch-sized LDS.
        lds_bytes = static_lds if static_lds else None
        sources['lds_bytes'] = field if lds_bytes is not None else 'missing launch LDS'
    else:
        sources['lds_bytes'] = 'caller / matched compilation metadata'
    if lds_bytes is not None and lds_bytes < 0:
        raise ValueError('LDS allocation must be nonnegative')
    llvm = occupancy.llvm_occupancy(text)
    if llvm is not None:
        reg, reg_source = llvm, 'llvm-comment'
    elif combined is not None and supported_wave:
        reg, _ = occupancy.waves_by_vgpr(combined, arch)
        reg_source = 'model'
    else:
        reg, reg_source = None, 'unknown'
    if not supported_wave:
        lds, detail = None, 'unknown or unsupported wave/SIMD geometry'
    elif workgroup_size is None and lds_bytes:
        lds, detail = None, 'no actual workgroup size -- compiler max-flat is only an upper bound'
    else:
        lds, detail = occupancy.waves_by_lds(lds_bytes, arch, workgroup_size,
                                           model=model, capacity=lds_per_cu)
    waves = reg if lds is None or reg is None else min(reg, lds)
    source = reg_source + ('+lds' if reg is not None and lds is not None else '')
    warning = None
    if reg is None:
        warning = 'register term WITHHELD: missing registers or unsupported target/wave; no default arch'
    elif lds is None:
        warning = f'register term only ({detail})'
    spill = _integer(text, r'^\s*\.amdhsa_private_segment_fixed_size\s+(\d+)')
    if spill is None:
        spill = _metadata(text, 'private_segment_fixed_size')
    return {'arch': arch, 'wave_size': wave, 'vgpr_arch': arch_vgpr, 'agpr': agpr,
            'vgpr_combined': combined, 'spill_bytes': spill, 'kernel_name': kernel_name,
            'wg_size': workgroup_size, 'wg_size_compile_bound': bound,
            'lds_bytes_per_wg': lds_bytes, 'lds_capacity_scope': 'CU' if occupancy.lds_per_cu(arch) else None,
            'waves_by_vgpr_term': reg, 'waves_by_lds_term': lds, 'waves_per_simd': waves,
            'waves_source': source, 'waves_warning': warning, 'lds_detail': detail,
            'sources': sources}
