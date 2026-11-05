"""Read actual reference files; report provenance and calibration suspicions."""
import csv
import hashlib
import json
import math
from pathlib import Path


def provenance(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def audit_rocprof_peaks(path, arch):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('empty roofline.csv')
    warnings = []
    for row in rows:
        bw = float(row['HBMBw'])
        if not math.isfinite(bw) or bw <= 0:
            raise ValueError('invalid HBM reference')
        if arch.startswith('gfx95'):
            # Calibration checks, not automatic replacements or proof of error.
            normalized = {}
            for key, factor in [('MFMABF16Flops',1),('MFMAF16Flops',1),('MFMAF8Flops',2),('MFMAF6F4Flops',4)]:
                if row.get(key):
                    value = float(row[key])
                    if value > 0 and math.isfinite(value):
                        normalized[key] = value / factor
            if normalized:
                best = max(normalized.values())
                warnings.extend(f"device={row.get('device')}: {key} is below 90% of scaled sibling reference"
                                for key, value in normalized.items() if value < .9 * best)
    return {**provenance(path), 'arch': arch, 'rows': rows, 'warnings': warnings}


def audit_tl_arch(path, compute_specs=(), rocprof_csv=None):
    config = json.loads(Path(path).read_text())
    peaks = config.get('max_achievable_tflops', {})
    bw = config.get('mem_bw_gbps')
    if not isinstance(bw, (int,float)) or not math.isfinite(bw) or bw <= 0:
        raise ValueError('TL arch needs a positive mem_bw_gbps')
    result = {**provenance(path), 'source': config.get('source', 'unverified'),
        'mem_bw_gbps': bw, 'selected_specs': {s: peaks.get(s) for s in compute_specs if s},
        'missing_specs': [s for s in compute_specs if s and not peaks.get(s)]}
    if rocprof_csv:
        with Path(rocprof_csv).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        result['rocprof_reference'] = provenance(rocprof_csv)
        mapping = {'matrix_bf16':'MFMABF16Flops','matrix_fp16':'MFMAF16Flops',
                   'matrix_fp8':'MFMAF8Flops','matrix_fp4':'MFMAF6F4Flops'}
        result['tl_over_rocprof_peak_ratios'] = []
        for row in rows:
            ratios = {'device':row.get('device'), 'hbm':bw/float(row['HBMBw'])}
            for key, column in mapping.items():
                if peaks.get(key) and row.get(column) and float(row[column]) > 0:
                    ratios[key] = peaks[key] * 1000 / float(row[column])
            result['tl_over_rocprof_peak_ratios'].append(ratios)
    return result
