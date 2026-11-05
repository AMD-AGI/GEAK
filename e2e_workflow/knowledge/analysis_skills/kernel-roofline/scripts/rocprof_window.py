"""Select production dispatches between per-call GPU markers, in every pass."""
import csv
import json
from collections import Counter
from pathlib import Path

BEGIN = '_krf_begin_marker'
END = '_krf_end_marker'


def select_dispatch_windows(profile_path, expected_calls, expected_dispatches=None, conditioning_kernel=None):
    if expected_calls <= 0:
        raise ValueError('expected_calls must be positive')
    passes = []
    for path in sorted(Path(profile_path).rglob('pmc_perf*.csv')):
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if not rows or not {'Dispatch_ID', 'Kernel_Name'} <= rows[0].keys():
            continue
        records = {}
        durations = {}
        devices = {r.get('GPU_ID', r.get('Agent_Id', '0')) for r in rows}
        if len(devices) != 1:
            raise ValueError('window selection requires exactly one GPU per command')
        for row in rows:
            identifier = int(row['Dispatch_ID'])
            if identifier in records:
                raise ValueError(f'repeated dispatch ID in {path}; separate counter passes before selection')
            records[identifier] = row['Kernel_Name']
            if row.get('Start_Timestamp') and row.get('End_Timestamp'):
                elapsed=(float(row['End_Timestamp'])-float(row['Start_Timestamp']))/1000
                if elapsed < 0:
                    raise ValueError('negative dispatch timestamp span')
                durations[identifier]=elapsed
        windows, current, selected = [], None, []
        audit = []
        conditioned=False
        for identifier, name in sorted(records.items()):
            reason = 'outside_measurement'
            if conditioning_kernel and conditioning_kernel in name:
                conditioned=True
            if BEGIN in name:
                if current is not None:
                    raise ValueError('nested measurement markers')
                current = []
                if conditioning_kernel and not conditioned:
                    raise ValueError('CACHE_DISPATCH_MISSING before measured call')
                conditioned=False
                reason = 'begin_marker'
            elif END in name:
                if current is None:
                    raise ValueError('end marker without begin')
                if not current:
                    raise ValueError('empty measurement window')
                windows.append(current)
                current = None
                reason = 'end_marker'
            elif current is not None:
                if '_krf_' in name:
                    raise ValueError('conditioning/probe kernel inside measurement window')
                current.append((identifier, name))
                selected.append(identifier)
                reason = 'counted'
            audit.append({'dispatch_id': identifier, 'kernel': name, 'reason': reason,'duration_us':durations.get(identifier)})
        if current is not None or len(windows) != expected_calls:
            raise ValueError(f'DISPATCH_COUNT_MISMATCH: {path}: expected {expected_calls} complete calls, got {len(windows)}')
        fingerprints = [Counter(name for _, name in window) for window in windows]
        expected = Counter(expected_dispatches) if expected_dispatches is not None else fingerprints[0]
        if any(fingerprint != expected for fingerprint in fingerprints):
            raise ValueError(f'DISPATCH_COUNT_MISMATCH: {path}: per-call kernel multiplicity differs')
        passes.append({'file': str(path), 'selected_ids': selected, 'per_call_dispatches': dict(expected),
                       'call_count': len(windows), 'audit': audit})
    if not passes:
        raise ValueError('no pmc_perf CSV with dispatch IDs and kernel names')
    expected_passes = {p.stem for p in (Path(profile_path)/'perfmon').glob('pmc_perf_*.txt')}
    retained_passes = {Path(p['file']).stem for p in passes}
    missing_passes = sorted(expected_passes - retained_passes)
    if missing_passes:
        raise ValueError('COUNTER_PASSES_MISSING: retain individual counter CSVs (rocprof-compute 3.4: -V): ' + ', '.join(missing_passes))
    # analyze -d applies one ID list across counter passes. Do not silently use it
    # when a tool/version remaps IDs differently in a later pass.
    first = passes[0]
    if any(p['selected_ids'] != first['selected_ids'] or p['per_call_dispatches'] != first['per_call_dispatches'] for p in passes[1:]):
        raise ValueError('COUNTER_PASS_WINDOW_MISMATCH: separate pass selection is required')
    return {'selected_ids': first['selected_ids'], 'passes': passes,
            'expected_calls': expected_calls, 'per_call_dispatches': first['per_call_dispatches'],
            'scope': 'production_dispatches_only'}


def save_window_audit(audit, path):
    Path(path).write_text(json.dumps(audit, indent=2) + '\n')
