#!/usr/bin/env python3
"""Validate profiler evidence, without interpreting bottlenecks or selecting optimizations.

Exit 0: matching usable evidence; 5: partial evidence; 4: unavailable evidence.
The caller must also check the profiler exit status. Files from a failed process
are never sufficient to claim a successful collection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def active_files(root, pattern):
    for path in sorted(root.rglob(pattern)):
        if not any('.old_' in part for part in path.relative_to(root).parts):
            yield path


def csv_rows(root, pattern):
    for path in active_files(root, pattern):
        try:
            with path.open() as stream:
                yield from csv.DictReader(stream)
        except (OSError, csv.Error, UnicodeError):
            continue


def inspect(root, mode, kernel, counters=()):
    root = Path(root)
    pattern = re.compile(kernel)
    result = {'state': 'unavailable', 'reason': 'no_matching_artifacts', 'matched_rows': 0}

    def matches(row):
        return bool(pattern.search(row.get('Kernel_Name') or row.get('kernel_name') or ''))

    if mode == 'pmc':
        present = set()
        for row in csv_rows(root, '*counter_collection.csv'):
            if matches(row) and number(row.get('Counter_Value')) is not None and row.get('Counter_Name'):
                present.add(row['Counter_Name'])
                result['matched_rows'] += 1
        missing = sorted(set(counters) - present)
        result.update(counter_names=sorted(present), missing_counters=missing)
        if present:
            result.update(state='partial' if missing else 'ok',
                          reason='missing_counters' if missing else 'matching_counters')
        else:
            result['reason'] = 'no_matching_counters'
        return result

    rows = [row for row in csv_rows(root, '*kernel_trace.csv') if matches(row)]
    if mode == 'trace':
        for row in rows:
            start, end = number(row.get('Start_Timestamp')), number(row.get('End_Timestamp'))
            if start is not None and end is not None and end > start:
                result['matched_rows'] += 1
        result.update(state='ok' if result['matched_rows'] else 'unavailable',
                      reason='matching_dispatches' if result['matched_rows'] else 'no_matching_dispatches')
        return result

    # Decoded ATT can contain a code table without a usable wave (including code:null).
    # Keep that distinction and never equate a directory with successful decoding.
    null_seen = False
    candidates = []
    for code_path in active_files(root, 'code.json'):
        try:
            code = json.loads(code_path.read_text()).get('code')
        except (OSError, ValueError, AttributeError):
            continue
        if code is None:
            null_seen = True
        if not isinstance(code, list) or not any(isinstance(row, list) and row and
                                                isinstance(row[0], str) for row in code):
            continue
        waves = 0
        for wave_path in code_path.parent.glob('se*_wv*.json'):
            try:
                wave = json.loads(wave_path.read_text()).get('wave', {})
                events = wave.get('instructions', [])
                if any(isinstance(event, list) and len(event) >= 5 and
                       number(event[0]) is not None and isinstance(event[4], int) and
                       0 <= event[4] < len(code) for event in events):
                    waves += 1
            except (OSError, ValueError, AttributeError):
                continue
        dispatch = re.search(r'dispatch_(\d+)', code_path.parent.name)
        identity = [row for row in rows if not dispatch or
                    str(row.get('Dispatch_Id') or row.get('Dispatch_ID') or '') == dispatch[1]]
        state = 'ok' if waves and identity else 'partial'
        reason = 'decoded_dispatch' if state == 'ok' else (
            'no_decoded_wave' if not waves else 'kernel_identity_unverified')
        candidates.append(dict(state=state, reason=reason, matched_rows=len(identity),
                               decoded_waves=waves, att_dispatch=str(code_path.parent)))
    if candidates:
        return next((item for item in candidates if item['state'] == 'ok'), candidates[0])
    result['reason'] = 'code_null' if null_seen else 'no_decoded_code'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=['trace', 'pmc', 'att', 'att-wide'], required=True)
    parser.add_argument('--kernel', required=True)
    parser.add_argument('--pmc', default='')
    args = parser.parse_args()
    try:
        result = inspect(args.out, args.mode, args.kernel, args.pmc.split())
    except (OSError, ValueError, re.error) as exc:
        result = {'state': 'unavailable', 'reason': f'invalid_artifacts: {exc}', 'matched_rows': 0}
    print(json.dumps(result))
    return {'ok': 0, 'partial': 5, 'unavailable': 4}[result['state']]


if __name__ == '__main__':
    raise SystemExit(main())
