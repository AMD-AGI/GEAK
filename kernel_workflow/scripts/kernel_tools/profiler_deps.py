#!/usr/bin/env python3
"""Inspect profiler CLI/decoder dependencies locally, without running GPU work or installing packages."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def tool(name):
    path = shutil.which(name)
    if not path:
        return {'status': 'absent', 'path': None, 'help': ''}
    result = {'status': 'broken', 'path': path, 'help': ''}
    try:
        for option, key in (('--version', 'version'), ('--help', 'help')):
            p = subprocess.run([path, option], capture_output=True, text=True, timeout=15)
            output = p.stdout + p.stderr
            result[key] = output if key == 'help' else output[-3000:]
            if p.returncode:
                result['error'] = f'{option} exited {p.returncode}'
                return result
    except (OSError, subprocess.TimeoutExpired) as exc:
        result['error'] = str(exc)
        return result
    result['status'] = 'cli_available'
    return result


def decoder_candidates():
    roots = []
    for value in (os.environ.get('ROCPROF_ATT_LIBRARY_PATH', ''),
                  os.environ.get('LD_LIBRARY_PATH', ''),
                  os.environ.get('KT_ATT_SYSTEM_DIRS',
                      f"{os.environ.get('ROCM_PATH', '/opt/rocm')}/lib:/opt/rocm/lib:/opt/rocm/lib64")):
        roots.extend(Path(x) for x in value.split(':') if x)
    seen = set()
    for root in roots:
        for pattern in ('librocprof-trace-decoder*.so*', 'libatt_decoder_trace*.so*'):
            for candidate in sorted(root.glob(pattern)):
                if candidate.is_file() and candidate not in seen:
                    seen.add(candidate)
                    yield candidate


def att_probe(sdk):
    result = {'status': 'no_rocprofv3', 'directory': None,
              'gpu_validation': 'not_attempted', 'libraries': []}
    if sdk['status'] != 'cli_available':
        return result
    help_text = sdk['help']
    if not re.search(r'--(?:att|advanced-thread-trace)(?:\s|[=,])', help_text):
        result['status'] = 'unsupported_cli'
        return result
    if '--att-library-path' not in help_text:
        # New SDKs may link/manage the decoder. A missing separate .so does not
        # establish failure, and CLI support does not establish GPU readiness.
        result['status'] = 'sdk_managed_unverified'
        return result
    result['status'] = 'decoder_absent'
    for candidate in decoder_candidates():
        entry = {'path': str(candidate)}
        try:
            library = ctypes.CDLL(str(candidate))
            entry['loadable'] = True
            entry['v1'] = hasattr(library, 'rocprof_trace_decoder_parse_data')
            entry['v2'] = hasattr(library, 'rocprof_trace_decoder_create_handle')
            if entry['v1']:
                result.update(status='external_candidate', directory=str(candidate.parent))
                result['libraries'].append(entry)
                return result
            entry['error'] = 'legacy --att-library-path requires the V1 parse_data API'
        except OSError as exc:
            entry.update(loadable=False, error=str(exc))
        result['libraries'].append(entry)
        result['status'] = 'decoder_incompatible'
    return result


def report(att_only=False):
    sdk = tool('rocprofv3')
    result = {'schema': 'geak.profiler-dependencies.v1', 'rocprofv3': sdk,
              'att': att_probe(sdk), 'gpu_validation': 'not_attempted'}
    if not att_only:
        result['rocprof_compute'] = tool('rocprof-compute')
        result['capabilities'] = {
            'trace': sdk['status'] == 'cli_available' and '--kernel-trace' in sdk['help'],
            'pmc': sdk['status'] == 'cli_available' and '--pmc' in sdk['help'],
            'compute': result['rocprof_compute']['status'] == 'cli_available',
        }
    # Preserve versions and errors, not the entire help banner in every artifact.
    for key in ('rocprofv3', 'rocprof_compute'):
        if key in result:
            result[key].pop('help', None)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--att-only', action='store_true')
    p.add_argument('--decoder-path', action='store_true', help='legacy file locator; does not validate ABI')
    args = p.parse_args()
    if args.decoder_path:
        first = next(decoder_candidates(), None)
        if first:
            print(first.parent)
            return 0
        return 1
    print(json.dumps(report(args.att_only), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
