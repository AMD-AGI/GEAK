#!/usr/bin/env python3
"""Check an installed calculator by running it; never install or fetch at query time."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'dependencies/matrix-calculator.json'


def interpreter(calc):
    if os.environ.get('GEAK_MATRIX_PYTHON'):
        return os.environ['GEAK_MATRIX_PYTHON']
    managed = Path(calc).parent.parent / 'matrix-venv/bin/python'
    return str(managed) if managed.is_file() else sys.executable


def inspect(calc=None):
    from amd_targets import calculator_path
    calc = Path(calc) if calc else calculator_path()
    result = {'status': 'absent', 'path': str(calc) if calc else None,
              'remedy': 'See kernel_tools/dependencies.md; prepare_matrix.py is an explicit setup command.'}
    if not calc or not calc.is_file():
        return result
    spec = json.loads(MANIFEST.read_text())
    result.update(python=interpreter(calc), sha256=hashlib.sha256(calc.read_bytes()).hexdigest())
    result['pinned_source'] = result['sha256'] == spec['files']['matrix_calculator.py']
    # Exercise imports AND target support; module imports alone miss stale calculators.
    checks = []
    try:
        for args in (['--version'], ['-a', 'cdna3', '-i', 'v_mfma_f32_16x16x16_f16', '-d'],
                     ['-a', 'rdna4', '-i', 'v_wmma_f32_16x16x16_f16', '-d']):
            proc = subprocess.run([result['python'], str(calc), *args], check=False, capture_output=True,
                                  text=True, timeout=15)
            checks.append({'argv': args, 'exit_code': proc.returncode,
                           'output': (proc.stdout + proc.stderr)[-3000:]})
            if proc.returncode or (args[0] != '--version' and 'Matrix Dimensions' not in proc.stdout):
                result.update(status='unusable', checks=checks)
                return result
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.update(status='unusable', error=str(exc), checks=checks)
        return result
    result.update(status='ready', version=checks[0]['output'].strip(), checks=checks,
                  validation='CPU CLI only; no GPU numerical validation')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--path', type=Path)
    p.add_argument('--python', action='store_true', help='print the selected calculator interpreter')
    args = p.parse_args()
    if args.python:
        from amd_targets import calculator_path
        calc = args.path or calculator_path()
        if not calc:
            p.error('calculator not found')
        print(interpreter(calc))
        return 0
    result = inspect(args.path)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'ready' else 3


if __name__ == '__main__':
    raise SystemExit(main())
