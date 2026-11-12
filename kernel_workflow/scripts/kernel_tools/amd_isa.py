#!/usr/bin/env python3
"""Shared AMD ISA queries for Triton, Gluon and HIP; --arch is required.

Examples: --arch gfx942 facts v_mfma_f32_16x16x16_f16
          --arch gfx1201 layout v_wmma_f32_16x16x16_f16 -A
          --arch gfx950 capabilities --json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from amd_targets import COMMANDS, MATRIX_COMMANDS, TARGETS, canonical, capabilities

HERE = Path(__file__).resolve().parent


def main(argv=None, default_arch=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if '--selftest' in args:
        assert canonical('gfx942') == 'cdna3' and canonical('gfx1201') == 'rdna4'
        assert capabilities('gfx1201')['queries']['waitcnt']['status'] == 'not_extracted'
        print('[amd_isa] SELFTEST PASS')
        return 0
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--arch', default=default_arch, required=default_arch is None)
    parser.add_argument('--json', action='store_true', help='machine-readable capabilities')
    opts, rest = parser.parse_known_args(args)
    try:
        key = canonical(opts.arch)
        if not rest:
            parser.error('a query is required: capabilities, ' + ', '.join(sorted(COMMANDS)))
        query, tail = rest[0], rest[1:]
        if query == 'capabilities':
            data = capabilities(opts.arch)
            if opts.json:
                print(json.dumps(data, indent=2))
            else:
                print(f"target={opts.arch} ISA family={key}")
                for name, info in data['queries'].items():
                    print(f"{name:12} {info['status']:20} {info['source'] or '-'}")
            return 0
        if opts.json:
            parser.error('--json currently applies to capabilities only')
        if query not in COMMANDS:
            parser.error(f'unknown query {query!r}')
        if query in MATRIX_COMMANDS and TARGETS[key]['calculator']:
            return calculator_query(key, query, tail)
        if key not in ('gfx950', 'cdna3', 'rdna3', 'rdna4'):
            raise ValueError(f'{query} data not extracted for {opts.arch}')
        return subprocess.call([sys.executable, str(HERE / '_amd_isa_db.py'), '--arch', key, query, *tail])
    except ValueError as exc:
        print(f'SCOPED TOOL-GAP: {exc}', file=sys.stderr)
        return 3


def calculator_query(arch, query, args):
    from _amd_matrix import _calc_path, _run, facts
    calc = _calc_path()
    if query == 'facts':
        if not args or any(arg.startswith('-') for arg in args):
            raise ValueError('facts requires instruction names, without extra flags')
        for instruction in args:
            facts(calc, arch, instruction, False)
        return 0
    if query == 'list':
        rc, out, err = _run(calc, '-a', arch, '-L')
        if rc:
            raise ValueError(err.strip() or 'calculator list failed')
        print('\n'.join(line for line in out.splitlines() if not args or args[0].lower() in line.lower()))
        return 0
    p = argparse.ArgumentParser(prog=f'amd_isa {query}')
    p.add_argument('instruction')
    if query == 'locate':
        p.add_argument('operand', choices=list('ABCD'))
        p.add_argument('row', type=int)
        p.add_argument('column', type=int)
    else:
        group = p.add_mutually_exclusive_group()
        for operand in 'ABCD':
            group.add_argument('-' + operand, dest='operand', action='store_const', const=operand)
        group.add_argument('--operand', choices=list('ABCD'))
        p.set_defaults(operand='A')
    p.add_argument('--block', type=int, default=0)
    p.add_argument('--markdown', action='store_true')
    p.add_argument('--wave-size', choices=['32', '64'])
    opts = p.parse_args(args)
    command = ['-a', arch, '-i', opts.instruction, '-' + opts.operand]
    if query == 'layout':
        if opts.block != 0:
            raise ValueError('calculator layout prints all blocks; use locate --block for a single block')
        command += ['-R', '--markdown']
    else:
        coordinates = {'A': ('-I', '-K'), 'B': ('-K', '-J'), 'C': ('-I', '-J'), 'D': ('-I', '-J')}
        x, y = coordinates[opts.operand]
        command += ['-g', x, str(opts.row), y, str(opts.column), '-b', str(opts.block)]
    if opts.wave_size:
        command += ['-w', opts.wave_size]
    rc, out, err = _run(calc, *command)
    if rc:
        raise ValueError(err.strip() or f'calculator {query} failed')
    print(out, end='')
    print(f'# provenance: AMD Matrix Instruction Calculator {arch} {opts.instruction}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
