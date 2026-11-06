#!/usr/bin/env python3
"""Compatibility CLI for matrix facts; shared dispatch lives in amd_isa.py."""
import sys
from amd_targets import COMMANDS as GFX950_SUBCOMMANDS
from amd_isa import main as query
from _amd_matrix import _elems_per_lane


def main(argv):
    if len(argv) == 2 and argv[1] == '--selftest':
        assert _elems_per_lane(4, 'BF16') == 8
        print('[layout_facts] SELFTEST PASS')
        return 0
    if len(argv) < 2 or argv[1] in ('--help', '-h', 'help'):
        print('Usage: layout_facts.py <arch> <instruction> [--layout] | <query> [args]')
        return 0 if len(argv) > 1 else 2
    arch, args = argv[1], argv[2:]
    if not args:
        return query(['--arch', arch])
    if args[0] in ('--list', '-L'):
        args[0] = 'list'
    if args[0] in GFX950_SUBCOMMANDS:
        return query(['--arch', arch, *args])
    want_layout = '--layout' in args
    instructions = [a for a in args if a != '--layout']
    for instruction in instructions:
        rc = query(['--arch', arch, 'facts', instruction])
        if rc:
            return rc
        if want_layout:
            rc = query(['--arch', arch, 'layout', instruction, '-A'])
            if rc:
                return rc
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
