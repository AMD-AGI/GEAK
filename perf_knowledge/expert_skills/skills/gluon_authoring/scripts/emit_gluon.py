#!/usr/bin/env python3
"""Gluon-specific transcription after the shared IR capture has succeeded."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['layouts', 'anchor', 'pipeline'], required=True)
    p.add_argument('--dir', type=Path, required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--arch', required=True)
    p.add_argument('--kernel', default='')
    a = p.parse_args()
    recover = Path(__file__).with_name('recover_gluon.py')
    ttgir = a.dir / (a.variant + '.ttgir')
    command = [sys.executable, str(recover), '--ttgir', str(ttgir),
               '--out', str(a.dir / (a.variant + '.gluon.py')), '--arch', a.arch]
    if a.mode != 'layouts':
        command += ['--with-skeleton']
        if a.kernel:
            command += ['--kernel', a.kernel]
    if a.mode == 'pipeline':
        command += ['--with-pipeline']
    rc = subprocess.call(command)
    if rc:
        return rc
    with (a.dir / (a.variant + '.transcribe_record.txt')).open('w') as record:
        return subprocess.call([sys.executable, str(recover), '--record', '--ttgir', str(ttgir)], stdout=record)


if __name__ == '__main__':
    raise SystemExit(main())
