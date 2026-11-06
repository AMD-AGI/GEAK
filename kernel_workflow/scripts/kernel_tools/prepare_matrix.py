#!/usr/bin/env python3
"""Explicitly prepare a pinned CPU calculator in a new prefix; never modifies workload Python."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import venv

from matrix_dependency import MANIFEST


def run(*command):
    subprocess.run([str(x) for x in command], check=True)


def prepare(prefix, source=None, wheelhouse=None, offline=False):
    if sys.version_info < (3, 10):
        raise ValueError('GEAK hardware query setup requires Python 3.10 or newer')
    spec = json.loads(MANIFEST.read_text())
    prefix = prefix.expanduser().resolve()
    if offline and (not source or not wheelhouse):
        raise ValueError('--offline requires --source (local git checkout) and --wheelhouse')
    if source and not (source / '.git').exists():
        raise ValueError('--source must be a local git checkout containing the pinned commit')
    if wheelhouse and not wheelhouse.is_dir():
        raise ValueError('--wheelhouse must be an existing directory')
    # Never overwrite an existing install, partial setup or user-managed tree.
    prefix.mkdir(parents=True, exist_ok=False)
    receipt = {'schema': 'geak.tool-install.v1', 'status': 'incomplete', 'dependency': spec,
               'prefix': str(prefix), 'offline': offline}
    record = prefix / 'matrix-install.json'
    record.write_text(json.dumps(receipt, indent=2) + '\n')
    calc = prefix / spec['name']
    try:
        run('git', 'clone', '--no-hardlinks', '--no-checkout', '--',
            str(source.resolve()) if source else spec['repository'], calc)
        run('git', '-C', calc, 'checkout', '--detach', spec['commit'])
        for name, expected in spec['files'].items():
            if hashlib.sha256((calc / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'pinned source checksum mismatch: {name}')
        # Use the caller's pip only as an installer into this new environment. This
        # also works on distributions whose system Python omits ensurepip.
        environment = prefix / 'matrix-venv'
        venv.EnvBuilder(with_pip=False).create(environment)
        python = environment / 'bin/python'
        site = subprocess.check_output([str(python), '-c',
            'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip()
        command = [sys.executable, '-m', 'pip', '--disable-pip-version-check', 'install',
                   '--target', site, '--require-hashes', '--only-binary=:all:',
                   '-r', str(MANIFEST.parent / spec['requirements'])]
        if wheelhouse:
            command += ['--find-links', str(wheelhouse.resolve())]
        if offline:
            command += ['--no-index']
        run(*command)
        # Ignore caller interpreter overrides while validating the newly prepared env.
        result = subprocess.run([str(python), str(MANIFEST.parent.parent / 'matrix_dependency.py'),
                                 '--path', str(calc / 'matrix_calculator.py')],
                                capture_output=True, text=True, env=clean_environment(), check=True)
        receipt.update(status='ready', check=json.loads(result.stdout))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        receipt.update(status='failed', error=str(exc))
        raise
    finally:
        record.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def clean_environment():
    import os
    env = dict(os.environ)
    for name in ('GEAK_MATRIX_PYTHON', 'PYTHONPATH', 'PYTHONHOME'):
        env.pop(name, None)
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', required=True, type=Path)
    p.add_argument('--source', type=Path, help='use a local git checkout instead of the network')
    p.add_argument('--wheelhouse', type=Path)
    p.add_argument('--offline', action='store_true')
    args = p.parse_args()
    try:
        result = prepare(args.prefix, args.source, args.wheelhouse, args.offline)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'prepare_matrix: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
