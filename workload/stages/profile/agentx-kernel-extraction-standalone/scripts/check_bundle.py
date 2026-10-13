#!/usr/bin/env python3
"""Verify standalone skill files; optionally run CPU-only regression tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def integrity():
    count = 0
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = (ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'invalid/missing bundle path: {relative}')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'checksum mismatch: {relative}')
        count += 1
    for md in [ROOT/'SKILL.md', *sorted((ROOT/'references').glob('*.md'))]:
        for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)', md.read_text()):
            if link.startswith(('http:', 'https:', '#')):
                continue
            dest = (md.parent/link.split('#',1)[0]).resolve()
            if not dest.is_relative_to(ROOT) or not dest.is_file():
                raise ValueError(f'unresolved local reference in {md.name}: {link}')
    # Imported JS paths must stay in this distribution. node: built-ins are fine.
    for script in (ROOT/'scripts').glob('*.mjs'):
        for rel in re.findall(r"from\s+['\"]([^'\"]+)['\"]", script.read_text()):
            if rel.startswith('node:'):
                continue
            dest=(script.parent/rel).resolve()
            if not rel.startswith('.') or not dest.is_relative_to(ROOT) or not dest.is_file():
                raise ValueError(f'external JS import: {script.name}: {rel}')
    return count


def self_tests(log_dir=None):
    if not shutil.which('node'):
        raise ValueError('Node.js 18+ is required for source/delivery tests')
    results=[]
    # Run each Python module in its own interpreter: several tests deliberately
    # replace torch/sys.modules with fakes and test sys.path isolation.
    suites=[(p.name,[sys.executable,str(p)]) for p in sorted((ROOT/'tests').glob('test_*.py'))]
    suites.append(('node-delivery', ['node','--test',*[str(p) for p in sorted((ROOT/'tests').glob('test_*.mjs'))]]))
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','NODE_PATH')}
    env.update(PYTHONPATH=str(ROOT/'scripts'),PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory(prefix='standalone skill cpu ') as tmp:
        for name,cmd in suites:
            proc=subprocess.run(cmd,cwd=tmp,env=env,capture_output=True,text=True,timeout=60)
            output=proc.stdout+'\n'+proc.stderr
            n=re.search(r'Ran (\d+) tests?',output)
            if not n:
                n=re.search(r'# tests (\d+)',output)
            row={'suite':name,'exit_code':proc.returncode,'passed':proc.returncode==0,
                 'tests':int(n.group(1)) if n else None}
            if log_dir:
                log_dir.mkdir(parents=True,exist_ok=True)
                (log_dir/(name+'.log')).write_text(output)
            if proc.returncode:
                row['failure_tail']=output[-4000:]
            results.append(row)
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--log-dir',type=Path,help='optional external test log directory')
    args=ap.parse_args()
    if sys.version_info < (3,10):
        raise ValueError('Python 3.10+ required')
    if args.log_dir:
        args.log_dir=args.log_dir.resolve()
        if args.log_dir.is_relative_to(ROOT):
            raise ValueError('put test logs outside the checksummed skill folder')
    report={'bundle_verified':True,'files_verified':integrity(),
            'external_repository_required':False,'gpu_executed':False}
    if args.self_test:
        report['cpu_suites']=self_tests(args.log_dir)
        report['cpu_passed']=all(r['passed'] for r in report['cpu_suites'])
    print(json.dumps(report,indent=2))
    return 0 if report.get('cpu_passed',True) else 1


if __name__=='__main__':
    try:
        sys.exit(main())
    except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
        print(json.dumps({'ok':False,'error':str(exc)}),file=sys.stderr)
        sys.exit(2)
