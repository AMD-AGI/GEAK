#!/usr/bin/env python3
"""Collect a rocprof leg against the original requested-case manifest."""
import argparse
import json
from pathlib import Path
from fold import with_manifest, compare_cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--cases',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--reference',choices=['empirical','datasheet'],default='empirical')
    args=p.parse_args()
    cases=json.loads(args.cases.read_text())['cases']
    arms={'baseline':[],'candidate':[]}
    for path in sorted(args.root.glob('*_roofline.json')):
        data=json.loads(path.read_text())
        if data.get('scope')!='measured-window':
            continue
        row=data.get('rows_by_reference',{}).get(args.reference)
        if row and row.get('arm') in arms:
            arms[row['arm']].append(row)
    args.output.mkdir(parents=True,exist_ok=True)
    for arm,rows in arms.items():
        arms[arm]=with_manifest(rows,cases)
        (args.output/f'rows_{arm}.json').write_text(json.dumps({'rows':arms[arm]},indent=2)+'\n')
    comparison=compare_cases(arms['baseline'],arms['candidate'])
    (args.output/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
